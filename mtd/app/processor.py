
import ast
from itertools import chain
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, SpooledTemporaryFile
from typing import Iterable, Optional
from nfstream import NFStreamer, NFPlugin
import pandas as pd

from mtd.app.plugins import Plugins
from mtd.data.detections.sigma import SigmaDetections
from mtd.data.enrichment.geoip import GeoIpEnrichment
from mtd.data.enrichment.greynoise import GreyNoiseEnrichment
from rich_dataframe import prettify
from rich import print
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.panel import Panel
from rich.text import Text
from rich.progress import SpinnerColumn, Progress
import plotext as plt

default_plugins = []


class TrafficProcessor():
    streamer: NFStreamer

    plugins: Plugins
    def __init__(self, source: Path | str, plugins: Optional[Iterable[NFPlugin]] = None, plugin_dirs: Optional[list[Path]] = None, **kwargs):
        self.plugins = Plugins(plugins)
        self.plugins.add_plugins([
            # GreyNoiseEnrichment(greynoise_api_key=kwargs.get("greynoise_api_key")),
            GeoIpEnrichment(),
            SigmaDetections(sigma_paths=kwargs.get("sigma_paths"))
        ])
        self.plugins.load_plugin_dir()
        for plugin_dir in plugin_dirs or []:
            self.plugins.load_plugin_dir(plugin_dir)
        self.plugins.load_prefixed_plugins()

        self.streamer = NFStreamer(source, statistical_analysis=True, udps=self.plugins, )
    def setup_plugins(self):
        self.streamer.udps = self.plugins

    def plot_detections(self, detections):
        def make_plot(width, height, detected, title):
            if (not len(detected)):
                return "no detections"
            bins = pd.cut(list(detected.keys()), bins=round(width / 5))
            detected = pd.Series(detected).groupby(bins, observed=False).sum().to_dict()
            plt.clt()
            plt.clf()
            plt.limit_size(True, True)
            plt.plotsize(width, height)
            plt.bar(list(map(lambda interval: round(interval.mid), detected.keys())), list(detected.values()), reset_ticks=True)
            plt.title(title)
            plt.xlabel("time [s]")
            plt.ylabel("count")
            plt.ylim(0, max(detected.values()) + 1)
            plt.theme("dark")
            return plt.build()
        class plotextMixin(JupyterMixin):
            def __init__(self, detections, title = ""):
                self.decoder = AnsiDecoder()
                self.detected = detections
                self.title = title

            def __rich_console__(self, console, options):
                self.width = options.max_width or console.width
                self.height = options.height or console.height
                canvas = make_plot(self.width, self.height, self.detected, self.title)
                self.rich_canvas = Group(*self.decoder.decode(canvas))
                yield self.rich_canvas
        return plotextMixin(detections, title="detections")
        
        

    def process(self):
        layout = Layout(name="plot", ratio=1)
        layout.split(
            Layout(name="main", ratio=5),
            Layout(name="bottom", ratio=1)
        )
        detections = {}
        progress = Progress("processing", SpinnerColumn(finished_text="✅"))
        processing_task = progress.add_task("processing flows", total=1)
        layout["bottom"].update(progress)
        with Live(layout, refresh_per_second=4) as live:
            for flow in self.streamer:
                # self.report(pd.DataFrame.from_records([flow.values()], columns=flow.keys()))
                timestamp_s = flow.bidirectional_first_seen_ms // 1000
                detections[timestamp_s] =  int(detections.get(timestamp_s, 0) + flow.udps.detections)
                plot = self.plot_detections(detections)
                layout["main"].update(Panel(plot))
                live.refresh()
            progress.advance(processing_task)
            plot = None
            plot = self.plot_detections(detections)
            layout["main"].update(Panel(plot))
        self.report(self.to_pandas())
        
    def report(self, df):
        console = Console()
        df = df.join(pd.json_normalize(df["udps.enrichments"].apply(ast.literal_eval)).add_prefix("udps.enrichments."))
        by_src_ip = df.groupby("src_ip")
        prettify(by_src_ip.agg({"src2dst_packets": "sum", "src2dst_bytes": "sum", "bidirectional_packets": "sum", "bidirectional_bytes": "sum", "src2dst_first_seen_ms": ["min", "max"], "src2dst_last_seen_ms": ["min", "max"], "udps.detections": "sum"}), clear_console=False, delay_time=0.1, row_limit=1024)
        console.rule("[bold yellow]enrichments and detections")

        prettify(df.filter(regex=r'(^udps\.enrichments\.|src_ip$|dst_ip$)', axis=1), col_limit=20, clear_console=False, delay_time=0.1, row_limit=20)

         

    def to_pandas(self):
        """ fixed streamer to pandas function (added escapechar) """
        with NamedTemporaryFile() as temp_file:
            total_flows = self.streamer.to_csv(path=temp_file.name, flows_per_file=0)
            if total_flows > 0:  # If there is flows, return Dataframe else return None.
                df = pd.read_csv(temp_file.name, escapechar='\\')
                if total_flows != df.shape[0]:
                    print("WARNING: {} flows ignored by pandas type conversion. Consider using to_csv() "
                        "method if drops are critical.".format(abs(df.shape[0] - total_flows)))
            else:
                df = None
            return df