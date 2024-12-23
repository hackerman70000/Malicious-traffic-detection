
from itertools import chain
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, SpooledTemporaryFile
from typing import Iterable, Optional
from nfstream import NFStreamer, NFPlugin
import pandas as pd

from src.app.plugins import Plugins
from src.data.detections.sigma import SigmaDetections
from src.data.enrichment.geoip import GeoIpEnrichment
from src.data.enrichment.greynoise import GreyNoiseEnrichment
from rich_dataframe import prettify
from rich import print

default_plugins = []

class TrafficProcessor():
    streamer: NFStreamer

    plugins: Plugins
    def __init__(self, source: Path | str, plugins: Optional[Iterable[NFPlugin]] = None, plugin_dirs: Optional[list[Path]] = None, **kwargs):
        self.plugins = Plugins(plugins)
        self.plugins.add_plugins([
            GreyNoiseEnrichment(greynoise_api_key=kwargs.get("greynoise_api_key")),
            GeoIpEnrichment(),
            SigmaDetections(sigma_paths=kwargs.get("sigma_paths"))
        ])
        self.plugins.load_plugin_dir()
        for plugin_dir in plugin_dirs or []:
            self.plugins.load_plugin_dir(plugin_dir)
        self.plugins.load_prefixed_plugins()

        self.streamer = NFStreamer(source, statistical_analysis=True, udps=self.plugins)
    def setup_plugins(self):
        self.streamer.udps = self.plugins

    def process(self):
        try:
            data = self.to_pandas()
        except Exception as e:
            print(f"Failed to process data: {e}")
            raise e
        print(prettify(data))
    

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