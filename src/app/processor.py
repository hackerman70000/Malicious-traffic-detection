
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional
from nfstream import NFStreamer, NFPlugin

from src.app.plugins import Plugins
from src.data.enrichment.geoip import GeoIpEnrichment
from src.data.enrichment.greynoise import GreyNoiseEnrichment


default_plugins = []

class TrafficProcessor():
    streamer: NFStreamer

    plugins: Plugins
    def __init__(self, source: Path | str, plugins: Optional[Iterable[NFPlugin]] = None, plugin_dirs: Optional[list[Path]] = None, **kwargs):
        self.plugins = Plugins(plugins)
        self.plugins.add_plugins([
            GreyNoiseEnrichment(greynoise_api_key=kwargs.get("greynoise_api_key")),
            GeoIpEnrichment(),
        ])
        self.plugins.load_plugin_dir()
        for plugin_dir in plugin_dirs or []:
            self.plugins.load_plugin_dir(plugin_dir)
        self.plugins.load_prefixed_plugins()

        self.streamer = NFStreamer(source, statistical_analysis=True, udps=self.plugins)
    
    def setup_plugins(self):
        self.streamer.udps = self.plugins

    def process(self):
        # todo
    