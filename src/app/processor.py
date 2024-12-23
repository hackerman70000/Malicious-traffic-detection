
from pathlib import Path
from typing import Iterable, Optional
from nfstream import NFStreamer, NFPlugin

from src.app.plugins import Plugins


class TrafficProcessor():
    streamer: NFStreamer

    plugins: Plugins
    def __init__(self, source: Path | str, plugins: Optional[Iterable[NFPlugin]] = None, plugin_dirs: Optional[list[Path]] = None):
        self.plugins = Plugins(plugins)
        self.plugins.load_plugin_dir()
        for plugin_dir in plugin_dirs or []:
            self.plugins.load_plugin_dir(plugin_dir)
        self.plugins.load_prefixed_plugins()

        self.streamer = NFStreamer(source, statistical_analysis=True, udps=self.plugins)
    
    def setup_plugins(self):
        self.streamer.udps = self.plugins

    def process(self):
        # main loop I guess implement processing
        pass
    