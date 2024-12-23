import importlib
import inspect
from pathlib import Path
import pkgutil
from typing import Iterable, Optional
from nfstream import NFPlugin

def nfsplugin_predicate(obj):
    return inspect.isclass(obj) and issubclass(obj, NFPlugin) and obj != NFPlugin

class Plugins:
    plugins: set[NFPlugin]
    def __init__(self, plugins: Optional[Iterable[NFPlugin]] = None):
        self.plugins = set(map(lambda c: c(), set(plugins))) if plugins else set()

    def load_plugin_dir(self, dir: Path):
        dir = dir or Path(__file__).parent.parent / "plugins"
        if dir.is_file():
            try:
                plugin = importlib.util.module_from_spec(importlib.util.spec_from_file_location(dir.stem, dir))
                self.plugins.union(map(lambda c: c(), inspect.getmembers(plugin, nfsplugin_predicate)))
            except Exception as e:
                print(f"Failed to load plugin {dir}: {e}")
        for path in dir.iterdir():
            if path.is_file() and not path.name.startswith((".", "_")) and path.suffix == ".py":
                try:
                    plugin = importlib.util.module_from_spec(importlib.util.spec_from_file_location(path.stem, path))
                    self.plugins.union(map(lambda c: c(), inspect.getmembers(plugin, nfsplugin_predicate)))
                except Exception as e:
                    print(f"Failed to load plugin {path}: {e}")
            elif path.is_dir():
                self.load_plugin_dir(path)
    def load_prefixed_plugins(self, prefix: str):
        new_plugins = [importlib.import_module(name)
            for _, name, _
            in pkgutil.iter_modules()
            if name.startswith(prefix or 'mtd_')
        ]
        self.plugins.union(map(lambda c: c(), inspect.getmembers(new_plugins, nfsplugin_predicate)))
    
    def add_plugins(self, plugins: Iterable[NFPlugin]):
        self.plugins.union(plugins)

    def __iter__(self):
        return iter(self.plugins)
    def __next__(self):
        return next(self.plugins)