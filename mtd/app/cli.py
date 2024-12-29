from mimetypes import read_mime_types
from pathlib import Path
import socket
from typing import Optional
import pydantic_typer
import typer
from click import Path as ClickPath
from click.shell_completion import CompletionItem
from typing_extensions import Annotated

from mtd.app.processor import TrafficProcessor

app = pydantic_typer.Typer()

class SourceParser(ClickPath):
    name = "fileOrInterface"
    def convert(self, value, param, ctx) -> Path | str:
        if value is None:
            return self.fail("No value provided", param, ctx)
        path = Path(value)
        if path.exists() and "application/vnd.tcpdump.pcap" in read_mime_types(path).values():
            return Path(value)
        names = map(lambda x: x[1], socket.if_nameindex())
        if value not in names:
            return self.fail(f"Value {value} is not a valid interface or pcap file", param, ctx)
        return value
    def shell_complete(self, ctx, param, incomplete):
        return [CompletionItem(name, help=f"Interface {name} at {index}") for index, name in socket.if_nameindex()] + super().shell_complete(ctx, param, incomplete)
    

def complete_intfname(incomplete: str):
    for item in ClickPath().shell_complete(None, None, incomplete):
        yield item.value
    for index, name in socket.if_nameindex():
        if name.startswith(incomplete):
            yield (name, f"interface {name} at index {index}")

@app.command()
def main(
    source: Annotated[Optional[Path | str], typer.Option(click_type=SourceParser)],
    plugins: Optional[list[Path]] = typer.Option(None, help="Directories or files to load plugins from"),
    sigma_paths: Optional[list[Path]] = typer.Option(None, help="Directories or files to load sigma rules from"),
    model_path: Optional[Path] = typer.Option(None, help="Path to the model directory containing model.json and metadata.json"),
    default_plugins: list[str] = typer.Option(["Sigma", "GeoIP", "ML", "GreyNoise"], help="By default all plugins are loaded, change this to load only specific plugins"),
    output: Optional[Path] = typer.Option(None, help="Output file to write detections to"),
   ):
    tp = TrafficProcessor(source, plugin_dirs=plugins, sigma_paths=sigma_paths, model_path=model_path, default_plugins=default_plugins, output=output)
    
    tp.process()



if __name__ == "__main__":
    app()