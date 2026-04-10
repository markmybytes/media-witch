"""Main CLI entry point."""

from __future__ import annotations

import click

from .nfo import nfo_command
from .organize import organize_command
from .subtitles import subtitles_command
from .torrent import torrent_command


@click.group()
@click.version_option(version="2.0.0", prog_name="media-witch")
def cli() -> None:
    """Media Witch - A modular CLI toolkit for media file organization."""
    pass


# Register subcommands
cli.add_command(organize_command)
cli.add_command(nfo_command)
cli.add_command(subtitles_command)
cli.add_command(torrent_command)


if __name__ == "__main__":
    cli()
