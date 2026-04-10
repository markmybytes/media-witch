"""CLI command for torrent fake file creation."""

from __future__ import annotations

from pathlib import Path

import click

from ..features.torrent.api import TorrentConfig, create_from_torrents
from .common import verbose_option


@click.command(name="torrent")
@click.argument("paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=".",
    help="Output directory for fake files (default: current directory)",
)
@verbose_option
def torrent_command(
    paths: tuple[str, ...],
    output_dir: str,
    verbose: bool,
) -> None:
    """Create fake file structure from .torrent files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect torrent files
    torrent_files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".torrent":
            torrent_files.append(path)
        elif path.is_dir():
            torrent_files.extend(path.glob("*.torrent"))

    if not torrent_files:
        click.echo("No .torrent files found", err=True)
        return

    if not verbose:
        click.echo(f"Found {len(torrent_files)} torrent file(s)")

    # Process torrents
    config = TorrentConfig(output_dir=output_path, verbose=verbose)
    results = create_from_torrents(torrent_files, config)

    # Summary
    total_files = sum(len(r.created_files) for r in results)
    total_errors = sum(len(r.errors) for r in results)

    click.echo(f"\nDone! Created {total_files} fake file(s)")
    if total_errors > 0:
        click.echo(f"Errors: {total_errors}", err=True)
        for result in results:
            for error in result.errors:
                click.echo(f"  {error}", err=True)
