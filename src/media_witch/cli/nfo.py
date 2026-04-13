"""CLI command for NFO generation."""

from __future__ import annotations

from pathlib import Path

import click

from ..core.media import is_video
from ..features.nfo.api import NFOConfig, generate_episode_nfos
from ..ui.prompts import ask_nfo_overrides
from .common import common_options


@click.command(name="nfo")
@click.argument("paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--season",
    type=int,
    required=True,
    help="Season number (required for TV episodes)",
)
@click.option(
    "--episode-start",
    type=int,
    default=1,
    help="Starting episode number",
)
@click.option(
    "--interactive",
    is_flag=True,
    default=False,
    help="Interactively override episode numbers",
)
@common_options
def nfo_command(
    paths: tuple[str, ...],
    season: int,
    episode_start: int,
    interactive: bool,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Generate NFO metadata files for video files."""
    for path_str in paths:
        path = Path(path_str)

        # Collect video files
        if path.is_file() and is_video(path):
            videos = [path]
        elif path.is_dir():
            videos = sorted([p for p in path.iterdir()
                            if p.is_file() and is_video(p)])
        else:
            click.echo(
                f"Skipping {path}: not a video file or directory", err=True)
            continue

        if not videos:
            if not quiet:
                click.echo(f"No video files found in {path}")
            continue

        # Get episode overrides if interactive mode
        episode_overrides = {}
        if interactive:
            episode_overrides = ask_nfo_overrides(videos, season)

        # Generate NFOs
        config = NFOConfig(
            season=season,
            episode_start=episode_start,
            episode_overrides=episode_overrides,
            dry_run=dry_run,
        )

        try:
            result = generate_episode_nfos(videos, config)

            if not quiet:
                if dry_run:
                    click.echo("[DRY-RUN] Preview mode")
                click.echo(f"NFOs created: {len(result.created)}")
                click.echo(f"Skipped: {len(result.skipped)}")
                if verbose:
                    for nfo in result.created:
                        click.echo(f"  Created: {nfo}")
                if result.errors:
                    click.echo(f"Errors: {len(result.errors)}", err=True)
                    for error in result.errors:
                        click.echo(f"  {error}", err=True)
        except Exception as e:
            click.echo(f"Error processing {path}: {e}", err=True)
            continue
