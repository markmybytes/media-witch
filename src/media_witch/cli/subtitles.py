"""CLI command for subtitle renaming."""

from __future__ import annotations

from pathlib import Path

import click

from ..core.media import is_subtitle, is_video
from ..features.subtitles.api import (SubtitleConfig, pair_subtitles,
                                      rename_subtitles)
from ..features.subtitles.locale import (LocaleMapper, load_csv_rules,
                                         parse_cli_rules)
from .common import common_options, locale_csv_option, locale_map_option


@click.command(name="subtitles")
@click.argument("paths", nargs=-1, type=click.Path(exists=True), required=True)
@locale_csv_option
@locale_map_option
@click.option(
    "--remove",
    is_flag=True,
    default=False,
    help="Remove subtitles whose locale is not in the mapping target list",
)
@common_options
def subtitles_command(
    paths: tuple[str, ...],
    map_csv: str | None,
    locale_maps: tuple[str, ...],
    remove: bool,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Rename and organize subtitle files."""
    # Set up locale mapper
    csv_rules = load_csv_rules(Path(map_csv) if map_csv else None)
    cli_rules = parse_cli_rules(list(locale_maps))
    mapper = LocaleMapper(csv_rules=csv_rules, cli_rules=cli_rules)

    for path_str in paths:
        path = Path(path_str)

        # Collect files
        if path.is_file():
            if is_subtitle(path):
                subtitles = [path]
                videos = [p for p in path.parent.iterdir() if is_video(p)]
            else:
                click.echo(f"Skipping {path}: not a subtitle file", err=True)
                continue
        elif path.is_dir():
            subtitles = [p for p in path.iterdir() if p.is_file()
                         and is_subtitle(p)]
            videos = [p for p in path.iterdir() if p.is_file() and is_video(p)]
        else:
            continue

        if not subtitles:
            if not quiet:
                click.echo(f"No subtitle files found in {path}")
            continue

        if not videos:
            if not quiet:
                click.echo(f"No video files found to pair with in {path}")
            continue

        # Pair subtitles with videos
        pairs = pair_subtitles(subtitles, videos)

        config = SubtitleConfig(
            locale_mapper=mapper,
            dry_run=dry_run,
            remove_unmapped=remove,
        )

        total_renamed = 0
        total_removed = 0
        total_skipped = 0
        total_errors = 0

        for video, subs in pairs.items():
            if not subs:
                continue

            try:
                result = rename_subtitles(subs, video, config)
                total_renamed += len(result.renamed)
                total_removed += len(result.removed)
                total_skipped += len(result.skipped)
                total_errors += len(result.errors)

                if verbose or dry_run:
                    for src, dst in result.renamed:
                        click.echo(f"  [RENAME] {src.name} → {dst.name}")
                    for sub in result.removed:
                        click.echo(f"  [REMOVE] {sub.name}")

            except Exception as e:
                click.echo(
                    f"Error processing subtitles for {video}: {e}", err=True)
                continue

        if not quiet:
            if dry_run:
                click.echo("[DRY-RUN] Preview mode")
            click.echo(f"Subtitles renamed: {total_renamed}")
            if remove:
                click.echo(f"Subtitles removed: {total_removed}")
            click.echo(f"Skipped: {total_skipped}")
            if total_errors > 0:
                click.echo(f"Errors: {total_errors}", err=True)
