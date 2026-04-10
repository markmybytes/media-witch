"""CLI command for organizing media files."""

from __future__ import annotations

from pathlib import Path

import click

from ..features.organize.api import OrganizeConfig, organize_directory
from ..features.subtitles.locale import (LocaleMapper, load_csv_rules,
                                         parse_cli_rules)
from .common import common_options, locale_csv_option, locale_map_option


@click.command(name="organize")
@click.argument("paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--mode",
    type=click.Choice(["interactive", "auto-tv", "auto-movie"]),
    default="interactive",
    help="Organization mode",
)
@click.option(
    "--generate-nfo/--no-nfo",
    default=False,
    help="Generate NFO files for TV episodes",
)
@click.option(
    "--batch-season",
    type=int,
    help="Season number for non-interactive TV mode",
)
@locale_csv_option
@locale_map_option
@common_options
def organize_command(
    paths: tuple[str, ...],
    mode: str,
    generate_nfo: bool,
    batch_season: int | None,
    map_csv: str | None,
    locale_maps: tuple[str, ...],
    dry_run: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Organize media files into TV show or movie structure."""
    # Set up locale mapper
    csv_rules = load_csv_rules(Path(map_csv) if map_csv else None)
    cli_rules = parse_cli_rules(list(locale_maps))
    mapper = LocaleMapper(csv_rules=csv_rules, cli_rules=cli_rules)

    # Process each path
    for path_str in paths:
        path = Path(path_str)

        if mode == "interactive":
            if not quiet:
                click.echo(f"\nProcessing: {path}")
            # In interactive mode, would use prompts here
            # For now, default to skip
            config = OrganizeConfig(mode="skip", dry_run=dry_run)
        elif mode == "auto-tv":
            season = batch_season or 1
            config = OrganizeConfig(
                mode="show",
                season=season,
                locale_mapper=mapper,
                generate_nfo=generate_nfo,
                dry_run=dry_run,
            )
        elif mode == "auto-movie":
            config = OrganizeConfig(
                mode="movie",
                locale_mapper=mapper,
                dry_run=dry_run,
            )
        else:
            config = OrganizeConfig(mode="skip", dry_run=dry_run)

        try:
            result = organize_directory(path, config)

            if not quiet:
                if dry_run:
                    click.echo("[DRY-RUN] Preview mode")
                click.echo(f"Files moved: {len(result.files_moved)}")
                click.echo(f"NFOs created: {len(result.nfos_created)}")
                if result.errors:
                    click.echo(f"Errors: {len(result.errors)}", err=True)
                    for error in result.errors[:5]:  # Show first 5 errors
                        click.echo(f"  {error}", err=True)
        except Exception as e:
            click.echo(f"Error processing {path}: {e}", err=True)
            continue
