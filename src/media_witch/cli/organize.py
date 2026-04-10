"""CLI command for organizing media files."""

from __future__ import annotations

from pathlib import Path

import click

from ..core.media import list_files_and_dirs
from ..features.organize.api import OrganizeConfig, organize_directory
from ..features.subtitles.locale import (LocaleMapper, load_csv_rules,
                                         parse_cli_rules)
from ..ui.prompts import (ask_extras_classification, ask_nfo_overrides,
                          ask_processing_choice, ask_season, ask_yes_no)
from .common import common_options, locale_csv_option, locale_map_option


def _process_interactive(
    path: Path,
    mapper: LocaleMapper,
    dry_run: bool,
    quiet: bool,
) -> None:
    """Process a single directory in interactive mode.

    Args:
        path: Directory to process
        mapper: Locale mapper for subtitles
        dry_run: Whether to preview changes
        quiet: Whether to suppress output
    """
    files, dirs = list_files_and_dirs(path)
    has_files = len(files) > 0

    # Ask user what to do
    choice = ask_processing_choice(path, has_files)

    if choice == "skip":
        if not quiet:
            click.echo(f"Skipped: {path}")
        return

    # Handle batch processing modes (subdirectories)
    if choice == "shows":
        # Process each subdir as a separate show
        for subdir in dirs:
            _process_interactive(subdir, mapper, dry_run, quiet)
        return
    elif choice == "seasons":
        # Process each subdir as a season
        generate_nfo = ask_yes_no("Generate NFO files for all seasons?", default=False)
        for season_num, subdir in enumerate(sorted(dirs), start=1):
            if not quiet:
                click.echo(f"\nProcessing Season {season_num}: {subdir.name}")
            _process_single_dir(
                subdir, "show", season_num, mapper, generate_nfo, dry_run, quiet
            )
        return
    elif choice == "movies":
        # Process each subdir as a movie
        for subdir in dirs:
            _process_single_dir(
                subdir, "movie", None, mapper, False, dry_run, quiet
            )
        return

    # Handle single directory modes
    if choice == "show":
        season = ask_season(default=1)
        generate_nfo = ask_yes_no("Generate NFO files?", default=False)
        _process_single_dir(
            path, "show", season, mapper, generate_nfo, dry_run, quiet
        )
    elif choice == "movie":
        _process_single_dir(
            path, "movie", None, mapper, False, dry_run, quiet
        )


def _process_single_dir(
    path: Path,
    mode: str,
    season: int | None,
    mapper: LocaleMapper,
    generate_nfo: bool,
    dry_run: bool,
    quiet: bool,
) -> None:
    """Process a single directory with given configuration.

    Args:
        path: Directory to process
        mode: 'show' or 'movie'
        season: Season number (for shows)
        mapper: Locale mapper
        generate_nfo: Whether to generate NFO files
        dry_run: Preview mode
        quiet: Suppress output
    """
    # Create config with interactive callbacks
    config = OrganizeConfig(
        mode=mode,  # type: ignore
        season=season,
        locale_mapper=mapper,
        generate_nfo=generate_nfo,
        dry_run=dry_run,
        extras_classifier=ask_extras_classification,
        nfo_override_callback=ask_nfo_overrides if generate_nfo else None,
    )

    try:
        result = organize_directory(path, config)

        if not quiet:
            if dry_run:
                click.echo("[DRY-RUN] Preview mode")
            click.echo(f"Files moved: {len(result.files_moved)}")
            click.echo(f"NFOs created: {len(result.nfos_created)}")
            if result.errors:
                click.echo(f"Errors: {len(result.errors)}", err=True)
                for error in result.errors[:5]:
                    click.echo(f"  {error}", err=True)
    except Exception as e:
        click.echo(f"Error processing {path}: {e}", err=True)


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
            _process_interactive(path, mapper, dry_run, quiet)
            continue

        # Auto modes
        if mode == "auto-tv":
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
