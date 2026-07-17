"""CLI command for organizing media files."""

from __future__ import annotations

from pathlib import Path

import click

from ..core.media import is_video, list_files_and_dirs
from ..core.patterns import natural_sort_key
from ..features.organize.api import OrganizeConfig, classify_extras_auto, organize_directory
from ..features.subtitles.locale import LocaleMapper, load_csv_rules, parse_cli_rules
from ..ui.prompts import (
    ask_extras_classification,
    ask_nfo_overrides,
    ask_processing_choice,
    ask_remove_unmapped_subtitles,
    ask_season,
    ask_yes_no,
)
from .common import dry_run_option, locale_csv_option, locale_map_option, quiet_option


def _find_leaf_directories(path: Path) -> list[Path]:
    """Recursively find directories that contain files (leaf directories).

    Args:
        path: Directory to search

    Returns:
        List of leaf directories (directories containing files)
    """
    files, dirs = list_files_and_dirs(path)
    if files:
        return [path]

    leafs = []
    for subdir in dirs:
        leafs.extend(_find_leaf_directories(subdir))
    return leafs


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

    if choice == 'skip':
        if not quiet:
            click.echo(f'⏭️  Skipped: {path}')
        return

    # If directory has files, process based on choice
    if has_files:
        if choice == 'show':
            # Find all leaf directories and process each as a season
            leafs = _find_leaf_directories(path)
            for leaf in leafs:
                if not quiet:
                    click.echo(f'\n{"─" * 60}')
                    click.echo(f'📂 [UNIT] {leaf}')
                    click.echo(f'{"─" * 60}')
                season = ask_season(default=1)
                # Always enable NFO generation, but callback will ask user
                _process_single_dir(leaf, 'show', season, mapper, True, dry_run, quiet)
            return
        elif choice == 'movie':
            _process_single_dir(path, 'movie', None, mapper, False, dry_run, quiet)
            return

    # Handle batch processing modes (no files in current directory)
    if choice == 'shows':
        # Process each subdir as a separate show
        for subdir in dirs:
            _process_interactive(subdir, mapper, dry_run, quiet)
        return
    elif choice == 'seasons':
        # Process each subdir as a season (ask for season number per subdir)
        # Files in each season subdir will be moved to parent's Season folders
        for subdir in sorted(dirs):
            if not quiet:
                click.echo(f'\n{"─" * 60}')
                click.echo(f'📺 [SEASON] {subdir}')
                click.echo(f'{"─" * 60}')
            season = ask_season(default=1)
            # Always enable NFO generation, but callback will ask user
            _process_single_dir_batch(subdir, 'show', season, path, mapper, True, dry_run, quiet)
        return
    elif choice == 'movies':
        # Process each subdir as a movie
        for subdir in dirs:
            if not quiet:
                click.echo(f'\n{"─" * 60}')
                click.echo(f'🎬 [MOVIE] {subdir}')
                click.echo(f'{"─" * 60}')
            _process_single_dir(subdir, 'movie', None, mapper, False, dry_run, quiet)
        return


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
    _process_single_dir_impl(path, mode, season, mapper, generate_nfo, dry_run, quiet, None)


def _process_single_dir_batch(
    path: Path,
    mode: str,
    season: int | None,
    root_dir: Path,
    mapper: LocaleMapper,
    generate_nfo: bool,
    dry_run: bool,
    quiet: bool,
) -> None:
    """Process a single directory in batch mode with output at root_dir level.

    Args:
        path: Directory to process (source of files)
        mode: 'show' or 'movie'
        season: Season number (for shows)
        root_dir: Root directory where Season folders are created
        mapper: Locale mapper
        generate_nfo: Whether to generate NFO files
        dry_run: Preview mode
        quiet: Suppress output
    """
    _process_single_dir_impl(path, mode, season, mapper, generate_nfo, dry_run, quiet, root_dir)


def _process_single_dir_impl(
    path: Path,
    mode: str,
    season: int | None,
    mapper: LocaleMapper,
    generate_nfo: bool,
    dry_run: bool,
    quiet: bool,
    root_dir: Path | None,
) -> None:
    """Internal implementation for processing a directory.

    Args:
        path: Directory to process
        mode: 'show' or 'movie'
        season: Season number (for shows)
        mapper: Locale mapper
        generate_nfo: Whether to generate NFO files
        dry_run: Preview mode
        quiet: Suppress output
        root_dir: Root directory for Season output (batch mode). If None, uses path.
    """
    # Step 1: Ask about subtitle removal
    remove_unmapped_subs = False
    if mapper:
        remove_unmapped_subs = ask_remove_unmapped_subtitles(mapper)

    # Step 2: Get items to classify for extras
    files, dirs = list_files_and_dirs(path)
    items = dirs + files

    # Step 3: Ask about extras classification (if items exist)
    extras_flags = None
    if items:
        defaults = classify_extras_auto(items)
        extras_flags = ask_extras_classification(items, defaults)

    # Step 4: Ask about NFO generation and episode overrides (TV shows only)
    skip_nfo = False
    episode_overrides = None

    if mode == 'show' and season is not None and generate_nfo:
        # Find which videos will actually be organized (not marked as extras)
        videos_kept = []
        if extras_flags:
            for i, item in enumerate(items):
                if i < len(extras_flags) and not extras_flags[i] and is_video(item):
                    videos_kept.append(item)
        else:
            videos_kept = [f for f in files if is_video(f)]

        videos_sorted = sorted(videos_kept, key=natural_sort_key)

        if videos_sorted:
            # Ask if user wants to generate NFO files
            generate_nfo_answer = ask_yes_no('Generate NFO files?', default=True)
            skip_nfo = not generate_nfo_answer

            # Ask for episode overrides if user said yes
            if not skip_nfo:
                episode_overrides = ask_nfo_overrides(videos_sorted, season)

    # Step 5: Create config with all answers (no more callbacks)
    config = OrganizeConfig(
        mode=mode,  # type: ignore
        season=season,
        locale_mapper=mapper,
        generate_nfo=generate_nfo,
        dry_run=dry_run,
        extras_flags=extras_flags,
        episode_overrides=episode_overrides,
        skip_nfo_generation=skip_nfo,
        root_dir=root_dir,
        remove_unmapped_subs=remove_unmapped_subs,
    )

    try:
        result = organize_directory(path, config)

        if not quiet:
            if dry_run:
                click.echo('[DRY-RUN] Preview mode')
            click.echo(f'Files moved: {len(result.files_moved)}')
            click.echo(f'NFOs created: {len(result.nfos_created)}')
            if result.errors:
                click.echo(f'Errors: {len(result.errors)}', err=True)
                for error in result.errors[:5]:
                    click.echo(f'  {error}', err=True)
    except Exception as e:
        click.echo(f'Error processing {path}: {e}', err=True)


@click.command(name='organize')
@click.argument('paths', nargs=-1, type=click.Path(exists=True), required=True)
@locale_csv_option
@locale_map_option
@dry_run_option
@quiet_option
def organize_command(
    paths: tuple[str, ...],
    map_csv: str | None,
    locale_maps: tuple[str, ...],
    dry_run: bool,
    quiet: bool,
) -> None:
    """Organize media files into TV show or movie structure (interactive mode).

    This command walks you through organizing your media files with prompts for:
    - TV show vs movie classification
    - Season numbers for TV shows
    - Extras classification
    - NFO file generation (always enabled)
    - Episode number overrides
    """
    try:
        # Set up locale mapper
        csv_rules = load_csv_rules(Path(map_csv) if map_csv else None)
        cli_rules = parse_cli_rules(list(locale_maps))
        mapper = LocaleMapper(csv_rules=csv_rules, cli_rules=cli_rules)

        # Process each path in interactive mode
        for path_str in paths:
            path = Path(path_str)
            _process_interactive(path, mapper, dry_run, quiet)
    except KeyboardInterrupt:
        click.echo('\n⚠️  Operation cancelled by user', err=True)
        raise SystemExit(1) from None
