"""Public API for media file organization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...core.actions import ActionQueue
from ...core.fileops import FileOps
from ...core.media import is_audio, is_subtitle, is_video, list_files_and_dirs
from ...core.patterns import has_episode_pattern, natural_sort_key
from ..nfo.api import NFOConfig, generate_episode_nfos
from ..subtitles.api import SubtitleConfig, rename_subtitles
from ..subtitles.locale import LocaleMapper


@dataclass
class OrganizeConfig:
    """Configuration for organization.

    Attributes:
        mode: Organization mode ('show', 'movie', 'skip')
        season: Season number for TV shows
        locale_mapper: Optional LocaleMapper for subtitle renaming
        generate_nfo: Whether to generate NFO files (TV only)
        dry_run: If True, preview changes without executing
        extras_flags: List of boolean flags for extra classification (True = extra, False = primary)
        episode_overrides: Dict mapping video index (1-based) to episode number
        skip_nfo_generation: If True, skip NFO generation even if generate_nfo is True
        root_dir: Root directory for creating Season folders (for batch processing)
                 If set, Season folders are created at root_dir instead of path
        remove_unmapped_subs: Whether to remove subtitles not in mapping target locales
    """

    mode: Literal['show', 'movie', 'skip']
    season: int | None = None
    locale_mapper: LocaleMapper | None = None
    generate_nfo: bool = False
    dry_run: bool = False
    extras_flags: list[bool] | None = None
    episode_overrides: dict[int, int] | None = None
    skip_nfo_generation: bool = False
    root_dir: Path | None = None
    remove_unmapped_subs: bool = False


@dataclass
class OrganizeResult:
    """Result of organization operation.

    Attributes:
        files_moved: List of (source, destination) tuples
        nfos_created: List of created NFO files
        errors: List of error messages
        skipped: List of skipped files
    """

    files_moved: list[tuple[Path, Path]]
    nfos_created: list[Path]
    errors: list[str]
    skipped: list[Path]


def classify_extras_auto(items: list[Path]) -> list[bool]:
    """Automatically classify items as primary media or extras.

    Args:
        items: Files/directories to classify

    Returns:
        Boolean list where True = extra, False = primary
    """
    return [p.is_dir() or not has_episode_pattern(p.name) for p in items]


def organize_tv_show(
    path: Path,
    season: int,
    config: OrganizeConfig,
    fops: FileOps,
) -> OrganizeResult:
    """Organize files as TV show episodes.

    Args:
        path: Directory to organize (source of files)
        season: Season number
        config: Organization configuration
        fops: FileOps instance

    Returns:
        Result object with operation details
    """
    files, dirs = list_files_and_dirs(path)
    items = dirs + files
    if not items:
        return OrganizeResult([], [], [], [])

    aq = ActionQueue(fops)
    files_moved = []
    nfos_created = []
    errors = []
    skipped: list[Path] = []

    # Classify extras (use provided flags or fall back to automatic)
    if config.extras_flags is not None:
        flags = config.extras_flags
    else:
        flags = classify_extras_auto(items)

    # Use root_dir if provided (for batch season processing), otherwise use path
    output_root = config.root_dir if config.root_dir else path
    season_dir = output_root / f'Season {season}'
    extra_dir = output_root / 'EXTRA' / f'Season {season}'

    moved_video_dsts: list[Path] = []

    # Process items
    for item, is_ex in zip(items, flags, strict=False):
        try:
            if item.is_dir():
                if is_ex:
                    dst = extra_dir / item.name
                    aq.add(fops.move_dir_atomic, item, dst, desc=f'[MOVE-DIR] {item} -> {dst}')
                    files_moved.append((item, dst))
                else:
                    aq.add(
                        fops.move_dir_contents_to,
                        item,
                        season_dir,
                        desc=f'[FLATTEN] {item} -> {season_dir}',
                    )
                    files_moved.append((item, season_dir))
                continue

            if is_ex:
                dst = extra_dir / item.name
                aq.add(fops.move_file, item, dst, desc=f'[MOVE] {item} -> {dst}')
                files_moved.append((item, dst))
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = season_dir / item.name
                aq.add(fops.move_file, item, dst, desc=f'[MOVE] {item} -> {dst}')
                files_moved.append((item, dst))
                if is_video(dst):
                    moved_video_dsts.append(dst)
                continue

            dst = season_dir / item.name
            aq.add(fops.move_file, item, dst, desc=f'[MOVE] {item} -> {dst}')
            files_moved.append((item, dst))

        except Exception as e:
            errors.append(f'Error processing {item}: {e}')

    # Handle subtitles
    if config.locale_mapper and moved_video_dsts:
        sub_config = SubtitleConfig(
            locale_mapper=config.locale_mapper,
            dry_run=config.dry_run,
            remove_unmapped=config.remove_unmapped_subs,
        )
        for vdst in sorted(moved_video_dsts, key=natural_sort_key):
            subs = [p for p in path.iterdir() if p.is_file() and is_subtitle(p)]
            if season_dir.exists():
                subs += [p for p in season_dir.iterdir() if p.is_file() and is_subtitle(p)]
            if subs:
                rename_subtitles(subs, vdst, sub_config)

    # Commit all file operations before NFO generation
    aq.commit()
    fops.remove_dir_if_empty(path)

    # Generate NFOs
    if config.generate_nfo and not config.skip_nfo_generation and moved_video_dsts:
        videos_sorted = sorted(moved_video_dsts, key=natural_sort_key)

        nfo_config = NFOConfig(
            season=season,
            episode_start=1,
            episode_overrides=config.episode_overrides or {},
            dry_run=config.dry_run,
        )
        nfo_result = generate_episode_nfos(videos_sorted, nfo_config)
        nfos_created.extend(nfo_result.created)
        errors.extend(nfo_result.errors)

    return OrganizeResult(
        files_moved=files_moved,
        nfos_created=nfos_created,
        errors=errors,
        skipped=skipped,
    )


def organize_movie(
    path: Path,
    config: OrganizeConfig,
    fops: FileOps,
) -> OrganizeResult:
    """Organize files as movie.

    Args:
        path: Directory to organize
        config: Organization configuration
        fops: FileOps instance

    Returns:
        Result object with operation details
    """
    files, dirs = list_files_and_dirs(path)
    items = dirs + files
    if not items:
        return OrganizeResult([], [], [], [])

    aq = ActionQueue(fops)
    files_moved = []
    errors = []
    skipped: list[Path] = []

    # Classify extras (use provided flags or fall back to automatic)
    if config.extras_flags is not None:
        flags = config.extras_flags
    else:
        flags = classify_extras_auto(items)

    extra_dir = path / 'EXTRA'
    moved_video_dsts: list[Path] = []

    # Process items
    for item, is_ex in zip(items, flags, strict=False):
        try:
            if item.is_dir():
                if is_ex:
                    dst = extra_dir / item.name
                    aq.add(fops.move_dir_atomic, item, dst, desc=f'[MOVE-DIR] {item} -> {dst}')
                    files_moved.append((item, dst))
                else:
                    aq.add(
                        fops.move_dir_contents_to, item, path, desc=f'[FLATTEN] {item} -> {path}'
                    )
                    files_moved.append((item, path))
                continue

            if is_ex:
                dst = extra_dir / item.name
                aq.add(fops.move_file, item, dst, desc=f'[MOVE] {item} -> {dst}')
                files_moved.append((item, dst))
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = path / item.name
                aq.add(fops.move_file, item, dst, desc=f'[MOVE] {item} -> {dst}')
                files_moved.append((item, dst))
                if is_video(dst):
                    moved_video_dsts.append(dst)
                continue

            dst = path / item.name
            aq.add(fops.move_file, item, dst, desc=f'[MOVE] {item} -> {dst}')
            files_moved.append((item, dst))

        except Exception as e:
            errors.append(f'Error processing {item}: {e}')

    # Handle subtitles
    if config.locale_mapper and moved_video_dsts:
        sub_config = SubtitleConfig(
            locale_mapper=config.locale_mapper,
            dry_run=config.dry_run,
            remove_unmapped=config.remove_unmapped_subs,
        )
        subs = [p for p in path.iterdir() if p.is_file() and is_subtitle(p)]
        for vdst in sorted(moved_video_dsts, key=natural_sort_key):
            if subs:
                rename_subtitles(subs, vdst, sub_config)

    aq.commit()

    return OrganizeResult(
        files_moved=files_moved,
        nfos_created=[],
        errors=errors,
        skipped=skipped,
    )


def organize_directory(
    path: Path,
    config: OrganizeConfig,
) -> OrganizeResult:
    """Organize media files in directory.

    Args:
        path: Directory to organize
        config: Organization configuration

    Returns:
        Result object with operation details

    Raises:
        ValueError: If path is not a directory
        PermissionError: If insufficient permissions
    """
    if not path.is_dir():
        raise ValueError(f'Not a directory: {path}')

    fops = FileOps(dry_run=config.dry_run)

    if config.mode == 'skip':
        return OrganizeResult([], [], [], [path])
    elif config.mode == 'show':
        if config.season is None:
            raise ValueError('Season number required for TV show mode')
        return organize_tv_show(path, config.season, config, fops)
    elif config.mode == 'movie':
        return organize_movie(path, config, fops)
    else:
        raise ValueError(f'Invalid mode: {config.mode}')
