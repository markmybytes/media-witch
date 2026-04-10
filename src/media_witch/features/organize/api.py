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
from ..subtitles.api import SubtitleService
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
    """
    mode: Literal["show", "movie", "skip"]
    season: int | None = None
    locale_mapper: LocaleMapper | None = None
    generate_nfo: bool = False
    dry_run: bool = False


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
        path: Directory to organize
        season: Season number
        config: Organization configuration
        fops: FileOps instance

    Returns:
        Result object with operation details
    """
    files, dirs = list_files_and_dirs(path
                                      )
    items = dirs + files
    if not items:
        return OrganizeResult([], [], [], [])

    aq = ActionQueue(fops)
    files_moved = []
    nfos_created = []
    errors = []
    skipped = []

    # Classify extras automatically
    flags = classify_extras_auto(items)

    season_dir = path / f"Season {season}"
    extra_dir = path / "EXTRA" / f"Season {season}"

    moved_video_dsts: list[Path] = []

    # Process items
    for item, is_ex in zip(items, flags):
        try:
            if item.is_dir():
                if is_ex:
                    dst = extra_dir / item.name
                    aq.add(fops.move_dir_atomic, item, dst,
                           desc=f"[MOVE-DIR] {item} -> {dst}")
                    files_moved.append((item, dst))
                else:
                    aq.add(fops.move_dir_contents_to, item, season_dir,
                           desc=f"[FLATTEN] {item} -> {season_dir}")
                    files_moved.append((item, season_dir))
                continue

            if is_ex:
                dst = extra_dir / item.name
                aq.add(fops.move_file, item, dst,
                       desc=f"[MOVE] {item} -> {dst}")
                files_moved.append((item, dst))
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = season_dir / item.name
                aq.add(fops.move_file, item, dst,
                       desc=f"[MOVE] {item} -> {dst}")
                files_moved.append((item, dst))
                if is_video(dst):
                    moved_video_dsts.append(dst)
                continue

            dst = season_dir / item.name
            aq.add(fops.move_file, item, dst,
                   desc=f"[MOVE] {item} -> {dst}")
            files_moved.append((item, dst))

        except Exception as e:
            errors.append(f"Error processing {item}: {e}")

    # Handle subtitles
    if config.locale_mapper and moved_video_dsts:
        subsvc = SubtitleService(config.locale_mapper, fops)
        for vdst in sorted(moved_video_dsts, key=natural_sort_key):
            subs = [p for p in path.iterdir() if p.is_file()
                    and is_subtitle(p)]
            if season_dir.exists():
                subs += [p for p in season_dir.iterdir()
                         if p.is_file() and is_subtitle(p)]
            subsvc.plan(subs, vdst, aq)

    # Generate NFOs
    if config.generate_nfo and moved_video_dsts:
        videos_sorted = sorted(moved_video_dsts, key=natural_sort_key)
        nfo_config = NFOConfig(
            season=season, episode_start=1, dry_run=config.dry_run)
        nfo_result = generate_episode_nfos(videos_sorted, nfo_config)
        nfos_created.extend(nfo_result.created)
        errors.extend(nfo_result.errors)

    aq.commit()
    fops.remove_dir_if_empty(path)

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
    skipped = []

    # Classify extras automatically
    flags = classify_extras_auto(items)

    extra_dir = path / "EXTRA"
    moved_video_dsts: list[Path] = []

    # Process items
    for item, is_ex in zip(items, flags):
        try:
            if item.is_dir():
                if is_ex:
                    dst = extra_dir / item.name
                    aq.add(fops.move_dir_atomic, item, dst,
                           desc=f"[MOVE-DIR] {item} -> {dst}")
                    files_moved.append((item, dst))
                else:
                    aq.add(fops.move_dir_contents_to, item, path,
                           desc=f"[FLATTEN] {item} -> {path}")
                    files_moved.append((item, path))
                continue

            if is_ex:
                dst = extra_dir / item.name
                aq.add(fops.move_file, item, dst,
                       desc=f"[MOVE] {item} -> {dst}")
                files_moved.append((item, dst))
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = path / item.name
                aq.add(fops.move_file, item, dst,
                       desc=f"[MOVE] {item} -> {dst}")
                files_moved.append((item, dst))
                if is_video(dst):
                    moved_video_dsts.append(dst)
                continue

            dst = path / item.name
            aq.add(fops.move_file, item, dst,
                   desc=f"[MOVE] {item} -> {dst}")
            files_moved.append((item, dst))

        except Exception as e:
            errors.append(f"Error processing {item}: {e}")

    # Handle subtitles
    if config.locale_mapper and moved_video_dsts:
        subsvc = SubtitleService(config.locale_mapper, fops)
        subs = [p for p in path.iterdir() if p.is_file() and is_subtitle(p)]
        for vdst in sorted(moved_video_dsts, key=natural_sort_key):
            subsvc.plan(subs, vdst, aq)

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
        raise ValueError(f"Not a directory: {path}")

    fops = FileOps(dry_run=config.dry_run)

    if config.mode == "skip":
        return OrganizeResult([], [], [], [path])
    elif config.mode == "show":
        if config.season is None:
            raise ValueError("Season number required for TV show mode")
        return organize_tv_show(path, config.season, config, fops)
    elif config.mode == "movie":
        return organize_movie(path, config, fops)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")
