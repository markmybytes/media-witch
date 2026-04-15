"""Public API for NFO file generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...core.fileops import FileOps


@dataclass
class NFOConfig:
    """Configuration for NFO generation.

    Attributes:
        season: Season number
        episode_start: Starting episode number (default: 1)
        episode_overrides: Optional dict mapping file index to episode number
        dry_run: If True, preview changes without executing
    """
    season: int
    episode_start: int = 1
    episode_overrides: dict[int, int] | None = None
    dry_run: bool = False


@dataclass
class NFOResult:
    """Result of NFO generation.

    Attributes:
        created: List of created NFO file paths
        skipped: List of files that were skipped
        errors: List of error messages
    """
    created: list[Path]
    skipped: list[Path]
    errors: list[str]


def generate_nfo_content(
    title: str,
    season: int,
    episode: int,
) -> str:
    """Generate XML content for NFO file.

    Args:
        title: Episode title
        season: Season number
        episode: Episode number

    Returns:
        XML content string
    """
    return (
        '<?xml version="1.0" encoding="utf-8" standalone="yes"?>'
        '<episodedetails>'
        f'<title>{title}</title>'
        f'<episode>{episode}</episode>'
        f'<season>{season}</season>'
        '</episodedetails>'
    )


def generate_episode_nfos(
    videos: list[Path],
    config: NFOConfig,
) -> NFOResult:
    """Generate .nfo files for video files.

    Args:
        videos: Sorted list of video files
        config: NFO generation configuration

    Returns:
        Result object with created NFO paths
    """
    fops = FileOps(dry_run=config.dry_run)
    created = []
    skipped = []
    errors = []

    # Build episode number mapping
    episode_overrides = config.episode_overrides or {}
    defaults = {p: i + config.episode_start for i, p in enumerate(videos)}

    for idx, video in enumerate(videos, start=1):
        try:
            ep = episode_overrides.get(idx, defaults[video])
            nfo_path = video.with_suffix(".nfo")

            if nfo_path.exists() and not config.dry_run:
                skipped.append(nfo_path)
                continue

            content = generate_nfo_content(
                title=video.stem,
                season=config.season,
                episode=ep,
            )

            fops.write_text_if_absent(nfo_path, content, label="[NFO]")
            created.append(nfo_path)

        except Exception as e:
            errors.append(f"Error creating NFO for {video}: {e}")

    return NFOResult(created=created, skipped=skipped, errors=errors)
