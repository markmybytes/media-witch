"""Public API for subtitle renaming and organization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...core.actions import ActionQueue
from ...core.fileops import FileOps
from .locale import LocaleMapper


@dataclass
class SubtitleConfig:
    """Configuration for subtitle operations.

    Attributes:
        locale_mapper: LocaleMapper instance for locale code resolution
        dry_run: If True, preview changes without executing
        remove_unmapped: If True, remove subtitles not in mapping target locales
    """
    locale_mapper: LocaleMapper
    dry_run: bool = False
    remove_unmapped: bool = False


@dataclass
class SubtitleResult:
    """Result of subtitle operations.

    Attributes:
        renamed: List of (source, destination) path tuples for renamed files
        skipped: List of files that were skipped
        removed: List of files that were removed
        errors: List of error messages
    """
    renamed: list[tuple[Path, Path]]
    skipped: list[Path]
    removed: list[Path]
    errors: list[str]


class SubtitleService:
    """Service for subtitle pairing and renaming."""

    def __init__(self, mapper: LocaleMapper, fops: FileOps) -> None:
        """Initialize SubtitleService.

        Args:
            mapper: LocaleMapper for locale code resolution
            fops: FileOps instance for file operations
        """
        self.mapper = mapper
        self.fops = fops

    @staticmethod
    def _right_most_token(sub: Path) -> str:
        """Extract rightmost token from subtitle filename.

        Args:
            sub: Subtitle file path

        Returns:
            Rightmost dot-separated token, or empty string
        """
        parts = sub.stem.split(".")
        return parts[-1] if len(parts) > 1 else ""

    @staticmethod
    def _stem_wo_token(sub: Path) -> str:
        """Get stem without the rightmost token.

        Args:
            sub: Subtitle file path

        Returns:
            Stem without rightmost token
        """
        parts = sub.stem.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else sub.stem

    @staticmethod
    def pairs_with(sub: Path, video: Path) -> bool:
        """Check if subtitle pairs with video file.

        Args:
            sub: Subtitle file path
            video: Video file path

        Returns:
            True if subtitle pairs with video
        """
        return sub.stem == video.stem or sub.stem.startswith(f"{video.stem}.")

    def normalized_target(self, sub: Path, video: Path) -> Path:
        """Generate normalized target path for subtitle.

        Applies locale mapping to subtitle language codes.

        Args:
            sub: Source subtitle path
            video: Video file to pair with

        Returns:
            Normalized target path
        """
        t = self._right_most_token(sub)
        mapped = self.mapper.resolve(t) if t else t
        stem = f"{self._stem_wo_token(sub)}.{mapped}" if t else video.stem
        return video.with_name(f"{stem}{sub.suffix.lower()}")

    def plan(self, subs: list[Path], video_dst: Path, aq: ActionQueue) -> None:
        """Plan subtitle operations for action queue.

        Args:
            subs: Subtitle files to process
            video_dst: Destination video file path
            aq: ActionQueue to add operations to
        """
        for sub in subs:
            if not self.pairs_with(sub, video_dst):
                continue
            dst = video_dst.parent / \
                self.normalized_target(sub, video_dst).name
            if sub.parent != video_dst.parent:
                tmp = video_dst.parent / sub.name
                aq.add(self.fops.move_file, sub, tmp,
                       desc=f"[MOVE] {sub} -> {tmp}")
                if tmp != dst:
                    aq.add(self.fops.rename_file, tmp, dst,
                           desc=f"[RENAME] {tmp.name} -> {dst.name}")
            else:
                if sub != dst:
                    aq.add(self.fops.rename_file, sub, dst,
                           desc=f"[RENAME] {sub.name} -> {dst.name}")


def rename_subtitles(
    subtitles: list[Path],
    video: Path,
    config: SubtitleConfig,
) -> SubtitleResult:
    """Rename subtitles to match video file.

    Args:
        subtitles: Subtitle files to rename
        video: Video file to pair with
        config: Subtitle configuration

    Returns:
        Result object with renamed paths
    """
    fops = FileOps(dry_run=config.dry_run)
    service = SubtitleService(config.locale_mapper, fops)

    renamed = []
    skipped = []
    removed = []
    errors = []

    # Get allowed target locales if remove_unmapped is enabled
    allowed_locales = config.locale_mapper.get_target_locales(
    ) if config.remove_unmapped else set()

    for sub in subtitles:
        try:
            if not service.pairs_with(sub, video):
                skipped.append(sub)
                continue

            # Check if subtitle should be removed (not in allowed locales)
            if config.remove_unmapped:
                token = service._right_most_token(sub)
                mapped_locale = config.locale_mapper.resolve(
                    token) if token else token
                if mapped_locale and mapped_locale not in allowed_locales:
                    if not config.dry_run:
                        fops.remove_file(sub, label="[REMOVE]")
                    removed.append(sub)
                    continue

            dst = video.parent / service.normalized_target(sub, video).name
            if sub != dst:
                if not config.dry_run:
                    if sub.parent != video.parent:
                        tmp = video.parent / sub.name
                        fops.move_file(sub, tmp)
                        if tmp != dst:
                            fops.rename_file(tmp, dst)
                        renamed.append((sub, dst))
                    else:
                        fops.rename_file(sub, dst)
                        renamed.append((sub, dst))
                else:
                    renamed.append((sub, dst))
        except Exception as e:
            errors.append(f"Error processing {sub}: {e}")

    return SubtitleResult(renamed=renamed, skipped=skipped, removed=removed, errors=errors)


def pair_subtitles(
    subtitles: list[Path],
    videos: list[Path],
) -> dict[Path, list[Path]]:
    """Pair subtitle files with video files.

    Args:
        subtitles: Subtitle files to pair
        videos: Video files to pair with

    Returns:
        Mapping of video -> list of paired subtitles
    """
    pairs: dict[Path, list[Path]] = {v: [] for v in videos}

    for sub in subtitles:
        for video in videos:
            if SubtitleService.pairs_with(sub, video):
                pairs[video].append(sub)
                break

    return pairs
