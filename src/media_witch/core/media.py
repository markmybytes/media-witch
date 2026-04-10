"""Media file type detection utilities."""

from __future__ import annotations

from pathlib import Path

VIDEO_EXTS: set[str] = {
    ".mkv", ".mp4", ".avi", ".mov", ".ts", ".m2ts", ".wmv"
}

AUDIO_EXTS: set[str] = {
    ".mka", ".aac", ".flac", ".dts", ".ac3", ".eac3",
    ".mp3", ".ogg", ".opus"
}

SUB_EXTS: set[str] = {
    ".ass", ".ssa", ".sup", ".srt"
}


def is_video(p: Path) -> bool:
    """Check if path is a video file.

    Args:
        p: Path to check

    Returns:
        True if file has a video extension
    """
    return p.suffix.lower() in VIDEO_EXTS


def is_audio(p: Path) -> bool:
    """Check if path is an audio file.

    Args:
        p: Path to check

    Returns:
        True if file has an audio extension
    """
    return p.suffix.lower() in AUDIO_EXTS


def is_subtitle(p: Path) -> bool:
    """Check if path is a subtitle file.

    Args:
        p: Path to check

    Returns:
        True if file has a subtitle extension
    """
    return p.suffix.lower() in SUB_EXTS


def list_files_and_dirs(path: Path) -> tuple[list[Path], list[Path]]:
    """List files and directories separately in a path.

    Args:
        path: Directory to scan

    Returns:
        Tuple of (files, directories) as lists of Path objects
    """
    files = [e for e in path.iterdir() if e.is_file()]
    dirs = [e for e in path.iterdir() if e.is_dir()]
    return files, dirs
