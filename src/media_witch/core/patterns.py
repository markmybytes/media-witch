"""Pattern matching for TV episodes and natural sorting."""

from __future__ import annotations

import re
from pathlib import Path

EPISODE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)\bS(\d{1,2})E(\d{1,3})\b"),  # S01E01, S1E1
    re.compile(r"\[(\d{1,3})\]"),  # [01], [1]
]


def has_episode_pattern(name: str) -> bool:
    """Check if a filename contains an episode pattern.

    Args:
        name: Filename to check

    Returns:
        True if filename matches any episode pattern
    """
    return any(p.search(name) for p in EPISODE_PATTERNS)


def natural_sort_key(p: Path) -> tuple:
    """Generate a natural sort key for a path.

    Splits filename into text and number parts for natural sorting
    (e.g., "file2.txt" comes before "file10.txt").

    Args:
        p: Path to generate key for

    Returns:
        Tuple that can be used as a sort key
    """
    return tuple(
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", p.name)
    )


def extract_season_episode(name: str) -> tuple[int | None, int | None]:
    """Extract season and episode numbers from filename.

    Tries to match S##E## pattern first, then falls back to [##] pattern.

    Args:
        name: Filename to parse

    Returns:
        Tuple of (season, episode) numbers, or (None, episode) for [##] pattern,
        or (None, None) if no pattern matches
    """
    # Try S##E## pattern
    for pattern in EPISODE_PATTERNS:
        match = pattern.search(name)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # S##E## format
                return int(groups[0]), int(groups[1])
            elif len(groups) == 1:
                # [##] format - no season info
                return None, int(groups[0])
    return None, None
