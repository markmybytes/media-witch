"""Pattern matching for TV episodes and natural sorting."""

from __future__ import annotations

import re
from pathlib import Path

EPISODE_PATTERNS: list[re.Pattern] = [
    re.compile(r'(?i)\bS(\d{1,2})E(\d{1,3})\b'),  # S01E01, S1E1
    re.compile(r'\[(\d{1,3})\]'),  # [01], [1]
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
    return tuple(int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.name))
