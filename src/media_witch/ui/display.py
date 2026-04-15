"""Display utilities for formatting output."""

from __future__ import annotations

from pathlib import Path


def print_section(title: str) -> None:
    """Print a section header.

    Args:
        title: Section title
    """
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}')


def print_tree(path: Path) -> str:
    """Generate a tree-style listing of directory contents.

    Args:
        path: Directory to list

    Returns:
        Formatted tree string, or empty string on error
    """
    try:
        entries = sorted(path.iterdir())
    except (PermissionError, NotADirectoryError):
        return ''

    lines = [
        f'{"└── " if i == len(entries) - 1 else "├── "}{entry.name}{"/" if entry.is_dir() else ""}'
        for i, entry in enumerate(entries)
    ]
    return '\n'.join(lines)
