"""Torrent file parsing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .decoder import bdecode


class TorrentInfo:
    """Parsed torrent file information."""

    def __init__(self, data: dict[bytes, Any]) -> None:
        """Initialize from decoded torrent data.

        Args:
            data: Decoded bencode dictionary
        """
        self._data = data
        self._info = data[b'info']

    @property
    def name(self) -> str:
        """Get torrent name."""
        name_bytes = self._info[b'name']
        if isinstance(name_bytes, bytes):
            return name_bytes.decode('utf-8', errors='replace')
        return str(name_bytes)

    @property
    def is_single_file(self) -> bool:
        """Check if torrent contains a single file."""
        return b'files' not in self._info

    @property
    def files(self) -> list[tuple[list[str], int]]:
        """Get list of files in torrent.

        Returns:
            List of (path_parts, size) tuples
        """
        if self.is_single_file:
            return [([self.name], self._info[b'length'])]

        result = []
        for file_dict in self._info[b'files']:
            path = [p.decode('utf-8', errors='replace')
                    for p in file_dict[b'path']]
            size = file_dict[b'length']
            result.append((path, size))
        return result

    @property
    def total_size(self) -> int:
        """Get total size of all files in bytes."""
        if self.is_single_file:
            length = self._info[b'length']
            return int(length) if isinstance(length, (int, float)) else 0
        return sum(
            int(f[b'length']) if isinstance(f[b'length'], (int, float)) else 0
            for f in self._info[b'files']
        )


def parse_torrent(torrent_path: Path) -> TorrentInfo:
    """Parse a .torrent file.

    Args:
        torrent_path: Path to .torrent file

    Returns:
        TorrentInfo object with parsed metadata

    Raises:
        FileNotFoundError: If torrent file doesn't exist
        ValueError: If file is not valid bencode
    """
    with torrent_path.open('rb') as f:
        data = bdecode(f.read())
    return TorrentInfo(data)
