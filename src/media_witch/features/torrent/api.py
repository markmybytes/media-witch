"""Public API for torrent fake file creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .parser import parse_torrent


@dataclass
class TorrentConfig:
    """Configuration for torrent operations.

    Attributes:
        output_dir: Output directory for created files
        verbose: Enable verbose output
    """
    output_dir: Path
    verbose: bool = False


@dataclass
class TorrentResult:
    """Result of torrent file creation.

    Attributes:
        created_files: List of created file paths
        created_dirs: List of created directory paths
        errors: List of error messages
    """
    created_files: list[Path]
    created_dirs: list[Path]
    errors: list[str]


def create_from_torrent(
    torrent_path: Path,
    config: TorrentConfig,
) -> TorrentResult:
    """Create fake file structure from .torrent file.

    Args:
        torrent_path: Path to .torrent file
        config: Torrent configuration

    Returns:
        Result object with created paths

    Raises:
        FileNotFoundError: If torrent file doesn't exist
        ValueError: If file is not valid torrent
    """
    created_files = []
    created_dirs = []
    errors = []

    try:
        info = parse_torrent(torrent_path)

        # Create base directory
        folder_name = torrent_path.stem
        base_dir = config.output_dir / folder_name
        base_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(base_dir)

        if config.verbose:
            print(f"Creating fake files in: {base_dir}")
            print(f"Torrent: {info.name}")
            print(f"Total files: {len(info.files)}")
            print(f"Total size: {info.total_size:,} bytes\n")

        # Create files
        for path_parts, _size in info.files:
            try:
                # Build path using pathlib
                file_path = base_dir
                for part in path_parts:
                    file_path = file_path / part

                # Create parent directories
                file_path.parent.mkdir(parents=True, exist_ok=True)
                if file_path.parent not in created_dirs:
                    created_dirs.append(file_path.parent)

                # Create empty file
                file_path.touch()
                created_files.append(file_path)

                if config.verbose and len(created_files) % 100 == 0:
                    try:
                        rel_path = file_path.relative_to(base_dir)
                        print(f"  Created: {rel_path} (0 bytes)")
                    except ValueError:
                        print(f"  Created: .../{file_path.name} (0 bytes)")

            except OSError as e:
                errors.append(f"Failed to create {file_path.name}: {e}")

        if config.verbose:
            print(f"\nDone! {len(created_files)} file(s) created")

    except Exception as e:
        errors.append(f"Error processing {torrent_path}: {e}")

    return TorrentResult(
        created_files=created_files,
        created_dirs=created_dirs,
        errors=errors,
    )


def create_from_torrents(
    torrent_paths: list[Path],
    config: TorrentConfig,
) -> list[TorrentResult]:
    """Create fake files from multiple torrents.

    Args:
        torrent_paths: List of .torrent file paths
        config: Torrent configuration

    Returns:
        List of TorrentResult objects, one per torrent
    """
    results = []
    for torrent_path in torrent_paths:
        result = create_from_torrent(torrent_path, config)
        results.append(result)
    return results
