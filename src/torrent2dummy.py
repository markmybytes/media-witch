#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This module is no longer maintained.

This file has been superseded by the refactored media-witch package (v2.0.0+).

Please use the new CLI instead:
  pip install -e .
  media-witch torrent ./torrents/*.torrent --output-dir ./output

For programmatic use, import from the modular package:
  from media_witch.features.torrent.api import create_from_torrent, TorrentConfig
  from pathlib import Path

  config = TorrentConfig(output_dir=Path("./output"))
  result = create_from_torrent(Path("file.torrent"), config)

This file will be removed in v2.1.0.
"""

import argparse
import os
import sys
from typing import Any, Dict, List


def bdecode(data: bytes) -> Any:
    """Fixed bencode decoder for Python 3."""
    def decode_any(i: int) -> tuple[Any, int]:
        c = data[i]

        if c == ord('i'):
            end = data.index(b'e', i)
            return int(data[i + 1:end]), end + 1
        elif c == ord('l'):
            result = []
            i += 1
            while data[i] != ord('e'):
                item, i = decode_any(i)
                result.append(item)
            return result, i + 1
        elif c == ord('d'):
            result: Dict[bytes, Any] = {}
            i += 1
            while data[i] != ord('e'):
                key, i = decode_any(i)
                if not isinstance(key, bytes):
                    raise ValueError("Dictionary keys must be bytes")
                value, i = decode_any(i)
                result[key] = value
            return result, i + 1
        elif 48 <= c <= 57:
            colon = data.index(b':', i)
            length = int(data[i:colon])
            return data[colon + 1 : colon + 1 + length], colon + 1 + length
        else:
            raise ValueError(f"Invalid bencode character at position {i}: {chr(c)}")

    result, _ = decode_any(0)
    return result


def read_torrent_info(torrent_path: str) -> Dict[bytes, Any]:
    with open(torrent_path, 'rb') as f:
        return bdecode(f.read())


def print_torrent_files(torrent: Dict[bytes, Any], torrent_filename: str) -> None:
    info = torrent[b'info']
    name = info[b'name'].decode('utf-8', errors='replace')
    print(f"\n{'='*60}")
    print(f"Processing: {torrent_filename}")
    print(f"Torrent Name: {name}")

    if b'files' in info:
        files_list = info[b'files']
        print(f"Total files: {len(files_list)}")
        for file_dict in files_list:
            path = '/'.join(p.decode('utf-8', errors='replace') for p in file_dict[b'path'])
            print(f"  {path}  ({file_dict[b'length']:,} bytes)")
        print(f"Total size: {sum(f[b'length'] for f in files_list):,} bytes")
    else:
        size = info[b'length']
        print(f"Single file: {name}  ({size:,} bytes)")
        print(f"Total size: {size:,} bytes")


def find_all_torrents(folder_path: str) -> List[str]:
    """Return sorted list of all .torrent files in the folder."""
    if not os.path.isdir(folder_path):
        return []
    return [os.path.join(folder_path, item) for item in sorted(os.listdir(folder_path))
            if item.lower().endswith('.torrent')]


def create_fake_files(torrent_path: str, output_dir: str = None) -> None:
    """Create empty fake files for one torrent."""
    folder_name = os.path.splitext(os.path.basename(torrent_path))[0]
    output_dir = output_dir or os.getcwd()
    base_dir = os.path.join(output_dir, folder_name)

    torrent = read_torrent_info(torrent_path)
    info = torrent[b'info']
    os.makedirs(base_dir, exist_ok=True)
    print(f"Creating empty fake files in: {base_dir}")

    created_count = 0
    if b'files' in info:
        for file_dict in info[b'files']:
            path_parts = [p.decode('utf-8', errors='replace') for p in file_dict[b'path']]
            full_path = os.path.join(base_dir, *path_parts)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            open(full_path, 'wb').close()
            print(f"  Created: {os.path.relpath(full_path, base_dir)}  (0 bytes)")
            created_count += 1
    else:
        name = info[b'name'].decode('utf-8', errors='replace')
        full_path = os.path.join(base_dir, name)
        open(full_path, 'wb').close()
        print(f"  Created: {name}  (0 bytes)")
        created_count = 1

    print(f"Done! {created_count} empty fake file(s) created in folder: {folder_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create empty fake files for all .torrent files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 torrent2dummy.py ./my_torrents -o ./output"
    )
    parser.add_argument("path", help="Path to .torrent file or folder containing .torrent files")
    parser.add_argument(
        "-o", "--output-dir",
        default=os.getcwd(),
        help="Output directory for fake files (default: current directory)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path not found → {args.path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.path):
        if not args.path.lower().endswith('.torrent'):
            print(f"Error: File must be a .torrent file → {args.path}", file=sys.stderr)
            sys.exit(1)
        torrent_files = [args.path]
    else:
        torrent_files = find_all_torrents(args.path)
        if not torrent_files:
            print(f"Error: No .torrent files found in folder: {args.path}", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(torrent_files)} .torrent file(s)\n")

    for torrent_path in torrent_files:
        try:
            torrent = read_torrent_info(torrent_path)
            print_torrent_files(torrent, os.path.basename(torrent_path))
            create_fake_files(torrent_path, args.output_dir)
        except Exception as e:
            print(f"Error processing {os.path.basename(torrent_path)}: {e}")
            print("Skipping to next torrent...\n")
            continue

    print(f"{'='*60}")
    print("All done! Processed all .torrent files.")


if __name__ == "__main__":
    main()

