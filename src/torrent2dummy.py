#!/usr/bin/env python3
"""
Torrent Fake File Creator - Process ALL .torrent files
- Input: a folder path
- Scans the folder and processes EVERY .torrent file found
- For each torrent: creates empty fake files (0 bytes)
- Output folder for each = torrent filename (without .torrent) in current directory
"""

import os
import sys
from typing import Any, Dict, List


# ====================== Fixed Pure Python Bencode Decoder ======================
def bdecode(data: bytes) -> Any:
    """Fixed bencode decoder for Python 3."""
    def decode_any(i: int) -> tuple[Any, int]:
        c = data[i]

        if c == ord('i'):                                   # Integer
            end = data.index(b'e', i)
            return int(data[i + 1:end]), end + 1

        elif c == ord('l'):                                 # List
            result = []
            i += 1
            while data[i] != ord('e'):
                item, i = decode_any(i)
                result.append(item)
            return result, i + 1

        elif c == ord('d'):                                 # Dictionary
            result: Dict[bytes, Any] = {}
            i += 1
            while data[i] != ord('e'):
                key, i = decode_any(i)
                if not isinstance(key, bytes):
                    raise ValueError("Dictionary keys must be bytes")
                value, i = decode_any(i)
                result[key] = value
            return result, i + 1

        elif 48 <= c <= 57:                                 # String
            colon = data.index(b':', i)
            length = int(data[i:colon])
            return data[colon + 1 : colon + 1 + length], colon + 1 + length

        else:
            raise ValueError(f"Invalid bencode character at position {i}: {chr(c)}")

    result, _ = decode_any(0)
    return result


# ====================== Main Functions ======================
def read_torrent_info(torrent_path: str) -> Dict[bytes, Any]:
    with open(torrent_path, 'rb') as f:
        data = f.read()
    return bdecode(data)


def print_torrent_files(torrent: Dict[bytes, Any], torrent_filename: str) -> None:
    info = torrent[b'info']
    name = info[b'name'].decode('utf-8', errors='replace')
    print(f"\n{'='*60}")
    print(f"Processing: {torrent_filename}")
    print(f"Torrent Name: {name}")

    if b'files' in info:
        print(f"Total files: {len(info[b'files'])}")
        total_size = sum(f[b'length'] for f in info[b'files'])
        for file_dict in info[b'files']:
            path = '/'.join(p.decode('utf-8', errors='replace') for p in file_dict[b'path'])
            size = file_dict[b'length']
            print(f"  {path}  ({size:,} bytes)")
        print(f"Total size: {total_size:,} bytes")
    else:
        size = info[b'length']
        print(f"Single file: {name}  ({size:,} bytes)")
        print(f"Total size: {size:,} bytes")


def find_all_torrents(folder_path: str) -> List[str]:
    """Return sorted list of all .torrent files in the folder."""
    if not os.path.isdir(folder_path):
        return []
    
    torrents = []
    for item in sorted(os.listdir(folder_path)):
        if item.lower().endswith('.torrent'):
            torrents.append(os.path.join(folder_path, item))
    
    return torrents


def create_fake_files(torrent_path: str) -> None:
    """Create empty fake files for one torrent."""
    torrent_name = os.path.basename(torrent_path)
    folder_name = os.path.splitext(torrent_name)[0]

    base_dir = os.path.join(os.getcwd(), folder_name)

    torrent = read_torrent_info(torrent_path)
    info = torrent[b'info']

    os.makedirs(base_dir, exist_ok=True)
    print(f"Creating empty fake files in: {base_dir}")

    created_count = 0

    if b'files' in info:  # Multi-file torrent
        for file_dict in info[b'files']:
            path_parts = [p.decode('utf-8', errors='replace') for p in file_dict[b'path']]
            full_path = os.path.join(base_dir, *path_parts)

            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Create empty file (0 bytes)
            with open(full_path, 'wb') as f:
                pass

            print(f"  Created: {os.path.relpath(full_path, base_dir)}  (0 bytes)")
            created_count += 1
    else:  # Single-file torrent
        name = info[b'name'].decode('utf-8', errors='replace')
        full_path = os.path.join(base_dir, name)

        with open(full_path, 'wb') as f:
            pass

        print(f"  Created: {name}  (0 bytes)")
        created_count = 1

    print(f"Done! {created_count} empty fake file(s) created in folder: {folder_name}\n")


# ====================== CLI ======================
def main():
    if len(sys.argv) < 2:
        script = os.path.basename(sys.argv[0])
        print("Usage:")
        print(f"  python3 {script} <folder_path>")
        print("\nExample:")
        print(f"  python3 {script} ./my_torrents")
        print("  → Processes ALL .torrent files in the folder one by one")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found → {folder_path}")
        sys.exit(1)

    torrent_files = find_all_torrents(folder_path)

    if not torrent_files:
        print(f"Error: No .torrent files found in folder: {folder_path}")
        sys.exit(1)

    print(f"Found {len(torrent_files)} .torrent file(s) in {folder_path}\n")

    for torrent_path in torrent_files:
        try:
            torrent = read_torrent_info(torrent_path)
            print_torrent_files(torrent, os.path.basename(torrent_path))
            create_fake_files(torrent_path)
        except Exception as e:
            print(f"Error processing {os.path.basename(torrent_path)}: {e}")
            print("Skipping to next torrent...\n")
            continue

    print(f"{'='*60}")
    print("All done! Processed all .torrent files.")


if __name__ == "__main__":
    main()