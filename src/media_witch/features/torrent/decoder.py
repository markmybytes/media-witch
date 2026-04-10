"""Bencode decoder for torrent files."""

from __future__ import annotations

from typing import Any


def bdecode(data: bytes) -> Any:
    """Decode bencode-encoded data.

    Bencode is the encoding used by BitTorrent for storing and transmitting
    loosely structured data.

    Args:
        data: Bencode-encoded bytes

    Returns:
        Decoded Python object (dict, list, int, or bytes)

    Raises:
        ValueError: If data is not valid bencode
    """
    def decode_any(i: int) -> tuple[Any, int]:
        """Decode any bencode type starting at position i.

        Returns:
            Tuple of (decoded value, next position)
        """
        c = data[i]

        # Integer: i<num>e
        if c == ord('i'):
            end = data.index(b'e', i)
            return int(data[i + 1:end]), end + 1

        # List: l<items>e
        elif c == ord('l'):
            result = []
            i += 1
            while data[i] != ord('e'):
                item, i = decode_any(i)
                result.append(item)
            return result, i + 1

        # Dictionary: d<key><value>...e
        elif c == ord('d'):
            result: dict[bytes, Any] = {}
            i += 1
            while data[i] != ord('e'):
                key, i = decode_any(i)
                if not isinstance(key, bytes):
                    raise ValueError("Dictionary keys must be bytes")
                value, i = decode_any(i)
                result[key] = value
            return result, i + 1

        # String: <length>:<data>
        elif 48 <= c <= 57:  # ASCII digits 0-9
            colon = data.index(b':', i)
            length = int(data[i:colon])
            return data[colon + 1: colon + 1 + length], colon + 1 + length

        else:
            raise ValueError(
                f"Invalid bencode character at position {i}: {chr(c)}")

    result, _ = decode_any(0)
    return result
