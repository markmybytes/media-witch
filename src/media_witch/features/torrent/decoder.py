"""Bencode encoder and decoder for torrent files."""

from __future__ import annotations

from typing import Any


def bencode(data: Any) -> bytes:
    """Encode Python data to bencode format.

    Bencode is the encoding used by BitTorrent for storing and transmitting
    loosely structured data.

    Args:
        data: Python object (dict, list, int, bytes, or str)

    Returns:
        Bencode-encoded bytes

    Raises:
        TypeError: If data contains unsupported types
        ValueError: If dict contains non-bytes/non-str keys
    """
    if isinstance(data, int):
        return f"i{data}e".encode()

    elif isinstance(data, bytes):
        return f"{len(data)}:".encode() + data

    elif isinstance(data, str):
        # Convert str to bytes for encoding
        data_bytes = data.encode('utf-8')
        return f"{len(data_bytes)}:".encode() + data_bytes

    elif isinstance(data, list):
        encoded_items = b"".join(bencode(item) for item in data)
        return b"l" + encoded_items + b"e"

    elif isinstance(data, dict):
        # Bencode requires keys to be sorted
        sorted_items = []
        for key, value in sorted(data.items()):
            if isinstance(key, str):
                key = key.encode('utf-8')
            elif not isinstance(key, bytes):
                raise ValueError(f"Dictionary keys must be bytes or str, got {type(key)}")
            sorted_items.append(bencode(key))
            sorted_items.append(bencode(value))
        return b"d" + b"".join(sorted_items) + b"e"

    else:
        raise TypeError(f"Unsupported type for bencode: {type(data)}")


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
        try:
            c = data[i]

            # Integer: i<num>e
            if c == ord('i'):
                end = data.index(b'e', i)
                return int(data[i + 1:end]), end + 1

            # List: l<items>e
            elif c == ord('l'):
                list_result: list[Any] = []
                i += 1
                while data[i] != ord('e'):
                    item, i = decode_any(i)
                    list_result.append(item)
                return list_result, i + 1

            # Dictionary: d<key><value>...e
            elif c == ord('d'):
                dict_result: dict[bytes, Any] = {}
                i += 1
                while data[i] != ord('e'):
                    key, i = decode_any(i)
                    if not isinstance(key, bytes):
                        raise ValueError("Dictionary keys must be bytes")
                    value, i = decode_any(i)
                    dict_result[key] = value
                return dict_result, i + 1

            # String: <length>:<data>
            elif 48 <= c <= 57:  # ASCII digits 0-9
                colon = data.index(b':', i)
                length = int(data[i:colon])
                end_pos = colon + 1 + length
                if end_pos > len(data):
                    raise ValueError(
                        f"String length {length} exceeds available data at position {i}")
                return data[colon + 1: end_pos], end_pos

            else:
                raise ValueError(
                    f"Invalid bencode character at position {i}: {chr(c)}")
        except IndexError as e:
            raise ValueError(f"Unexpected end of data at position {i}") from e

    try:
        result, _ = decode_any(0)
        return result
    except IndexError as e:
        raise ValueError(
            "Cannot decode empty or malformed bencode data") from e
