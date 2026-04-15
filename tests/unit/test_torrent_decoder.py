"""Unit tests for bencode decoder."""

import pytest

from media_witch.features.torrent.decoder import bdecode


class TestBdecodeIntegers:
    """Tests for decoding bencode integers."""

    def test_decode_positive_integer(self) -> None:
        """Test decoding positive integers."""
        assert bdecode(b"i42e") == 42
        assert bdecode(b"i0e") == 0
        assert bdecode(b"i123456789e") == 123456789

    def test_decode_negative_integer(self) -> None:
        """Test decoding negative integers."""
        assert bdecode(b"i-42e") == -42
        assert bdecode(b"i-1e") == -1
        assert bdecode(b"i-999e") == -999

    def test_decode_zero(self) -> None:
        """Test decoding zero."""
        assert bdecode(b"i0e") == 0


class TestBdecodeStrings:
    """Tests for decoding bencode strings."""

    def test_decode_simple_string(self) -> None:
        """Test decoding simple strings."""
        assert bdecode(b"4:spam") == b"spam"
        assert bdecode(b"0:") == b""
        assert bdecode(b"3:foo") == b"foo"

    def test_decode_long_string(self) -> None:
        """Test decoding longer strings."""
        data = b"test data with spaces"
        encoded = f"{len(data)}:".encode() + data
        assert bdecode(encoded) == data

    def test_decode_string_with_special_chars(self) -> None:
        """Test decoding strings with special characters."""
        assert bdecode(b"5:hello") == b"hello"
        assert bdecode(b"11:hello world") == b"hello world"

    def test_decode_empty_string(self) -> None:
        """Test decoding empty string."""
        assert bdecode(b"0:") == b""


class TestBdecodeLists:
    """Tests for decoding bencode lists."""

    def test_decode_empty_list(self) -> None:
        """Test decoding empty list."""
        assert bdecode(b"le") == []

    def test_decode_simple_list(self) -> None:
        """Test decoding list of integers."""
        assert bdecode(b"li1ei2ei3ee") == [1, 2, 3]

    def test_decode_mixed_list(self) -> None:
        """Test decoding list with mixed types."""
        assert bdecode(b"l4:spami42ee") == [b"spam", 42]

    def test_decode_nested_list(self) -> None:
        """Test decoding nested lists."""
        assert bdecode(b"lli1eeli2eee") == [[1], [2]]

    def test_decode_list_with_strings(self) -> None:
        """Test decoding list of strings."""
        assert bdecode(b"l3:foo3:bar3:baze") == [b"foo", b"bar", b"baz"]


class TestBdecodeDictionaries:
    """Tests for decoding bencode dictionaries."""

    def test_decode_empty_dict(self) -> None:
        """Test decoding empty dictionary."""
        assert bdecode(b"de") == {}

    def test_decode_simple_dict(self) -> None:
        """Test decoding simple dictionary."""
        result = bdecode(b"d3:bar4:spam3:fooi42ee")
        assert result == {b"bar": b"spam", b"foo": 42}

    def test_decode_dict_with_list_value(self) -> None:
        """Test decoding dictionary with list as value."""
        result = bdecode(b"d4:listli1ei2ei3eee")
        assert result == {b"list": [1, 2, 3]}

    def test_decode_nested_dict(self) -> None:
        """Test decoding nested dictionaries."""
        result = bdecode(b"d5:innerd3:keyi42eee")
        assert result == {b"inner": {b"key": 42}}

    def test_decode_dict_maintains_order(self) -> None:
        """Test that dictionary keys are decoded correctly."""
        result = bdecode(b"d1:ai1e1:bi2e1:ci3ee")
        assert result[b"a"] == 1
        assert result[b"b"] == 2
        assert result[b"c"] == 3


class TestBdecodeComplexStructures:
    """Tests for decoding complex bencode structures."""

    def test_decode_torrent_like_structure(self) -> None:
        """Test decoding a structure similar to torrent metadata."""
        # Simplified torrent info dict: d4:name4:test6:lengthi1024ee
        result = bdecode(b"d4:name4:test6:lengthi1024ee")
        assert result == {b"name": b"test", b"length": 1024}

    def test_decode_multi_file_structure(self) -> None:
        """Test decoding multi-file torrent structure."""
        # d5:filesl d4:name5:file1 6:lengthi100e e e e
        data = b"d5:filesld4:name5:file16:lengthi100eeee"
        result = bdecode(data)
        assert b"files" in result
        assert isinstance(result[b"files"], list)
        assert len(result[b"files"]) == 1

    def test_decode_deeply_nested(self) -> None:
        """Test decoding deeply nested structures."""
        data = b"d1:ad1:bd1:cd1:di42eeeeee"
        result = bdecode(data)
        assert result[b"a"][b"b"][b"c"][b"d"] == 42


class TestBdecodeErrors:
    """Tests for bencode decoder error handling."""

    def test_invalid_start_character(self) -> None:
        """Test that invalid start character raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bencode character"):
            bdecode(b"x42e")

    def test_malformed_integer(self) -> None:
        """Test that malformed integer raises error."""
        with pytest.raises(ValueError):
            bdecode(b"i42")  # Missing 'e'

    def test_malformed_string(self) -> None:
        """Test that malformed string raises error."""
        with pytest.raises(ValueError):
            bdecode(b"5:ab")  # String too short

    def test_malformed_list(self) -> None:
        """Test that malformed list raises error."""
        with pytest.raises(ValueError):
            bdecode(b"li1e")  # Missing 'e'

    def test_malformed_dict(self) -> None:
        """Test that malformed dictionary raises error."""
        with pytest.raises(ValueError):
            bdecode(b"d3:foo")  # Missing value and 'e'

    def test_dict_with_non_bytes_key(self) -> None:
        """Test that dictionary with non-bytes key raises error."""
        with pytest.raises(ValueError, match="Dictionary keys must be bytes"):
            bdecode(b"di1ei2ee")  # Integer key instead of bytes

    def test_empty_input(self) -> None:
        """Test that empty input raises error."""
        with pytest.raises((ValueError, IndexError)):
            bdecode(b"")


class TestBdecodeRealWorldExamples:
    """Tests with real-world bencode examples."""

    def test_announce_url(self) -> None:
        """Test decoding announce URL from torrent."""
        data = b"d8:announce23:http://tracker.test:806e"
        result = bdecode(data)
        assert result[b"announce"] == b"http://tracker.test:806"

    def test_piece_length(self) -> None:
        """Test decoding piece length."""
        data = b"d12:piece lengthi262144ee"
        result = bdecode(data)
        assert result[b"piece length"] == 262144

    def test_info_hash_structure(self) -> None:
        """Test decoding basic info dictionary structure."""
        data = b"d4:infod4:name8:testfile6:lengthi1024eee"
        result = bdecode(data)
        assert b"info" in result
        assert result[b"info"][b"name"] == b"testfile"
        assert result[b"info"][b"length"] == 1024
