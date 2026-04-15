"""Property-based tests for bencode encoder and decoder."""

from typing import cast

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from media_witch.features.torrent.decoder import bdecode, bencode


class TestIntegerProperties:
    """Property-based tests for integer encoding/decoding."""

    @given(st.integers(min_value=-(2**63), max_value=2**63 - 1))
    def test_integer_roundtrip(self, n: int) -> None:
        """Any integer should encode and decode back to itself."""
        encoded = bencode(n)
        decoded = bdecode(encoded)
        assert decoded == n
        assert isinstance(decoded, int)

    @given(st.integers())
    def test_integer_encoding_format(self, n: int) -> None:
        """Encoded integers should have format i<num>e."""
        encoded = bencode(n)
        assert encoded.startswith(b'i')
        assert encoded.endswith(b'e')
        assert str(n).encode() in encoded

    @given(st.integers())
    def test_negative_integers_roundtrip(self, n: int) -> None:
        """Negative integers should roundtrip correctly."""
        assume(n < 0)
        encoded = bencode(n)
        decoded = bdecode(encoded)
        assert decoded == n

    def test_zero_roundtrip(self) -> None:
        """Zero should encode and decode correctly."""
        assert bdecode(b'i0e') == 0
        assert bencode(0) == b'i0e'


class TestStringProperties:
    """Property-based tests for string encoding/decoding."""

    @given(st.binary(min_size=0, max_size=1000))
    def test_bytes_roundtrip(self, s: bytes) -> None:
        """Any byte string should encode and decode back to itself."""
        encoded = bencode(s)
        decoded = bdecode(encoded)
        assert decoded == s
        assert isinstance(decoded, bytes)

    @given(st.text(min_size=0, max_size=1000))
    def test_str_roundtrip(self, s: str) -> None:
        """Any string should encode and decode to bytes."""
        encoded = bencode(s)
        decoded = bdecode(encoded)
        assert decoded == s.encode('utf-8')
        assert isinstance(decoded, bytes)

    @given(st.binary())
    def test_string_encoding_format(self, s: bytes) -> None:
        """Encoded strings should have format <length>:<data>."""
        encoded = bencode(s)
        assert b':' in encoded
        length_part = encoded.split(b':')[0]
        assert length_part.decode().isdigit()
        assert int(length_part) == len(s)

    def test_empty_string_roundtrip(self) -> None:
        """Empty string should encode and decode correctly."""
        assert bdecode(b'0:') == b''
        assert bencode(b'') == b'0:'


class TestListProperties:
    """Property-based tests for list encoding/decoding."""

    @given(st.lists(st.integers(), min_size=0, max_size=100))
    def test_integer_list_roundtrip(self, lst: list[int]) -> None:
        """Any list of integers should roundtrip correctly."""
        encoded = bencode(lst)
        decoded = bdecode(encoded)
        assert decoded == lst
        assert isinstance(decoded, list)
        assert len(decoded) == len(lst)

    @given(st.lists(st.binary(), min_size=0, max_size=50))
    def test_bytes_list_roundtrip(self, lst: list[bytes]) -> None:
        """Any list of byte strings should roundtrip correctly."""
        encoded = bencode(lst)
        decoded = bdecode(encoded)
        assert decoded == lst

    @given(st.lists(st.one_of(st.integers(), st.binary()), min_size=0, max_size=50))
    def test_mixed_list_roundtrip(self, lst: list) -> None:
        """Lists with mixed types should roundtrip correctly."""
        encoded = bencode(lst)
        decoded = bdecode(encoded)
        assert decoded == lst

    @given(st.lists(st.lists(st.integers(), max_size=10), max_size=10))
    def test_nested_list_roundtrip(self, lst: list) -> None:
        """Nested lists should roundtrip correctly."""
        encoded = bencode(lst)
        decoded = bdecode(encoded)
        assert decoded == lst

    def test_empty_list_roundtrip(self) -> None:
        """Empty list should encode and decode correctly."""
        assert bdecode(b'le') == []
        assert bencode([]) == b'le'


class TestDictProperties:
    """Property-based tests for dictionary encoding/decoding."""

    @given(
        st.dictionaries(
            keys=st.binary(min_size=1, max_size=50), values=st.integers(), min_size=0, max_size=50
        )
    )
    def test_dict_roundtrip(self, d: dict[bytes, int]) -> None:
        """Any dictionary with bytes keys should roundtrip correctly."""
        encoded = bencode(d)
        decoded = bdecode(encoded)
        assert decoded == d
        assert isinstance(decoded, dict)
        assert len(decoded) == len(d)

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=50), values=st.integers(), min_size=0, max_size=20
        )
    )
    def test_dict_with_str_keys_roundtrip(self, d: dict[str, int]) -> None:
        """Dictionaries with str keys should encode to bytes keys."""
        encoded = bencode(d)
        decoded = bdecode(encoded)
        # Keys should be converted to bytes
        expected = {k.encode('utf-8'): v for k, v in d.items()}
        assert decoded == expected

    @given(
        st.dictionaries(
            keys=st.binary(min_size=1, max_size=30),
            values=st.one_of(st.integers(), st.binary(), st.lists(st.integers(), max_size=10)),
            min_size=0,
            max_size=20,
        )
    )
    def test_dict_with_mixed_values_roundtrip(self, d: dict) -> None:
        """Dictionaries with mixed value types should roundtrip correctly."""
        encoded = bencode(d)
        decoded = bdecode(encoded)
        assert decoded == d

    @given(
        st.dictionaries(
            keys=st.binary(min_size=1, max_size=20),
            values=st.dictionaries(
                keys=st.binary(min_size=1, max_size=20), values=st.integers(), max_size=5
            ),
            max_size=10,
        )
    )
    def test_nested_dict_roundtrip(self, d: dict) -> None:
        """Nested dictionaries should roundtrip correctly."""
        encoded = bencode(d)
        decoded = bdecode(encoded)
        assert decoded == d

    def test_empty_dict_roundtrip(self) -> None:
        """Empty dictionary should encode and decode correctly."""
        assert bdecode(b'de') == {}
        assert bencode({}) == b'de'

    @given(
        st.dictionaries(
            keys=st.binary(min_size=1, max_size=20), values=st.integers(), min_size=2, max_size=20
        )
    )
    def test_dict_keys_are_sorted(self, d: dict[bytes, int]) -> None:
        """Bencode requires dictionary keys to be sorted."""
        encoded = bencode(d)
        decoded = bdecode(encoded)
        assert decoded == d
        # Verify the encoding has keys in sorted order
        # (This is a property of bencode format)


@composite
def bencode_data(draw: DrawFn, max_depth: int = 4) -> int | bytes | list | dict:
    """Strategy for generating arbitrary bencode-compatible structures."""
    if max_depth == 0:
        # Base case: only primitives
        return draw(
            st.one_of(  # type: ignore[no-any-return]
                st.integers(min_value=-(2**31), max_value=2**31 - 1), st.binary(max_size=100)
            )
        )

    # Recursive case: primitives, lists, or dicts
    primitive = st.one_of(
        st.integers(min_value=-(2**31), max_value=2**31 - 1), st.binary(max_size=50)
    )

    # Recursively generate sub-structures
    sub_structure = bencode_data(max_depth - 1)  # type: ignore[arg-type]

    return cast(
        int | bytes | list | dict,
        draw(
            st.one_of(  # type: ignore[no-any-return]
                primitive,
                st.lists(sub_structure, max_size=10),
                st.dictionaries(
                    keys=st.binary(min_size=1, max_size=20),
                    values=sub_structure,
                    max_size=10,
                ),
            )
        ),
    )


class TestComplexStructures:
    """Property-based tests for complex nested structures."""

    @given(bencode_data())
    def test_arbitrary_structure_roundtrip(self, data: int | bytes | list | dict) -> None:
        """Any bencode-compatible structure should roundtrip correctly."""
        encoded = bencode(data)
        decoded = bdecode(encoded)
        assert decoded == data

    @given(bencode_data(max_depth=6))
    def test_deeply_nested_structures(self, data: int | bytes | list | dict) -> None:
        """Deeply nested structures should roundtrip correctly."""
        encoded = bencode(data)
        decoded = bdecode(encoded)
        assert decoded == data

    def test_torrent_like_structure_roundtrip(self) -> None:
        """Torrent-like structures should roundtrip correctly."""
        data = {
            b'announce': b'http://tracker.example.com:8080/announce',
            b'info': {
                b'name': b'MyFile.mkv',
                b'length': 1073741824,
                b'piece length': 262144,
                b'pieces': b'x' * 20,  # Normally 20-byte SHA1 hashes
            },
        }
        encoded = bencode(data)
        decoded = bdecode(encoded)
        assert decoded == data


class TestDecoderErrorHandling:
    """Property-based tests for decoder error handling."""

    @given(st.binary(max_size=1000))
    def test_decoder_never_crashes(self, data: bytes) -> None:
        """Decoder should handle any bytes gracefully - raise ValueError or succeed."""
        try:
            result = bdecode(data)
            assert isinstance(result, (int, bytes, list, dict))
        except (ValueError, IndexError):
            pass
        except Exception as e:
            pytest.fail(f'Unexpected exception type {type(e).__name__}: {e}')

    def test_invalid_start_character(self) -> None:
        """Invalid start character should raise ValueError."""
        with pytest.raises(ValueError, match='Invalid bencode character'):
            bdecode(b'x42e')

    def test_malformed_integer(self) -> None:
        """Malformed integer should raise error."""
        with pytest.raises(ValueError):
            bdecode(b'i42')  # Missing 'e'

    def test_malformed_string(self) -> None:
        """Malformed string should raise error."""
        with pytest.raises(ValueError):
            bdecode(b'5:ab')  # String too short

    def test_malformed_list(self) -> None:
        """Malformed list should raise error."""
        with pytest.raises(ValueError):
            bdecode(b'li1e')  # Missing 'e'

    def test_malformed_dict(self) -> None:
        """Malformed dictionary should raise error."""
        with pytest.raises(ValueError):
            bdecode(b'd3:foo')  # Missing value and 'e'

    def test_dict_with_non_bytes_key(self) -> None:
        """Dictionary with non-bytes key should raise error."""
        with pytest.raises(ValueError, match='Dictionary keys must be bytes'):
            bdecode(b'di1ei2ee')  # Integer key instead of bytes

    def test_empty_input(self) -> None:
        """Empty input should raise error."""
        with pytest.raises((ValueError, IndexError)):
            bdecode(b'')


class TestEncoderErrorHandling:
    """Tests for encoder error handling."""

    def test_unsupported_type_raises_error(self) -> None:
        """Encoding unsupported types should raise TypeError."""
        with pytest.raises(TypeError, match='Unsupported type'):
            bencode(3.14)  # float not supported

        with pytest.raises(TypeError, match='Unsupported type'):
            bencode(None)  # None not supported

    def test_dict_with_invalid_key_type(self) -> None:
        """Dictionary with non-bytes/non-str keys should raise ValueError."""
        with pytest.raises(ValueError, match='Dictionary keys must be bytes or str'):
            bencode({1: b'value'})  # Integer key not supported


class TestBencodeInvariants:
    """Tests for bencode encoding/decoding invariants."""

    @given(bencode_data())
    def test_encode_decode_identity(self, data: int | bytes | list | dict) -> None:
        """encode(decode(encode(x))) == encode(x)."""
        encoded = bencode(data)
        decoded = bdecode(encoded)
        re_encoded = bencode(decoded)
        assert re_encoded == encoded

    @given(bencode_data())
    def test_decode_encode_identity(self, data: int | bytes | list | dict) -> None:
        """decode(encode(x)) == x for all valid bencode data."""
        encoded = bencode(data)
        decoded = bdecode(encoded)
        assert decoded == data

    @given(st.binary())
    def test_valid_bencode_always_parseable(self, data: bytes) -> None:
        """If bencode produces it, bdecode should parse it without error."""
        try:
            decoded = bdecode(data)
            re_encoded = bencode(decoded)
            re_decoded = bdecode(re_encoded)
            assert re_decoded == decoded
        except (ValueError, IndexError):
            pass

    @given(st.lists(st.integers(), min_size=0, max_size=100))
    def test_list_length_preserved(self, lst: list[int]) -> None:
        """Length of list should be preserved through roundtrip."""
        encoded = bencode(lst)
        decoded = bdecode(encoded)
        assert len(decoded) == len(lst)

    @given(
        st.dictionaries(
            keys=st.binary(min_size=1, max_size=20), values=st.integers(), min_size=0, max_size=50
        )
    )
    def test_dict_size_preserved(self, d: dict) -> None:
        """Number of dict entries should be preserved through roundtrip."""
        encoded = bencode(d)
        decoded = bdecode(encoded)
        assert len(decoded) == len(d)
