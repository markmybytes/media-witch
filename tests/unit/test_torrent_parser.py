"""Property-based tests for torrent parser."""

from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.strategies import composite

from media_witch.features.torrent.parser import TorrentInfo, parse_torrent


@composite
def valid_filename_bytes(draw, min_size: int = 1, max_size: int = 100):
    """Generate valid filename as bytes."""
    name = draw(st.text(
        alphabet=st.characters(
            min_codepoint=0x20,  # Space
            max_codepoint=0x7E,  # ~
            blacklist_characters='<>:"/\\|?*\x00',  # Invalid filename chars
        ),
        min_size=min_size,
        max_size=max_size
    ))

    ext = draw(st.sampled_from(['.mkv', '.mp4', '.avi', '.txt', '.zip', '']))
    return (name.strip() + ext).encode('utf-8', errors='replace')


@composite
def single_file_torrent_data(draw):
    """Generate valid single-file torrent data."""
    name = draw(valid_filename_bytes())
    length = draw(st.integers(min_value=0, max_value=10**12))

    return {
        b"info": {
            b"name": name,
            b"length": length
        }
    }


@composite
def file_entry(draw):
    """Generate a single file entry for multi-file torrents."""
    num_components = draw(st.integers(min_value=1, max_value=5))
    path = [
        draw(valid_filename_bytes(min_size=1, max_size=50))
        for _ in range(num_components)
    ]
    length = draw(st.integers(min_value=0, max_value=10**10))

    return {
        b"path": path,
        b"length": length
    }


@composite
def multi_file_torrent_data(draw):
    """Generate valid multi-file torrent data."""
    name = draw(valid_filename_bytes())
    num_files = draw(st.integers(min_value=0, max_value=100))
    files = [draw(file_entry()) for _ in range(num_files)]

    return {
        b"info": {
            b"name": name,
            b"files": files
        }
    }


@composite
def any_torrent_data(draw):
    """Generate either single-file or multi-file torrent data."""
    return draw(st.one_of(
        single_file_torrent_data(),
        multi_file_torrent_data()
    ))


class TestSingleFileTorrentProperties:
    """Property-based tests for single-file torrent handling."""

    @given(single_file_torrent_data())
    def test_single_file_is_single_file(self, torrent_data: dict) -> None:
        """Single-file torrents should have is_single_file=True."""
        info = TorrentInfo(torrent_data)
        assert info.is_single_file is True

    @given(single_file_torrent_data())
    def test_single_file_total_size_equals_length(self, torrent_data: dict) -> None:
        """Single-file torrent total_size should equal length field."""
        info = TorrentInfo(torrent_data)
        expected_size = torrent_data[b"info"][b"length"]
        assert info.total_size == expected_size

    @given(single_file_torrent_data())
    def test_single_file_has_one_file_entry(self, torrent_data: dict) -> None:
        """Single-file torrents should have exactly one file entry."""
        info = TorrentInfo(torrent_data)
        assert len(info.files) == 1

    @given(single_file_torrent_data())
    def test_single_file_name_decoded_correctly(self, torrent_data: dict) -> None:
        """Single-file torrent name should be decoded from bytes to str."""
        info = TorrentInfo(torrent_data)
        assert isinstance(info.name, str)

        original_name = torrent_data[b"info"][b"name"]
        assert info.name == original_name.decode('utf-8', errors='replace')

    @given(
        valid_filename_bytes(),
        st.integers(min_value=0, max_value=10**15)
    )
    def test_single_file_with_large_size(self, name: bytes, size: int) -> None:
        """Single-file torrents should handle very large file sizes."""
        data = {
            b"info": {
                b"name": name,
                b"length": size
            }
        }
        info = TorrentInfo(data)
        assert info.total_size == size

    @given(single_file_torrent_data())
    def test_single_file_files_list_matches_structure(self, torrent_data: dict) -> None:
        """Single-file torrent files list should match name and length."""
        info = TorrentInfo(torrent_data)
        path_parts, file_size = info.files[0]

        expected_name = torrent_data[b"info"][b"name"].decode(
            'utf-8', errors='replace')
        expected_size = torrent_data[b"info"][b"length"]

        assert path_parts == [expected_name]
        assert file_size == expected_size


class TestMultiFileTorrentProperties:
    """Property-based tests for multi-file torrent handling."""

    @given(multi_file_torrent_data())
    def test_multi_file_is_not_single_file(self, torrent_data: dict) -> None:
        """Multi-file torrents should have is_single_file=False."""
        info = TorrentInfo(torrent_data)
        assert info.is_single_file is False

    @given(multi_file_torrent_data())
    def test_multi_file_total_size_is_sum(self, torrent_data: dict) -> None:
        """Multi-file torrent total_size should equal sum of all file lengths."""
        info = TorrentInfo(torrent_data)
        files_data = torrent_data[b"info"][b"files"]
        expected_size = sum(f[b"length"] for f in files_data)

        assert info.total_size == expected_size

    @given(multi_file_torrent_data())
    def test_multi_file_count_matches(self, torrent_data: dict) -> None:
        """Number of files should match input."""
        info = TorrentInfo(torrent_data)
        expected_count = len(torrent_data[b"info"][b"files"])

        assert len(info.files) == expected_count

    @given(multi_file_torrent_data())
    def test_multi_file_name_decoded_correctly(self, torrent_data: dict) -> None:
        """Multi-file torrent name should be decoded correctly."""
        info = TorrentInfo(torrent_data)
        assert isinstance(info.name, str)

        original_name = torrent_data[b"info"][b"name"]
        assert info.name == original_name.decode('utf-8', errors='replace')

    @given(multi_file_torrent_data())
    def test_multi_file_paths_decoded_correctly(self, torrent_data: dict) -> None:
        """File paths should be decoded from bytes to str."""
        info = TorrentInfo(torrent_data)

        for (path_parts, _), file_data in zip(info.files, torrent_data[b"info"][b"files"]):
            assert all(isinstance(part, str) for part in path_parts)

            expected_path = [
                part.decode('utf-8', errors='replace')
                for part in file_data[b"path"]
            ]
            assert path_parts == expected_path

    @given(multi_file_torrent_data())
    def test_multi_file_sizes_match(self, torrent_data: dict) -> None:
        """File sizes should match input data."""
        info = TorrentInfo(torrent_data)

        for (_, size), file_data in zip(info.files, torrent_data[b"info"][b"files"]):
            assert size == file_data[b"length"]

    @given(st.integers(min_value=0, max_value=1000))
    def test_empty_files_list(self, total_size_should_be_zero: int) -> None:
        """Multi-file torrent with empty files list should have zero total size."""
        data = {
            b"info": {
                b"name": b"Empty",
                b"files": []
            }
        }
        info = TorrentInfo(data)

        assert info.files == []
        assert info.total_size == 0
        assert info.is_single_file is False


class TestTorrentEncodingProperties:
    """Property-based tests for handling different encodings."""

    @given(st.text(min_size=1, max_size=100))
    def test_utf8_name_roundtrip(self, name: str) -> None:
        """UTF-8 encoded names should decode correctly."""
        data = {
            b"info": {
                b"name": name.encode('utf-8'),
                b"length": 1024
            }
        }
        info = TorrentInfo(data)
        assert info.name == name

    @given(st.binary(min_size=1, max_size=100))
    def test_invalid_utf8_handled_gracefully(self, invalid_bytes: bytes) -> None:
        """Invalid UTF-8 should not crash, uses replacement character."""
        data = {
            b"info": {
                b"name": invalid_bytes,
                b"length": 1024
            }
        }

        info = TorrentInfo(data)
        assert isinstance(info.name, str)

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
    def test_utf8_file_paths(self, path_parts: list[str]) -> None:
        """UTF-8 encoded file paths should decode correctly."""
        data = {
            b"info": {
                b"name": b"Root",
                b"files": [
                    {
                        b"path": [part.encode('utf-8') for part in path_parts],
                        b"length": 100
                    }
                ]
            }
        }
        info = TorrentInfo(data)
        assert info.files[0][0] == path_parts


class TestParseTorrentProperties:
    """Property-based tests for parse_torrent file reading."""

    def test_parse_single_file_torrent(self, tmp_path: Path) -> None:
        """Parsing single-file torrent from file should work."""
        from media_witch.features.torrent.decoder import bencode

        torrent_data = {
            b"info": {
                b"name": b"test.file",
                b"length": 1024
            }
        }

        torrent_content = bencode(torrent_data)
        torrent_path = tmp_path / "test.torrent"
        torrent_path.write_bytes(torrent_content)

        info = parse_torrent(torrent_path)

        assert info.is_single_file is True
        assert info.total_size == 1024

    def test_parse_multi_file_torrent(self, tmp_path: Path) -> None:
        """Parsing multi-file torrent from file should work."""
        from media_witch.features.torrent.decoder import bencode

        torrent_data = {
            b"info": {
                b"name": b"Test",
                b"files": [
                    {b"path": [b"file1.txt"], b"length": 100},
                    {b"path": [b"file2.txt"], b"length": 200},
                ]
            }
        }

        torrent_content = bencode(torrent_data)
        torrent_path = tmp_path / "test.torrent"
        torrent_path.write_bytes(torrent_content)

        info = parse_torrent(torrent_path)

        assert info.is_single_file is False
        assert len(info.files) == 2

    def test_parse_nonexistent_file_raises_error(self) -> None:
        """Parsing non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_torrent(Path("/nonexistent/path/file.torrent"))

    def test_parse_invalid_bencode_raises_error(self, tmp_path: Path) -> None:
        """Parsing invalid bencode should raise ValueError."""
        torrent_path = tmp_path / "invalid.torrent"
        torrent_path.write_bytes(b"invalid bencode data")

        with pytest.raises((ValueError, IndexError)):
            parse_torrent(torrent_path)

    def test_parse_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Parsing empty file should raise error."""
        torrent_path = tmp_path / "empty.torrent"
        torrent_path.write_bytes(b"")

        with pytest.raises((ValueError, IndexError)):
            parse_torrent(torrent_path)


class TestTorrentInvariants:
    """Tests for invariants that should hold for all torrents."""

    @given(any_torrent_data())
    def test_name_is_always_string(self, torrent_data: dict) -> None:
        """Torrent name should always be a string."""
        info = TorrentInfo(torrent_data)
        assert isinstance(info.name, str)

    @given(any_torrent_data())
    def test_total_size_is_non_negative(self, torrent_data: dict) -> None:
        """Total size should never be negative."""
        info = TorrentInfo(torrent_data)
        assert info.total_size >= 0

    @given(any_torrent_data())
    def test_files_list_is_list_of_tuples(self, torrent_data: dict) -> None:
        """Files should be a list of (path_list, size) tuples."""
        info = TorrentInfo(torrent_data)
        assert isinstance(info.files, list)

        for item in info.files:
            assert isinstance(item, tuple)
            assert len(item) == 2
            path_parts, size = item
            assert isinstance(path_parts, list)
            assert isinstance(size, int)
            assert all(isinstance(part, str) for part in path_parts)

    @given(any_torrent_data())
    def test_is_single_file_is_boolean(self, torrent_data: dict) -> None:
        """is_single_file should always be a boolean."""
        info = TorrentInfo(torrent_data)
        assert isinstance(info.is_single_file, bool)

    @given(any_torrent_data())
    def test_file_sizes_sum_to_total_size(self, torrent_data: dict) -> None:
        """Sum of individual file sizes should equal total_size."""
        info = TorrentInfo(torrent_data)
        calculated_size = sum(size for _, size in info.files)
        assert calculated_size == info.total_size

    @given(any_torrent_data())
    def test_all_file_sizes_non_negative(self, torrent_data: dict) -> None:
        """All individual file sizes should be non-negative."""
        info = TorrentInfo(torrent_data)
        for _, size in info.files:
            assert size >= 0

    @given(any_torrent_data())
    def test_all_paths_non_empty(self, torrent_data: dict) -> None:
        """All file paths should have at least one component."""
        info = TorrentInfo(torrent_data)
        for path_parts, _ in info.files:
            assert len(path_parts) >= 1
            # Note: path parts can be empty strings (edge case from empty name bytes)
            # This is valid according to bencode spec


class TestRealWorldScenarios:
    """Property-based tests for realistic torrent structures."""

    @given(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz',
                min_size=1, max_size=30),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=50)
    )
    def test_tv_show_episode_structure(
        self, show_name: str, season: int, num_episodes: int
    ) -> None:
        """TV show season torrent structure."""
        files = []
        for ep in range(1, num_episodes + 1):
            files.append({
                b"path": [f"{show_name}.S{season:02d}E{ep:02d}.mkv".encode()],
                b"length": 1500000000 + ep * 1000
            })

        data = {
            b"info": {
                b"name": f"{show_name}.Season.{season}".encode(),
                b"files": files
            }
        }

        info = TorrentInfo(data)
        assert info.is_single_file is False
        assert len(info.files) == num_episodes

    @given(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz',
                min_size=1, max_size=30),
        st.integers(min_value=1000000, max_value=100000000000)
    )
    def test_movie_with_extras_structure(self, movie_name: str, main_size: int) -> None:
        """Movie torrent with main file and extras."""
        data = {
            b"info": {
                b"name": f"{movie_name}.2024.1080p".encode(),
                b"files": [
                    {
                        b"path": [f"{movie_name}.2024.1080p.mkv".encode()],
                        b"length": main_size
                    },
                    {
                        b"path": [b"Extras", b"Behind.The.Scenes.mkv"],
                        b"length": main_size // 10
                    },
                    {
                        b"path": [b"Extras", b"Trailer.mkv"],
                        b"length": main_size // 50
                    }
                ]
            }
        }

        info = TorrentInfo(data)
        assert len(info.files) == 3
        assert info.total_size == main_size + main_size // 10 + main_size // 50

    @given(st.integers(min_value=1, max_value=10000))
    def test_many_small_files(self, num_files: int) -> None:
        """Torrent with many small files."""
        files = [
            {b"path": [f"file{i}.txt".encode()], b"length": 100}
            for i in range(num_files)
        ]

        data = {
            b"info": {
                b"name": b"ManyFiles",
                b"files": files
            }
        }

        info = TorrentInfo(data)
        assert len(info.files) == num_files
        assert info.total_size == num_files * 100

    @given(st.integers(min_value=10**9, max_value=10**12))
    def test_very_large_file(self, size: int) -> None:
        """Test handling of very large files (>1GB)."""
        data = {
            b"info": {
                b"name": b"LargeFile.mkv",
                b"length": size
            }
        }

        info = TorrentInfo(data)
        assert info.total_size == size
        assert info.files[0][1] == size
