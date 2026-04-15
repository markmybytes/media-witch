"""Unit tests for torrent parser."""

from pathlib import Path

import pytest

from media_witch.features.torrent.parser import TorrentInfo, parse_torrent


class TestTorrentInfoSingleFile:
    """Tests for TorrentInfo with single-file torrents."""

    def test_single_file_name(self) -> None:
        """Test getting name from single-file torrent."""
        data = {
            b"info": {
                b"name": b"testfile.mkv",
                b"length": 1024,
            }
        }
        info = TorrentInfo(data)
        assert info.name == "testfile.mkv"

    def test_single_file_is_single_file(self) -> None:
        """Test is_single_file property for single-file torrent."""
        data = {
            b"info": {
                b"name": b"testfile.mkv",
                b"length": 1024,
            }
        }
        info = TorrentInfo(data)
        assert info.is_single_file is True

    def test_single_file_files_list(self) -> None:
        """Test files property for single-file torrent."""
        data = {
            b"info": {
                b"name": b"movie.mp4",
                b"length": 2048,
            }
        }
        info = TorrentInfo(data)
        files = info.files
        assert len(files) == 1
        assert files[0] == (["movie.mp4"], 2048)

    def test_single_file_total_size(self) -> None:
        """Test total_size for single-file torrent."""
        data = {
            b"info": {
                b"name": b"video.mkv",
                b"length": 5000,
            }
        }
        info = TorrentInfo(data)
        assert info.total_size == 5000


class TestTorrentInfoMultiFile:
    """Tests for TorrentInfo with multi-file torrents."""

    def test_multi_file_name(self) -> None:
        """Test getting name from multi-file torrent."""
        data = {
            b"info": {
                b"name": b"MyShow.Season.1",
                b"files": [
                    {b"path": [b"episode1.mkv"], b"length": 1000},
                    {b"path": [b"episode2.mkv"], b"length": 2000},
                ],
            }
        }
        info = TorrentInfo(data)
        assert info.name == "MyShow.Season.1"

    def test_multi_file_is_single_file(self) -> None:
        """Test is_single_file property for multi-file torrent."""
        data = {
            b"info": {
                b"name": b"Files",
                b"files": [
                    {b"path": [b"file1.txt"], b"length": 100},
                ],
            }
        }
        info = TorrentInfo(data)
        assert info.is_single_file is False

    def test_multi_file_files_list(self) -> None:
        """Test files property for multi-file torrent."""
        data = {
            b"info": {
                b"name": b"Root",
                b"files": [
                    {b"path": [b"file1.txt"], b"length": 100},
                    {b"path": [b"dir", b"file2.txt"], b"length": 200},
                    {b"path": [b"file3.txt"], b"length": 300},
                ],
            }
        }
        info = TorrentInfo(data)
        files = info.files

        assert len(files) == 3
        assert files[0] == (["file1.txt"], 100)
        assert files[1] == (["dir", "file2.txt"], 200)
        assert files[2] == (["file3.txt"], 300)

    def test_multi_file_nested_paths(self) -> None:
        """Test files with deeply nested paths."""
        data = {
            b"info": {
                b"name": b"Root",
                b"files": [
                    {b"path": [b"a", b"b", b"c", b"file.txt"], b"length": 500},
                ],
            }
        }
        info = TorrentInfo(data)
        files = info.files

        assert len(files) == 1
        assert files[0] == (["a", "b", "c", "file.txt"], 500)

    def test_multi_file_total_size(self) -> None:
        """Test total_size for multi-file torrent."""
        data = {
            b"info": {
                b"name": b"Files",
                b"files": [
                    {b"path": [b"file1.txt"], b"length": 100},
                    {b"path": [b"file2.txt"], b"length": 200},
                    {b"path": [b"file3.txt"], b"length": 300},
                ],
            }
        }
        info = TorrentInfo(data)
        assert info.total_size == 600

    def test_multi_file_empty_files_list(self) -> None:
        """Test multi-file torrent with empty files list."""
        data = {
            b"info": {
                b"name": b"Empty",
                b"files": [],
            }
        }
        info = TorrentInfo(data)
        assert info.files == []
        assert info.total_size == 0


class TestTorrentInfoEncoding:
    """Tests for handling different encodings."""

    def test_utf8_name(self) -> None:
        """Test handling UTF-8 encoded names."""
        data = {
            b"info": {
                b"name": "Test File 日本語.mkv".encode("utf-8"),
                b"length": 1024,
            }
        }
        info = TorrentInfo(data)
        assert info.name == "Test File 日本語.mkv"

    def test_invalid_utf8_name(self) -> None:
        """Test handling invalid UTF-8 with error replacement."""
        data = {
            b"info": {
                b"name": b"\xff\xfe Invalid",
                b"length": 1024,
            }
        }
        info = TorrentInfo(data)
        # Should not raise error, uses replacement character
        assert isinstance(info.name, str)

    def test_utf8_file_paths(self) -> None:
        """Test handling UTF-8 encoded file paths."""
        data = {
            b"info": {
                b"name": b"Root",
                b"files": [
                    {
                        b"path": ["субтитры".encode("utf-8"), b"file.srt"],
                        b"length": 100,
                    },
                ],
            }
        }
        info = TorrentInfo(data)
        files = info.files
        assert files[0][0] == ["субтитры", "file.srt"]


class TestParseTorrent:
    """Tests for parse_torrent function."""

    def test_parse_single_file_torrent(self, tmp_path: Path) -> None:
        """Test parsing a single-file torrent."""
        # Create a valid bencode torrent file
        torrent_content = b"d4:infod4:name9:test.file6:lengthi1024eee"
        torrent_path = tmp_path / "test.torrent"
        torrent_path.write_bytes(torrent_content)

        info = parse_torrent(torrent_path)

        assert info.name == "test.file"
        assert info.is_single_file is True
        assert info.total_size == 1024

    def test_parse_multi_file_torrent(self, tmp_path: Path) -> None:
        """Test parsing a multi-file torrent."""
        # d4:infod4:name4:Test5:filesld4:pathl5:file1e6:lengthi100eeee
        torrent_content = (
            b"d4:infod4:name4:Test5:filesl"
            b"d4:pathl9:file1.txt9:file2.txte6:lengthi100ee"
            b"d4:pathl9:file3.txte6:lengthi200ee"
            b"eee"
        )
        torrent_path = tmp_path / "multi.torrent"
        torrent_path.write_bytes(torrent_content)

        info = parse_torrent(torrent_path)

        assert info.name == "Test"
        assert info.is_single_file is False
        assert len(info.files) == 2

    def test_parse_nonexistent_file(self) -> None:
        """Test that parsing nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_torrent(Path("/nonexistent/file.torrent"))

    def test_parse_invalid_bencode(self, tmp_path: Path) -> None:
        """Test that parsing invalid bencode raises ValueError."""
        torrent_path = tmp_path / "invalid.torrent"
        torrent_path.write_bytes(b"invalid bencode data")

        with pytest.raises(ValueError):
            parse_torrent(torrent_path)

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Test that parsing empty file raises error."""
        torrent_path = tmp_path / "empty.torrent"
        torrent_path.write_bytes(b"")

        with pytest.raises((ValueError, IndexError)):
            parse_torrent(torrent_path)


class TestTorrentInfoRealWorldScenarios:
    """Tests with realistic torrent structures."""

    def test_tv_show_season_torrent(self) -> None:
        """Test parsing TV show season torrent structure."""
        data = {
            b"info": {
                b"name": b"Show.Name.S01.1080p.WEB-DL",
                b"files": [
                    {
                        b"path": [b"Show.Name.S01E01.mkv"],
                        b"length": 1500000000,
                    },
                    {
                        b"path": [b"Show.Name.S01E02.mkv"],
                        b"length": 1600000000,
                    },
                    {
                        b"path": [b"Subs", b"Show.Name.S01E01.en.srt"],
                        b"length": 50000,
                    },
                    {
                        b"path": [b"Subs", b"Show.Name.S01E02.en.srt"],
                        b"length": 52000,
                    },
                ],
            }
        }
        info = TorrentInfo(data)

        assert "Show.Name" in info.name
        assert info.is_single_file is False
        assert len(info.files) == 4
        assert info.total_size == 3100102000

        # Check specific file paths
        episode1_path, episode1_size = info.files[0]
        assert episode1_path == ["Show.Name.S01E01.mkv"]
        assert episode1_size == 1500000000

        sub_path, sub_size = info.files[2]
        assert sub_path == ["Subs", "Show.Name.S01E01.en.srt"]
        assert sub_size == 50000

    def test_movie_with_extras(self) -> None:
        """Test parsing movie torrent with extras."""
        data = {
            b"info": {
                b"name": b"Movie.Name.2024.1080p.BluRay",
                b"files": [
                    {
                        b"path": [b"Movie.Name.2024.1080p.BluRay.mkv"],
                        b"length": 8000000000,
                    },
                    {
                        b"path": [b"Extras", b"Behind.The.Scenes.mkv"],
                        b"length": 500000000,
                    },
                    {
                        b"path": [b"Extras", b"Trailer.mkv"],
                        b"length": 100000000,
                    },
                ],
            }
        }
        info = TorrentInfo(data)

        assert "Movie.Name" in info.name
        assert len(info.files) == 3
        assert info.total_size == 8600000000

        # Verify main movie
        main_path, main_size = info.files[0]
        assert "Movie.Name.2024" in main_path[0]
        assert main_size == 8000000000

    def test_large_file_sizes(self) -> None:
        """Test handling large file sizes (>4GB)."""
        large_size = 50 * 1024 * 1024 * 1024  # 50 GB
        data = {
            b"info": {
                b"name": b"Large.File.mkv",
                b"length": large_size,
            }
        }
        info = TorrentInfo(data)

        assert info.total_size == large_size
        assert info.files[0][1] == large_size

    def test_many_small_files(self) -> None:
        """Test torrent with many small files."""
        files_list = [
            {b"path": [f"file{i}.txt".encode()], b"length": 100}
            for i in range(1000)
        ]
        data = {
            b"info": {
                b"name": b"ManyFiles",
                b"files": files_list,
            }
        }
        info = TorrentInfo(data)

        assert len(info.files) == 1000
        assert info.total_size == 100000
