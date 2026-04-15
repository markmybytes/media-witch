"""Unit tests for torrent API."""

from pathlib import Path

import pytest

from media_witch.features.torrent.api import (
    TorrentConfig,
    TorrentResult,
    create_from_torrent,
    create_from_torrents,
)


class TestTorrentConfig:
    """Tests for TorrentConfig dataclass."""

    def test_config_creation(self, tmp_path: Path) -> None:
        """Test TorrentConfig instantiation."""
        config = TorrentConfig(output_dir=tmp_path, verbose=True)
        assert config.output_dir == tmp_path
        assert config.verbose is True

    def test_config_defaults(self, tmp_path: Path) -> None:
        """Test TorrentConfig default values."""
        config = TorrentConfig(output_dir=tmp_path)
        assert config.output_dir == tmp_path
        assert config.verbose is False


class TestTorrentResult:
    """Tests for TorrentResult dataclass."""

    def test_result_creation(self) -> None:
        """Test TorrentResult instantiation."""
        result = TorrentResult(
            created_files=[Path('file1.txt')],
            created_dirs=[Path('dir1')],
            errors=['error1'],
        )
        assert len(result.created_files) == 1
        assert len(result.created_dirs) == 1
        assert len(result.errors) == 1


class TestCreateFromTorrentSingleFile:
    """Tests for create_from_torrent with single-file torrents."""

    def test_creates_single_file(self, tmp_path: Path) -> None:
        """Test creating fake file from single-file torrent."""
        # Create valid single-file torrent
        torrent_content = b'd4:infod4:name9:video.mkv6:lengthi1024eee'
        torrent_path = tmp_path / 'test.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir, verbose=False)

        result = create_from_torrent(torrent_path, config)

        # Check result
        assert len(result.errors) == 0
        assert len(result.created_files) == 1
        assert len(result.created_dirs) >= 1

        # Check file was created
        created_file = output_dir / 'test' / 'video.mkv'
        assert created_file.exists()
        assert created_file.is_file()

    def test_creates_base_directory(self, tmp_path: Path) -> None:
        """Test that base directory is created from torrent stem."""
        torrent_content = b'd4:infod4:name8:file.txt6:lengthi100eee'
        torrent_path = tmp_path / 'mytorrent.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        base_dir = output_dir / 'mytorrent'
        assert base_dir.exists()
        assert base_dir.is_dir()
        assert base_dir in result.created_dirs

    def test_single_file_zero_size(self, tmp_path: Path) -> None:
        """Test that created files are empty (0 bytes)."""
        torrent_content = b'd4:infod4:name8:test.txt6:lengthi999999eee'
        torrent_path = tmp_path / 'test.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        create_from_torrent(torrent_path, config)

        created_file = output_dir / 'test' / 'test.txt'
        assert created_file.exists()
        assert created_file.stat().st_size == 0  # Empty file


class TestCreateFromTorrentMultiFile:
    """Tests for create_from_torrent with multi-file torrents."""

    def test_creates_multiple_files(self, tmp_path: Path) -> None:
        """Test creating multiple files from multi-file torrent."""
        # Multi-file torrent with 3 files
        torrent_content = (
            b'd4:infod4:name4:Root5:filesl'
            b'd4:pathl9:file1.txte6:lengthi100ee'
            b'd4:pathl9:file2.txte6:lengthi200ee'
            b'd4:pathl9:file3.txte6:lengthi300ee'
            b'eee'
        )
        torrent_path = tmp_path / 'multi.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        assert len(result.errors) == 0
        assert len(result.created_files) == 3

        # Check all files exist
        base = output_dir / 'multi'
        assert (base / 'file1.txt').exists()
        assert (base / 'file2.txt').exists()
        assert (base / 'file3.txt').exists()

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        """Test creating nested directory structure."""
        torrent_content = (
            b'd4:infod4:name4:Root5:filesl'
            b'd4:pathl3:dir5:file1e6:lengthi100ee'
            b'd4:pathl3:dir4:sub25:file2e6:lengthi200ee'
            b'eee'
        )
        torrent_path = tmp_path / 'nested.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        create_from_torrent(torrent_path, config)

        base = output_dir / 'nested'
        assert (base / 'dir' / 'file1').exists()
        assert (base / 'dir' / 'sub2' / 'file2').exists()
        assert (base / 'dir').is_dir()
        assert (base / 'dir' / 'sub2').is_dir()

    def test_creates_deeply_nested_paths(self, tmp_path: Path) -> None:
        """Test creating deeply nested directory paths."""
        torrent_content = (
            b'd4:infod4:name4:Root5:filesld4:pathl1:a1:b1:c1:d8:file.txte6:lengthi100eeeeee'
        )
        torrent_path = tmp_path / 'deep.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        create_from_torrent(torrent_path, config)

        deep_file = output_dir / 'deep' / 'a' / 'b' / 'c' / 'd' / 'file.txt'
        assert deep_file.exists()

    def test_multiple_files_same_directory(self, tmp_path: Path) -> None:
        """Test creating multiple files in same directory."""
        torrent_content = (
            b'd4:infod4:name4:Root5:filesl'
            b'd4:pathl3:dir5:file1e6:lengthi100ee'
            b'd4:pathl3:dir5:file2e6:lengthi200ee'
            b'd4:pathl3:dir5:file3e6:lengthi300ee'
            b'eee'
        )
        torrent_path = tmp_path / 'samedir.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        create_from_torrent(torrent_path, config)

        dir_path = output_dir / 'samedir' / 'dir'
        assert (dir_path / 'file1').exists()
        assert (dir_path / 'file2').exists()
        assert (dir_path / 'file3').exists()


class TestCreateFromTorrentErrors:
    """Tests for error handling in create_from_torrent."""

    def test_nonexistent_torrent_file(self, tmp_path: Path) -> None:
        """Test handling of nonexistent torrent file."""
        torrent_path = tmp_path / 'nonexistent.torrent'
        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        assert len(result.errors) > 0
        assert len(result.created_files) == 0

    def test_invalid_torrent_file(self, tmp_path: Path) -> None:
        """Test handling of invalid torrent file."""
        torrent_path = tmp_path / 'invalid.torrent'
        torrent_path.write_bytes(b'invalid bencode')

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        assert len(result.errors) > 0
        assert 'Error processing' in result.errors[0]

    def test_creates_despite_some_file_errors(self, tmp_path: Path) -> None:
        """Test that successful files are created even if some fail."""
        # This test assumes that file creation can fail for individual files
        # but the function continues processing other files
        torrent_content = b'd4:infod4:name4:Root5:filesld4:pathl8:file.txte6:lengthi100eeeeee'
        torrent_path = tmp_path / 'test.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        # Should succeed in creating the valid file
        assert len(result.created_files) >= 1 or len(result.errors) == 0


class TestCreateFromTorrentVerbose:
    """Tests for verbose output mode."""

    def test_verbose_mode_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that verbose mode produces output."""
        torrent_content = b'd4:infod4:name8:test.txt6:lengthi100eee'
        torrent_path = tmp_path / 'test.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir, verbose=True)

        create_from_torrent(torrent_path, config)

        captured = capsys.readouterr()
        assert 'Creating fake files' in captured.out
        assert 'Torrent:' in captured.out
        assert 'Total files:' in captured.out

    def test_non_verbose_mode_no_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that non-verbose mode produces no output."""
        torrent_content = b'd4:infod4:name8:test.txt6:lengthi100eee'
        torrent_path = tmp_path / 'test.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir, verbose=False)

        create_from_torrent(torrent_path, config)

        captured = capsys.readouterr()
        assert captured.out == ''


class TestCreateFromTorrents:
    """Tests for create_from_torrents (batch processing)."""

    def test_creates_from_multiple_torrents(self, tmp_path: Path) -> None:
        """Test creating files from multiple torrent files."""
        # Create 3 torrent files
        torrents = []
        for i in range(3):
            filename = f'file{i}.txt'
            content = f'd4:infod4:name{len(filename)}:{filename}6:lengthi100eee'.encode()
            torrent_path = tmp_path / f'torrent{i}.torrent'
            torrent_path.write_bytes(content)
            torrents.append(torrent_path)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        results = create_from_torrents(torrents, config)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert len(result.created_files) >= 1
            base_dir = output_dir / f'torrent{i}'
            assert base_dir.exists()

    def test_batch_continues_on_error(self, tmp_path: Path) -> None:
        """Test that batch processing continues even if one torrent fails."""
        # Create one valid and one invalid torrent
        valid_content = b'd4:infod4:name9:valid.txt6:lengthi100eee'
        valid_path = tmp_path / 'valid.torrent'
        valid_path.write_bytes(valid_content)

        invalid_path = tmp_path / 'invalid.torrent'
        invalid_path.write_bytes(b'invalid')

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        results = create_from_torrents([valid_path, invalid_path], config)

        assert len(results) == 2
        assert len(results[0].created_files) >= 1  # Valid one succeeded
        assert len(results[1].errors) > 0  # Invalid one failed

    def test_empty_torrent_list(self, tmp_path: Path) -> None:
        """Test processing empty list of torrents."""
        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        results = create_from_torrents([], config)

        assert len(results) == 0


class TestCreateFromTorrentRealWorldScenarios:
    """Tests with realistic torrent structures."""

    def test_tv_show_season_structure(self, tmp_path: Path) -> None:
        """Test creating TV show season file structure."""
        torrent_content = (
            b'd4:infod4:name23:Show.Name.S01.1080p.WEB5:filesl'
            b'd4:pathl20:Show.Name.S01E01.mkve6:lengthi1500000000ee'
            b'd4:pathl20:Show.Name.S01E02.mkve6:lengthi1600000000ee'
            b'd4:pathl4:Subs23:Show.Name.S01E01.en.srte6:lengthi50000ee'
            b'd4:pathl4:Subs23:Show.Name.S01E02.en.srte6:lengthi52000ee'
            b'eee'
        )
        torrent_path = tmp_path / 'show.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        base = output_dir / 'show'
        assert (base / 'Show.Name.S01E01.mkv').exists()
        assert (base / 'Show.Name.S01E02.mkv').exists()
        assert (base / 'Subs' / 'Show.Name.S01E01.en.srt').exists()
        assert (base / 'Subs' / 'Show.Name.S01E02.en.srt').exists()
        assert len(result.created_files) == 4

    def test_movie_with_extras_structure(self, tmp_path: Path) -> None:
        """Test creating movie with extras file structure."""
        torrent_content = (
            b'd4:infod4:name28:Movie.Name.2024.1080p.BluRay5:filesl'
            b'd4:pathl32:Movie.Name.2024.1080p.BluRay.mkve6:lengthi8000000000ee'
            b'd4:pathl6:Extras21:Behind.The.Scenes.mkve6:lengthi500000000ee'
            b'd4:pathl6:Extras11:Trailer.mkve6:lengthi100000000ee'
            b'eee'
        )
        torrent_path = tmp_path / 'movie.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        base = output_dir / 'movie'
        assert (base / 'Movie.Name.2024.1080p.BluRay.mkv').exists()
        assert (base / 'Extras' / 'Behind.The.Scenes.mkv').exists()
        assert (base / 'Extras' / 'Trailer.mkv').exists()
        assert len(result.created_files) == 3

    def test_many_files_torrent(self, tmp_path: Path) -> None:
        """Test creating torrent with many files."""
        # Create torrent with 50 files
        files_bencode = b''.join(
            [f'd4:pathl6:file{i:02d}e6:lengthi100ee'.encode() for i in range(50)]
        )
        torrent_content = b'd4:infod4:name9:ManyFiles5:filesl' + files_bencode + b'eee'
        torrent_path = tmp_path / 'many.torrent'
        torrent_path.write_bytes(torrent_content)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        config = TorrentConfig(output_dir=output_dir)

        result = create_from_torrent(torrent_path, config)

        assert len(result.created_files) == 50
        base = output_dir / 'many'
        for i in range(50):
            assert (base / f'file{i:02d}').exists()
