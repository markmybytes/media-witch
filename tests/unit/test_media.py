"""Unit tests for media file detection utilities."""

from pathlib import Path

from media_witch.core.media import is_audio, is_subtitle, is_video, list_files_and_dirs


class TestIsVideo:
    """Tests for is_video function."""

    def test_video_extensions(self) -> None:
        """Test recognition of video file extensions."""
        assert is_video(Path("movie.mkv")) is True
        assert is_video(Path("movie.mp4")) is True
        assert is_video(Path("movie.avi")) is True
        assert is_video(Path("movie.mov")) is True
        assert is_video(Path("movie.ts")) is True
        assert is_video(Path("movie.m2ts")) is True
        assert is_video(Path("movie.wmv")) is True

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert is_video(Path("movie.MKV")) is True
        assert is_video(Path("movie.Mp4")) is True

    def test_non_video(self) -> None:
        """Test non-video files."""
        assert is_video(Path("audio.mp3")) is False
        assert is_video(Path("subtitle.srt")) is False
        assert is_video(Path("document.txt")) is False


class TestIsAudio:
    """Tests for is_audio function."""

    def test_audio_extensions(self) -> None:
        """Test recognition of audio file extensions."""
        assert is_audio(Path("track.mka")) is True
        assert is_audio(Path("track.aac")) is True
        assert is_audio(Path("track.flac")) is True
        assert is_audio(Path("track.mp3")) is True
        assert is_audio(Path("track.ogg")) is True

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert is_audio(Path("track.MP3")) is True
        assert is_audio(Path("track.Flac")) is True

    def test_non_audio(self) -> None:
        """Test non-audio files."""
        assert is_audio(Path("video.mkv")) is False
        assert is_audio(Path("subtitle.srt")) is False


class TestIsSubtitle:
    """Tests for is_subtitle function."""

    def test_subtitle_extensions(self) -> None:
        """Test recognition of subtitle file extensions."""
        assert is_subtitle(Path("sub.ass")) is True
        assert is_subtitle(Path("sub.ssa")) is True
        assert is_subtitle(Path("sub.sup")) is True
        assert is_subtitle(Path("sub.srt")) is True

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert is_subtitle(Path("sub.ASS")) is True
        assert is_subtitle(Path("sub.Srt")) is True

    def test_non_subtitle(self) -> None:
        """Test non-subtitle files."""
        assert is_subtitle(Path("video.mkv")) is False
        assert is_subtitle(Path("audio.mp3")) is False


class TestListFilesAndDirs:
    """Tests for list_files_and_dirs function."""

    def test_separates_files_and_dirs(self, tmp_path: Path) -> None:
        """Test that files and directories are separated correctly."""
        # Create test structure
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()

        files, dirs = list_files_and_dirs(tmp_path)

        assert len(files) == 2
        assert len(dirs) == 2
        assert all(f.is_file() for f in files)
        assert all(d.is_dir() for d in dirs)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test with empty directory."""
        files, dirs = list_files_and_dirs(tmp_path)
        assert len(files) == 0
        assert len(dirs) == 0

    def test_only_files(self, tmp_path: Path) -> None:
        """Test directory with only files."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()

        files, dirs = list_files_and_dirs(tmp_path)
        assert len(files) == 2
        assert len(dirs) == 0

    def test_only_dirs(self, tmp_path: Path) -> None:
        """Test directory with only subdirectories."""
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()

        files, dirs = list_files_and_dirs(tmp_path)
        assert len(files) == 0
        assert len(dirs) == 2
