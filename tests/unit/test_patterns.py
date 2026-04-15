"""Unit tests for pattern matching utilities."""

from pathlib import Path

import pytest

from media_witch.core.patterns import has_episode_pattern, natural_sort_key


class TestHasEpisodePattern:
    """Tests for has_episode_pattern function."""

    def test_s01e01_format(self) -> None:
        """Test standard S##E## format."""
        assert has_episode_pattern("Show.S01E01.mkv") is True
        assert has_episode_pattern("Show.S1E1.mkv") is True
        assert has_episode_pattern("Show.S10E99.mkv") is True

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert has_episode_pattern("Show.s01e01.mkv") is True
        assert has_episode_pattern("Show.S01e01.mkv") is True

    def test_bracket_format(self) -> None:
        """Test [##] format."""
        assert has_episode_pattern("[01].mkv") is True
        assert has_episode_pattern("[1].mkv") is True
        assert has_episode_pattern("[123].mkv") is True

    def test_no_pattern(self) -> None:
        """Test files without episode patterns."""
        assert has_episode_pattern("Random.Movie.mkv") is False
        assert has_episode_pattern("No.Pattern.Here.mp4") is False


class TestNaturalSortKey:
    """Tests for natural_sort_key function."""

    def test_sorts_numbers_naturally(self) -> None:
        """Test that numbers are sorted numerically."""
        files = [
            Path("file10.txt"),
            Path("file2.txt"),
            Path("file1.txt"),
        ]
        sorted_files = sorted(files, key=natural_sort_key)
        assert [f.name for f in sorted_files] == [
            "file1.txt", "file2.txt", "file10.txt"]

    def test_sorts_episodes_naturally(self) -> None:
        """Test episode sorting."""
        files = [
            Path("Show.S01E10.mkv"),
            Path("Show.S01E2.mkv"),
            Path("Show.S01E1.mkv"),
        ]
        sorted_files = sorted(files, key=natural_sort_key)
        assert [f.name for f in sorted_files] == [
            "Show.S01E1.mkv",
            "Show.S01E2.mkv",
            "Show.S01E10.mkv",
        ]

    def test_case_insensitive(self) -> None:
        """Test case-insensitive sorting."""
        files = [
            Path("FileC.txt"),
            Path("filea.txt"),
            Path("FileB.txt"),
        ]
        sorted_files = sorted(files, key=natural_sort_key)
        assert [f.name for f in sorted_files] == [
            "filea.txt",
            "FileB.txt",
            "FileC.txt",
        ]
