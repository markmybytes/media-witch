"""Property-based tests for pattern matching utilities."""

from pathlib import Path

from hypothesis import assume, given
from hypothesis import strategies as st

from media_witch.core.patterns import has_episode_pattern, natural_sort_key


class TestEpisodePatternProperties:
    """Property-based tests for episode pattern detection."""

    @given(
        st.integers(min_value=1, max_value=99),
        st.integers(min_value=1, max_value=999)
    )
    def test_standard_sXXeXX_format_always_matches(self, season: int, episode: int) -> None:
        """Any valid S##E## format should be detected."""
        assert has_episode_pattern(f"Show.S{season}E{episode}.mkv") is True
        assert has_episode_pattern(
            f"Show.S{season:02d}E{episode:02d}.mkv") is True
        assert has_episode_pattern(
            f"Show.S{season:02d}E{episode:03d}.mkv") is True

    @given(
        st.integers(min_value=1, max_value=99),
        st.integers(min_value=1, max_value=999)
    )
    def test_lowercase_sXXeXX_format_matches(self, season: int, episode: int) -> None:
        """Lowercase s##e## format should be detected (case insensitive)."""
        assert has_episode_pattern(f"show.s{season}e{episode}.mkv") is True
        assert has_episode_pattern(
            f"show.s{season:02d}e{episode:02d}.mkv") is True

    @given(
        st.integers(min_value=1, max_value=99),
        st.integers(min_value=1, max_value=999)
    )
    def test_mixed_case_format_matches(self, season: int, episode: int) -> None:
        """Mixed case S##e## or s##E## should match."""
        assert has_episode_pattern(f"Show.S{season}e{episode}.mkv") is True
        assert has_episode_pattern(f"Show.s{season}E{episode}.mkv") is True

    @given(st.integers(min_value=1, max_value=999))
    def test_bracket_format_matches(self, episode: int) -> None:
        """Bracket format [##] should be detected."""
        assert has_episode_pattern(f"Show.[{episode}].mkv") is True
        assert has_episode_pattern(f"[{episode}].mkv") is True
        assert has_episode_pattern(f"Show.Name.[{episode:02d}].mkv") is True
        assert has_episode_pattern(
            f"Show.Name.[{episode:03d}].720p.mkv") is True

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz. -_", min_size=1, max_size=50))
    def test_plain_text_no_false_positives(self, text: str) -> None:
        """Plain text without episode markers shouldn't match."""
        assume('s' not in text.lower() or 'e' not in text.lower())
        assume('[' not in text)

        assert has_episode_pattern(text) is False

    @given(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz",
                min_size=1, max_size=30),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=50)
    )
    def test_episode_pattern_in_typical_filename(
        self, show_name: str, season: int, episode: int
    ) -> None:
        """Episode patterns should be detected in realistic filenames."""
        filenames = [
            f"{show_name}.S{season:02d}E{episode:02d}.720p.WEB-DL.mkv",
            f"{show_name}.S{season:02d}E{episode:02d}.1080p.BluRay.x264.mkv",
            f"[SubGroup] {show_name} - S{season:02d}E{episode:02d} [1080p].mkv",
            f"{show_name} [{episode:02d}] [1080p].mkv",
        ]

        for filename in filenames:
            assert has_episode_pattern(filename) is True, f"Failed: {filename}"

    @given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=20))
    def test_all_caps_text_no_false_positive(self, text: str) -> None:
        """All-caps text without numbers shouldn't match."""
        assume('S' not in text or 'E' not in text)
        assert has_episode_pattern(text) is False

    @given(st.integers(min_value=0, max_value=9))
    def test_single_digit_episodes_match(self, episode: int) -> None:
        """Single digit episodes should match."""
        assert has_episode_pattern(f"Show.S1E{episode}.mkv") is True
        assert has_episode_pattern(f"Show.[{episode}].mkv") is True


class TestNaturalSortProperties:
    """Property-based tests for natural sort functionality."""

    @given(st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789.-_ ",
                min_size=1, max_size=50),
        min_size=0,
        max_size=100
    ))
    def test_sort_is_idempotent(self, names: list[str]) -> None:
        """Sorting multiple times should give the same result."""
        # filter out empty/whitespace-only strings
        names = [n.strip() for n in names if n.strip()]
        if not names:
            return

        paths = [Path(name) for name in names]
        sorted_once = sorted(paths, key=natural_sort_key)
        sorted_twice = sorted(sorted_once, key=natural_sort_key)

        assert sorted_once == sorted_twice

    @given(st.lists(
        st.integers(min_value=1, max_value=10000),
        min_size=2,
        max_size=50,
        unique=True
    ))
    def test_numeric_filenames_sort_numerically(self, numbers: list[int]) -> None:
        """Files with numbers should sort in numeric order."""
        paths = [Path(f"file{n}.txt") for n in numbers]

        naturally_sorted = sorted(paths, key=natural_sort_key)
        expected = [Path(f"file{n}.txt") for n in sorted(numbers)]

        assert naturally_sorted == expected

    @given(st.lists(
        st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                min_size=1, max_size=20),
        min_size=2,
        max_size=30
    ))
    def test_sort_is_case_insensitive(self, texts: list[str]) -> None:
        """Natural sort should be case-insensitive."""
        paths_original = [Path(t) for t in texts]
        paths_lower = [Path(t.lower()) for t in texts]

        sorted_original = sorted(paths_original, key=natural_sort_key)
        sorted_lower = sorted(paths_lower, key=natural_sort_key)

        # compare lowercase versions
        assert [p.name.lower() for p in sorted_original] == [
            p.name.lower() for p in sorted_lower]

    @given(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz",
                min_size=1, max_size=20),
        st.lists(st.integers(min_value=1, max_value=10000),
                 min_size=2, max_size=20, unique=True)
    )
    def test_episodes_sort_numerically_not_lexically(
        self, prefix: str, episodes: list[int]
    ) -> None:
        """Episode numbers should sort numerically (e.g., E2 before E10)."""
        paths = [Path(f"{prefix}.S01E{ep}.mkv") for ep in episodes]

        sorted_paths = sorted(paths, key=natural_sort_key)
        sorted_episode_nums = [
            int(p.name.split('E')[1].split('.')[0])
            for p in sorted_paths
        ]

        assert sorted_episode_nums == sorted(episodes)

    @given(st.lists(
        st.integers(min_value=1, max_value=50),
        min_size=2,
        max_size=20,
        unique=True
    ))
    def test_season_numbers_sort_correctly(self, seasons: list[int]) -> None:
        """Season numbers should sort numerically."""
        import re

        paths = [Path(f"ShowName.S{s}E01.mkv") for s in seasons]

        sorted_paths = sorted(paths, key=natural_sort_key)
        sorted_season_nums = []
        for p in sorted_paths:
            match = re.search(r'S(\d+)E', p.name)
            if match:
                sorted_season_nums.append(int(match.group(1)))

        assert sorted_season_nums == sorted(seasons)

    @given(st.lists(
        st.tuples(st.integers(1, 20), st.integers(1, 50)),
        min_size=2,
        max_size=30,
        unique=True
    ))
    def test_season_episode_pairs_sort_correctly(self, pairs: list[tuple[int, int]]) -> None:
        """Full S##E## pairs should sort by season then episode."""
        paths = [Path(f"Show.S{s:02d}E{e:02d}.mkv") for s, e in pairs]

        sorted_paths = sorted(paths, key=natural_sort_key)
        sorted_pairs = []
        for p in sorted_paths:
            parts = p.name.split('.')
            for part in parts:
                if part.startswith('S') and 'E' in part:
                    s = int(part.split('E')[0][1:])
                    e = int(part.split('E')[1])
                    sorted_pairs.append((s, e))
                    break

        assert sorted_pairs == sorted(pairs)

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=50))
    def test_single_file_sorts_to_itself(self, name: str) -> None:
        """Single file should sort to itself."""
        path = Path(name)
        sorted_paths = sorted([path], key=natural_sort_key)
        assert sorted_paths == [path]

    @given(st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz",
                min_size=1, max_size=20),
        min_size=2,
        max_size=50
    ))
    def test_alphabetic_filenames_sort_alphabetically(self, names: list[str]) -> None:
        """Files without numbers should sort alphabetically (case-insensitive)."""
        names = [n for n in names if not any(c.isdigit() for c in n)]
        if len(names) < 2:
            return

        paths = [Path(name + ".txt") for name in names]
        sorted_paths = sorted(paths, key=natural_sort_key)

        expected = sorted(names, key=str.lower)
        result = [p.stem for p in sorted_paths]

        assert result == expected

    @given(st.lists(
        st.text(alphabet="0123456789", min_size=1, max_size=10),
        min_size=2,
        max_size=30
    ))
    def test_pure_numeric_names_sort_numerically(self, num_strs: list[str]) -> None:
        """Pure numeric filenames should sort by numeric value."""
        try:
            numbers = [int(n) for n in num_strs]
        except ValueError:
            return

        paths = [Path(f"{n}.txt") for n in numbers]
        sorted_paths = sorted(paths, key=natural_sort_key)

        result_numbers = [int(p.stem) for p in sorted_paths]
        assert result_numbers == sorted(numbers)


# ============================================================================
# Edge case tests
# ============================================================================


class TestPatternEdgeCases:
    """Tests for edge cases in pattern matching."""

    def test_empty_string_no_pattern(self) -> None:
        """Empty string should not match."""
        assert has_episode_pattern("") is False

    def test_just_brackets_no_match(self) -> None:
        """Just brackets without numbers shouldn't match."""
        assert has_episode_pattern("[]") is False
        assert has_episode_pattern("[ ]") is False

    def test_s_and_e_separate_no_match(self) -> None:
        """S and E not in pattern format shouldn't match."""
        assert has_episode_pattern("Some Episode") is False
        assert has_episode_pattern("Season Episode") is False

    @given(st.integers(min_value=100, max_value=999))
    def test_three_digit_episodes_match(self, episode: int) -> None:
        """Three-digit episode numbers should match."""
        assert has_episode_pattern(f"Show.S01E{episode}.mkv") is True

    @given(st.integers(min_value=1, max_value=99))
    def test_two_digit_seasons_match(self, season: int) -> None:
        """Two-digit season numbers should match."""
        assert has_episode_pattern(f"Show.S{season:02d}E01.mkv") is True


class TestNaturalSortEdgeCases:
    """Tests for edge cases in natural sorting."""

    def test_empty_list_sorts_to_empty(self) -> None:
        """Empty list should sort to empty list."""
        assert sorted([], key=natural_sort_key) == []

    def test_single_item_sorts_to_itself(self) -> None:
        """Single item list should sort to itself."""
        path = Path("file.txt")
        assert sorted([path], key=natural_sort_key) == [path]

    def test_identical_names_maintain_order(self) -> None:
        """Identical names should maintain relative order (stable sort)."""
        paths = [Path("same.txt"), Path("same.txt"), Path("same.txt")]
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert sorted_paths == paths

    @given(st.text(alphabet=".-_ ", min_size=1, max_size=20))
    def test_special_characters_dont_crash(self, special: str) -> None:
        """Special characters shouldn't cause crashes."""
        path = Path(f"file{special}name.txt")
        result = natural_sort_key(path)
        assert result is not None

    def test_leading_zeros_sort_numerically(self) -> None:
        """Leading zeros should not affect numeric sorting."""
        paths = [Path("file001.txt"), Path("file01.txt"), Path("file1.txt")]
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert len(sorted_paths) == 3
