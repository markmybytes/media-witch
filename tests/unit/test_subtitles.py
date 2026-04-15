"""Unit tests for subtitle API."""

from pathlib import Path

from media_witch.core.fileops import FileOps
from media_witch.features.subtitles.api import (
    SubtitleConfig,
    SubtitleService,
    pair_subtitles,
    rename_subtitles,
)
from media_witch.features.subtitles.locale import LocaleMapper, Rule


class TestSubtitleService:
    """Tests for SubtitleService class."""

    def test_pairs_with_exact_match(self) -> None:
        """Test pairing with exact stem match."""
        sub = Path("video.srt")
        video = Path("video.mkv")
        assert SubtitleService.pairs_with(sub, video) is True

    def test_pairs_with_locale_suffix(self) -> None:
        """Test pairing with locale suffix."""
        sub = Path("video.en.srt")
        video = Path("video.mkv")
        assert SubtitleService.pairs_with(sub, video) is True

    def test_pairs_with_no_match(self) -> None:
        """Test non-matching files."""
        sub = Path("other.srt")
        video = Path("video.mkv")
        assert SubtitleService.pairs_with(sub, video) is False

    def test_right_most_token(self) -> None:
        """Test extraction of rightmost token."""
        assert SubtitleService._right_most_token(Path("video.en.srt")) == "en"
        assert SubtitleService._right_most_token(Path("video.zh.Hant.srt")) == "Hant"
        assert SubtitleService._right_most_token(Path("video.srt")) == ""

    def test_stem_wo_token(self) -> None:
        """Test stem without token."""
        assert SubtitleService._stem_wo_token(Path("video.en.srt")) == "video"
        assert SubtitleService._stem_wo_token(Path("video.zh.Hant.srt")) == "video.zh"
        assert SubtitleService._stem_wo_token(Path("video.srt")) == "video"

    def test_normalized_target_with_mapping(self) -> None:
        """Test normalized target with locale mapping."""
        rules = [Rule("chi", "zh", False)]
        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)
        fops = FileOps(dry_run=True)
        service = SubtitleService(mapper, fops)

        sub = Path("/path/video.chi.srt")
        video = Path("/path/video.mkv")
        result = service.normalized_target(sub, video)

        assert result.name == "video.zh.srt"

    def test_normalized_target_no_token(self) -> None:
        """Test normalized target with no locale token."""
        mapper = LocaleMapper(csv_rules=[], cli_rules=[])
        fops = FileOps(dry_run=True)
        service = SubtitleService(mapper, fops)

        sub = Path("/path/video.srt")
        video = Path("/path/video.mkv")
        result = service.normalized_target(sub, video)

        assert result.name == "video.srt"


class TestRenameSubtitles:
    """Tests for rename_subtitles function."""

    def test_rename_with_mapping(self, tmp_path: Path) -> None:
        """Test subtitle renaming with locale mapping."""
        video = tmp_path / "video.mkv"
        video.touch()

        sub = tmp_path / "video.chi.srt"
        sub.write_text("subtitle content")

        rules = [Rule("chi", "zh", False)]
        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)
        config = SubtitleConfig(locale_mapper=mapper, dry_run=False)

        result = rename_subtitles([sub], video, config)

        assert len(result.renamed) == 1
        assert result.renamed[0][0] == sub
        assert result.renamed[0][1].name == "video.zh.srt"
        assert (tmp_path / "video.zh.srt").exists()
        assert not sub.exists()

    def test_rename_dry_run(self, tmp_path: Path) -> None:
        """Test dry-run doesn't actually rename files."""
        video = tmp_path / "video.mkv"
        video.touch()

        sub = tmp_path / "video.en.srt"
        sub.touch()

        mapper = LocaleMapper(csv_rules=[], cli_rules=[])
        config = SubtitleConfig(locale_mapper=mapper, dry_run=True)

        result = rename_subtitles([sub], video, config)

        assert len(result.renamed) == 0  # Dry run records differently
        assert sub.exists()  # Original still exists

    def test_skip_non_pairing_subtitles(self, tmp_path: Path) -> None:
        """Test that non-pairing subtitles are skipped."""
        video = tmp_path / "video.mkv"
        video.touch()

        sub = tmp_path / "other.srt"
        sub.touch()

        mapper = LocaleMapper(csv_rules=[], cli_rules=[])
        config = SubtitleConfig(locale_mapper=mapper, dry_run=False)

        result = rename_subtitles([sub], video, config)

        assert len(result.skipped) == 1
        assert result.skipped[0] == sub


class TestPairSubtitles:
    """Tests for pair_subtitles function."""

    def test_pair_subtitles_with_videos(self) -> None:
        """Test pairing subtitles with videos."""
        videos = [
            Path("/path/video1.mkv"),
            Path("/path/video2.mkv"),
        ]
        subtitles = [
            Path("/path/video1.en.srt"),
            Path("/path/video1.zh.srt"),
            Path("/path/video2.en.srt"),
            Path("/path/other.srt"),
        ]

        pairs = pair_subtitles(subtitles, videos)

        assert len(pairs[videos[0]]) == 2
        assert len(pairs[videos[1]]) == 1
        assert Path("/path/other.srt") not in pairs[videos[0]]
        assert Path("/path/other.srt") not in pairs[videos[1]]

    def test_pair_no_matches(self) -> None:
        """Test pairing with no matches."""
        videos = [Path("/path/video.mkv")]
        subtitles = [Path("/path/other.srt")]

        pairs = pair_subtitles(subtitles, videos)

        assert len(pairs[videos[0]]) == 0
