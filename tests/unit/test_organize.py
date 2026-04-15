"""Unit tests for organize module."""

from pathlib import Path

import pytest

from media_witch.core.fileops import FileOps
from media_witch.features.organize.api import (
    OrganizeConfig,
    OrganizeResult,
    classify_extras_auto,
    organize_directory,
    organize_movie,
    organize_tv_show,
)
from media_witch.features.subtitles.locale import LocaleMapper, Rule


class TestClassifyExtrasAuto:
    """Tests for classify_extras_auto function."""

    def test_classifies_directories_as_extras(self, tmp_path: Path) -> None:
        """Test that directories are classified as extras."""
        dir1 = tmp_path / "extras_dir"
        dir1.mkdir()
        file1 = tmp_path / "Show.S01E01.mkv"
        file1.touch()

        items = [dir1, file1]
        flags = classify_extras_auto(items)

        assert flags[0] is True  # Directory is extra
        assert flags[1] is False  # File with pattern is primary

    def test_classifies_files_without_pattern_as_extras(self) -> None:
        """Test that files without episode patterns are extras."""
        items = [
            Path("Show.S01E01.mkv"),
            Path("Behind.The.Scenes.mkv"),
            Path("random.file.txt"),
            Path("[01].mkv"),
        ]
        flags = classify_extras_auto(items)

        assert flags[0] is False  # Has S01E01 pattern
        assert flags[1] is True  # No pattern
        assert flags[2] is True  # No pattern
        assert flags[3] is False  # Has [01] pattern

    def test_empty_list(self) -> None:
        """Test with empty list."""
        flags = classify_extras_auto([])
        assert flags == []

    def test_all_primary_files(self) -> None:
        """Test with all primary (patterned) files."""
        items = [
            Path("Show.S01E01.mkv"),
            Path("Show.S01E02.mkv"),
            Path("[03].mkv"),
        ]
        flags = classify_extras_auto(items)

        assert all(flag is False for flag in flags)

    def test_all_extra_files(self) -> None:
        """Test with all extra (non-patterned) files."""
        items = [
            Path("extras.mkv"),
            Path("bonus.content.mp4"),
            Path("readme.txt"),
        ]
        flags = classify_extras_auto(items)

        assert all(flag is True for flag in flags)


class TestOrganizeTvShow:
    """Tests for organize_tv_show function."""

    def test_organizes_video_files_to_season_folder(self, tmp_path: Path) -> None:
        """Test organizing video files into Season folder."""
        # Create test files
        video1 = tmp_path / "Show.S01E01.mkv"
        video2 = tmp_path / "Show.S01E02.mkv"
        video1.touch()
        video2.touch()

        config = OrganizeConfig(mode="show", season=1, dry_run=False)
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        # Check files moved to Season 1 folder
        season_dir = tmp_path / "Season 1"
        assert season_dir.exists()
        assert (season_dir / "Show.S01E01.mkv").exists()
        assert (season_dir / "Show.S01E02.mkv").exists()
        assert not video1.exists()
        assert not video2.exists()

    def test_organizes_with_custom_root_dir(self, tmp_path: Path) -> None:
        """Test organizing with custom root directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        root_dir = tmp_path / "root"
        root_dir.mkdir()

        video = source_dir / "Show.S02E01.mkv"
        video.touch()

        config = OrganizeConfig(mode="show", season=2,
                                dry_run=False, root_dir=root_dir)
        fops = FileOps(dry_run=False)

        organize_tv_show(source_dir, 2, config, fops)

        # Season folder should be created at root_dir, not source_dir
        season_dir = root_dir / "Season 2"
        assert season_dir.exists()
        assert (season_dir / "Show.S02E01.mkv").exists()

    def test_moves_extras_to_extra_folder(self, tmp_path: Path) -> None:
        """Test that extras are moved to EXTRA folder."""
        video = tmp_path / "Show.S01E01.mkv"
        extra_file = tmp_path / "behind.the.scenes.mkv"
        video.touch()
        extra_file.touch()

        config = OrganizeConfig(mode="show", season=1, dry_run=False)
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        # Primary video in Season folder
        assert (tmp_path / "Season 1" / "Show.S01E01.mkv").exists()

        # Extra in EXTRA folder
        assert (tmp_path / "EXTRA" / "Season 1" /
                "behind.the.scenes.mkv").exists()

    def test_respects_extras_flags(self, tmp_path: Path) -> None:
        """Test that provided extras_flags are respected."""
        file1 = tmp_path / "Show.S01E01.mkv"
        file2 = tmp_path / "Show.S01E02.mkv"
        file1.touch()
        file2.touch()

        # Mark file1 as extra (True), file2 as primary (False)
        config = OrganizeConfig(
            mode="show",
            season=1,
            dry_run=False,
            extras_flags=[False, True],  # Reverse of auto-classification
        )
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        # file1 should be in Season 1 (primary)
        assert (tmp_path / "Season 1" / "Show.S01E01.mkv").exists()

        # file2 should be in EXTRA (marked as extra)
        assert (tmp_path / "EXTRA" / "Season 1" / "Show.S01E02.mkv").exists()

    def test_flattens_primary_directories(self, tmp_path: Path) -> None:
        """Test that primary directories are flattened into Season folder."""
        video_dir = tmp_path / "VideoFolder"
        video_dir.mkdir()
        video = video_dir / "Show.S01E01.mkv"
        video.touch()

        config = OrganizeConfig(mode="show", season=1,
                                dry_run=False, extras_flags=[False])
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        # Video should be flattened into Season 1
        assert (tmp_path / "Season 1" / "Show.S01E01.mkv").exists()
        assert not video_dir.exists()

    def test_moves_extra_directories_atomically(self, tmp_path: Path) -> None:
        """Test that extra directories are moved atomically."""
        extra_dir = tmp_path / "BonusContent"
        extra_dir.mkdir()
        extra_file = extra_dir / "behind.the.scenes.mkv"
        extra_file.touch()

        config = OrganizeConfig(mode="show", season=1, dry_run=False)
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        # Directory moved to EXTRA atomically
        moved_dir = tmp_path / "EXTRA" / "Season 1" / "BonusContent"
        assert moved_dir.exists()
        assert (moved_dir / "behind.the.scenes.mkv").exists()
        assert not extra_dir.exists()

    def test_handles_subtitles_with_locale_mapper(self, tmp_path: Path) -> None:
        """Test subtitle handling with locale mapper."""
        video = tmp_path / "Show.S01E01.mkv"
        sub = tmp_path / "Show.S01E01.chi.srt"
        video.touch()
        sub.write_text("subtitle content")

        rules = [Rule("chi", "zh", False)]
        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)
        config = OrganizeConfig(mode="show", season=1,
                                dry_run=False, locale_mapper=mapper)
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        # Subtitle should be renamed and moved
        season_dir = tmp_path / "Season 1"
        assert (season_dir / "Show.S01E01.mkv").exists()
        assert (season_dir / "Show.S01E01.zh.srt").exists()

    def test_generates_nfo_files_when_enabled(self, tmp_path: Path) -> None:
        """Test NFO file generation when enabled."""
        video1 = tmp_path / "Show.S01E01.mkv"
        video2 = tmp_path / "Show.S01E02.mkv"
        video1.touch()
        video2.touch()

        config = OrganizeConfig(mode="show", season=1,
                                dry_run=False, generate_nfo=True)
        fops = FileOps(dry_run=False)

        result = organize_tv_show(tmp_path, 1, config, fops)

        # NFO files should be created
        season_dir = tmp_path / "Season 1"
        assert (season_dir / "Show.S01E01.nfo").exists()
        assert (season_dir / "Show.S01E02.nfo").exists()
        assert len(result.nfos_created) == 2

    def test_skips_nfo_when_skip_flag_set(self, tmp_path: Path) -> None:
        """Test that NFO generation is skipped when skip_nfo_generation is True."""
        video = tmp_path / "Show.S01E01.mkv"
        video.touch()

        config = OrganizeConfig(
            mode="show",
            season=1,
            dry_run=False,
            generate_nfo=True,
            skip_nfo_generation=True,
        )
        fops = FileOps(dry_run=False)

        result = organize_tv_show(tmp_path, 1, config, fops)

        season_dir = tmp_path / "Season 1"
        assert not (season_dir / "Show.S01E01.nfo").exists()
        assert len(result.nfos_created) == 0

    def test_episode_overrides_in_nfo(self, tmp_path: Path) -> None:
        """Test episode number overrides in NFO generation."""
        video1 = tmp_path / "Show.S01E01.mkv"
        video2 = tmp_path / "Show.S01E02.mkv"
        video1.touch()
        video2.touch()

        config = OrganizeConfig(
            mode="show",
            season=1,
            dry_run=False,
            generate_nfo=True,
            episode_overrides={2: 10},  # Second video should be episode 10
        )
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        # Check NFO content
        season_dir = tmp_path / "Season 1"
        nfo1_content = (season_dir / "Show.S01E01.nfo").read_text()
        nfo2_content = (season_dir / "Show.S01E02.nfo").read_text()

        assert "<episode>1</episode>" in nfo1_content
        assert "<episode>10</episode>" in nfo2_content

    def test_dry_run_no_changes(self, tmp_path: Path) -> None:
        """Test that dry-run doesn't make actual changes."""
        video = tmp_path / "Show.S01E01.mkv"
        video.touch()

        config = OrganizeConfig(mode="show", season=1, dry_run=True)
        fops = FileOps(dry_run=True)

        organize_tv_show(tmp_path, 1, config, fops)

        # Files should not be moved
        assert video.exists()
        assert not (tmp_path / "Season 1").exists()

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test organizing empty directory."""
        config = OrganizeConfig(mode="show", season=1, dry_run=False)
        fops = FileOps(dry_run=False)

        result = organize_tv_show(tmp_path, 1, config, fops)

        assert len(result.files_moved) == 0
        assert len(result.errors) == 0

    def test_handles_audio_files(self, tmp_path: Path) -> None:
        """Test that audio files are handled correctly."""
        audio = tmp_path / "audio.mka"
        audio.touch()

        config = OrganizeConfig(mode="show", season=1,
                                dry_run=False, extras_flags=[False])
        fops = FileOps(dry_run=False)

        organize_tv_show(tmp_path, 1, config, fops)

        season_dir = tmp_path / "Season 1"
        assert (season_dir / "audio.mka").exists()


class TestOrganizeMovie:
    """Tests for organize_movie function."""

    def test_organizes_video_files_in_place(self, tmp_path: Path) -> None:
        """Test organizing movie files in place."""
        video = tmp_path / "Movie.2024.mkv"
        video.touch()

        config = OrganizeConfig(
            mode="movie", dry_run=False, extras_flags=[False])
        fops = FileOps(dry_run=False)

        organize_movie(tmp_path, config, fops)

        # Video stays in place (moved to same directory)
        assert (tmp_path / "Movie.2024.mkv").exists()

    def test_moves_extras_to_extra_folder(self, tmp_path: Path) -> None:
        """Test that extras are moved to EXTRA folder."""
        video = tmp_path / "Movie.2024.mkv"
        extra = tmp_path / "behind.the.scenes.mkv"
        video.touch()
        extra.touch()

        config = OrganizeConfig(
            mode="movie", dry_run=False, extras_flags=[True, False])
        fops = FileOps(dry_run=False)

        organize_movie(tmp_path, config, fops)

        # Main video in place
        assert (tmp_path / "Movie.2024.mkv").exists()

        # Extra in EXTRA folder (movie mode uses just "EXTRA", not "EXTRA/Season X")
        assert (tmp_path / "EXTRA" / "behind.the.scenes.mkv").exists()

    def test_flattens_primary_directories(self, tmp_path: Path) -> None:
        """Test flattening primary directories."""
        video_dir = tmp_path / "MovieFolder"
        video_dir.mkdir()
        video = video_dir / "Movie.2024.mkv"
        video.touch()

        config = OrganizeConfig(
            mode="movie", dry_run=False, extras_flags=[False])
        fops = FileOps(dry_run=False)

        organize_movie(tmp_path, config, fops)

        # Video flattened to parent directory
        assert (tmp_path / "Movie.2024.mkv").exists()
        assert not video_dir.exists()

    def test_moves_extra_directories_atomically(self, tmp_path: Path) -> None:
        """Test moving extra directories atomically."""
        extra_dir = tmp_path / "Extras"
        extra_dir.mkdir()
        extra = extra_dir / "trailer.mkv"
        extra.touch()

        config = OrganizeConfig(mode="movie", dry_run=False)
        fops = FileOps(dry_run=False)

        organize_movie(tmp_path, config, fops)

        # Directory moved atomically
        moved_dir = tmp_path / "EXTRA" / "Extras"
        assert moved_dir.exists()
        assert (moved_dir / "trailer.mkv").exists()
        assert not extra_dir.exists()

    def test_handles_subtitles_with_locale_mapper(self, tmp_path: Path) -> None:
        """Test subtitle handling with locale mapper."""
        video = tmp_path / "Movie.2024.mkv"
        sub = tmp_path / "Movie.2024.chi.srt"
        video.touch()
        sub.write_text("subtitle content")

        rules = [Rule("chi", "zh", False)]
        mapper = LocaleMapper(csv_rules=[], cli_rules=rules)
        config = OrganizeConfig(
            mode="movie", dry_run=False, locale_mapper=mapper, extras_flags=[False, False])
        fops = FileOps(dry_run=False)

        organize_movie(tmp_path, config, fops)

        # Subtitle renamed
        assert (tmp_path / "Movie.2024.zh.srt").exists()

    def test_no_nfo_generation_for_movies(self, tmp_path: Path) -> None:
        """Test that NFO files are not generated for movies."""
        video = tmp_path / "Movie.2024.mkv"
        video.touch()

        config = OrganizeConfig(mode="movie", dry_run=False, generate_nfo=True)
        fops = FileOps(dry_run=False)

        result = organize_movie(tmp_path, config, fops)

        # No NFOs created for movies
        assert len(result.nfos_created) == 0

    def test_dry_run_no_changes(self, tmp_path: Path) -> None:
        """Test dry-run mode doesn't make changes."""
        video = tmp_path / "Movie.2024.mkv"
        video.touch()

        config = OrganizeConfig(
            mode="movie", dry_run=True, extras_flags=[False])
        fops = FileOps(dry_run=True)

        organize_movie(tmp_path, config, fops)

        # File should still exist in original location
        assert video.exists()

    def test_respects_extras_flags(self, tmp_path: Path) -> None:
        """Test that provided extras_flags are respected."""
        file1 = tmp_path / "file1.mkv"
        file2 = tmp_path / "file2.mkv"
        file1.touch()
        file2.touch()

        # Mark file2 as extra
        config = OrganizeConfig(
            mode="movie", dry_run=False, extras_flags=[False, True])
        fops = FileOps(dry_run=False)

        organize_movie(tmp_path, config, fops)

        # file1 stays in place
        assert (tmp_path / "file1.mkv").exists()

        # file2 moved to EXTRA
        assert (tmp_path / "EXTRA" / "file2.mkv").exists()

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test organizing empty directory."""
        config = OrganizeConfig(mode="movie", dry_run=False)
        fops = FileOps(dry_run=False)

        result = organize_movie(tmp_path, config, fops)

        assert len(result.files_moved) == 0
        assert len(result.errors) == 0


class TestOrganizeDirectory:
    """Tests for organize_directory function."""

    def test_organizes_as_tv_show(self, tmp_path: Path) -> None:
        """Test organizing directory as TV show."""
        video = tmp_path / "Show.S01E01.mkv"
        video.touch()

        config = OrganizeConfig(mode="show", season=1, dry_run=False)

        organize_directory(tmp_path, config)

        season_dir = tmp_path / "Season 1"
        assert season_dir.exists()
        assert (season_dir / "Show.S01E01.mkv").exists()

    def test_organizes_as_movie(self, tmp_path: Path) -> None:
        """Test organizing directory as movie."""
        video = tmp_path / "Movie.2024.mkv"
        video.touch()

        config = OrganizeConfig(
            mode="movie", dry_run=False, extras_flags=[False])

        organize_directory(tmp_path, config)

        assert (tmp_path / "Movie.2024.mkv").exists()

    def test_skip_mode_returns_skipped(self, tmp_path: Path) -> None:
        """Test that skip mode returns the path in skipped list."""
        config = OrganizeConfig(mode="skip", dry_run=False)

        result = organize_directory(tmp_path, config)

        assert len(result.skipped) == 1
        assert result.skipped[0] == tmp_path
        assert len(result.files_moved) == 0

    def test_raises_error_for_non_directory(self, tmp_path: Path) -> None:
        """Test that organizing non-directory raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.touch()

        config = OrganizeConfig(mode="movie", dry_run=False)

        with pytest.raises(ValueError, match="Not a directory"):
            organize_directory(file_path, config)

    def test_raises_error_for_tv_show_without_season(self, tmp_path: Path) -> None:
        """Test that TV show mode without season raises ValueError."""
        config = OrganizeConfig(mode="show", season=None, dry_run=False)

        with pytest.raises(ValueError, match="Season number required"):
            organize_directory(tmp_path, config)

    def test_raises_error_for_invalid_mode(self, tmp_path: Path) -> None:
        """Test that invalid mode raises ValueError."""
        config = OrganizeConfig(mode="invalid", dry_run=False)  # type: ignore

        with pytest.raises(ValueError, match="Invalid mode"):
            organize_directory(tmp_path, config)

    def test_with_dry_run(self, tmp_path: Path) -> None:
        """Test organizing with dry-run enabled."""
        video = tmp_path / "Show.S01E01.mkv"
        video.touch()

        config = OrganizeConfig(mode="show", season=1, dry_run=True)

        organize_directory(tmp_path, config)

        # No actual changes
        assert video.exists()
        assert not (tmp_path / "Season 1").exists()


class TestOrganizeConfigDataclass:
    """Tests for OrganizeConfig dataclass."""

    def test_default_values(self) -> None:
        """Test OrganizeConfig default values."""
        config = OrganizeConfig(mode="show")

        assert config.mode == "show"
        assert config.season is None
        assert config.locale_mapper is None
        assert config.generate_nfo is False
        assert config.dry_run is False
        assert config.extras_flags is None
        assert config.episode_overrides is None
        assert config.skip_nfo_generation is False
        assert config.root_dir is None
        assert config.remove_unmapped_subs is False

    def test_custom_values(self, tmp_path: Path) -> None:
        """Test OrganizeConfig with custom values."""
        mapper = LocaleMapper(csv_rules=[], cli_rules=[])
        config = OrganizeConfig(
            mode="movie",
            season=5,
            locale_mapper=mapper,
            generate_nfo=True,
            dry_run=True,
            extras_flags=[True, False],
            episode_overrides={1: 10},
            skip_nfo_generation=True,
            root_dir=tmp_path,
            remove_unmapped_subs=True,
        )

        assert config.mode == "movie"
        assert config.season == 5
        assert config.locale_mapper is mapper
        assert config.generate_nfo is True
        assert config.dry_run is True
        assert config.extras_flags == [True, False]
        assert config.episode_overrides == {1: 10}
        assert config.skip_nfo_generation is True
        assert config.root_dir == tmp_path
        assert config.remove_unmapped_subs is True


class TestOrganizeResultDataclass:
    """Tests for OrganizeResult dataclass."""

    def test_result_creation(self) -> None:
        """Test OrganizeResult instantiation."""
        result = OrganizeResult(
            files_moved=[(Path("a"), Path("b"))],
            nfos_created=[Path("c.nfo")],
            errors=["error"],
            skipped=[Path("d")],
        )

        assert len(result.files_moved) == 1
        assert len(result.nfos_created) == 1
        assert len(result.errors) == 1
        assert len(result.skipped) == 1
