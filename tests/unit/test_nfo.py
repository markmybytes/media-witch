"""Unit tests for NFO generation."""

from pathlib import Path

from media_witch.features.nfo.api import (NFOConfig, generate_episode_nfos,
                                          generate_nfo_content)


class TestGenerateNfoContent:
    """Tests for generate_nfo_content function."""

    def test_generates_valid_xml(self) -> None:
        """Test that valid XML is generated."""
        content = generate_nfo_content(
            title="Episode Title",
            season=1,
            episode=5,
        )

        assert '<?xml version="1.0"' in content
        assert '<episodedetails>' in content
        assert '<title>Episode Title</title>' in content
        assert '<episode>5</episode>' in content
        assert '<season>1</season>' in content
        assert '</episodedetails>' in content

    def test_special_characters_in_title(self) -> None:
        """Test handling of special characters in title."""
        content = generate_nfo_content(
            title="Episode & Title",
            season=2,
            episode=10,
        )

        assert '<title>Episode & Title</title>' in content


class TestGenerateEpisodeNfos:
    """Tests for generate_episode_nfos function."""

    def test_generates_nfos_for_videos(self, tmp_path: Path) -> None:
        """Test NFO generation for multiple videos."""
        videos = [
            tmp_path / "episode1.mkv",
            tmp_path / "episode2.mkv",
            tmp_path / "episode3.mkv",
        ]
        for v in videos:
            v.touch()

        config = NFOConfig(season=1, episode_start=1, dry_run=False)
        result = generate_episode_nfos(videos, config)

        assert len(result.created) == 3
        assert len(result.errors) == 0

        # Check NFO files were created
        for i, video in enumerate(videos, start=1):
            nfo = video.with_suffix(".nfo")
            assert nfo.exists()
            content = nfo.read_text()
            assert f'<episode>{i}</episode>' in content
            assert '<season>1</season>' in content

    def test_episode_start_offset(self, tmp_path: Path) -> None:
        """Test episode numbering with start offset."""
        videos = [
            tmp_path / "episode1.mkv",
            tmp_path / "episode2.mkv",
        ]
        for v in videos:
            v.touch()

        config = NFOConfig(season=2, episode_start=5, dry_run=False)
        generate_episode_nfos(videos, config)

        nfo1 = videos[0].with_suffix(".nfo")
        nfo2 = videos[1].with_suffix(".nfo")

        content1 = nfo1.read_text()
        content2 = nfo2.read_text()

        assert '<episode>5</episode>' in content1
        assert '<episode>6</episode>' in content2

    def test_episode_overrides(self, tmp_path: Path) -> None:
        """Test episode number overrides."""
        videos = [
            tmp_path / "episode1.mkv",
            tmp_path / "episode2.mkv",
            tmp_path / "episode3.mkv",
        ]
        for v in videos:
            v.touch()

        # Override: episode 2 should be numbered as 10
        config = NFOConfig(
            season=1,
            episode_start=1,
            episode_overrides={2: 10},
            dry_run=False
        )
        generate_episode_nfos(videos, config)

        nfo1 = videos[0].with_suffix(".nfo")
        nfo2 = videos[1].with_suffix(".nfo")
        nfo3 = videos[2].with_suffix(".nfo")

        assert '<episode>1</episode>' in nfo1.read_text()
        assert '<episode>10</episode>' in nfo2.read_text()
        assert '<episode>3</episode>' in nfo3.read_text()

    def test_dry_run_no_creation(self, tmp_path: Path) -> None:
        """Test dry-run doesn't create files."""
        videos = [tmp_path / "episode1.mkv"]
        videos[0].touch()

        config = NFOConfig(season=1, episode_start=1, dry_run=True)
        generate_episode_nfos(videos, config)

        nfo = videos[0].with_suffix(".nfo")
        assert not nfo.exists()

    def test_skips_existing_nfos(self, tmp_path: Path) -> None:
        """Test that existing NFO files are skipped."""
        video = tmp_path / "episode1.mkv"
        video.touch()

        nfo = video.with_suffix(".nfo")
        nfo.write_text("existing content")

        config = NFOConfig(season=1, episode_start=1, dry_run=False)
        result = generate_episode_nfos([video], config)

        assert len(result.skipped) == 1
        assert nfo.read_text() == "existing content"
