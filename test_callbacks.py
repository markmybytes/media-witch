"""Integration test for interactive organize mode."""

from pathlib import Path

from src.media_witch.features.organize.api import OrganizeConfig


def mock_extras_classifier(items: list[Path], defaults: list[bool]) -> list[bool]:
    """Mock classifier that marks items ending with 'extra' as extras."""
    return [p.name.endswith("extra") or default for p, default in zip(items, defaults)]


def mock_nfo_overrides(videos: list[Path], season: int) -> dict[int, int]:
    """Mock NFO override that doesn't change anything."""
    return {}


def test_config_with_callbacks():
    """Test that OrganizeConfig accepts callables."""
    config = OrganizeConfig(
        mode="show",
        season=1,
        generate_nfo=True,
        dry_run=True,
        extras_classifier=mock_extras_classifier,
        nfo_override_callback=mock_nfo_overrides,
    )

    print(f"✓ Config created successfully")
    print(f"  Mode: {config.mode}")
    print(f"  Season: {config.season}")
    print(f"  Has extras_classifier: {config.extras_classifier is not None}")
    print(
        f"  Has nfo_override_callback: {config.nfo_override_callback is not None}")

    # Test calling the callbacks
    test_items = [Path("video.mkv"), Path("sample_extra")]
    test_defaults = [False, True]
    result = config.extras_classifier(test_items, test_defaults)
    print(f"✓ Extras classifier returned: {result}")

    test_videos = [Path("ep1.mkv"), Path("ep2.mkv")]
    overrides = config.nfo_override_callback(test_videos, 1)
    print(f"✓ NFO overrides returned: {overrides}")


if __name__ == "__main__":
    try:
        test_config_with_callbacks()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
