"""Shared pytest fixtures for all tests."""

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Verbosity, settings

from media_witch.core.fileops import FileOps

# ============================================================================
# Hypothesis configuration
# ============================================================================

# Register profiles for different environments
settings.register_profile(
    'ci',
    max_examples=1000,
    deadline=1000,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    'dev',
    max_examples=100,
    deadline=500,
)

settings.register_profile(
    'debug',
    max_examples=10,
    deadline=None,
    verbosity=Verbosity.verbose,
)

# Load profile based on environment variable
profile = os.getenv('HYPOTHESIS_PROFILE', 'dev')
settings.load_profile(profile)


@pytest.fixture
def tmp_media_dir(tmp_path: Path) -> Path:
    """Temporary directory for media files.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        Path to a temporary media directory
    """
    media_dir = tmp_path / 'media'
    media_dir.mkdir()
    return media_dir


@pytest.fixture
def file_ops_real(tmp_path: Path) -> FileOps:
    """Real FileOps instance (no dry-run).

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        FileOps instance with dry_run=False
    """
    logs: list[str] = []
    fops = FileOps(dry_run=False, logger=logs.append)
    fops._test_logs = logs  # Attach logs for test inspection
    return fops


@pytest.fixture
def file_ops_dry(tmp_path: Path) -> FileOps:
    """Dry-run FileOps instance.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        FileOps instance with dry_run=True
    """
    logs: list[str] = []
    fops = FileOps(dry_run=True, logger=logs.append)
    fops._test_logs = logs  # Attach logs for test inspection
    return fops
