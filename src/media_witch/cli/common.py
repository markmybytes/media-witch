"""Common CLI options and decorators."""

from __future__ import annotations

import click

# Shared options
dry_run_option = click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Preview changes without executing them",
)

verbose_option = click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)

quiet_option = click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress non-essential output",
)

locale_csv_option = click.option(
    "--map-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="CSV file containing subtitle locale mappings",
)

locale_map_option = click.option(
    "--map",
    "locale_maps",
    multiple=True,
    help="Inline locale mapping: source,target,case_sensitive (repeatable)",
)


def common_options(f):
    """Decorator to add common options to commands."""
    f = dry_run_option(f)
    f = verbose_option(f)
    f = quiet_option(f)
    return f
