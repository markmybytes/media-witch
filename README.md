
# Media Witch

A modular CLI toolkit for media file organization. Transform disorganized media libraries into properly structured directories with automatic episode detection, subtitle renaming, NFO generation, and torrent file mock creation.

[![Tests](https://github.com/yourusername/media-witch/workflows/Tests/badge.svg)](https://github.com/yourusername/media-witch/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)

## Features

- **Organize Media Files** - Automatically structure TV shows and movies into season folders
- **Generate NFO Files** - Create episode metadata files for media servers
- **Rename Subtitles** - Match subtitles to videos with automatic locale code mapping
- **Create Torrent Structures** - Generate fake file trees from .torrent files
- **Modular APIs** - Import features programmatically in your own scripts
- **Extensible** - Easy to add new features and organization strategies

## Installation

### From Source

```bash
git clone https://github.com/yourusername/media-witch.git
cd media-witch

# Install in development mode
pip install -e .

# With development dependencies (testing, type checking, linting)
pip install -e ".[dev]"
```

## Quick Start

### Organize Media Files (Interactive)
```bash
media-witch organize ./Downloads/MyShow/
# Interactive prompts guide you through:
# - TV Show, Movie, or Skip?
# - Season number (for TV shows)
# - Which files are extras? (checkbox selection)
# - NFO files generated automatically
# - Episode number overrides (optional)
```

### Organize Multiple Shows
```bash
media-witch organize ./Downloads/TV-Shows/
# Detects multiple shows and processes each interactively
```

### Generate NFO Files
```bash
media-witch nfo ./Season1/*.mkv --season 1 --episode-start 1
```

### Rename Subtitles with Locale Mapping
```bash
media-witch subtitles ./Movies/ \
  --map "chi,zh,false" \
  --map "cht,zh-Hant,false"
```

Or from CSV file:
```bash
media-witch subtitles ./Movies/ --map-csv locales.csv
```

### Create Files from Torrents
```bash
media-witch torrent ./torrents/*.torrent --output-dir ./structures
```

## CLI Commands

### organize
Organize media files into TV show or movie structure with interactive prompts.

```bash
media-witch organize [OPTIONS] PATHS...

Options:
  --map-csv PATH                           CSV file with subtitle locale mappings
  --map TEXT                               Inline locale mapping rule (repeatable)
  --dry-run / --no-dry-run                 Preview changes without executing
  -q, --quiet                              Suppress non-essential output
```

**Features:**
- **Interactive prompts** for TV/Movie classification
- **Automatic leaf-finding** for complex directory structures
- **NFO generation** enabled by default for TV shows
- **Extras classification** via checkbox selection
- **Episode number overrides** when needed
- **Batch processing** for multiple shows/seasons/movies

### nfo
```bash
media-witch nfo [OPTIONS] PATHS...

Options:
  --season INTEGER (required)              Season number
  --episode-start INTEGER                  Starting episode number (default: 1)
  --dry-run / --no-dry-run
  -v, --verbose
  -q, --quiet
```

### subtitles
```bash
media-witch subtitles [OPTIONS] PATHS...

Options:
  --map-csv PATH                           CSV file with locale mappings
  --map TEXT                               Inline locale mapping rule (repeatable)
  --dry-run / --no-dry-run
  -v, --verbose
  -q, --quiet
```

### torrent
```bash
media-witch torrent [OPTIONS] PATHS...

Options:
  --output-dir PATH                        Output directory for files (default: .)
  -v, --verbose                            Enable verbose output
```

## Programmatic Usage

### Organize a Directory
```python
from pathlib import Path
from media_witch.features.organize.api import organize_directory, OrganizeConfig
from media_witch.features.subtitles.locale import LocaleMapper

config = OrganizeConfig(
    mode="show",
    season=1,
    locale_mapper=LocaleMapper(csv_rules=[], cli_rules=[]),
    generate_nfo=True,
    dry_run=False,
)

result = organize_directory(Path("./media"), config)
print(f"Moved {len(result.files_moved)} files")
print(f"Created {len(result.nfos_created)} NFO files")
```

### Generate NFO Files
```python
from pathlib import Path
from media_witch.features.nfo.api import generate_episode_nfos, NFOConfig

videos = sorted(Path("./Season1").glob("*.mkv"))
config = NFOConfig(season=1, episode_start=1)
result = generate_episode_nfos(videos, config)

print(f"Created {len(result.created)} NFO files")
```

### Rename Subtitles
```python
from pathlib import Path
from media_witch.features.subtitles.api import rename_subtitles, SubtitleConfig
from media_witch.features.subtitles.locale import LocaleMapper, Rule

mapper = LocaleMapper(
    csv_rules=[],
    cli_rules=[Rule("chi", "zh", False)]
)
config = SubtitleConfig(locale_mapper=mapper)

result = rename_subtitles(
    subtitles=[Path("movie.chi.srt")],
    video=Path("movie.mkv"),
    config=config
)
```

### Create from Torrent
```python
from pathlib import Path
from media_witch.features.torrent.api import create_from_torrent, TorrentConfig

config = TorrentConfig(output_dir=Path("./output"), verbose=True)
result = create_from_torrent(Path("download.torrent"), config)

print(f"Created {len(result.created_files)} files")
```

## Locale Mapping CSV

Format for `--map-csv` option:

```csv
source,target,is_case_sensitive
chi,zh,false
cht,zh-Hant,false
eng,en,false
jpn,ja,false
EN,en-US,true
```

Columns:
- **source**: Locale code to match
- **target**: Locale code to map to
- **is_case_sensitive**: Whether matching is case-sensitive (true/false)

## Architecture

```
src/media_witch/                    Package root
├── cli/                            CLI commands using Click
│   ├── main.py                     Command group (media-witch)
│   ├── organize.py                 organize subcommand
│   ├── nfo.py                      nfo subcommand
│   ├── subtitles.py                subtitles subcommand
│   ├── torrent.py                  torrent subcommand
│   └── common.py                   Shared options
├── core/                           Core utilities
│   ├── fileops.py                  File operations with dry-run
│   ├── actions.py                  Action queue pattern
│   ├── media.py                    Media file detection
│   └── patterns.py                 Episode pattern matching
├── features/                       Feature implementations
│   ├── organize/                   Media organization
│   │   └── api.py
│   ├── nfo/                        NFO generation
│   │   └── api.py
│   ├── subtitles/                  Subtitle processing
│   │   ├── locale.py
│   │   └── api.py
│   └── torrent/                    Torrent processing
│       ├── decoder.py
│       ├── parser.py
│       └── api.py
└── ui/                             User interface
    ├── prompts.py                  Interactive prompts
    └── display.py                  Display formatting
```

## Testing

Run unit tests:
```bash
pytest tests/unit/ -v
```

Run with coverage:
```bash
pytest tests/unit/ --cov=media_witch --cov-report=html
```

Current test coverage: **86 tests**, covering all core modules and feature APIs.

## Development

### Code Style
```bash
# Check code style
ruff check src/media_witch tests

# Format code
ruff format src/media_witch tests
```

### Type Checking
```bash
mypy src/media_witch
```

### Install Dev Dependencies
```bash
pip install -e ".[dev]"
```

## Deprecation Notice

**Version 2.0.0+ is a major refactoring.** The old scripts (`src/main.py`, `src/torrent2dummy.py`) are deprecated and will be removed in v2.1.0. Please migrate to the new CLI:

### Migration Guide

| Old Usage                                  | New Usage                          |
| ------------------------------------------ | ---------------------------------- |
| `python src/main.py ./path`                | `media-witch organize ./path`      |
| `python src/torrent2dummy.py file.torrent` | `media-witch torrent file.torrent` |
| Import from monolithic file                | Import from feature modules        |

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Run tests and linting (`pytest`, `ruff check`)
5. Commit with clear messages
6. Push and create a Pull Request

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Changelog

### v2.0.0 (2026-04-10)
- **BREAKING CHANGE:** Complete refactoring from monolithic to modular architecture
- New Click-based CLI with distinct subcommands
- Public APIs for programmatic use
- Comprehensive test suite (86+ tests)
- Package installable via pip
- Type hints throughout codebase
- CI/CD pipeline with GitHub Actions

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/media-witch/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/media-witch/discussions)

---

Made with ❤️ for media enthusiasts
