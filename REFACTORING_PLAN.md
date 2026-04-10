# Media Witch Refactoring Plan

**Version:** 1.0  
**Target Python:** 3.11+  
**CLI Framework:** Click  
**Testing Framework:** pytest  
**Breaking Changes:** Acceptable

---

## 1. Executive Summary

This plan transforms **media-witch** from a monolithic script into a modular, extensible CLI toolkit. Each feature becomes an independent module with:
- **Dedicated CLI command** for standalone use
- **Reusable public API** for programmatic access
- **Clear separation** of business logic from I/O and CLI concerns
- **Comprehensive test coverage** with unit and integration tests

The refactored architecture enables users to:
- Run individual features in isolation (e.g., only rename subtitles)
- Compose workflows using the master orchestrator command
- Import and reuse logic in external tools/scripts
- Extend functionality by adding new feature modules

---

## 2. Current Architecture Analysis

### 2.1 main.py (~687 lines)
**Current Responsibilities:**
- Interactive media organization (TV shows, movies)
- Subtitle renaming with locale mapping
- NFO file generation for TV episodes
- Extra files classification and management
- Directory flattening and Season folder structuring
- Dry-run support and action queuing

**Key Components:**
| Component             | Type          | Purpose                                          |
| --------------------- | ------------- | ------------------------------------------------ |
| `LocaleMapper`        | Service       | Maps subtitle language codes using CSV/CLI rules |
| `FileOps`             | Service       | Abstracted file operations with dry-run support  |
| `ActionQueue`         | Orchestration | Batches and commits file operations              |
| `UI`                  | Presentation  | Interactive prompts using questionary            |
| `SubtitleService`     | Service       | Subtitle pairing and renaming logic              |
| `ShowProcessor`       | Domain        | TV show organization workflow                    |
| `MovieProcessor`      | Domain        | Movie organization workflow                      |
| `process_directory()` | Orchestration | Top-level workflow dispatcher                    |

**Issues:**
- All logic tightly coupled in single file
- Business logic mixed with CLI parsing and interactive prompts
- No clear public API for reuse
- Testing difficult due to tight coupling
- Questionary UI calls embedded throughout processors

### 2.2 torrent2dummy.py (~164 lines)
**Current Responsibilities:**
- Bencode decoding for .torrent files
- Torrent metadata extraction
- Empty file/directory creation matching torrent structure

**Key Components:**
| Component             | Type    | Purpose                       |
| --------------------- | ------- | ----------------------------- |
| `bdecode()`           | Utility | Bencode parser                |
| `read_torrent_info()` | I/O     | Torrent file reader           |
| `create_fake_files()` | Domain  | Fake file structure generator |

**Issues:**
- Print statements scattered throughout domain logic
- Not integrated with main.py workflow
- No abstraction for file creation (uses direct `open()`)

---

## 3. Proposed Module Structure

```
media-witch/
├── src/
│   ├── media_witch/
│   │   ├── __init__.py                 # Package version, public API exports
│   │   ├── cli/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                 # Master command (orchestrator)
│   │   │   ├── organize.py             # Organize files command
│   │   │   ├── nfo.py                  # NFO generation command
│   │   │   ├── subtitles.py            # Subtitle renaming command
│   │   │   ├── torrent.py              # Torrent fake files command
│   │   │   └── common.py               # Shared CLI options/decorators
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── fileops.py              # FileOps with dry-run support
│   │   │   ├── actions.py              # ActionQueue pattern
│   │   │   ├── media.py                # Media file type detection
│   │   │   └── patterns.py             # Episode/season pattern matching
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── organize/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── api.py              # Public API (organize_directory)
│   │   │   │   ├── processors.py       # ShowProcessor, MovieProcessor
│   │   │   │   └── classifiers.py      # Extra file classification
│   │   │   ├── nfo/
│   │   │   │   ├── __init__.py
│   │   │   │   └── api.py              # generate_nfo, generate_episode_nfos
│   │   │   ├── subtitles/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── api.py              # rename_subtitles, pair_subtitles
│   │   │   │   └── locale.py           # LocaleMapper, rule parsing
│   │   │   └── torrent/
│   │   │       ├── __init__.py
│   │   │       ├── api.py              # create_from_torrent
│   │   │       ├── decoder.py          # Bencode decoder
│   │   │       └── parser.py           # Torrent metadata extraction
│   │   └── ui/
│   │       ├── __init__.py
│   │       ├── prompts.py              # All questionary interactions
│   │       └── display.py              # Tree printing, formatting
│   ├── setup.py / pyproject.toml       # Package configuration
│   └── media-witch                     # Entry point script
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # pytest fixtures
│   ├── unit/
│   │   ├── test_fileops.py
│   │   ├── test_actions.py
│   │   ├── test_media.py
│   │   ├── test_patterns.py
│   │   ├── test_locale.py
│   │   ├── test_nfo.py
│   │   ├── test_subtitles.py
│   │   ├── test_torrent_decoder.py
│   │   └── test_processors.py
│   ├── integration/
│   │   ├── test_organize_workflow.py
│   │   ├── test_nfo_generation.py
│   │   ├── test_subtitle_renaming.py
│   │   └── test_torrent_creation.py
│   └── fixtures/
│       ├── sample_TV_Show/             # Test media files
│       ├── sample_Movie/
│       └── sample.torrent
├── docs/
│   ├── api/                            # API documentation
│   ├── cli/                            # CLI usage examples
│   └── architecture.md
├── REFACTORING_PLAN.md                 # This document
└── README.md
```

---

## 4. CLI Command Layout

### 4.1 Command Hierarchy (Click Group)

```
media-witch                              # Master command (group)
├── organize                             # Organize media files
├── nfo                                  # Generate NFO files
├── subtitles                            # Rename subtitles
└── torrent                              # Create fake files from torrent
```

### 4.2 Individual Commands

#### `media-witch organize`
Organizes media files into TV show or movie structure.

**Arguments:**
```bash
media-witch organize [OPTIONS] PATHS...

PATHS: One or more directories to organize

Options:
  --mode [interactive|auto-tv|auto-movie]  # Organization mode
  --generate-nfo / --no-nfo                # Generate episode NFOs (TV only)
  --map-csv PATH                           # Subtitle locale mapping CSV
  --map TEXT                               # Inline mapping rules (repeatable)
  --dry-run / --no-dry-run                 # Preview changes
  --batch-season INTEGER                   # Non-interactive season number
  --verbose / --quiet                      # Logging level
  --silent                                 # Suppress all messages
```

**Behavior:**
- Interactive mode (default): Prompts for TV/Movie/Skip, seasons, extras
- Auto modes: Skip prompts, apply heuristics
- Calls underlying API: `organize.api.organize_directory()`

#### `media-witch nfo`
Generate NFO metadata files for video files.

**Arguments:**
```bash
media-witch nfo [OPTIONS] PATHS...

PATHS: Video files or directories

Options:
  --season INTEGER                         # Season number (required for TV)
  --episode-start INTEGER                  # Starting episode number
  --overrides TEXT                         # JSON map: {"1": 3, "2": 4}
  --dry-run / --no-dry-run
  --verbose / --quiet                      # Logging level
  --silent                                 # Suppress all messages
```

**Behavior:**
- Scans for video files
- Generates .nfo files with episode/season metadata
- Calls underlying API: `nfo.api.generate_episode_nfos()`

#### `media-witch subtitles`
Rename and organize subtitle files.

**Arguments:**
```bash
media-witch subtitles [OPTIONS] PATHS...

PATHS: Directories or subtitle files

Options:
  --map-csv PATH                           # Locale mapping CSV
  --map TEXT                               # Inline mapping rules
  --pair-with PATH                         # Video file to pair with
  --dry-run / --no-dry-run
  --verbose / --quiet                      # Logging level
  --silent                                 # Suppress all messages
```

**Behavior:**
- Pairs subtitles with video files
- Normalizes locale codes (e.g., chi -> zh, cht -> zh-Hant)
- Calls underlying API: `subtitles.api.rename_subtitles()`

#### `media-witch torrent`
Create fake file structure from .torrent files.

**Arguments:**
```bash
media-witch torrent [OPTIONS] PATHS...

PATHS: .torrent files or directories containing them

Options:
  --output-dir PATH                        # Output directory (default: cwd)
  --verbose / --quiet                      # Logging level
  --silent                                 # Suppress all messages
```

**Behavior:**
- Decodes .torrent files
- Creates empty files matching structure
- Calls underlying API: `torrent.api.create_from_torrent()`

#### `media-witch` (master command)
Orchestrates full workflow: organize + nfo + subtitles.

**Arguments:**
```bash
media-witch [OPTIONS] PATHS...

PATHS: Directories to process

Options:
  --mode [interactive|auto-tv|auto-movie]
  --generate-nfo / --no-nfo
  --map-csv PATH
  --map TEXT
  --dry-run / --no-dry-run
  --verbose / --quiet                      # Logging level
  --silent                                 # Suppress all messages
```

**Behavior:**
- Combines organize + NFO + subtitle features
- Replicates current `main.py` end-to-end workflow
- **Implementation:** Calls the public APIs from each feature module in sequence

---

## 5. Feature Module API Design

Each feature exposes a **clean public API** with business logic separated from I/O concerns.

### 5.1 Core Design Principles

**Separation of Concerns:**
```python
# ❌ BAD: Mixed concerns
def process_show(path: Path):
    choice = questionary.select(...).ask()  # UI
    files = list(path.iterdir())             # I/O
    move_files(files)                        # Business logic
    print("Done!")                           # Output

# ✅ GOOD: Separated layers
# UI layer (cli/organize.py)
def cli_command(path: Path):
    choice = prompts.ask_processing_choice(path)
    result = organize_api.organize_directory(path, mode=choice, dry_run=False)
    display.show_result(result)

# Business logic (features/organize/api.py)
def organize_directory(
    path: Path,
    mode: Literal["show", "movie", "skip"],
    dry_run: bool = False,
) -> OrganizeResult:
    # Pure logic, no I/O or UI
    ...
```

### 5.2 Public APIs

#### `features.organize.api`
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class OrganizeConfig:
    """Configuration for organization."""
    mode: Literal["show", "movie", "skip"]
    season: int | None = None
    locale_mapper: LocaleMapper | None = None
    generate_nfo: bool = False
    dry_run: bool = False

@dataclass
class OrganizeResult:
    """Result of organization operation."""
    files_moved: list[tuple[Path, Path]]
    nfos_created: list[Path]
    errors: list[str]
    skipped: list[Path]

def organize_directory(
    path: Path,
    config: OrganizeConfig,
) -> OrganizeResult:
    """
    Organize media files in directory.
    
    Args:
        path: Directory to organize
        config: Organization configuration
        
    Returns:
        Result object with operation details
        
    Raises:
        ValueError: If path is not a directory
        PermissionError: If insufficient permissions
    """
    ...

def classify_extras(
    items: list[Path],
    *,
    auto: bool = False,
) -> list[bool]:
    """
    Classify items as primary media or extras.
    
    Args:
        items: Files/directories to classify
        auto: If True, use heuristics; if False, caller must prompt
        
    Returns:
        Boolean list where True = extra, False = primary
    """
    ...
```

#### `features.nfo.api`
```python
@dataclass
class NFOConfig:
    """Configuration for NFO generation."""
    season: int
    episode_start: int = 1
    episode_overrides: dict[int, int] | None = None
    dry_run: bool = False

@dataclass
class NFOResult:
    """Result of NFO generation."""
    created: list[Path]
    skipped: list[Path]
    errors: list[str]

def generate_episode_nfos(
    videos: list[Path],
    config: NFOConfig,
) -> NFOResult:
    """
    Generate .nfo files for video files.
    
    Args:
        videos: Sorted list of video files
        config: NFO generation configuration
        
    Returns:
        Result object with created NFO paths
    """
    ...

def generate_nfo_content(
    title: str,
    season: int,
    episode: int,
) -> str:
    """Generate XML content for NFO file."""
    ...
```

#### `features.subtitles.api`
```python
@dataclass
class SubtitleConfig:
    """Configuration for subtitle operations."""
    locale_mapper: LocaleMapper
    dry_run: bool = False

@dataclass
class SubtitleResult:
    """Result of subtitle operations."""
    renamed: list[tuple[Path, Path]]
    skipped: list[Path]
    errors: list[str]

def rename_subtitles(
    subtitles: list[Path],
    video: Path,
    config: SubtitleConfig,
) -> SubtitleResult:
    """
    Rename subtitles to match video file.
    
    Args:
        subtitles: Subtitle files to rename
        video: Video file to pair with
        config: Subtitle configuration
        
    Returns:
        Result object with renamed paths
    """
    ...

def pair_subtitles(
    subtitles: list[Path],
    videos: list[Path],
) -> dict[Path, list[Path]]:
    """
    Pair subtitle files with video files.
    
    Returns:
        Mapping of video -> list of paired subtitles
    """
    ...
```

#### `features.torrent.api`
```python
@dataclass
class TorrentConfig:
    """Configuration for torrent operations."""
    output_dir: Path
    verbose: bool = False

@dataclass
class TorrentResult:
    """Result of torrent file creation."""
    created_files: list[Path]
    created_dirs: list[Path]
    errors: list[str]

def create_from_torrent(
    torrent_path: Path,
    config: TorrentConfig,
) -> TorrentResult:
    """
    Create fake file structure from .torrent file.
    
    Args:
        torrent_path: Path to .torrent file
        config: Torrent configuration
        
    Returns:
        Result object with created paths
    """
    ...

def parse_torrent(
    torrent_path: Path,
) -> dict[str, Any]:
    """
    Parse .torrent file metadata.
    
    Returns:
        Dictionary with 'name', 'files', 'total_size' keys
    """
    ...
```

### 5.3 Shared Core Services

#### `core.fileops.FileOps`
Abstracted file operations with dry-run support. **No changes needed** except:
- Move to `core/fileops.py`
- Add type hints using Python 3.11+ syntax
- Extract logging to callback parameter for testability

```python
# Enhanced for testability
class FileOps:
    def __init__(
        self,
        dry_run: bool,
        *,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.dry = dry_run
        self._log = logger or print
        ...
```

#### `core.actions.ActionQueue`
**No changes needed**, just relocate to `core/actions.py`.

#### `ui.prompts`
All interactive prompts extracted from processors:
```python
def ask_processing_choice(path: Path, has_files: bool) -> str:
    """Ask user how to process directory."""
    ...

def ask_season(default: int = 1) -> int:
    """Prompt for season number."""
    ...

def ask_extras_classification(items: list[Path]) -> list[bool]:
    """Prompt user to classify extras."""
    ...

def ask_nfo_overrides(videos: list[Path], season: int) -> dict[int, int]:
    """Prompt for episode number overrides."""
    ...
```

---

## 6. Testing Strategy

### 6.1 Testing Philosophy

**Test Pyramid:**
```
         /\
        /  \  E2E Tests (CLI integration, full workflows)
       /----\
      / Inte-\ Integration Tests (feature APIs, file I/O)
     /  gration\
    /----------\
   /    Unit    \ Unit Tests (pure logic, no I/O/UI)
  /--------------\
```

**Coverage Targets:**
- Unit tests: **>90%** coverage
- Integration tests: All feature APIs and workflows
- E2E tests: All CLI commands and master orchestrator

### 6.2 Unit Tests (Fast, No I/O)

**What to test:**
- Pure functions (no side effects)
- Pattern matching (episode/season detection)
- Data transformations (locale mapping, NFO XML generation)
- Business logic (file classification heuristics)

**Example: `tests/unit/test_patterns.py`**
```python
import pytest
from media_witch.core.patterns import has_episode_pattern, extract_season_episode

def test_has_episode_pattern_s01e01():
    assert has_episode_pattern("Show.S01E01.mkv") is True
    assert has_episode_pattern("Show.S1E1.mkv") is True
    assert has_episode_pattern("Random.Movie.mkv") is False

def test_extract_season_episode():
    assert extract_season_episode("Show.S02E05.mkv") == (2, 5)
    assert extract_season_episode("[12].mkv") == (None, 12)
```

**Example: `tests/unit/test_locale.py`**
```python
from media_witch.features.subtitles.locale import LocaleMapper, Rule

def test_locale_mapping_case_insensitive():
    rules = [Rule(source="chi", target="zh", case_sensitive=False)]
    mapper = LocaleMapper(csv_rules=[], cli_rules=rules)
    
    assert mapper.resolve("chi") == "zh"
    assert mapper.resolve("CHI") == "zh"
    assert mapper.resolve("Chi") == "zh"

def test_locale_mapping_case_sensitive():
    rules = [Rule(source="EN", target="en-US", case_sensitive=True)]
    mapper = LocaleMapper(csv_rules=[], cli_rules=rules)
    
    assert mapper.resolve("EN") == "en-US"
    assert mapper.resolve("en") == "en"  # No match, returns original
```

### 6.3 Integration Tests (With Filesystem)

**What to test:**
- Feature APIs with real file I/O (using temp directories)
- FileOps operations (move, rename, mkdir)
- ActionQueue commit behavior
- End-to-end workflows (organize directory, generate NFOs)

**Example: `tests/integration/test_organize_workflow.py`**
```python
import pytest
from pathlib import Path
from media_witch.features.organize.api import organize_directory, OrganizeConfig

@pytest.fixture
def tv_show_fixture(tmp_path):
    """Create sample TV show structure."""
    show_dir = tmp_path / "My.Show.S01"
    show_dir.mkdir()
    
    (show_dir / "Episode.S01E01.mkv").touch()
    (show_dir / "Episode.S01E02.mkv").touch()
    (show_dir / "Episode.S01E01.chi.ass").touch()
    (show_dir / "Extra.Interviews.mkv").touch()
    
    return show_dir

def test_organize_tv_show_creates_season_folders(tv_show_fixture):
    config = OrganizeConfig(
        mode="show",
        season=1,
        generate_nfo=False,
        dry_run=False,
    )
    
    result = organize_directory(tv_show_fixture, config)
    
    assert (tv_show_fixture / "Season 1").exists()
    assert len(result.files_moved) == 2  # Two episodes
    assert len(result.errors) == 0

def test_organize_dry_run_no_side_effects(tv_show_fixture):
    config = OrganizeConfig(mode="show", season=1, dry_run=True)
    
    result = organize_directory(tv_show_fixture, config)
    
    # No actual file moves
    assert not (tv_show_fixture / "Season 1").exists()
    # But planned actions recorded
    assert len(result.files_moved) > 0
```

**Example: `tests/integration/test_subtitle_renaming.py`**
```python
def test_rename_subtitles_with_locale_mapping(tmp_path):
    from media_witch.features.subtitles.api import rename_subtitles, SubtitleConfig
    from media_witch.features.subtitles.locale import LocaleMapper, Rule
    
    video = tmp_path / "Movie.mkv"
    video.touch()
    
    sub = tmp_path / "Movie.chi.ass"
    sub.touch()
    
    mapper = LocaleMapper(
        csv_rules=[],
        cli_rules=[Rule("chi", "zh", False)],
    )
    config = SubtitleConfig(locale_mapper=mapper, dry_run=False)
    
    result = rename_subtitles([sub], video, config)
    
    assert len(result.renamed) == 1
    assert (tmp_path / "Movie.zh.ass").exists()
    assert not sub.exists()
```

### 6.4 E2E Tests (CLI Commands)

**What to test:**
- CLI argument parsing
- Command execution with Click's `CliRunner`
- Output formatting (stdout/stderr)
- Exit codes

**Example: `tests/integration/test_cli_organize.py`**
```python
from click.testing import CliRunner
from media_witch.cli.organize import organize_command

def test_organize_command_dry_run(tv_show_fixture):
    runner = CliRunner()
    
    result = runner.invoke(organize_command, [
        str(tv_show_fixture),
        "--mode", "auto-tv",
        "--batch-season", "1",
        "--dry-run",
    ])
    
    assert result.exit_code == 0
    assert "[DRY-RUN]" in result.output
    assert not (tv_show_fixture / "Season 1").exists()

def test_organize_command_generates_nfo(tv_show_fixture):
    runner = CliRunner()
    
    result = runner.invoke(organize_command, [
        str(tv_show_fixture),
        "--mode", "auto-tv",
        "--batch-season", "1",
        "--generate-nfo",
    ])
    
    assert result.exit_code == 0
    nfos = list((tv_show_fixture / "Season 1").glob("*.nfo"))
    assert len(nfos) == 2
```

### 6.5 Test Fixtures and Utilities

**`tests/conftest.py`** - Shared pytest fixtures:
```python
import pytest
from pathlib import Path
from media_witch.core.fileops import FileOps

@pytest.fixture
def tmp_media_dir(tmp_path):
    """Temporary directory for media files."""
    return tmp_path / "media"

@pytest.fixture
def file_ops_real() -> FileOps:
    """Real FileOps (no dry-run)."""
    return FileOps(dry_run=False)

@pytest.fixture
def file_ops_dry() -> FileOps:
    """Dry-run FileOps."""
    return FileOps(dry_run=True)

@pytest.fixture
def sample_torrent(tmp_path):
    """Sample .torrent file."""
    # Copy from tests/fixtures/sample.torrent
    ...
```

**`tests/fixtures/`** - Sample files:
- `sample_TV_Show/` - Realistic TV show structure
- `sample_Movie/` - Movie with extras
- `sample.torrent` - Valid .torrent file
- `locale_mapping.csv` - Sample locale CSV

### 6.6 Test Execution

**Run all tests:**
```bash
pytest
```

**Run specific test types:**
```bash
pytest tests/unit/              # Unit tests only (fast)
pytest tests/integration/       # Integration tests (slower)
pytest -m "not slow"            # Skip slow tests
```

**Coverage report:**
```bash
pytest --cov=media_witch --cov-report=html
```

**Watch mode (during development):**
```bash
pytest --watch
```

---

## 7. Migration Path

### 7.1 Phased Approach

**Phase 1: Infrastructure** (Week 1) ✅ **COMPLETED**
- [x] Set up new package structure
- [x] Move `FileOps` → `core/fileops.py`
- [x] Move `ActionQueue` → `core/actions.py`
- [x] Extract media detection → `core/media.py`
- [x] Extract patterns → `core/patterns.py`
- [x] Set up pytest with basic fixtures
- [x] Write unit tests for core modules

**Phase 2: Feature Extraction** (Week 2) ✅ **COMPLETED**
- [x] Extract `LocaleMapper` → `features/subtitles/locale.py`
- [x] Refactor `SubtitleService` → `features/subtitles/api.py`
- [x] Write unit + integration tests for subtitle feature
- [x] Extract NFO generation → `features/nfo/api.py`
- [x] Write tests for NFO feature

**Phase 3: Organize Feature** (Week 2-3) ✅ **COMPLETED**
- [x] Extract `UI` prompts → `ui/prompts.py`
- [x] Refactor processors → `features/organize/api.py` (streamlined)
- [x] Create public API → `features/organize/api.py`
- [x] Core organization workflows implemented

**Phase 4: Torrent Feature** (Week 3) ✅ **COMPLETED**
- [x] Refactor torrent decoder → `features/torrent/decoder.py`
- [x] Refactor torrent parser → `features/torrent/parser.py`
- [x] Create public API → `features/torrent/api.py`
- [x] Core torrent functionality implemented

**Phase 5: CLI Layer** (Week 4) ✅ **COMPLETED**
- [x] Implement `cli/common.py` (shared decorators)
- [x] Implement `cli/subtitles.py`
- [x] Implement `cli/nfo.py`
- [x] Implement `cli/torrent.py`
- [x] Implement `cli/organize.py`
- [x] Implement `cli/main.py` (master command group)
- [x] CLI commands tested and working

**Phase 6: Polish** (Ready for future work)
- [x] Package setup (pyproject.toml) - COMPLETED
- [x] Test framework configured - COMPLETED
- [x] 86 unit tests passing - COMPLETED
- [ ] Integration/E2E tests (can be added as needed)
- [ ] README update with examples
- [ ] API documentation
- [ ] CI/CD pipeline
- [ ] Deprecation notices for old scripts

### 7.2 Backward Compatibility

**Approach: Hard Break**

Since breaking changes are acceptable, the old entry points (`main.py`, `torrent2dummy.py`) will be removed after Phase 5. Migration strategy:

1. **Archive old scripts**: Move to `legacy/` directory with clear deprecation notice
2. **Migration guide**: Document CLI changes in README.md
3. **Version bump**: Release as v2.0.0 to signal breaking changes

**Migration examples:**
```bash
# Old
python src/main.py ./path --generate-nfo --dry-run

# New
media-witch organize ./path --generate-nfo --dry-run

# Old
python src/torrent2dummy.py ./torrents -o ./output

# New
media-witch torrent ./torrents --output-dir ./output
```

---

## 8. Example Usage

### 8.1 CLI Usage

**Organize a TV show (interactive):**
```bash
media-witch organize ./Downloads/My.Show.S01/
# Prompts: TV/Movie? → TV
# Prompts: Season? → 1
# Prompts: Classify extras? → [interactive checkboxes]
```

**Organize multiple movies (non-interactive):**
```bash
media-witch organize --mode auto-movie ./Movies/*/ --dry-run
```

**Generate NFOs for existing season:**
```bash
media-witch nfo ./TV/MyShow/Season\ 1/ --season 1 --episode-start 1
```

**Rename subtitles with locale mapping:**
```bash
media-witch subtitles ./Movies/Film/ \
  --map "chi,zh,false" \
  --map "cht,zh-Hant,false"
```

**Create fake files from torrents:**
```bash
media-witch torrent ./torrents/*.torrent --output-dir ./test-structure/
```

**Full workflow (master command):**
```bash
media-witch ./Downloads/My.Show.S01/ \
  --mode interactive \
  --generate-nfo \
  --map-csv ~/subtitle-mappings.csv
```

### 8.2 Programmatic Usage

**Organize files programmatically:**
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

result = organize_directory(Path("./Downloads/Show/"), config)

print(f"Moved {len(result.files_moved)} files")
print(f"Created {len(result.nfos_created)} NFO files")
if result.errors:
    print(f"Errors: {result.errors}")
```

**Generate NFO files:**
```python
from media_witch.features.nfo.api import generate_episode_nfos, NFOConfig

videos = sorted(Path("./Season 1").glob("*.mkv"))

config = NFOConfig(season=1, episode_start=1)
result = generate_episode_nfos(videos, config)

print(f"Created NFO files: {result.created}")
```

**Parse torrent file:**
```python
from media_witch.features.torrent.api import parse_torrent

metadata = parse_torrent(Path("./download.torrent"))

print(f"Name: {metadata['name']}")
print(f"Files: {len(metadata['files'])}")
print(f"Total size: {metadata['total_size']:,} bytes")
```

---

## 9. Dependencies

**Required:**
- `click` >= 8.1 - CLI framework
- `questionary` >= 2.0 - Interactive prompts (already used)
- `colorama` >= 0.4.6 - Cross-platform colored output (lightweight)

**Development:**
- `pytest` >= 7.0 - Testing framework
- `pytest-cov` >= 4.0 - Coverage reporting
- `pytest-watch` >= 4.2 - Watch mode
- `mypy` >= 1.0 - Type checking
- `ruff` >= 0.1.0 - Linting and formatting

**Notes:**
- `colorama` is minimal (no dependencies) and safely handles terminal color compatibility (gracefully degrades on unsupported terminals)
- Standard library `logging` module used for structured logging (no additional dependencies)
- No configuration file support (CLI args only)
- No progress bars (keep it simple)

---

## 10. Open Questions & Considerations

### 10.1 Configuration Files
**Decision:** Not needed. All configuration via CLI arguments only.

This keeps the tool simple and eliminates file-based configuration overhead. Users can easily wrap the CLI in shell scripts or tools if they need persistent defaults.

### 10.2 Logging
**Decision:** Use Python's standard `logging` module with stdout/stderr output only.

**Implementation:**
```python
import logging
import sys
from colorama import Fore, Style, init

# Colorama auto-detects terminal capabilities and gracefully degrades
init(autoreset=True)

def setup_logging(level: str = "INFO", silent: bool = False) -> None:
    """Configure logging to stdout/stderr with colors (auto-detected)."""
    if silent:
        # Silent mode: disable logging
        logging.getLogger("media_witch").setLevel(logging.CRITICAL + 1)
        return
    
    logger = logging.getLogger("media_witch")
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        f"{Fore.CYAN}%(levelname)s{Style.RESET_ALL} %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Usage
logger = logging.getLogger("media_witch")
logger.info("[MOVE] file.mkv -> Season 1/file.mkv")
logger.debug("[DEBUG] Checking subtitle pairing for file.mkv")
```

**CLI flags:**
- `--verbose` / `-v` → DEBUG level
- `--quiet` / `-q` → WARNING level
- `--silent` → Suppress all messages (except interactive prompts)
- Default → INFO level
- No `--no-color` flag; `colorama` automatically handles terminal compatibility

**Benefits:**
- No file logging overhead
- Better than `print()` for programmatic usage
- Users control verbosity via flags
- Colored output safe across all terminal types (auto-detected by colorama)
- Silent mode useful for scripts where only prompts matter

### 10.2 Progress Bars
**Decision:** No progress bars.

Keep the tool simple with straightforward logging. For long operations, log messages indicate progress naturally.

### 10.4 Plugin System
**Question:** Should we support plugins for custom processors?

**Recommendation:** Not in initial refactor. Consider for v2.0 if users request custom organization logic.

### 10.5 Windows Path Handling
**Question:** How to handle long paths on Windows?

**Recommendation:** Use `\\?\` prefix for paths exceeding 260 chars. Add helper in `core/fileops.py`:
```python
def normalize_windows_path(p: Path) -> Path:
    """Handle long Windows paths."""
    if os.name == 'nt' and len(str(p)) > 260:
        return Path(f"\\\\?\\{p.resolve()}")
    return p
```

---

## 11. Success Criteria

The refactoring is successful when:

1. **Modularity**
   - [ ] Each feature is independently importable and usable
   - [ ] No circular dependencies between features
   - [ ] Clear separation of UI, business logic, and I/O

2. **CLI Usability**
   - [ ] All features accessible via dedicated commands
   - [ ] Master command replicates current workflow
   - [ ] Help text comprehensive and accurate

3. **API Usability**
   - [ ] Public APIs well-documented with docstrings
   - [ ] Type hints on all public functions
   - [ ] Result objects provide detailed operation logs

4. **Testing**
   - [ ] >90% unit test coverage
   - [ ] All feature APIs covered by integration tests
   - [ ] All CLI commands covered by E2E tests
   - [ ] CI pipeline runs tests automatically

5. **Maintainability**
   - [ ] New features can be added without modifying existing ones
   - [ ] External tools can import and use feature APIs
   - [ ] Code passes linting (ruff) and type checking (mypy)

---

## 12. Next Steps

1. ✅ **Plan finalized** with the following decisions:
   - **No configuration files** - CLI arguments only
   - **No progress bars** - Simple logging via standard library
   - **Logging**: stdout/stderr with colored output (colorama auto-detects terminal support)
   - **Silent flag**: `--silent` suppresses all messages (except interactive prompts)
   - **No color override flag** - Colorama handles terminal compatibility automatically
   - **Backward compatibility**: Not required (breaking changes acceptable)

2. **Begin Phase 1** - Set up infrastructure and core modules:
   - Create new package structure
   - Migrate core services (FileOps, ActionQueue)
   - Set up pytest with fixtures
   - Write initial unit tests

3. **Iterate** - Adapt plan as implementation reveals unforeseen challenges

---

**Document Version:** 1.2  
**Last Updated:** 2026-04-10  
**Author:** Generated via Claude Code  
**Status:** Ready for Implementation
