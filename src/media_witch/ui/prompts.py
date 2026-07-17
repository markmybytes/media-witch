"""Interactive prompts using questionary."""

from __future__ import annotations

from pathlib import Path

import questionary


def ask_processing_choice(path: Path, has_files: bool) -> str:
    """Ask user how to process directory.

    Args:
        path: Directory being processed
        has_files: Whether directory contains files

    Returns:
        Processing choice: 'show', 'movie', 'skip', 'shows', 'seasons', 'movies'
    """
    print()
    print('=' * 60)
    print('📂 PROCESSING FOLDER')
    print(f'   Current: {path.name}')
    print('=' * 60)

    # Import display here to avoid circular import
    from .display import print_tree

    print(print_tree(path) or 'Failed to scan the directory.')
    print()

    if has_files:
        return (
            questionary.select(
                'Select how to process this folder:',
                choices=[
                    questionary.Choice('TV Show', 'show'),
                    questionary.Choice('Movie', 'movie'),
                    questionary.Choice('Skip this folder', 'skip'),
                ],
                default='skip',
            ).unsafe_ask()
            or 'skip'
        )
    else:
        print('No files found. This folder likely contains subdirectories.\n')
        return (
            questionary.select(
                'What does this folder represent?',
                choices=[
                    questionary.Choice('Skip this folder', 'skip'),
                    questionary.Choice('Contains multiple SHOWS', 'shows'),
                    questionary.Choice('Contains multiple SEASONS of the same show', 'seasons'),
                    questionary.Choice('Contains multiple MOVIE SEQUELS', 'movies'),
                ],
                default='skip',
            ).unsafe_ask()
            or 'skip'
        )


def ask_season(default: int = 1) -> int:
    """Prompt for season number.

    Args:
        default: Default season number

    Returns:
        Season number
    """
    while True:
        ans = questionary.text('Season number?', default=str(default)).unsafe_ask()
        if ans is None:
            return default
        ans = ans.strip()
        if ans.isdigit() and int(ans) > -1:
            return int(ans)
        print('Enter a non-negative integer.')


def ask_extras_classification(items: list[Path], defaults: list[bool]) -> list[bool]:
    """Prompt user to classify extras.

    Args:
        items: Files/directories to classify
        defaults: Default classification (True = extra, False = primary)

    Returns:
        Boolean list where True = extra, False = primary
    """
    choices = [
        questionary.Choice(title=f'{p.name} [{"EXTRA" if d else "PRIMARY"}]', value=i, checked=d)
        for i, (p, d) in enumerate(zip(items, defaults, strict=False))
    ]
    selected = set(questionary.checkbox('Select EXTRAS', choices=choices).unsafe_ask() or [])
    return [i in selected for i in range(len(items))]


def ask_nfo_overrides(videos: list[Path], season: int) -> dict[int, int]:
    """Prompt for episode number overrides.

    Args:
        videos: Sorted list of video files
        season: Season number

    Returns:
        Mapping of video index (1-based) to episode number
    """
    defaults = {p: i + 1 for i, p in enumerate(videos)}
    print('\n' + '─' * 60)
    print('📺 DEFAULT EPISODE NUMBERING:')
    print('─' * 60)
    for i, p in enumerate(videos, start=1):
        print(f'  {i:2d}. {p.name:40s} → Episode {defaults[p]}')
    print()

    if not questionary.confirm('Override any episode numbers?', default=False).unsafe_ask():
        return {}

    print('\n' + '─' * 60)
    print('✏️  EPISODE NUMBER OVERRIDES:')
    print('─' * 60)
    out: dict[int, int] = {}
    for idx, p in enumerate(videos, start=1):
        while True:
            v = questionary.text(
                f'[{idx}/{len(videos)}] {p.name}', default=str(defaults[p])
            ).unsafe_ask()
            if v is None:
                v = str(defaults[p])
            v = v.strip()
            if v.isdigit() and int(v) > 0:
                if int(v) != defaults[p]:
                    out[idx] = int(v)
                break
            print('  ⚠️  Enter positive integer.')
    return out


def ask_yes_no(question: str, default: bool = False) -> bool:
    """Ask a yes/no question.

    Args:
        question: Question to ask
        default: Default answer (used if user cancels)

    Returns:
        True for yes, False for no
    """
    answer = questionary.confirm(question, default=default).unsafe_ask()
    return answer if answer is not None else default


def ask_remove_unmapped_subtitles(mapper: object) -> bool:
    """Ask if unmapped subtitles should be removed.

    Args:
        mapper: LocaleMapper instance

    Returns:
        True to remove unmapped subtitles, False to keep them
    """
    # Import here to avoid circular dependency
    from ..features.subtitles.locale import LocaleMapper

    if not isinstance(mapper, LocaleMapper):
        return False

    target_locales = mapper.get_target_locales()
    if not target_locales:
        return False

    locales_str = ', '.join(sorted(target_locales))
    question = f'Remove subtitles NOT in these locales: {locales_str}?'
    return questionary.confirm(question, default=False).unsafe_ask() or False
