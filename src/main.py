#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import inquirer

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".ts", ".m2ts", ".wmv"}
AUDIO_EXTS = {".mka", ".aac", ".flac", ".dts",
              ".ac3", ".eac3", ".mp3", ".ogg", ".opus"}
SUB_EXTS = {".ass", ".ssa", ".sup", ".srt"}

EPISODE_PATTERNS = [
    re.compile(r"(?i)\bS(\d{1,2})E(\d{1,3})\b"),
    re.compile(r"\[(\d{1,3})\]")
]


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS


def is_subtitle(p: Path) -> bool:
    return p.suffix.lower() in SUB_EXTS


def has_episode_pattern(name: str) -> bool:
    return any(p.search(name) for p in EPISODE_PATTERNS)


def natural_sort_key(p: Path) -> Tuple:
    return tuple(int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name))


def log(msg: str) -> None:
    print(msg)


def list_files_and_dirs(path: Path) -> Tuple[List[Path], List[Path]]:
    files, dirs = [], []
    for e in path.iterdir():
        if e.is_file():
            files.append(e)
        elif e.is_dir():
            dirs.append(e)
    return files, dirs


@dataclass
class Rule:
    source: str
    target: str
    case_sensitive: bool


class LocaleMapper:
    def __init__(self, csv_rules: List[Rule], cli_rules: List[Rule]) -> None:
        self.csv_rules = csv_rules
        self.cli_rules = cli_rules

    @staticmethod
    def _match(rule: Rule, token: str) -> bool:
        return token == rule.source if rule.case_sensitive else token.lower() == rule.source.lower()

    def resolve(self, token: str) -> str:
        for table in (self.cli_rules, self.csv_rules):
            for rule in table:
                if self._match(rule, token):
                    return rule.target
        return token


def parse_cli_rules(cli_rules: Sequence[str]) -> List[Rule]:
    out = []
    for spec in cli_rules:
        parts = [x.strip() for x in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "Mapping rule must be: source,target,case_sensitive")
        out.append(Rule(parts[0], parts[1],
                   parts[2].lower() in ("1", "true", "yes")))
    return out


def load_csv_rules(csv_path: Optional[Path]) -> List[Rule]:
    if not csv_path:
        return []
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rules = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field = {k.lower(): k for k in reader.fieldnames}
        for row in reader:
            src = row[field["source"]].strip()
            tgt = row[field["target"]].strip()
            cs = row[field["is_case_sensitive"]
                     ].strip().lower() in ("1", "true", "yes")
            rules.append(Rule(src, tgt, cs))
    return rules


class FileOps:
    def __init__(self, dry: bool) -> None:
        self.dry = dry

    def ensure_dir(self, p: Path) -> None:
        if not p.exists():
            log(f"[MKDIR] {p}")
            if not self.dry:
                p.mkdir(parents=True, exist_ok=True)

    def move_file(self, src: Path, dst: Path) -> None:
        if src.resolve() == dst.resolve():
            log(f"[SKIP] Same location: {src}")
            return
        self.ensure_dir(dst.parent)
        if dst.exists():
            log(f"[SKIP] Exists: {dst}")
            return
        log(f"[MOVE] {src} -> {dst}")
        if not self.dry:
            shutil.move(str(src), str(dst))

    def move_dir_atomic(self, src: Path, dst: Path) -> None:
        if src.resolve() == dst.resolve():
            log(f"[SKIP] Same dir: {src}")
            return
        if not dst.exists():
            self.ensure_dir(dst.parent)
            log(f"[MOVE-DIR] {src} -> {dst}")
            if not self.dry:
                shutil.move(str(src), str(dst))
            return
        log(f"[MERGE-DIR] {src} -> {dst}")
        self.move_tree_merge(src, dst)

    def move_tree_merge(self, src_dir: Path, dst_dir: Path) -> None:
        self.ensure_dir(dst_dir)
        for root, dirs, files in os.walk(src_dir):
            rel = Path(root).relative_to(src_dir)
            for d in dirs:
                self.ensure_dir(dst_dir / rel / d)
            for f in files:
                self.move_file(Path(root) / f, dst_dir / rel / f)
        self.remove_dir_if_empty(src_dir)

    def remove_dir_if_empty(self, p: Path) -> None:
        try:
            next(p.iterdir())
        except StopIteration:
            log(f"[RMDIR] {p}")
            if not self.dry:
                p.rmdir()
        except PermissionError:
            return


class UI:
    @staticmethod
    def ask_processing_choice(path: Path, has_files: bool) -> str:
        """
        Returns one of:
          - if has_files: 'show' | 'movie' | 'skip'
          - if no files:  'skip' | 'shows' | 'seasons' | 'movies'
        """
        if has_files:
            ans = inquirer.prompt([
                inquirer.List(
                    "c",
                    message=f"Folder contains files:\n{path}\nSelect how to process:",
                    choices=[
                        ("TV Show", "show"),
                        ("Movie", "movie"),
                        ("Skip this folder", "skip"),
                    ],
                    default="skip",
                )
            ])
            return ans["c"]
        else:
            ans = inquirer.prompt([
                inquirer.List(
                    "c",
                    message=f"No files in folder:\n{path}\nWhat does this folder represent?",
                    choices=[
                        ("Skip this folder", "skip"),
                        ("Contains multiple SHOWS", "shows"),
                        ("Contains multiple SEASONS of the same show", "seasons"),
                        ("Contains multiple MOVIE SEQUELS", "movies"),
                    ],
                    default="skip",
                )
            ])
            return ans["c"]

    @staticmethod
    def ask_season() -> int:
        while True:
            a = inquirer.prompt(
                [inquirer.Text("s", message="Season number?", default="1")])["s"]
            if a.isdigit() and int(a) > 0:
                return int(a)
            print("Enter positive integer.")

    @staticmethod
    def checkbox_extras(items: List[Path], defaults: List[bool]) -> List[bool]:
        choices = [(f"{p.name} [{'EXTRA' if d else 'PRIMARY'}]", i)
                   for i, (p, d) in enumerate(zip(items, defaults))]
        checked = set(inquirer.prompt([
            inquirer.Checkbox(
                "sel",
                message="Select EXTRAS",
                choices=choices,
                default=[i for i, v in enumerate(defaults) if v],
                carousel=True,
            )
        ]).get("sel", []))
        return [(i in checked) for i in range(len(items))]

    @staticmethod
    def ask_nfo_overrides(videos: List[Path], season: int) -> Dict[int, int]:
        defaults = {p: i+1 for i, p in enumerate(videos)}
        print("\nDefault episode numbers:")
        for p in videos:
            print(f"{p.name} -> {defaults[p]}")
        if not inquirer.prompt([inquirer.Confirm("o", message="Override any?", default=False)])["o"]:
            return {}
        out: Dict[int, int] = {}
        for idx, p in enumerate(videos, start=1):
            if inquirer.prompt([inquirer.Confirm("c", message=f"Override {p.name}?", default=False)])["c"]:
                while True:
                    val = inquirer.prompt([inquirer.Text("n", message=f"Episode number for {p.name}",
                                                         default=str(defaults[p]))])["n"]
                    if val.isdigit() and int(val) > 0:
                        out[idx] = int(val)
                        break
                    print("Enter positive integer.")
        return out


class SubtitleService:
    def __init__(self, mapper: LocaleMapper, fops: FileOps) -> None:
        self.mapper = mapper
        self.fops = fops

    @staticmethod
    def _rightmost_token(sub: Path) -> str:
        parts = sub.stem.split(".")
        return parts[-1] if len(parts) > 1 else ""

    @staticmethod
    def _stem_wo_token(sub: Path) -> str:
        parts = sub.stem.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else sub.stem

    @staticmethod
    def pairs_with(sub: Path, video: Path) -> bool:
        return sub.stem == video.stem or sub.stem.startswith(video.stem + ".")

    def normalized_target(self, sub: Path, video: Path) -> Path:
        tok = self._rightmost_token(sub)
        mapped = self.mapper.resolve(tok) if tok else tok
        stem = f"{self._stem_wo_token(sub)}.{mapped}" if tok else video.stem
        return video.with_name(stem + sub.suffix.lower())

    def process(self, subs: Iterable[Path], video: Path) -> None:
        for sub in subs:
            if not self.pairs_with(sub, video):
                continue
            dst = video.parent / self.normalized_target(sub, video).name
            if sub.parent != video.parent:
                tmp = video.parent / sub.name
                self.fops.move_file(sub, tmp)
                if tmp != dst:
                    self.fops.rename_file(tmp, dst)
            else:
                self.fops.rename_file(sub, dst)


class BaseProcessor:
    def __init__(self, root: Path, mapper: LocaleMapper, fops: FileOps) -> None:
        self.root = root
        self.mapper = mapper
        self.fops = fops
        self.subsvc = SubtitleService(mapper, fops)

    def classify_extras(self, items: List[Path]) -> List[bool]:
        defaults = [p.is_dir() or not is_video(p) and not is_audio(p) and not is_subtitle(p)
                    if p.is_dir() else not has_episode_pattern(p.name)
                    for p in items]
        return UI.checkbox_extras(items, defaults)


class ShowProcessor(BaseProcessor):
    def process_unit(self, unit: Path, season: int) -> None:
        files, dirs = list_files_and_dirs(unit)
        items = dirs + files
        flags = self.classify_extras(items)

        season_dir = self.root / f"Season {season}"
        extra_dir = self.root / "EXTRA" / f"Season {season}"

        moved_videos = []

        for item, ex in zip(items, flags):
            if item.is_dir():
                if ex:
                    self.fops.move_dir_atomic(item, extra_dir / item.name)
                else:
                    self.fops.move_dir_contents_to(item, season_dir)
                continue

            if ex:
                self.fops.move_file(item, extra_dir / item.name)
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = season_dir / item.name
                self.fops.move_file(item, dst)
                if is_video(dst):
                    moved_videos.append(dst)
                continue

            self.fops.move_file(item, season_dir / item.name)

        for v in sorted(moved_videos, key=natural_sort_key):
            subs = [p for p in (unit.iterdir() if unit.exists() else [
            ]) if p.is_file() and is_subtitle(p)]
            subs += [p for p in (season_dir.iterdir() if season_dir.exists()
                                 else []) if p.is_file() and is_subtitle(p)]
            self.subsvc.process(subs, v)

    def process(self) -> None:
        leafs = self.find_leafs(self.root)
        for unit in leafs:
            log("\n" + "-"*50)
            log(f"[UNIT] {unit}")
            season = UI.ask_season()
            self.process_unit(unit, season)
            self.fops.remove_dir_if_empty(unit)

    def find_leafs(self, root: Path) -> List[Path]:
        files, dirs = list_files_and_dirs(root)
        if files:
            return [root]
        out = []
        for d in dirs:
            out.extend(self.find_leafs(d))
        return out


class MovieProcessor(BaseProcessor):
    def process(self) -> None:
        files, dirs = list_files_and_dirs(self.root)
        items = dirs + files
        flags = self.classify_extras(items)

        extra_dir = self.root / "EXTRA"
        moved_videos = []

        for item, ex in zip(items, flags):
            if item.is_dir():
                if ex:
                    self.fops.move_dir_atomic(item, extra_dir / item.name)
                else:
                    self.fops.move_dir_contents_to(item, self.root)
                continue

            if ex:
                self.fops.move_file(item, extra_dir / item.name)
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = self.root / item.name
                self.fops.move_file(item, dst)
                if is_video(dst):
                    moved_videos.append(dst)
                continue

            self.fops.move_file(item, self.root / item.name)

        subs = [p for p in self.root.iterdir() if p.is_file()
                and is_subtitle(p)]
        for v in sorted(moved_videos, key=natural_sort_key):
            self.subsvc.process(subs, v)


def PROCESS(root: Path, mapper: LocaleMapper, fops: FileOps, gen_nfo: bool) -> None:
    files, dirs = list_files_and_dirs(root)
    has_files = len(files) > 0

    choice = UI.ask_processing_choice(root, has_files)

    if choice == "skip":
        log("[SKIP]")
        return

    if has_files:
        if choice == "show":
            ShowProcessor(root, mapper, fops).process()
            return
        if choice == "movie":
            MovieProcessor(root, mapper, fops).process()
            return
        # 'skip' handled above; no other options when has_files=True
        return

    # has_files == False
    if choice == "shows":
        for d in dirs:
            PROCESS(root / d.name, mapper, fops, gen_nfo)
        return

    if choice == "seasons":
        sp = ShowProcessor(root, mapper, fops)
        for d in dirs:
            unit = root / d.name
            log(f"\n--- Processing season folder: {unit} ---")
            season = UI.ask_season()
            sp.process_unit(unit, season)
        return

    if choice == "movies":
        for d in dirs:
            unit = root / d.name
            log(f"\n--- Processing sequel movie folder: {unit} ---")
            MovieProcessor(unit, mapper, fops).process()
        return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Interactive media organizer")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--map-csv", type=Path)
    parser.add_argument("--map", action="append", default=[])
    parser.add_argument("--generate-nfo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    mapper = LocaleMapper(load_csv_rules(args.map_csv),
                          parse_cli_rules(args.map))

    for root in args.roots:
        r = root.resolve()
        if not r.exists() or not r.is_dir():
            print(f"[ERROR] Not a directory: {r}")
            continue

        fops = FileOps(args.dry_run)
        PROCESS(r, mapper, fops, args.generate_nfo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
