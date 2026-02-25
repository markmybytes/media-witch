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
from typing import (Callable, Dict, Iterable, List, Optional, Sequence, Set,
                    Tuple)

import questionary

VIDEO_EXTS: Set[str] = {".mkv", ".mp4", ".avi", ".mov", ".ts", ".m2ts", ".wmv"}
AUDIO_EXTS: Set[str] = {".mka", ".aac", ".flac",
                        ".dts", ".ac3", ".eac3", ".mp3", ".ogg", ".opus"}
SUB_EXTS:   Set[str] = {".ass", ".ssa", ".sup", ".srt"}

EPISODE_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bS(\d{1,2})E(\d{1,3})\b"),
    re.compile(r"\[(\d{1,3})\]"),
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
    def _match(r: Rule, t: str) -> bool:
        return (t == r.source) if r.case_sensitive else (t.lower() == r.source.lower())

    def resolve(self, token: str) -> str:
        for src in (self.cli_rules, self.csv_rules):
            for r in src:
                if self._match(r, token):
                    return r.target
        return token


def parse_cli_rules(cli_rules: Sequence[str]) -> List[Rule]:
    out: List[Rule] = []
    for spec in cli_rules:
        parts = [x.strip() for x in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "Mapping rule must be: source,target,case_sensitive")
        out.append(Rule(parts[0], parts[1],
                   parts[2].lower() in ("1", "true", "yes", "y")))
    return out


def load_csv_rules(csv_path: Optional[Path]) -> List[Rule]:
    if not csv_path:
        return []
    if not csv_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {csv_path}")
    rules: List[Rule] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rules
        field = {k.lower(): k for k in reader.fieldnames}
        for row in reader:
            src = row[field.get("source", "source")].strip()
            if not src:
                continue
            tgt = row[field.get("target", "target")].strip()
            cs = row[field.get("is_case_sensitive",
                               "is_case_sensitive")].strip()
            rules.append(Rule(src, tgt, cs.lower()
                         in ("1", "true", "yes", "y")))
    return rules


class FileOps:
    def __init__(self, dry_run: bool) -> None:
        self.dry = dry_run
        self._ensured: Set[Path] = set()

    def _norm(self, p: Path) -> Path:
        try:
            return p.resolve()
        except Exception:
            return p.absolute()

    def ensure_dir(self, p: Path) -> None:
        n = self._norm(p)
        if n in self._ensured:
            return
        if n.exists():
            self._ensured.add(n)
            return
        log(f"[MKDIR] {n}")
        if not self.dry:
            n.mkdir(parents=True, exist_ok=True)
        self._ensured.add(n)

    def ensure_parent(self, p: Path) -> None:
        self.ensure_dir(self._norm(p).parent)

    def move_file(self, src: Path, dst: Path) -> None:
        s, d = self._norm(src), self._norm(dst)
        if s == d:
            log(f"[SKIP] Already at dest: {src}")
            return
        self.ensure_parent(d)
        if d.exists():
            log(f"[SKIP] Exists: {dst}")
            return
        log(f"[MOVE] {src} -> {dst}")
        if not self.dry:
            shutil.move(str(src), str(dst))

    def rename_file(self, src: Path, dst: Path) -> None:
        s, d = self._norm(src), self._norm(dst)
        if s == d:
            return
        self.ensure_parent(d)
        if d.exists():
            log(f"[SKIP] Exists: {dst}")
            return
        log(f"[RENAME] {src.name} -> {dst.name}")
        if not self.dry:
            src.rename(dst)

    def write_text_if_absent(self, path: Path, content: str, label: str = "[WRITE]") -> None:
        self.ensure_parent(path)
        if path.exists():
            log(f"[SKIP] Exists: {path}")
            return
        log(f"{label} {path}")
        if not self.dry:
            path.write_text(content, encoding="utf-8")

    def remove_dir_if_empty(self, dir_path: Path) -> None:
        try:
            next(dir_path.iterdir())
        except StopIteration:
            n = self._norm(dir_path)
            log(f"[RMDIR] {n}")
            if not self.dry:
                n.rmdir()
            if n in self._ensured:
                self._ensured.remove(n)
        except PermissionError:
            return

    def move_tree_merge(self, src_dir: Path, dst_dir: Path) -> None:
        self.ensure_dir(dst_dir)
        for root, dirs, files in os.walk(src_dir):
            rel = Path(root).relative_to(src_dir)
            for d in dirs:
                self.ensure_dir(dst_dir / rel / d)
            for f in files:
                self.move_file(Path(root) / f, dst_dir / rel / f)
        self.remove_dir_if_empty(src_dir)

    def move_dir_contents_to(self, src_dir: Path, dst_dir: Path) -> None:
        self.ensure_dir(dst_dir)
        for e in src_dir.iterdir():
            t = dst_dir / e.name
            if e.is_dir():
                self.move_tree_merge(e, t)
            else:
                self.move_file(e, t)
        self.remove_dir_if_empty(src_dir)

    def move_dir_atomic(self, src_dir: Path, dst_dir: Path) -> None:
        s, d = self._norm(src_dir), self._norm(dst_dir)
        if s == d:
            log(f"[SKIP] Already at dest: {src_dir}")
            return
        if not d.exists():
            self.ensure_parent(d)
            log(f"[MOVE-DIR] {src_dir} -> {dst_dir}")
            if not self.dry:
                shutil.move(str(src_dir), str(dst_dir))
            self._ensured.add(d)
            return
        log(f"[MERGE-DIR] {src_dir} -> {dst_dir}")
        self.move_tree_merge(src_dir, dst_dir)


class ActionQueue:
    def __init__(self, fops: FileOps) -> None:
        self.fops = fops
        self._q: List[Tuple[Callable, Tuple, Dict, str]] = []

    def add(self, func: Callable, *args, desc: str = "", **kwargs) -> None:
        self._q.append((func, args, kwargs, desc))

    def commit(self) -> None:
        if self.fops.dry:
            if not self._q:
                log("[DRY-RUN] No actions.")
                return
            log("[DRY-RUN] Planned actions:")
            for _, _, _, desc in self._q:
                log(desc if desc else "[DRY-RUN] action")
            return
        for func, args, kwargs, _ in self._q:
            func(*args, **kwargs)

    def clear(self) -> None:
        self._q.clear()

    def __len__(self) -> int:
        return len(self._q)


class UI:
    @staticmethod
    def ask_processing_choice(path: Path, has_files: bool) -> str:
        if has_files:
            return questionary.select(
                f"Folder contains files:\n{path}\nSelect how to process:",
                choices=[
                    questionary.Choice("TV Show", "show"),
                    questionary.Choice("Movie", "movie"),
                    questionary.Choice("Skip this folder", "skip"),
                ],
                default="skip",
            ).ask() or "skip"
        else:
            return questionary.select(
                f"No files in folder:\n{path}\nWhat does this folder represent?",
                choices=[
                    questionary.Choice("Skip this folder", "skip"),
                    questionary.Choice("Contains multiple SHOWS", "shows"),
                    questionary.Choice(
                        "Contains multiple SEASONS of the same show", "seasons"),
                    questionary.Choice(
                        "Contains multiple MOVIE SEQUELS", "movies"),
                ],
                default="skip",
            ).ask() or "skip"

    @staticmethod
    def ask_season(default: int = 1) -> int:
        while True:
            ans = questionary.text(
                "Season number?", default=str(default)).ask()
            if ans is None:
                return default
            ans = ans.strip()
            if ans.isdigit() and int(ans) > 0:
                return int(ans)
            print("Enter positive integer.")

    @staticmethod
    def checkbox_extras(items: List[Path], defaults: List[bool]) -> List[bool]:
        choices = [
            questionary.Choice(
                title=f"{p.name} [{'EXTRA' if d else 'PRIMARY'}]", value=i, checked=d)
            for i, (p, d) in enumerate(zip(items, defaults))
        ]
        selected = questionary.checkbox(
            "Select EXTRAS", choices=choices).ask() or []
        selected_set = set(selected)
        return [(i in selected_set) for i in range(len(items))]

    @staticmethod
    def ask_nfo_overrides(videos: List[Path], season: int) -> Dict[int, int]:
        defaults = {p: i + 1 for i, p in enumerate(videos)}
        print("\nDefault episode numbers:")
        for p in videos:
            print(f"  {p.name} -> {defaults[p]}")
        if not (questionary.confirm("Override any?", default=False).ask() or False):
            return {}
        out: Dict[int, int] = {}
        for idx, p in enumerate(videos, start=1):
            if questionary.confirm(f"Override {p.name}? (default {defaults[p]})", default=False).ask():
                while True:
                    v = questionary.text(
                        f"Episode number for {p.name}", default=str(defaults[p])).ask()
                    if v and v.isdigit() and int(v) > 0:
                        out[idx] = int(v)
                        break
                    print("Enter positive integer.")
        return out


class SubtitleService:
    def __init__(self, mapper: LocaleMapper, fops: FileOps) -> None:
        self.mapper = mapper
        self.fops = fops

    @staticmethod
    def _right_most_token(sub: Path) -> str:
        parts = sub.stem.split(".")
        return parts[-1] if len(parts) > 1 else ""

    @staticmethod
    def _stem_wo_token(sub: Path) -> str:
        parts = sub.stem.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else sub.stem

    @staticmethod
    def pairs_with(sub: Path, video: Path) -> bool:
        return sub.stem == video.stem or sub.stem.startswith(f"{video.stem}.")

    def normalized_target(self, sub: Path, video: Path) -> Path:
        t = self._right_most_token(sub)
        mapped = self.mapper.resolve(t) if t else t
        stem = f"{self._stem_wo_token(sub)}.{mapped}" if t else video.stem
        return video.with_name(f"{stem}{sub.suffix.lower()}")

    def plan(self, subs: Iterable[Path], video_dst: Path, aq: ActionQueue) -> None:
        for sub in subs:
            if not self.pairs_with(sub, video_dst):
                continue
            dst = video_dst.parent / \
                self.normalized_target(sub, video_dst).name
            if sub.parent != video_dst.parent:
                tmp = video_dst.parent / sub.name
                aq.add(self.fops.move_file, sub, tmp,
                       desc=f"[MOVE] {sub} -> {tmp}")
                if tmp != dst:
                    aq.add(self.fops.rename_file, tmp, dst,
                           desc=f"[RENAME] {tmp.name} -> {dst.name}")
            else:
                if sub != dst:
                    aq.add(self.fops.rename_file, sub, dst,
                           desc=f"[RENAME] {sub.name} -> {dst.name}")


class BaseProcessor:
    def __init__(self, root: Path, mapper: LocaleMapper, fops: FileOps) -> None:
        self.root = root
        self.mapper = mapper
        self.fops = fops
        self.subsvc = SubtitleService(mapper, fops)

    def classify_extras(self, items: List[Path]) -> List[bool]:
        defaults: List[bool] = []
        for p in items:
            if p.is_dir():
                defaults.append(True)
            else:
                defaults.append(not has_episode_pattern(p.name))
        return UI.checkbox_extras(items, defaults)


class ShowProcessor(BaseProcessor):
    def __init__(self, root: Path, mapper: LocaleMapper, fops: FileOps, generate_nfo: bool) -> None:
        super().__init__(root, mapper, fops)
        self.generate_nfo = generate_nfo

    def find_leafs(self, folder: Path) -> List[Path]:
        files, dirs = list_files_and_dirs(folder)
        if files:
            return [folder]
        out: List[Path] = []
        for d in dirs:
            out.extend(self.find_leafs(d))
        return out

    def process(self) -> None:
        for unit in self.find_leafs(self.root):
            log("\n" + "-" * 50)
            log(f"[UNIT] {unit}")
            try:
                season = UI.ask_season(1)
                self._stage_and_commit_unit(unit, season)
            except KeyboardInterrupt:
                log("[ABORT] No changes committed for this unit.")
                return

    def process_unit(self, unit: Path, season: int) -> None:
        try:
            self._stage_and_commit_unit(unit, season)
        except KeyboardInterrupt:
            log("[ABORT] No changes committed for this unit.")
            return

    def _stage_and_commit_unit(self, unit: Path, season: int) -> None:
        files, dirs = list_files_and_dirs(unit)
        items = dirs + files
        if not items:
            return

        aq = ActionQueue(self.fops)

        flags = self.classify_extras(items)

        season_dir = self.root / f"Season {season}"
        extra_dir = self.root / "EXTRA" / f"Season {season}"

        moved_video_dsts: List[Path] = []

        for item, is_ex in zip(items, flags):
            if item.is_dir():
                if is_ex:
                    aq.add(self.fops.move_dir_atomic, item, extra_dir / item.name,
                           desc=f"[MOVE-DIR] {item} -> {extra_dir / item.name}")
                else:
                    aq.add(self.fops.move_dir_contents_to, item, season_dir,
                           desc=f"[FLATTEN->Season] {item} -> {season_dir}")
                continue

            if is_ex:
                aq.add(self.fops.move_file, item, extra_dir / item.name,
                       desc=f"[MOVE] {item} -> {extra_dir / item.name}")
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = season_dir / item.name
                aq.add(self.fops.move_file, item, dst,
                       desc=f"[MOVE] {item} -> {dst}")
                if is_video(dst):
                    moved_video_dsts.append(dst)
                continue

            aq.add(self.fops.move_file, item, season_dir / item.name,
                   desc=f"[MOVE] {item} -> {season_dir / item.name}")

        for vdst in sorted(moved_video_dsts, key=natural_sort_key):
            subs = []
            if unit.exists():
                subs += [p for p in unit.iterdir() if p.is_file()
                         and is_subtitle(p)]
            if season_dir.exists():
                subs += [p for p in season_dir.iterdir() if p.is_file()
                         and is_subtitle(p)]
            self.subsvc.plan(subs, vdst, aq)

        if self.generate_nfo and moved_video_dsts:
            videos_sorted = sorted(moved_video_dsts, key=natural_sort_key)
            overrides = UI.ask_nfo_overrides(videos_sorted, season)
            defaults = {p: i + 1 for i, p in enumerate(videos_sorted)}
            for idx, p in enumerate(videos_sorted, start=1):
                ep = overrides.get(idx, defaults[p])
                nfo_path = p.with_suffix(".nfo")
                content = (
                    '<?xml version="1.0" encoding="utf-8" standalone="yes"?>'
                    '<episodedetails>'
                    f'<episode>{ep}</episode><season>{season}</season>'
                    '</episodedetails>'
                )
                aq.add(self.fops.write_text_if_absent, nfo_path, content, label="[NFO]",
                       desc=f"[NFO] Create {nfo_path.name} (episode={ep}, season={season})")

        aq.commit()

        self.fops.remove_dir_if_empty(unit)


class MovieProcessor(BaseProcessor):
    def process(self) -> None:
        files, dirs = list_files_and_dirs(self.root)
        items = dirs + files
        if not items:
            return

        aq = ActionQueue(self.fops)

        flags = self.classify_extras(items)

        extra_dir = self.root / "EXTRA"
        moved_video_dsts: List[Path] = []

        for item, is_ex in zip(items, flags):
            if item.is_dir():
                if is_ex:
                    aq.add(self.fops.move_dir_atomic, item, extra_dir / item.name,
                           desc=f"[MOVE-DIR] {item} -> {extra_dir / item.name}")
                else:
                    aq.add(self.fops.move_dir_contents_to, item, self.root,
                           desc=f"[FLATTEN->Root] {item} -> {self.root}")
                continue

            if is_ex:
                aq.add(self.fops.move_file, item, extra_dir / item.name,
                       desc=f"[MOVE] {item} -> {extra_dir / item.name}")
                continue

            if is_subtitle(item):
                continue

            if is_video(item) or is_audio(item):
                dst = self.root / item.name
                aq.add(self.fops.move_file, item, dst,
                       desc=f"[MOVE] {item} -> {dst}")
                if is_video(dst):
                    moved_video_dsts.append(dst)
                continue

            aq.add(self.fops.move_file, item, self.root / item.name,
                   desc=f"[MOVE] {item} -> {self.root / item.name}")

        subs = [p for p in self.root.iterdir() if p.is_file()
                and is_subtitle(p)]
        for vdst in sorted(moved_video_dsts, key=natural_sort_key):
            self.subsvc.plan(subs, vdst, aq)

        aq.commit()


def process_directory(root: Path, mapper: LocaleMapper, fops: FileOps, gen_nfo: bool) -> None:
    files, dirs = list_files_and_dirs(root)
    has_files = len(files) > 0

    try:
        choice = UI.ask_processing_choice(root, has_files)
    except KeyboardInterrupt:
        log("[ABORT] No changes committed.")
        return

    if choice == "skip":
        log("[SKIP]")
        return

    if has_files:
        try:
            if choice == "show":
                ShowProcessor(root, mapper, fops, gen_nfo).process()
                return
            if choice == "movie":
                MovieProcessor(root, mapper, fops).process()
                return
        except KeyboardInterrupt:
            log("[ABORT] No changes committed.")
            return
        return

    if choice == "shows":
        for d in dirs:
            process_directory(root / d.name, mapper, fops, gen_nfo)
        return

    if choice == "seasons":
        sp = ShowProcessor(root, mapper, fops, gen_nfo)
        for d in dirs:
            unit = root / d.name
            log(f"\n--- Processing season folder: {unit} ---")
            try:
                season = UI.ask_season(1)
                sp.process_unit(unit, season)
            except KeyboardInterrupt:
                log("[ABORT] No changes committed for this season folder.")
                return
        return

    if choice == "movies":
        for d in dirs:
            unit = root / d.name
            log(f"\n--- Processing sequel movie folder: {unit} ---")
            try:
                MovieProcessor(unit, mapper, fops).process()
            except KeyboardInterrupt:
                log("[ABORT] No changes committed for this movie folder.")
                return
        return


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Interactive media organizer (questionary-based, staged commits)")
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--map-csv", type=Path,
                   help="CSV: source,target,is_case_sensitive")
    p.add_argument("--map", action="append", default=[],
                   help="Inline rule 'source,target,case_sensitive'")
    p.add_argument("--generate-nfo", action="store_true",
                   help="Generate .nfo per episode (TV mode only).")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    try:
        mapper = LocaleMapper(load_csv_rules(
            a.map_csv), parse_cli_rules(a.map))
    except Exception as e:
        print(f"[ERROR] Mapping rules: {e}")
        return 2

    for root in [r.resolve() for r in a.roots]:
        if not root.exists() or not root.is_dir():
            print(f"[ERROR] Not a directory: {root}")
            continue

        fops = FileOps(dry_run=a.dry_run)

        log("=" * 72)
        log(f"[START] Root={root} DryRun={a.dry_run}")
        if a.map_csv:
            log(f"[MAP-CSV] {a.map_csv}")
        for r in a.map:
            log(f"[MAP-CLI] {r}")

        try:
            process_directory(root, mapper, fops, a.generate_nfo)
        except KeyboardInterrupt:
            print("\n[ABORTED] by user. No changes committed.")
            return 130

        log("[DONE]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
