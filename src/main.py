import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import inquirer

VIDEO_EXTS: Set[str] = {".mkv", ".mp4", ".avi", ".mov", ".ts", ".m2ts", ".wmv"}
AUDIO_EXTS: Set[str] = {".mka", ".aac", ".flac",
                        ".dts", ".ac3", ".eac3", ".mp3", ".ogg", ".opus"}
SUB_EXTS: Set[str] = {".ass", ".ssa", ".sup", ".srt"}

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
                f"Invalid CLI rule: {spec} (expected 'source,target,case_sensitive')")
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
        r = csv.DictReader(f)
        field = {k.lower(): k for k in (r.fieldnames or [])}
        for row in r:
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

    def ensure_dir(self, p: Path) -> None:
        if not p.exists():
            print(f"[MKDIR] {p}")
            if not self.dry:
                p.mkdir(parents=True, exist_ok=True)

    def move_file(self, src: Path, dst: Path) -> None:
        if src.resolve() == dst.resolve():
            print(f"[SKIP] Already at dest: {src}")
            return
        self.ensure_dir(dst.parent)
        if dst.exists():
            print(f"[SKIP] Exists: {dst}")
            return
        print(f"[MOVE] {src} -> {dst}")
        if not self.dry:
            shutil.move(str(src), str(dst))

    def rename_file(self, src: Path, dst: Path) -> None:
        if src.resolve() == dst.resolve():
            return
        self.ensure_dir(dst.parent)
        if dst.exists():
            print(f"[SKIP] Exists: {dst}")
            return
        print(f"[RENAME] {src.name} -> {dst.name}")
        if not self.dry:
            src.rename(dst)

    def remove_dir_if_empty(self, dir_path: Path) -> None:
        try:
            next(dir_path.iterdir())
        except StopIteration:
            print(f"[RMDIR] {dir_path}")
            if not self.dry:
                dir_path.rmdir()
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
            self.move_tree_merge(e, t) if e.is_dir() else self.move_file(e, t)
        self.remove_dir_if_empty(src_dir)

    def move_dir_atomic(self, src_dir: Path, dst_dir: Path) -> None:
        if src_dir.resolve() == dst_dir.resolve():
            print(f"[SKIP] Already at dest: {src_dir}")
            return
        if not dst_dir.exists():
            self.ensure_dir(dst_dir.parent)
            print(f"[MOVE-DIR] {src_dir} -> {dst_dir}")
            if not self.dry:
                shutil.move(str(src_dir), str(dst_dir))
            return
        print(f"[MERGE-DIR] {src_dir} -> {dst_dir}")
        self.move_tree_merge(src_dir, dst_dir)


class UI:
    @staticmethod
    def ask_mode_for(folder: Path) -> str:
        a = inquirer.prompt([
            inquirer.List("mode", message=f"Select folder type for: {folder}",
                          choices=[("TV Show", "show"), ("Movie", "movie")], default="show")
        ])
        return a["mode"]

    @staticmethod
    def ask_yes_no(msg: str, default: bool = True) -> bool:
        return bool(inquirer.prompt([inquirer.Confirm("yn", message=msg, default=default)])["yn"])

    @staticmethod
    def ask_season_number(default: int = 1) -> int:
        while True:
            s = inquirer.prompt([inquirer.Text("s", message="Enter season number for this unit",
                                               default=str(default))])["s"].strip()
            if s.isdigit() and int(s) > 0:
                return int(s)
            print("Please enter a positive integer.")

    @staticmethod
    def checkbox_extras(items: List[Path], defaults_is_extra: List[bool]) -> List[bool]:
        choices = [(f"{p.name}  [{'EXTRA' if d else 'PRIMARY'} · {'DIR' if p.is_dir() else 'FILE'}]", i)
                   for i, (p, d) in enumerate(zip(items, defaults_is_extra))]

        checked = set(inquirer.prompt([
            inquirer.Checkbox("ex", message="Mark items as EXTRAS (checked) vs PRIMARY (unchecked)",
                              choices=choices, default=[
                                  i for i, ex in enumerate(defaults_is_extra) if ex],
                              carousel=True)
        ]).get("ex", []))

        res = [(i in checked) for i in range(len(items))]

        print("\nFinal classification:")
        for i, (p, ex) in enumerate(zip(items, res), 1):
            print(f"{i:2d}) [{'EXTRA' if ex else 'PRIMARY'}] {p.name}")

        return res

    @staticmethod
    def ask_nfo_overrides(videos_sorted: List[Path], season: int) -> Dict[int, int]:
        defaults = {p: i + 1 for i, p in enumerate(videos_sorted)}

        print("\nNFO Generation – default episode numbers:")
        for p in videos_sorted:
            print(f"  {p.name} -> episode {defaults[p]} (season {season})")

        if not UI.ask_yes_no("Override any episode numbers?", default=False):
            return {}

        ov: Dict[int, int] = {}
        for idx, p in enumerate(videos_sorted, start=1):
            if UI.ask_yes_no(f"Override for {p.name}? (default {defaults[p]})", default=False):
                while True:
                    n = inquirer.prompt([inquirer.Text("n", message=f"Enter episode number for {p.name}",
                                                       default=str(defaults[p]))])["n"].strip()
                    if n.isdigit() and int(n) > 0:
                        ov[idx] = int(n)
                        break
                    print("Please enter a positive integer.")
        return ov


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

    def pair_and_normalize(self, subs: Iterable[Path], video: Path) -> None:
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

    @staticmethod
    def _candidate_items(files: List[Path], dirs: List[Path]) -> List[Path]:
        return dirs + [f for f in files]

    @staticmethod
    def _default_is_extra_dir() -> bool:
        return True

    @staticmethod
    def _default_is_extra_file(name: str) -> bool:
        return not has_episode_pattern(name)

    def classify_extras(self, items: List[Path]) -> List[bool]:
        return UI.checkbox_extras(
            items,
            [self._default_is_extra_dir() if p.is_dir(
            ) else self._default_is_extra_file(p.name) for p in items],
        )


class ShowProcessor(BaseProcessor):
    def __init__(self, root: Path, mapper: LocaleMapper, fops: FileOps, generate_nfo: bool) -> None:
        super().__init__(root, mapper, fops)
        self.generate_nfo = generate_nfo

    def _find_leaf_units(self, folder: Path) -> List[Path]:
        files, dirs = list_files_and_dirs(folder)
        if files:
            return [folder]
        units: List[Path] = []
        for d in dirs:
            units.extend(self._find_leaf_units(d))
        return units

    def process(self) -> None:
        print(f"[SHOW] Processing: {self.root}")

        units = self._find_leaf_units(self.root)
        if not units:
            print("[WARN] No files found under this root.")
            return

        for unit in units:
            print("\n" + "-" * 60)
            print(f"[UNIT] {unit}")

            season = UI.ask_season_number(1)

            files, dirs = list_files_and_dirs(unit)
            items = self._candidate_items(files, dirs)
            if not items:
                print("[INFO] No relevant files in this unit; skipping.")
                continue

            flags = self.classify_extras(items)

            season_dir = self.root / f"Season {season}"
            extra_season_dir = self.root / "EXTRA" / f"Season {season}"

            moved_videos: List[Path] = []

            for item, is_ex in zip(items, flags):
                if item.is_dir():
                    if is_ex:
                        self.fops.move_dir_atomic(
                            item, extra_season_dir / item.name)
                    else:
                        self.fops.move_dir_contents_to(item, season_dir)
                    continue

                if is_ex:
                    self.fops.move_file(item, extra_season_dir / item.name)
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
                subs = []
                if unit.exists():
                    subs += [p for p in unit.iterdir() if p.is_file()
                             and is_subtitle(p)]
                if season_dir.exists():
                    subs += [p for p in season_dir.iterdir() if p.is_file()
                             and is_subtitle(p)]
                self.subsvc.pair_and_normalize(subs, v)

            if self.generate_nfo:
                self._generate_nfo_for(season_dir, season)

            self.fops.remove_dir_if_empty(unit)

    def _generate_nfo_for(self, season_dir: Path, season: int) -> None:
        if not season_dir.exists():
            print(f"[NFO] Season folder doesn't exist: {season_dir}")
            return

        videos = sorted([p for p in season_dir.iterdir()
                        if p.is_file() and is_video(p)], key=natural_sort_key)
        if not videos:
            print("[NFO] No videos found.")
            return

        overrides = UI.ask_nfo_overrides(videos, season)

        for idx, p in enumerate(videos, start=1):
            ep = overrides.get(idx, idx)
            nfo = p.with_suffix(".nfo")
            if nfo.exists():
                print(f"[SKIP] NFO exists: {nfo.name}")
                continue
            print(f"[NFO] Create {nfo.name} (episode={ep}, season={season})")
            if not self.fops.dry:
                nfo.write_text(
                    '<?xml version="1.0" encoding="utf-8" standalone="yes"?>'
                    '<episodedetails>'
                    f'<episode>{ep}</episode><season>{season}</season>'
                    '</episodedetails>',
                    encoding="utf-8",
                )


class MovieProcessor(BaseProcessor):
    def process(self) -> None:
        print(f"[MOVIE] Processing: {self.root}")

        files, dirs = list_files_and_dirs(self.root)
        items = self._candidate_items(
            files, [d for d in dirs if d.name != "EXTRA"])
        if not items:
            print("[WARN] No relevant files found in movie root.")
            return

        flags = self.classify_extras(items)

        extra_dir = self.root / "EXTRA"
        moved_videos: List[Path] = []

        for item, is_ex in zip(items, flags):
            if item.is_dir():
                if is_ex:
                    self.fops.move_dir_atomic(item, extra_dir / item.name)
                else:
                    self.fops.move_dir_contents_to(item, self.root)
                continue

            if is_ex:
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
            self.subsvc.pair_and_normalize(subs, v)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Interactive media organizer (per-folder mode).")
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--map-csv", type=Path,
                   help="CSV: source,target,is_case_sensitive")
    p.add_argument("--map", action="append", default=[],
                   help="Inline rule 'source,target,case_sensitive'")
    p.add_argument("--generate-nfo", action="store_true")
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
            print(f"[ERROR] Root not found: {root}")
            return 2

        fops = FileOps(dry_run=a.dry_run)

        print("=" * 72)
        print(f"[START] Root={root} DryRun={a.dry_run}")
        if a.map_csv:
            print(f"[MAP-CSV] {a.map_csv}")
        for r in a.map:
            print(f"[MAP-CLI] {r}")

        mode = UI.ask_mode_for(root)

        try:
            if mode == "show":
                ShowProcessor(root, mapper, fops, a.generate_nfo).process()
            else:
                MovieProcessor(root, mapper, fops).process()
        except KeyboardInterrupt:
            print("\n[ABORTED] by user.")
            return 130

        print("[DONE]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
