"""File operations with dry-run support."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Set


class FileOps:
    """Abstracted file operations with dry-run support.

    All file system operations are logged and can be previewed using dry-run mode.
    Tracks ensured directories to avoid redundant operations.
    """

    def __init__(
        self,
        dry_run: bool,
        *,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize FileOps.

        Args:
            dry_run: If True, log operations but don't execute them
            logger: Optional logging function (defaults to print)
        """
        self.dry = dry_run
        self._log = logger or print
        self._ensured: Set[Path] = set()

    def _norm(self, p: Path) -> Path:
        """Normalize a path to absolute form."""
        try:
            return p.resolve()
        except Exception:
            return p.absolute()

    def ensure_dir(self, p: Path) -> None:
        """Ensure directory exists, creating it if necessary.

        Args:
            p: Directory path to ensure
        """
        n = self._norm(p)
        if n in self._ensured:
            return
        if n.exists():
            self._ensured.add(n)
            return
        self._log(f"[MKDIR] {n}")
        if not self.dry:
            n.mkdir(parents=True, exist_ok=True)
        self._ensured.add(n)

    def ensure_parent(self, p: Path) -> None:
        """Ensure parent directory of given path exists.

        Args:
            p: Path whose parent directory should be ensured
        """
        self.ensure_dir(self._norm(p).parent)

    def move_file(self, src: Path, dst: Path) -> None:
        """Move a file from source to destination.

        Args:
            src: Source file path
            dst: Destination file path
        """
        s, d = self._norm(src), self._norm(dst)
        if s == d:
            self._log(f"[SKIP] Already at dest: {src}")
            return
        self.ensure_parent(d)
        if d.exists():
            self._log(f"[SKIP] Exists: {dst}")
            return
        self._log(f"[MOVE] {src} -> {dst}")
        if not self.dry:
            shutil.move(str(src), str(dst))

    def rename_file(self, src: Path, dst: Path) -> None:
        """Rename a file from source to destination.

        Args:
            src: Source file path
            dst: Destination file path
        """
        s, d = self._norm(src), self._norm(dst)
        if s == d:
            return
        self.ensure_parent(d)
        if d.exists():
            self._log(f"[SKIP] Exists: {dst}")
            return
        self._log(f"[RENAME] {src.name} -> {dst.name}")
        if not self.dry:
            src.rename(dst)

    def remove_file(self, path: Path, label: str = "[REMOVE]") -> None:
        """Remove a file.

        Args:
            path: File path to remove
            label: Log message label
        """
        p = self._norm(path)
        if not p.exists():
            self._log(f"[SKIP] Not found: {path}")
            return
        self._log(f"{label} {path}")
        if not self.dry:
            p.unlink()

    def write_text_if_absent(
        self,
        path: Path,
        content: str,
        label: str = "[WRITE]"
    ) -> None:
        """Write text content to file if it doesn't exist.

        Args:
            path: File path to write to
            content: Text content to write
            label: Log message label
        """
        self.ensure_parent(path)
        if path.exists():
            self._log(f"[SKIP] Exists: {path}")
            return
        self._log(f"{label} {path}")
        if not self.dry:
            path.write_text(content, encoding="utf-8")

    def remove_dir_if_empty(self, dir_path: Path) -> None:
        """Remove directory if it's empty.

        Args:
            dir_path: Directory path to potentially remove
        """
        try:
            next(dir_path.iterdir())
        except StopIteration:
            n = self._norm(dir_path)
            self._log(f"[RMDIR] {n}")
            if not self.dry:
                n.rmdir()
            if n in self._ensured:
                self._ensured.remove(n)
        except PermissionError:
            return

    def move_tree_merge(self, src_dir: Path, dst_dir: Path) -> None:
        """Move entire directory tree, merging with destination.

        Args:
            src_dir: Source directory path
            dst_dir: Destination directory path
        """
        self.ensure_dir(dst_dir)
        for root, dirs, files in os.walk(src_dir):
            rel = Path(root).relative_to(src_dir)
            for d in dirs:
                self.ensure_dir(dst_dir / rel / d)
            for f in files:
                self.move_file(Path(root) / f, dst_dir / rel / f)
        self.remove_dir_if_empty(src_dir)

    def move_dir_contents_to(self, src_dir: Path, dst_dir: Path) -> None:
        """Move all contents of source directory to destination.

        Args:
            src_dir: Source directory path
            dst_dir: Destination directory path
        """
        self.ensure_dir(dst_dir)
        for e in src_dir.iterdir():
            t = dst_dir / e.name
            if e.is_dir():
                self.move_tree_merge(e, t)
            else:
                self.move_file(e, t)
        self.remove_dir_if_empty(src_dir)

    def move_dir_atomic(self, src_dir: Path, dst_dir: Path) -> None:
        """Move entire directory atomically, merging if destination exists.

        Args:
            src_dir: Source directory path
            dst_dir: Destination directory path
        """
        s, d = self._norm(src_dir), self._norm(dst_dir)
        if s == d:
            self._log(f"[SKIP] Already at dest: {src_dir}")
            return
        if not d.exists():
            self.ensure_parent(d)
            self._log(f"[MOVE-DIR] {src_dir} -> {dst_dir}")
            if not self.dry:
                shutil.move(str(src_dir), str(dst_dir))
            self._ensured.add(d)
            return
        self._log(f"[MERGE-DIR] {src_dir} -> {dst_dir}")
        self.move_tree_merge(src_dir, dst_dir)
