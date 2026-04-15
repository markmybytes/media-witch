"""Action queue pattern for batching operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .fileops import FileOps


class ActionQueue:
    """Queue for batching file operations with dry-run support.

    Actions are accumulated and executed together on commit(), allowing
    for preview in dry-run mode before actual execution.
    """

    def __init__(self, fops: FileOps) -> None:
        """Initialize ActionQueue.

        Args:
            fops: FileOps instance for executing operations
        """
        self.fops = fops
        self._q: list[tuple[Callable, tuple, dict, str]] = []

    def add(
        self,
        func: Callable,
        *args: Any,
        desc: str = "",
        **kwargs: Any
    ) -> None:
        """Add an action to the queue.

        Args:
            func: Callable to execute
            *args: Positional arguments for func
            desc: Description for dry-run preview
            **kwargs: Keyword arguments for func
        """
        self._q.append((func, args, kwargs, desc))

    def commit(self) -> None:
        """Execute all queued actions.

        In dry-run mode, logs planned actions without executing.
        """
        if self.fops.dry:
            if not self._q:
                self.fops._log("[DRY-RUN] No actions.")
                return
            self.fops._log("[DRY-RUN] Planned actions:")
            for _, _, _, desc in self._q:
                self.fops._log(desc if desc else "[DRY-RUN] action")
            return
        for func, args, kwargs, _ in self._q:
            func(*args, **kwargs)

    def clear(self) -> None:
        """Clear all queued actions without executing."""
        self._q.clear()
