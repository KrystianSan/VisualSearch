"""
search/base.py
Base class providing shared state and UI helpers to every search module.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ui.search_controller import SearchController


class BaseSearch:
    """
    Shared scaffolding for all search strategies.

    Subclasses must implement:
        start(files: list, include_sub: bool) -> None
    """

    def __init__(self, controller: "SearchController"):
        self.ctrl = controller

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def app(self):
        return self.ctrl.app

    @property
    def tree(self):
        return self.ctrl.app.tree

    @property
    def root(self):
        return self.ctrl.app.root

    @property
    def status(self):
        return self.ctrl.app.status

    @property
    def progress(self):
        return self.ctrl.app.progress

    @property
    def stop_flag(self) -> threading.Event:
        return self.ctrl.stop_flag

    @property
    def threshold(self) -> int:
        return int(self.app.sim.get())

    @property
    def target_path(self) -> str:
        return self.app.target_image_path

    # ------------------------------------------------------------------
    # Thread-safe UI helpers
    # ------------------------------------------------------------------

    def is_query_image(self, path) -> bool:
        """Return True if *path* resolves to the same file as the query image."""
        if not self.target_path:
            return False
        try:
            return Path(path).resolve() == Path(self.target_path).resolve()
        except (OSError, ValueError, TypeError):
            return False

    def insert_row(self, path: str, label: str) -> bool:
        """
        Insert a result row into the treeview from any thread.
        Silently skips the query image — no search module needs to check.
        Columns: (filename, path, similarity) with alternating row colors.

        Returns True if a row was inserted, False if skipped (query image) —
        callers that track their own "found" counter should only increment
        it when this returns True, or the count will run ahead of what's
        actually in the treeview.
        """
        if self.is_query_image(path):
            return False
        filename = os.path.basename(path)
        self.root.after(
            0,
            lambda fn=filename, p=path, s=label:
                self._insert_row_on_main(fn, p, s)
        )
        return True

    def _insert_row_on_main(self, filename: str, path: str, label: str) -> None:
        """Runs on the main thread: reads current row count for striping,
        inserts, then updates the result counter — all against the tree's
        actual state, not a value read from the calling worker thread."""
        row_count = len(self.tree.get_children())
        tag = "evenrow" if row_count % 2 == 0 else "oddrow"
        self.tree.insert("", tk.END, values=(filename, path, label), tags=(tag,))
        self.app.update_result_count()

    def set_status(self, msg: str) -> None:
        self.root.after(0, lambda m=msg: self.status.set(m))

    def t(self, key: str, **kwargs) -> str:
        """Translate *key* to the current UI language, formatting placeholders if given."""
        from i18n import get_text
        text = get_text(self.app.current_language, key)
        return text.format(**kwargs) if kwargs else text

    def no_results_hint(self) -> str:
        """
        Return a short suffix suggesting the user enable 'subfolders', if any
        currently-added folder has it disabled. Returns "" when every added
        folder already has subfolders on (the hint wouldn't help) or there
        are no folders at all.
        """
        folder_subfolders = getattr(self.app, "folder_subfolders", {}) or {}
        added_folders = getattr(self.app, "added_folders", []) or []
        if any(not folder_subfolders.get(f, False) for f in added_folders):
            return "  " + self.t("hint_no_results_subfolders")
        return ""

    def set_progress(self, value: int) -> None:
        self.root.after(0, lambda v=value: self.progress.__setitem__("value", v))

    def done(self) -> None:
        """Call at the end of every search thread to reset UI state."""
        self.root.after(0, self.ctrl._reset_search_ui)
        self.root.after(0, self.app.update_result_count)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def start(self, files: list, folder_subfolders: dict) -> None:
        raise NotImplementedError
