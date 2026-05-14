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

    def insert_row(self, path: str, label: str) -> None:
        """
        Insert a result row into the treeview from any thread.
        Silently skips the query image — no search module needs to check.
        Columns: (filename, path, similarity) with alternating row colors.
        """
        if self.is_query_image(path):
            return
        filename = os.path.basename(path)
        # Determine row parity for alternating colors
        row_count = len(self.tree.get_children())
        tag = "evenrow" if row_count % 2 == 0 else "oddrow"
        self.root.after(
            0,
            lambda fn=filename, p=path, s=label, t=tag:
                self.tree.insert("", tk.END, values=(fn, p, s), tags=(t,))
        )
        # Update result counter after each insert
        self.root.after(0, self.app.update_result_count)

    def set_status(self, msg: str) -> None:
        self.root.after(0, lambda m=msg: self.status.set(m))

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
