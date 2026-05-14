"""
search/duplicate_search.py
Exact-duplicate finder for a single query image using SHA-256 hashing.
Uses a quick partial hash as a fast pre-filter before full comparison.
"""

from __future__ import annotations

import logging
import os
import time
import threading

log = logging.getLogger(__name__)

from core.algorithms import calculate_image_hash, calculate_quick_hash
from .base import BaseSearch


class DuplicateSearch(BaseSearch):

    def start(self, files: list, folder_subfolders: dict) -> None:
        self.ctrl.search_thread = threading.Thread(
            target=self._thread, args=(files,), daemon=True
        )
        self.ctrl.search_thread.start()

    # ------------------------------------------------------------------

    def _thread(self, files: list) -> None:
        target_hash = calculate_image_hash(self.target_path)
        if not target_hash:
            self.set_status("Error: could not hash query image.")
            self.done()
            return
        target_quick = calculate_quick_hash(self.target_path)
        target_size = os.path.getsize(self.target_path)
        start = time.time()
        total = len(files)
        found = 0

        for count, file in enumerate(files, 1):
            if self.stop_flag.is_set():
                break

            self.set_status(f"Analysing ({count}/{total}) – {found} duplicate(s)")

            try:
                if os.path.getsize(file) != target_size:
                    continue
                if calculate_quick_hash(file) != target_quick:
                    continue
                if calculate_image_hash(file) == target_hash:
                    self.insert_row(str(file), "Duplicate")
                    found += 1
            except Exception as exc:
                log.warning("DuplicateSearch: %s: %s", file, exc)

            self.set_progress(count)

        elapsed = time.time() - start
        if not self.stop_flag.is_set():
            self.set_status(f"Found {found} duplicate(s) in {elapsed:.1f}s")
        self.set_progress(0)
        self.done()
