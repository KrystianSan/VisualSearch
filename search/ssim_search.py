"""
search/ssim_search.py
Structural Similarity Index (SSIM) comparison search.
Only compares images with identical dimensions to the query image.
"""

from __future__ import annotations

import time
import threading

from core.algorithms import compare_ssim
from .base import BaseSearch


class SSIMSearch(BaseSearch):

    def start(self, files: list, folder_subfolders: dict) -> None:
        self.ctrl.search_thread = threading.Thread(
            target=self._thread, args=(files,), daemon=True
        )
        self.ctrl.search_thread.start()

    # ------------------------------------------------------------------

    def _thread(self, files: list) -> None:
        start = time.time()
        found = 0
        total = len(files)

        for count, file in enumerate(files, 1):
            if self.stop_flag.is_set():
                break

            self.status.set(f"Analysing ({count}/{total}) – {found} matches")

            score = compare_ssim(self.target_path, str(file))
            if score is not None and score >= self.threshold:
                self.insert_row(str(file), f"{score:.2f}")
                found += 1

            self.set_progress(count)

        elapsed = time.time() - start
        if not self.stop_flag.is_set():
            self.status.set(f"Done in {elapsed:.2f}s – {found} matches")
        self.set_progress(0)
        self.done()
