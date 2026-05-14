"""
search/histogram_search.py
HSV hue-channel histogram intersection similarity search.
"""

from __future__ import annotations

import time
import threading

import cv2
import numpy as np

from core.algorithms import calculate_histogram, compare_histograms
from .base import BaseSearch


class HistogramSearch(BaseSearch):

    def start(self, files: list, folder_subfolders: dict) -> None:
        img = cv2.imdecode(
            np.fromfile(self.target_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            self.status.set("Error: cannot read query image.")
            return

        hist1 = calculate_histogram(img)
        self.ctrl.search_thread = threading.Thread(
            target=self._thread, args=(files, hist1), daemon=True
        )
        self.ctrl.search_thread.start()

    # ------------------------------------------------------------------

    def _thread(self, files: list, hist1: np.ndarray) -> None:
        start = time.time()
        found = 0
        total = len(files)

        for count, file in enumerate(files, 1):
            if self.stop_flag.is_set():
                break

            self.status.set(f"Analysing ({count}/{total}) – {found} matches")

            img = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                hist2 = calculate_histogram(img)
                sim = compare_histograms(hist1, hist2)
                if sim >= self.threshold:
                    self.insert_row(str(file), f"{sim:.2f}")
                    found += 1

            self.set_progress(count)

        elapsed = time.time() - start
        if not self.stop_flag.is_set():
            self.status.set(f"Done in {elapsed:.2f}s – {found} matches found")
        self.set_progress(0)
        self.done()
