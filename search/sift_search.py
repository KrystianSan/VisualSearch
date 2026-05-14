"""
search/sift_search.py
SIFT keypoint-based feature matching search.
Uses parallel processing for performance.
"""

from __future__ import annotations

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
import cv2
from tkinter import messagebox

log = logging.getLogger(__name__)

from config import SIFT_MIN_MATCHES
from core.algorithms import initialize_sift, sift_similarity
from .base import BaseSearch


class SIFTSearch(BaseSearch):

    def start(self, files: list, folder_subfolders: dict) -> None:
        self.ctrl.search_thread = threading.Thread(
            target=self._thread, args=(files,), daemon=True
        )
        self.ctrl.search_thread.start()

    # ------------------------------------------------------------------

    def _thread(self, files: list) -> None:
        sift = initialize_sift()
        if sift is None:
            messagebox.showerror("SIFT Error", "SIFT is unavailable. Install opencv-contrib-python.")
            self.done()
            return

        query_gray = cv2.imread(self.target_path, cv2.IMREAD_GRAYSCALE)
        if query_gray is None:
            self.status.set("Error: cannot read query image.")
            self.done()
            return

        _, des1 = sift.detectAndCompute(query_gray, None)
        if des1 is None or len(des1) < SIFT_MIN_MATCHES:
            self.status.set("No SIFT features found in query image.")
            self.done()
            return

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        workers = max(2, os.cpu_count() // 2)
        start = time.time()
        found = 0
        total = len(files)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(sift_similarity, des1, f, bf): f for f in files}
            for idx, future in enumerate(as_completed(futures), 1):
                if self.stop_flag.is_set():
                    break
                try:
                    file_path = futures[future]
                    sim = future.result()
                    if sim is not None and sim >= self.threshold:
                        path_str, sim_str = str(futures[future]), f"{sim:.2f}"
                        self.insert_row(path_str, sim_str)
                        found += 1
                except Exception as exc:
                    log.warning("SIFTSearch: %s", exc)

                self.set_progress(idx)
                self.status.set(f"Analysing ({idx}/{total}) – {found} matches")

        elapsed = time.time() - start
        if not self.stop_flag.is_set():
            self.status.set(f"Done in {elapsed:.2f}s – {found} matches")
        self.set_progress(0)
        self.done()
