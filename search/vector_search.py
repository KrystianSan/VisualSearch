"""
search/vector_search.py
ResNet18 vector-based cosine similarity search.
Requires folders to be pre-processed (vectors indexed on disk).
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path

import logging
import numpy as np
import pandas as pd
from tkinter import messagebox

log = logging.getLogger(__name__)

from config import METADATA_FILE, MAX_SEARCH_RESULTS, VECTOR_MAX_DISTANCE
from core.algorithms import cosine_similarity_batch, score_vector_similarity
from core.vector_db import VectorDatabase
from .base import BaseSearch


class VectorSearch(BaseSearch):

    def start(self, files: list, folder_subfolders: dict) -> None:
        # start() is called from the bootstrap worker thread — schedule all
        # UI interaction and thread launching on the main thread.
        self.root.after(0, lambda: self._start_on_main(files, folder_subfolders))

    def _start_on_main(self, files: list, folder_subfolders: dict) -> None:
        query_vec = self.app.vector_extractor.extract(Path(self.target_path))
        if query_vec is None:
            messagebox.showerror("Error", "Feature extraction failed.")
            self.ctrl._reset_search_ui()
            return

        vec_files = self._collect_vector_files(folder_subfolders)
        if not vec_files:
            messagebox.showinfo("Info", "No vector data found. Run 'Process Folders' first.")
            self.ctrl._reset_search_ui()
            return

        incomplete = self._find_incomplete_folders(folder_subfolders)
        if incomplete:
            names = "\n".join(f"  • {f}" for f in incomplete)
            answer = messagebox.askyesno(
                "Incomplete Vector Data",
                f"The following folder(s) have not been fully processed:\n\n{names}\n\n"
                "Search results may be incomplete. Continue anyway?",
            )
            if not answer:
                self.ctrl._reset_search_ui()
                return

        threshold = self.threshold
        self.progress.configure(maximum=len(vec_files), value=0)
        self.ctrl.search_thread = threading.Thread(
            target=self._thread,
            args=(vec_files, query_vec, threshold),
            daemon=True,
        )
        self.ctrl.search_thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_incomplete_folders(self, folder_subfolders: dict) -> list[str]:
        """Return folder paths whose vector index doesn't cover all images on disk."""
        from config import SUPPORTED_EXTENSIONS
        incomplete = []

        for folder_str in self.app.added_folders:
            fp = Path(folder_str)
            inc_sub = folder_subfolders.get(folder_str, False)

            if inc_sub:
                # All images are indexed into the root folder's metadata
                img_count = sum(
                    1 for e in fp.rglob("*")
                    if e.is_file() and e.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                if img_count == 0:
                    continue
                _, meta_path = VectorDatabase.get_paths(fp)
                if not meta_path.exists():
                    incomplete.append(folder_str)
                    continue
                try:
                    meta_count = sum(1 for _ in open(meta_path)) - 1
                except Exception:
                    meta_count = 0
                if img_count > meta_count:
                    incomplete.append(folder_str)
            else:
                img_count = sum(
                    1 for e in fp.glob("*")
                    if e.is_file() and e.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                if img_count == 0:
                    continue
                _, meta_path = VectorDatabase.get_paths(fp)
                if not meta_path.exists():
                    incomplete.append(folder_str)
                    continue
                try:
                    meta_count = sum(1 for _ in open(meta_path)) - 1
                except Exception:
                    meta_count = 0
                if img_count > meta_count:
                    incomplete.append(folder_str)

        return incomplete

    def _collect_vector_files(self, folder_subfolders: dict) -> list[Path]:
        result = []
        for folder_str in self.app.added_folders:
            fp = Path(folder_str)
            inc_sub = folder_subfolders.get(folder_str, False)
            if inc_sub:
                for root, _, _ in os.walk(fp):
                    vf = VectorDatabase.get_paths(root)[0]
                    if vf.exists():
                        result.append(vf)
            else:
                vf = VectorDatabase.get_paths(fp)[0]
                if vf.exists():
                    result.append(vf)
        return result

    def _thread(self, vec_files: list[Path], query_vec: np.ndarray, threshold: int) -> None:
        try:
            if query_vec.shape[0] != 512:
                self.root.after(0, messagebox.showerror, "Error", "Query vector dimension mismatch (expected 512).")
                return

            query_path = Path(self.target_path).resolve()
            results: list[tuple[str, float]] = []
            start = time.time()
            total = len(vec_files)

            for idx, vf in enumerate(vec_files, 1):
                if self.stop_flag.is_set():
                    break

                folder_label = str(vf.parent)
                self.set_progress(idx)
                self.set_status(f"Searching {folder_label} ({idx}/{total})")

                try:
                    vectors = np.load(vf)
                    meta_df = pd.read_csv(vf.parent / METADATA_FILE)

                    raw_sims = cosine_similarity_batch(vectors, query_vec)
                    dists    = np.linalg.norm(vectors - query_vec, axis=1)

                    for i, (raw_sim, dist) in enumerate(zip(raw_sims, dists)):
                        if Path(meta_df.iloc[i]["path"]) == query_path:
                            continue
                        if dist > VECTOR_MAX_DISTANCE:
                            continue
                        score = score_vector_similarity(float(raw_sim))
                        if score >= threshold:
                            results.append((meta_df.iloc[i]["path"], score))

                except Exception as exc:
                    log.warning("VectorSearch: %s: %s", vf, exc)

            elapsed = time.time() - start
            if self.stop_flag.is_set():
                self.set_status(f"Search stopped — {len(results)} result(s) collected so far")
                self.set_progress(0)
                self.done()
            else:
                self.root.after(0, lambda r=results, e=elapsed: self._finalize(r, e))

        except Exception as exc:
            log.exception("VectorSearch: thread error")
            self.done()

    def _finalize(self, results: list, elapsed: float) -> None:
        self.tree.delete(*self.tree.get_children())
        top = sorted(results, key=lambda x: x[1], reverse=True)[:MAX_SEARCH_RESULTS]
        for path, score in top:
            filename = os.path.basename(path)
            row_count = len(self.tree.get_children())
            tag = "evenrow" if row_count % 2 == 0 else "oddrow"
            self.tree.insert("", "end", values=(filename, path, f"{score:.1f}%"), tags=(tag,))
        count = len(self.tree.get_children())
        self.status.set(
            f"Found {count} match{'es' if count != 1 else ''} in {elapsed:.2f}s"
            f"  [threshold: {self.threshold}%]"
        )
        self.progress["value"] = 0
        self.app.update_result_count()
        self.ctrl._reset_search_ui()
