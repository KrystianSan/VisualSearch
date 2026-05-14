"""
search/duplicate_groups.py
Finds all groups of identical files across the selected folders
by grouping files with matching SHA-256 hashes.
"""

from __future__ import annotations

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from core.algorithms import calculate_image_hash
from .base import BaseSearch


class DuplicateGroupsSearch(BaseSearch):

    def start(self, files: list, folder_subfolders: dict) -> None:
        self.root.after(0, lambda: self.progress.configure(maximum=len(files), value=0))
        self.ctrl.search_thread = threading.Thread(
            target=self._thread, args=(files,), daemon=True
        )
        self.ctrl.search_thread.start()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _thread(self, files: list) -> None:
        start = time.time()
        hash_groups: dict = {}       # hash → {"files": [...], "total_size": int}
        tree_items:  dict = {}       # hash → treeview item id (once group has 2+ files)
        total = len(files)
        groups_so_far = 0

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(self._hash_file, f): f for f in files}
            for idx, future in enumerate(as_completed(futures), 1):
                if self.stop_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                file_hash, size, path, dims = future.result()
                if file_hash:
                    g = hash_groups.setdefault(file_hash, {"files": [], "total_size": 0})
                    prev_len = len(g["files"])
                    g["files"].append((path, size, dims))
                    g["total_size"] += size

                    if prev_len == 1:
                        # Just became a group — insert header + both files
                        groups_so_far += 1
                        g_copy = dict(g)
                        h = file_hash
                        self.root.after(0, lambda gc=g_copy, fh=h, gi=groups_so_far:
                                        self._insert_new_group(gc, fh, gi, tree_items))
                    elif prev_len >= 2 and file_hash in tree_items:
                        # Existing group grew — append new child and update header
                        entry = (path, size, dims)
                        f_idx = prev_len + 1
                        item_id = tree_items.get(file_hash)
                        new_total = g["total_size"]
                        n_files = len(g["files"])
                        self.root.after(0, lambda iid=item_id, e=entry, fi=f_idx,
                                        nt=new_total, nf=n_files, gi=groups_so_far:
                                        self._append_to_group(iid, e, fi, nt, nf, gi))

                self.set_status(f"Hashing ({idx}/{total}) – {groups_so_far} group(s) found")
                self.set_progress(idx)

        elapsed = time.time() - start
        stopped = self.stop_flag.is_set()
        msg = (
            f"Search stopped — {groups_so_far} group(s) found so far"
            if stopped else
            f"Found {groups_so_far} duplicate group(s) in {elapsed:.2f}s"
        )
        self.root.after(0, lambda m=msg: self._finish(m))

    def _insert_new_group(self, group: dict, file_hash: str,
                          g_idx: int, tree_items: dict) -> None:
        """Called on main thread when a hash first becomes a duplicate group."""
        size_mb = group["total_size"] / (1024 * 1024)
        n = len(group["files"])
        parent = self.tree.insert(
            "", "end",
            text=f"Group {g_idx} – {size_mb:.2f} MB ({n} files)",
            values=("", group["total_size"], f"{size_mb:.2f} MB", "", n),
            tags=("group_header",),
            open=True,
        )
        tree_items[file_hash] = parent
        for f_idx, (path, size, dims) in enumerate(group["files"], 1):
            self.tree.insert(
                parent, "end",
                text=os.path.basename(str(path)),
                values=(str(path), size, f"{size / 1024:.2f} KB", dims, ""),
                tags=("evenrow" if f_idx % 2 else "oddrow",),
            )
        self.app.update_result_count()

    def _append_to_group(self, parent_id: str, entry: tuple,
                         f_idx: int, new_total: int, n_files: int,
                         g_idx: int) -> None:
        """Called on main thread when a new file joins an existing group."""
        path, size, dims = entry
        self.tree.insert(
            parent_id, "end",
            text=os.path.basename(str(path)),
            values=(str(path), size, f"{size / 1024:.2f} KB", dims, ""),
            tags=("evenrow" if f_idx % 2 else "oddrow",),
        )
        # Update header text, raw size, and FileCount
        size_mb = new_total / (1024 * 1024)
        current_text = self.tree.item(parent_id, "text")
        group_num = current_text.split(" –")[0]
        self.tree.item(parent_id,
                       text=f"{group_num} – {size_mb:.2f} MB ({n_files} files)",
                       values=("", new_total, f"{size_mb:.2f} MB", "", n_files))
        self.app.update_result_count()

    def _finish(self, msg: str) -> None:
        self.status.set(msg)
        self.progress["value"] = 0
        self.app.update_result_count()
        self.ctrl._reset_search_ui()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path) -> tuple:
        try:
            file_hash = calculate_image_hash(path)
            size = os.path.getsize(path)
            try:
                with Image.open(path) as img:
                    dims = f"{img.width}x{img.height}"
            except Exception:
                dims = "N/A"
            return file_hash, size, path, dims
        except Exception:
            return None, 0, path, "N/A"
