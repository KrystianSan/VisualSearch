"""
ui/search_controller.py
Thin dispatcher: validates input, resets UI, then delegates to the
appropriate search module in search/.
Also owns folder processing (vector indexing) and save/load logic.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

from config import DEFAULT_RESULTS_DIR
from core.vector_db import VectorDatabase
from utils.file_utils import list_image_files, count_image_files
from search import (
    VectorSearch, HistogramSearch, DuplicateSearch,
    DuplicateGroupsSearch, SSIMSearch, SIFTSearch,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ui.app import VisualSearch


class SearchController:
    """
    Owns shared search state (stop_flag, search_thread) and
    coordinates which search module runs.
    """

    def __init__(self, app: "VisualSearch"):
        self.app = app
        self.stop_flag = threading.Event()
        self.search_thread: threading.Thread | None = None
        self.processing_thread: threading.Thread | None = None
        self.processing_flag = threading.Event()

        # Progress tracking for folder processing
        self.total_files = 0
        self.processed_files = 0
        self.session_files = 0
        self._is_resume = False
        self.start_time = 0.0
        self.processing_active = False

        # Map search mode names to search classes
        self._registry = {
            "Vector Similarity":    VectorSearch,
            "Histogram Similarity": HistogramSearch,
            "Find Duplicates":      DuplicateSearch,
            "Duplicate Groups":     DuplicateGroupsSearch,
            "SSIM Compare":         SSIMSearch,
            "SIFT Compare":         SIFTSearch,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def tree(self):
        return self.app.tree

    @property
    def status(self):
        return self.app.status

    @property
    def progress(self):
        return self.app.progress

    @property
    def root(self):
        return self.app.root

    # ------------------------------------------------------------------
    # Public: run / stop search
    # ------------------------------------------------------------------

    def run_search(self) -> None:
        if self.search_thread and self.search_thread.is_alive():
            messagebox.showinfo("Info", "A search is already running.")
            return

        mode = self.app.search_combobox.get()
        has_folders = bool(self.app.added_folders)

        # Validate inputs
        if mode == "Duplicate Groups":
            if not has_folders:
                messagebox.showinfo("Info", "Please add search folders.")
                return
        else:
            if not self.app.query_image:
                messagebox.showinfo("Info", "Upload a query image first.")
                return
            if not has_folders:
                messagebox.showinfo("Info", "Please add search folders.")
                return

        # Snapshot UI values before disabling controls
        folders = list(self.app.added_folders)
        folder_subfolders = dict(self.app.folder_subfolders)

        # Update UI state
        self.stop_flag.clear()
        self.app.set_searching()
        self.progress["value"] = 0
        self.status.set("Scanning folders...")

        # Clear rows immediately so the UI looks responsive.
        # For non-groups modes, also ensure we have the standard column schema
        # (in case the last search was Duplicate Groups).
        self.tree.delete(*self.tree.get_children())
        if mode != "Duplicate Groups":
            try:
                cols = list(self.tree["columns"])
            except Exception:
                cols = []
            if cols != ["filename", "path", "similarity"]:
                self._reset_treeview()

        # Launch a bootstrap thread that collects files then delegates to the
        # real search — keeps the main thread completely free
        self.search_thread = threading.Thread(
            target=self._bootstrap,
            args=(mode, folders, folder_subfolders),
            daemon=True,
        )
        self.search_thread.start()

    def _bootstrap(self, mode: str, folders: list, folder_subfolders: dict) -> None:
        """Collect file list on a worker thread, then start the real search."""
        files = list_image_files(folders, folder_subfolders=folder_subfolders)
        self.app.files_list = files
        self.root.after(0, lambda: self.progress.configure(maximum=max(len(files), 1), value=0))
        self.root.after(0, lambda: self.status.set("Starting search..."))

        search_cls = self._registry.get(mode)
        if not search_cls:
            return

        if mode == "Duplicate Groups":
            def rebuild_then_start():
                self.app.build_groups_treeview()
                t = threading.Thread(
                    target=search_cls(self).start,
                    args=(files, folder_subfolders),
                    daemon=True,
                )
                t.start()
                self.search_thread = t
            self.root.after(0, rebuild_then_start)
        else:
            search_cls(self).start(files, folder_subfolders)

    def stop_search(self) -> None:
        self.stop_flag.set()
        self.app.set_idle()
        self.progress["value"] = 0
        self.status.set("Search stopped.")

    # ------------------------------------------------------------------
    # Treeview helpers (called by search modules)
    # ------------------------------------------------------------------

    def _is_query_path(self, path: str) -> bool:
        """Return True if path matches the current query image."""
        if not self.app.target_image_path:
            return False
        try:
            return Path(path).resolve() == Path(self.app.target_image_path).resolve()
        except Exception:
            return False

    def _reset_treeview(self) -> None:
        self.app._build_standard_treeview()

    def _reset_search_ui(self) -> None:
        self.app.set_idle()
        self.progress["value"] = 0
        self.search_thread = None
        self.stop_flag.clear()

    # ------------------------------------------------------------------
    # Folder processing / vector indexing
    # ------------------------------------------------------------------

    def process_folders(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            return  # already running — button should have toggled to stop

        folders = list(self.app.added_folders)
        folder_subfolders = dict(self.app.folder_subfolders)

        if not folders:
            messagebox.showinfo("Info", "Please add search folders first.")
            return

        self.processing_flag.clear()
        self.root.after(0, self.app.set_process_button_running)
        self.processing_thread = threading.Thread(
            target=self._process_all_folders,
            args=(folders, folder_subfolders),
            daemon=True,
        )
        self.processing_thread.start()

    def stop_processing(self) -> None:
        self.processing_flag.set()
        self.status.set("Stopping…")

    def _process_all_folders(self, folders: list, folder_subfolders: dict) -> None:
        stopped_early = False
        try:
            self.total_files = count_image_files(folders, folder_subfolders=folder_subfolders)

            # Count already-indexed files so the progress bar resumes correctly.
            already_indexed = sum(VectorDatabase.count_indexed(f) for f in folders)
            self.processed_files = min(already_indexed, self.total_files)
            self._is_resume = already_indexed > 0
            self.session_files = 0   # files processed in this session only (for fps)
            self.start_time = time.time()
            self.processing_active = True

            status_t = threading.Thread(target=self._status_updater, daemon=True)
            status_t.start()

            initial_value = self.processed_files
            initial_msg = (
                f"Resuming — {initial_value}/{self.total_files} already indexed"
                if initial_value > 0
                else f"Processing 0/{self.total_files} files"
            )
            self.root.after(0, lambda: [
                self.progress.configure(maximum=max(self.total_files, 1), value=initial_value),
                self.status.set(initial_msg),
            ])

            for folder in folders:
                if self.processing_flag.is_set():
                    stopped_early = True
                    break
                include_sub = folder_subfolders.get(folder, False)
                self._index_folder(folder, include_sub)

            if self.processing_flag.is_set():
                stopped_early = True

        except Exception as exc:
            log.exception("Processing failed")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{exc}"))
        finally:
            self.processing_active = False
            status_t.join(1.0)

            elapsed = time.time() - self.start_time
            m, s = divmod(elapsed, 60)

            if stopped_early:
                msg = f"Stopped — {self.processed_files}/{self.total_files} file(s) indexed in {int(m)}m {s:.1f}s"
                self.root.after(0, lambda: self.status.set(msg))
            else:
                summary = f"Done in {int(m)}m {s:.1f}s — {self.processed_files} file(s) indexed"
                self.root.after(0, lambda: [
                    self.status.set(summary),
                    messagebox.showinfo("Done", f"All folders processed.\n{self.processed_files} file(s) indexed."),
                ])

            self.root.after(0, self.app.set_process_button_idle)

    def _index_folder(self, folder_path, include_sub: bool) -> None:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            return

        existing_meta = VectorDatabase.read_metadata(folder_path)
        pattern = folder_path.rglob("*") if include_sub else folder_path.glob("*")
        new_vectors, new_meta = [], []
        current_files: set[str] = set()

        try:
            for entry in pattern:
                if self.processing_flag.is_set():
                    break
                if not (entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm"}):
                    continue

                entry_str = entry.as_posix()
                current_files.add(entry_str)
                mtime = entry.stat().st_mtime

                if existing_meta.get(entry_str) == mtime:
                    # On resume, cached files are already counted in already_indexed
                    # so we skip ticking to avoid inflating processed_files past total
                    if not self._is_resume:
                        self._tick_progress()
                    continue

                vec = self.app.vector_extractor.extract(entry)
                if vec is not None:
                    new_vectors.append(vec)
                    new_meta.append({"path": entry_str, "mtime": mtime})
                self._tick_progress()

        except Exception:
            log.exception("_index_folder error")

        if new_vectors:
            VectorDatabase.save(folder_path, np.array(new_vectors), new_meta)

        # Prune entries for files no longer on disk (only when not interrupted)
        if current_files and not self.processing_flag.is_set():
            VectorDatabase.prune(folder_path, current_files)

    def _tick_progress(self) -> None:
        self.processed_files += 1
        self.session_files += 1
        v = self.processed_files
        self.root.after(0, lambda: self.progress.__setitem__("value", v))

    def _status_updater(self) -> None:
        while self.processing_active:
            elapsed = time.time() - self.start_time
            fps = self.session_files / elapsed if elapsed > 0 else 0
            m, s = divmod(elapsed, 60)
            msg = (
                f"Processing: {self.processed_files}/{self.total_files} files "
                f"({int(m)}m {s:.1f}s, {fps:.1f} files/sec)"
            )
            self.root.after(0, self.status.set, msg)
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Save / Load results
    # ------------------------------------------------------------------

    def save_results(self) -> None:
        if not self.tree.get_children():
            messagebox.showinfo("Info", "No results to save.")
            return

        mode = self.app.search_combobox.get()
        is_groups = self.app._is_groups_treeview()

        # Collect results from treeview
        results = []
        if is_groups:
            for group_item in self.tree.get_children():
                group_text = self.tree.item(group_item, "text")
                children = []
                for child in self.tree.get_children(group_item):
                    v = self.tree.item(child, "values")
                    children.append({
                        "path": str(v[0]),
                        "size": int(v[1]) if v[1] else 0,
                        "dims": str(v[3]) if len(v) > 3 else "",
                    })
                results.append({"group": group_text, "files": children})
        else:
            for item in self.tree.get_children():
                v = self.tree.item(item, "values")
                if len(v) >= 3:
                    results.append({"path": str(v[1]), "similarity": str(v[2])})

        payload = {
            "version":           3,
            "saved_at":          datetime.now().isoformat(timespec="seconds"),
            "mode":              mode,
            "query_image":       self.app.target_image_path or "",
            "threshold":         int(self.app.sim.get()),
            "folders":           list(self.app.added_folders),
            "folder_subfolders": dict(self.app.folder_subfolders),
            "result_count":      len(results),
            "results":           results,
        }

        os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = mode.replace(" ", "_").lower()
        dest = filedialog.asksaveasfilename(
            initialdir=DEFAULT_RESULTS_DIR,
            title="Save Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files",  "*.csv"),
                ("All files",  "*.*"),
            ],
            initialfile=f"{safe_mode}_{ts}.json",
        )
        if not dest:
            return

        if dest.lower().endswith(".csv"):
            self._save_csv(dest, results, is_groups, payload)
        else:
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            self.status.set(f"Saved {len(results)} result(s) → {Path(dest).name}")
            messagebox.showinfo("Saved", f"Saved {len(results)} result(s) to:\n{dest}")

    def _save_csv(self, dest: str, results: list, is_groups: bool, meta: dict) -> None:
        """Export results as CSV (flat format; groups are flattened with a group column)."""
        try:
            with open(dest, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if is_groups:
                    writer.writerow(["Group", "Path", "Size (bytes)", "Dimensions"])
                    for g in results:
                        for f in g.get("files", []):
                            writer.writerow([
                                g.get("group", ""),
                                f.get("path", ""),
                                f.get("size", ""),
                                f.get("dims", ""),
                            ])
                else:
                    writer.writerow(["Path", "Similarity", "Mode", "Query Image"])
                    query = meta.get("query_image", "")
                    mode  = meta.get("mode", "")
                    for r in results:
                        writer.writerow([r.get("path", ""), r.get("similarity", ""), mode, query])
            self.status.set(f"Exported {len(results)} result(s) → {Path(dest).name}")
            messagebox.showinfo("Exported", f"Exported {len(results)} result(s) to:\n{dest}")
        except Exception as exc:
            messagebox.showerror("Export Error", f"Could not write CSV:\n{exc}")

    def load_results(self) -> None:
        src = filedialog.askopenfilename(
            title="Load Results",
            initialdir=DEFAULT_RESULTS_DIR,
            filetypes=[
                ("Result files", "*.json *.csv"),
                ("JSON files",   "*.json"),
                ("CSV files",    "*.csv"),
                ("All files",    "*.*"),
            ],
        )
        if not src:
            return

        if src.lower().endswith(".json"):
            self._load_json(src)
        else:
            self._load_csv_legacy(src)

    def _load_json(self, src: str) -> None:
        try:
            with open(src, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read file:\n{exc}")
            return

        mode    = data.get("mode", "")
        results = data.get("results", [])
        saved_at = data.get("saved_at", "unknown")
        warnings: list[str] = []

        # --- Restore search mode ---
        from config import SEARCH_MODES
        if mode in SEARCH_MODES:
            self.app.search_combobox.set(mode)
        elif mode:
            warnings.append(f"Unknown search mode '{mode}' — kept current.")

        # --- Restore threshold ---
        threshold = data.get("threshold")
        if threshold is not None:
            try:
                self.app.sim.set(int(threshold))
            except Exception:
                pass

        # --- Restore folders (warn about missing ones) ---
        folders = data.get("folders", [])
        if folders:
            missing = [f for f in folders if not Path(f).exists()]
            present = [f for f in folders if Path(f).exists()]
            if missing:
                warnings.append(
                    f"{len(missing)} folder(s) no longer exist and were skipped:\n"
                    + "\n".join(f"  • {f}" for f in missing[:5])
                    + ("\n  …" if len(missing) > 5 else "")
                )
            self.app.added_folders = present
            self.app.folder_subfolders = {
                k: v for k, v in data.get("folder_subfolders", {}).items()
                if k in present
            }
            self.app._rebuild_folder_rows()

        # --- Restore query image ---
        query = data.get("query_image", "")
        if query:
            if Path(query).exists():
                self.app.target_image_path = query
                try:
                    self.app.query_image = Image.open(query)
                    self.app._display_uploaded(self.app.query_image)
                except Exception:
                    pass
            else:
                warnings.append(f"Query image not found:\n  {query}")

        # --- Rebuild treeview and populate results ---
        is_groups = bool(results and isinstance(results[0], dict) and "group" in results[0])

        if is_groups:
            self.app.build_groups_treeview()
        else:
            if self.app._is_groups_treeview():
                self.app._build_standard_treeview()
        self.tree.delete(*self.tree.get_children())

        missing_files = 0
        if is_groups:
            for g_idx, group in enumerate(results, 1):
                files = group.get("files", [])
                size_total = sum(f.get("size", 0) for f in files)
                parent = self.tree.insert(
                    "", "end",
                    text=group.get("group", f"Group {g_idx}"),
                    values=("", size_total, f"{size_total / (1024*1024):.2f} MB", "", len(files)),
                    tags=("group_header",),
                    open=True,
                )
                for f_idx, f in enumerate(files, 1):
                    p = f.get("path", "")
                    if not Path(p).exists():
                        missing_files += 1
                    self.tree.insert(
                        parent, "end",
                        text=Path(p).name if p else "",
                        values=(
                            p,
                            f.get("size", 0),
                            f"{f.get('size', 0) / 1024:.2f} KB",
                            f.get("dims", ""),
                            "",
                        ),
                        tags=(f'{"even" if f_idx % 2 else "odd"}_row',),
                    )
        else:
            for idx, r in enumerate(results):
                path = r.get("path", "")
                sim  = r.get("similarity", "")
                if not path:
                    continue
                if not Path(path).exists():
                    missing_files += 1
                fn  = Path(path).name
                tag = "evenrow" if idx % 2 == 0 else "oddrow"
                self.tree.insert("", tk.END, values=(fn, path, sim), tags=(tag,))

        if missing_files:
            warnings.append(f"{missing_files} result file(s) no longer exist on disk.")

        self.app.update_result_count()
        n = len(results)
        self.status.set(f"Loaded {n} result(s) — {mode} — saved {saved_at}")

        summary = f"Loaded {n} result(s)\nMode: {mode or 'unknown'}\nSaved: {saved_at}"
        if warnings:
            summary += "\n\n⚠ Warnings:\n" + "\n".join(warnings)
            messagebox.showwarning("Loaded with warnings", summary)
        else:
            messagebox.showinfo("Loaded", summary)

    def _load_csv_legacy(self, src: str) -> None:
        """Load old CSV format for backwards compatibility."""
        if self.app._is_groups_treeview():
            self.app._build_standard_treeview()
        self.tree.delete(*self.tree.get_children())
        loaded = 0
        try:
            with open(src, "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row or row[0] in (
                        "Target Image", "Similarity Threshold (%)",
                        "Search Subfolders", "Name", "Path",
                    ):
                        continue
                    if len(row) >= 2:
                        path, sim = row[0], row[1]
                        if not self._is_query_path(path):
                            fn  = os.path.basename(path)
                            tag = "evenrow" if loaded % 2 == 0 else "oddrow"
                            self.tree.insert("", tk.END,
                                             values=(fn, path, sim), tags=(tag,))
                            loaded += 1
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read CSV:\n{exc}")
            return

        self.app.update_result_count()
        self.status.set(f"Loaded {loaded} result(s) from legacy CSV.")
        messagebox.showinfo("Loaded", f"Loaded {loaded} result(s) from legacy CSV.")
