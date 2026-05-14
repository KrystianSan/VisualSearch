"""
core/vector_db.py
Manages persistent storage of image feature vectors and metadata.

Directory layout (mirrors the scanned folder hierarchy):
    vector_db/<drive>/<path_to_folder>/
        vectors.npy      – stacked (N, 512) float32 array
        metadata.csv     – columns: path, mtime
"""

import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

from config import VECTOR_ROOT, VECTOR_FILE, METADATA_FILE


class VectorDatabase:
    """Thin wrapper around on-disk .npy + .csv pairs."""

    @staticmethod
    def get_paths(folder_path) -> tuple[Path, Path]:
        """Return (vectors_path, metadata_path) for a given folder."""
        folder_path = Path(folder_path).resolve()
        rel = folder_path.relative_to(folder_path.anchor)
        db_dir = VECTOR_ROOT / rel
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / VECTOR_FILE, db_dir / METADATA_FILE

    @staticmethod
    def load(folder_path) -> tuple[np.ndarray | None, pd.DataFrame | None]:
        """Load vectors and metadata. Returns (None, None) on missing/corrupt data."""
        vec_path, meta_path = VectorDatabase.get_paths(folder_path)
        if not vec_path.exists() or not meta_path.exists():
            return None, None
        try:
            vectors = np.load(vec_path)   # fully in-memory — avoids Windows file locks
            meta_df = pd.read_csv(meta_path)
            return vectors, meta_df
        except Exception as exc:
            log.error("VectorDatabase.load: error loading %s: %s", folder_path, exc)
            return None, None

    @staticmethod
    def save(folder_path, vectors: np.ndarray, meta_records: list[dict]) -> bool:
        """Append new vectors and metadata; deduplicates by path."""
        vec_path, meta_path = VectorDatabase.get_paths(folder_path)
        tmp_vec  = vec_path.with_suffix(".tmp.npy")
        tmp_meta = meta_path.with_suffix(".tmp.csv")
        try:
            new_paths = {r["path"] for r in meta_records}

            if vec_path.exists():
                existing_vecs = np.load(vec_path)   # fully in-memory
                existing_meta = pd.read_csv(meta_path).to_dict("records") if meta_path.exists() else []
                keep_idx = [i for i, r in enumerate(existing_meta) if r["path"] not in new_paths]
                filtered_existing_meta = [existing_meta[i] for i in keep_idx]
                merged_vecs = np.vstack([existing_vecs[keep_idx], vectors]) if keep_idx else np.array(vectors)
                merged_meta = filtered_existing_meta + meta_records
            else:
                merged_vecs = np.array(vectors)
                merged_meta = meta_records

            # Write to temp files first, then rename atomically.
            # This prevents Windows file-lock conflicts on the live .npy file.
            np.save(tmp_vec, merged_vecs)
            pd.DataFrame(merged_meta).to_csv(tmp_meta, index=False)
            tmp_vec.replace(vec_path)
            tmp_meta.replace(meta_path)
            return True

        except Exception as exc:
            log.error("VectorDatabase.save: error saving %s: %s", folder_path, exc)
            tmp_vec.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)
            return False

    @staticmethod
    def delete(folder_path) -> None:
        """Remove vector and metadata files for a folder."""
        vec_path, meta_path = VectorDatabase.get_paths(folder_path)
        vec_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    @staticmethod
    def prune(folder_path, existing_files: set[str]) -> int:
        """Remove metadata/vector entries whose files no longer exist on disk."""
        vec_path, meta_path = VectorDatabase.get_paths(folder_path)
        if not vec_path.exists() or not meta_path.exists():
            return 0
        tmp_vec  = vec_path.with_suffix(".tmp.npy")
        tmp_meta = meta_path.with_suffix(".tmp.csv")
        try:
            meta_df = pd.read_csv(meta_path)
            keep_mask = meta_df["path"].isin(existing_files)
            removed = int((~keep_mask).sum())
            if removed == 0:
                return 0
            keep_idx = list(meta_df.index[keep_mask])
            vectors = np.load(vec_path)   # fully in-memory
            np.save(tmp_vec, np.array(vectors[keep_idx]))
            meta_df[keep_mask].reset_index(drop=True).to_csv(tmp_meta, index=False)
            tmp_vec.replace(vec_path)
            tmp_meta.replace(meta_path)
            return removed
        except Exception as exc:
            log.error("VectorDatabase.prune: error pruning %s: %s", folder_path, exc)
            tmp_vec.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)
            return 0

    @staticmethod
    def count_indexed(folder_path) -> int:
        """Return the number of entries currently in the metadata CSV."""
        _, meta_path = VectorDatabase.get_paths(folder_path)
        if not meta_path.exists():
            return 0
        try:
            return max(0, sum(1 for _ in open(meta_path)) - 1)  # minus header
        except Exception:
            return 0

    @staticmethod
    def read_metadata(folder_path) -> dict[str, float]:
        """Return {path: mtime} mapping from the saved metadata CSV."""
        _, meta_path = VectorDatabase.get_paths(folder_path)
        if not meta_path.exists():
            return {}
        try:
            df = pd.read_csv(meta_path)
            return dict(zip(df["path"], df["mtime"]))
        except Exception:
            return {}
