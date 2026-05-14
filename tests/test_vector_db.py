"""
tests/test_vector_db.py
Unit tests for core/vector_db.py

Covers: save/load round-trip, append/deduplication, prune, delete,
count_indexed, read_metadata, and atomic-write safety.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# vector_db is reloaded by the vector_db_folder fixture so VECTOR_ROOT is
# redirected to tmp_path — always import after the fixture is active.
from core.vector_db import VectorDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vectors(n: int, dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_meta(n: int, prefix: str = "file") -> list[dict]:
    return [{"path": f"/img/{prefix}_{i}.png", "mtime": float(i)} for i in range(n)]


# ---------------------------------------------------------------------------
# get_paths
# ---------------------------------------------------------------------------

class TestGetPaths:
    def test_returns_two_paths(self, vector_db_folder):
        vec_p, meta_p = VectorDatabase.get_paths(vector_db_folder)
        assert vec_p.suffix == ".npy"
        assert meta_p.suffix == ".csv"

    def test_creates_db_dir(self, vector_db_folder):
        vec_p, _ = VectorDatabase.get_paths(vector_db_folder)
        assert vec_p.parent.exists()


# ---------------------------------------------------------------------------
# save + load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_fresh_save_then_load(self, vector_db_folder):
        vecs = _make_vectors(3)
        meta = _make_meta(3)
        assert VectorDatabase.save(vector_db_folder, vecs, meta)

        loaded_vecs, loaded_df = VectorDatabase.load(vector_db_folder)
        assert loaded_vecs is not None
        assert loaded_df is not None
        assert loaded_vecs.shape == (3, 512)
        assert len(loaded_df) == 3

    def test_vectors_preserved_exactly(self, vector_db_folder):
        vecs = _make_vectors(5)
        meta = _make_meta(5)
        VectorDatabase.save(vector_db_folder, vecs, meta)

        loaded, _ = VectorDatabase.load(vector_db_folder)
        np.testing.assert_array_almost_equal(loaded, vecs)

    def test_metadata_paths_preserved(self, vector_db_folder):
        meta = _make_meta(4)
        VectorDatabase.save(vector_db_folder, _make_vectors(4), meta)

        _, df = VectorDatabase.load(vector_db_folder)
        assert list(df["path"]) == [r["path"] for r in meta]

    def test_load_missing_returns_none_none(self, vector_db_folder):
        vecs, df = VectorDatabase.load(vector_db_folder)
        assert vecs is None
        assert df is None

    def test_save_returns_true_on_success(self, vector_db_folder):
        result = VectorDatabase.save(vector_db_folder, _make_vectors(2), _make_meta(2))
        assert result is True


# ---------------------------------------------------------------------------
# append / deduplication
# ---------------------------------------------------------------------------

class TestAppendAndDeduplication:
    def test_append_new_entries(self, vector_db_folder):
        VectorDatabase.save(vector_db_folder, _make_vectors(3), _make_meta(3))
        VectorDatabase.save(vector_db_folder, _make_vectors(2, seed=1),
                            _make_meta(2, prefix="extra"))

        vecs, df = VectorDatabase.load(vector_db_folder)
        assert len(df) == 5
        assert vecs.shape[0] == 5

    def test_duplicate_path_is_replaced_not_appended(self, vector_db_folder):
        meta = _make_meta(3)
        VectorDatabase.save(vector_db_folder, _make_vectors(3), meta)

        # Re-save the same path with a different vector
        new_vec = _make_vectors(1, seed=99)
        VectorDatabase.save(vector_db_folder, new_vec, [meta[0]])

        _, df = VectorDatabase.load(vector_db_folder)
        # Still 3 entries — no duplicates
        assert len(df) == 3
        assert list(df["path"]).count(meta[0]["path"]) == 1

    def test_updated_vector_is_new_value(self, vector_db_folder):
        """After re-saving, the stored vector equals the new one."""
        meta = _make_meta(1)
        VectorDatabase.save(vector_db_folder, _make_vectors(1), meta)

        new_vec = _make_vectors(1, seed=77)
        VectorDatabase.save(vector_db_folder, new_vec, meta)

        loaded, _ = VectorDatabase.load(vector_db_folder)
        np.testing.assert_array_almost_equal(loaded, new_vec)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_files(self, vector_db_folder):
        VectorDatabase.save(vector_db_folder, _make_vectors(2), _make_meta(2))
        VectorDatabase.delete(vector_db_folder)

        vec_p, meta_p = VectorDatabase.get_paths(vector_db_folder)
        assert not vec_p.exists()
        assert not meta_p.exists()

    def test_delete_on_empty_folder_is_safe(self, vector_db_folder):
        VectorDatabase.delete(vector_db_folder)  # should not raise

    def test_load_after_delete_returns_none_none(self, vector_db_folder):
        VectorDatabase.save(vector_db_folder, _make_vectors(2), _make_meta(2))
        VectorDatabase.delete(vector_db_folder)
        vecs, df = VectorDatabase.load(vector_db_folder)
        assert vecs is None and df is None


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

class TestPrune:
    def test_prune_removes_missing_entries(self, vector_db_folder):
        meta = _make_meta(4)
        VectorDatabase.save(vector_db_folder, _make_vectors(4), meta)

        # Keep only paths 0 and 1
        keep = {meta[0]["path"], meta[1]["path"]}
        removed = VectorDatabase.prune(vector_db_folder, keep)

        assert removed == 2
        _, df = VectorDatabase.load(vector_db_folder)
        assert len(df) == 2
        assert set(df["path"]) == keep

    def test_prune_keeps_vectors_aligned(self, vector_db_folder):
        vecs = _make_vectors(3)
        meta = _make_meta(3)
        VectorDatabase.save(vector_db_folder, vecs, meta)

        keep = {meta[2]["path"]}
        VectorDatabase.prune(vector_db_folder, keep)

        loaded, df = VectorDatabase.load(vector_db_folder)
        assert loaded.shape == (1, 512)
        np.testing.assert_array_almost_equal(loaded[0], vecs[2])

    def test_prune_nothing_to_remove_returns_zero(self, vector_db_folder):
        meta = _make_meta(3)
        VectorDatabase.save(vector_db_folder, _make_vectors(3), meta)
        keep = {r["path"] for r in meta}
        assert VectorDatabase.prune(vector_db_folder, keep) == 0

    def test_prune_empty_db_returns_zero(self, vector_db_folder):
        assert VectorDatabase.prune(vector_db_folder, {"anything"}) == 0

    def test_prune_all_entries(self, vector_db_folder):
        meta = _make_meta(3)
        VectorDatabase.save(vector_db_folder, _make_vectors(3), meta)
        removed = VectorDatabase.prune(vector_db_folder, set())
        assert removed == 3
        _, df = VectorDatabase.load(vector_db_folder)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# count_indexed
# ---------------------------------------------------------------------------

class TestCountIndexed:
    def test_empty_returns_zero(self, vector_db_folder):
        assert VectorDatabase.count_indexed(vector_db_folder) == 0

    def test_matches_saved_count(self, vector_db_folder):
        n = 7
        VectorDatabase.save(vector_db_folder, _make_vectors(n), _make_meta(n))
        assert VectorDatabase.count_indexed(vector_db_folder) == n

    def test_count_after_append(self, vector_db_folder):
        VectorDatabase.save(vector_db_folder, _make_vectors(3), _make_meta(3))
        VectorDatabase.save(vector_db_folder, _make_vectors(2, seed=1),
                            _make_meta(2, prefix="b"))
        assert VectorDatabase.count_indexed(vector_db_folder) == 5

    def test_count_after_prune(self, vector_db_folder):
        meta = _make_meta(5)
        VectorDatabase.save(vector_db_folder, _make_vectors(5), meta)
        keep = {meta[0]["path"], meta[1]["path"]}
        VectorDatabase.prune(vector_db_folder, keep)
        assert VectorDatabase.count_indexed(vector_db_folder) == 2


# ---------------------------------------------------------------------------
# read_metadata
# ---------------------------------------------------------------------------

class TestReadMetadata:
    def test_empty_returns_empty_dict(self, vector_db_folder):
        assert VectorDatabase.read_metadata(vector_db_folder) == {}

    def test_returns_path_to_mtime_mapping(self, vector_db_folder):
        meta = _make_meta(3)
        VectorDatabase.save(vector_db_folder, _make_vectors(3), meta)
        result = VectorDatabase.read_metadata(vector_db_folder)
        assert isinstance(result, dict)
        for r in meta:
            assert r["path"] in result
            assert result[r["path"]] == pytest.approx(r["mtime"])

    def test_updated_entry_has_new_mtime(self, vector_db_folder):
        meta = [{"path": "/img/a.png", "mtime": 1000.0}]
        VectorDatabase.save(vector_db_folder, _make_vectors(1), meta)

        updated = [{"path": "/img/a.png", "mtime": 9999.0}]
        VectorDatabase.save(vector_db_folder, _make_vectors(1, seed=1), updated)

        result = VectorDatabase.read_metadata(vector_db_folder)
        assert result["/img/a.png"] == pytest.approx(9999.0)


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------

class TestAtomicWrites:
    def test_no_tmp_files_left_after_successful_save(self, vector_db_folder):
        VectorDatabase.save(vector_db_folder, _make_vectors(3), _make_meta(3))
        vec_p, meta_p = VectorDatabase.get_paths(vector_db_folder)
        assert not vec_p.with_suffix(".tmp.npy").exists()
        assert not meta_p.with_suffix(".tmp.csv").exists()

    def test_original_intact_if_tmp_deleted_mid_save(self, vector_db_folder, monkeypatch):
        """
        Simulate an interrupted save by making tmp_vec.replace() raise.
        The original files must remain readable.
        """
        # First save — establishes a good baseline
        VectorDatabase.save(vector_db_folder, _make_vectors(3), _make_meta(3))

        original_vecs, _ = VectorDatabase.load(vector_db_folder)

        # Patch Path.replace to fail on the second call (the vec rename)
        call_count = {"n": 0}
        original_replace = Path.replace

        def fail_on_second(self, target):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated disk full")
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", fail_on_second)

        result = VectorDatabase.save(
            vector_db_folder, _make_vectors(2, seed=99), _make_meta(2, prefix="new")
        )
        assert result is False

        # Restore and verify original data is still intact
        monkeypatch.undo()
        loaded, df = VectorDatabase.load(vector_db_folder)
        assert loaded is not None
        assert len(df) == 3
        np.testing.assert_array_almost_equal(loaded, original_vecs)
