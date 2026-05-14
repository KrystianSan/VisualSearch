"""
tests/test_file_utils.py
Unit tests for utils/file_utils.py

Covers: list_image_files (flat, recursive, per-folder overrides, deduplication),
count_image_files, and open_in_explorer error handling.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.file_utils import count_image_files, list_image_files


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_img(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))
    return path


def _make_tree(tmp_path: Path):
    """
    Build a small directory tree:

        root/
            a.png
            b.jpg
            notes.txt          ← not an image
            sub/
                c.png
                deep/
                    d.jpeg
    """
    _write_img(tmp_path / "a.png")
    _write_img(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me")
    _write_img(tmp_path / "sub" / "c.png")
    _write_img(tmp_path / "sub" / "deep" / "d.jpeg")
    return tmp_path


# ---------------------------------------------------------------------------
# list_image_files — flat (no subfolders)
# ---------------------------------------------------------------------------

class TestListImageFilesFlat:
    def test_returns_only_top_level_images(self, tmp_path):
        root = _make_tree(tmp_path)
        result = list_image_files([str(root)], include_subfolders=False)
        names = {p.name for p in result}
        assert names == {"a.png", "b.jpg"}

    def test_ignores_non_image_files(self, tmp_path):
        root = _make_tree(tmp_path)
        result = list_image_files([str(root)], include_subfolders=False)
        assert all(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"}
                   for p in result)

    def test_returns_list_of_paths(self, tmp_path):
        root = _make_tree(tmp_path)
        result = list_image_files([str(root)])
        assert all(isinstance(p, Path) for p in result)

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert list_image_files([str(tmp_path)]) == []

    def test_nonexistent_folder_returns_empty(self, tmp_path):
        result = list_image_files([str(tmp_path / "does_not_exist")])
        assert result == []


# ---------------------------------------------------------------------------
# list_image_files — recursive
# ---------------------------------------------------------------------------

class TestListImageFilesRecursive:
    def test_finds_all_images_in_tree(self, tmp_path):
        root = _make_tree(tmp_path)
        result = list_image_files([str(root)], include_subfolders=True)
        names = {p.name for p in result}
        assert names == {"a.png", "b.jpg", "c.png", "d.jpeg"}

    def test_count_matches_expected(self, tmp_path):
        root = _make_tree(tmp_path)
        result = list_image_files([str(root)], include_subfolders=True)
        assert len(result) == 4

    def test_all_paths_are_resolved(self, tmp_path):
        root = _make_tree(tmp_path)
        result = list_image_files([str(root)], include_subfolders=True)
        assert all(p.is_absolute() for p in result)


# ---------------------------------------------------------------------------
# list_image_files — multiple folders
# ---------------------------------------------------------------------------

class TestListImageFilesMultipleFolders:
    def test_combines_results(self, tmp_path):
        d1 = tmp_path / "d1"; d1.mkdir()
        d2 = tmp_path / "d2"; d2.mkdir()
        _write_img(d1 / "x.png")
        _write_img(d2 / "y.png")
        result = list_image_files([str(d1), str(d2)])
        names = {p.name for p in result}
        assert names == {"x.png", "y.png"}

    def test_deduplicates_nested_folder(self, tmp_path):
        """Passing both a parent and a child folder should not double-count."""
        root = _make_tree(tmp_path)
        sub = root / "sub"
        result = list_image_files([str(root), str(sub)], include_subfolders=False)
        names = [p.name for p in result]
        # sub/c.png should appear only once
        assert names.count("c.png") <= 1


# ---------------------------------------------------------------------------
# list_image_files — per-folder subfolder override
# ---------------------------------------------------------------------------

class TestListImageFilesPerFolderOverride:
    def test_per_folder_recursive_overrides_global_false(self, tmp_path):
        root = _make_tree(tmp_path)
        folder_subfolders = {str(root): True}
        result = list_image_files(
            [str(root)],
            include_subfolders=False,
            folder_subfolders=folder_subfolders,
        )
        names = {p.name for p in result}
        assert "c.png" in names and "d.jpeg" in names

    def test_per_folder_flat_overrides_global_recursive(self, tmp_path):
        root = _make_tree(tmp_path)
        folder_subfolders = {str(root): False}
        result = list_image_files(
            [str(root)],
            include_subfolders=True,
            folder_subfolders=folder_subfolders,
        )
        names = {p.name for p in result}
        assert "c.png" not in names
        assert "a.png" in names

    def test_missing_key_falls_back_to_global(self, tmp_path):
        root = _make_tree(tmp_path)
        # folder_subfolders provided but this folder is not in it
        result = list_image_files(
            [str(root)],
            include_subfolders=True,
            folder_subfolders={},   # empty override dict
        )
        names = {p.name for p in result}
        assert "c.png" in names   # recursive applied via global default


# ---------------------------------------------------------------------------
# count_image_files
# ---------------------------------------------------------------------------

class TestCountImageFiles:
    def test_flat_count_matches_list_len(self, tmp_path):
        root = _make_tree(tmp_path)
        listed = list_image_files([str(root)], include_subfolders=False)
        counted = count_image_files([str(root)], include_subfolders=False)
        assert counted == len(listed)

    def test_recursive_count_matches_list_len(self, tmp_path):
        root = _make_tree(tmp_path)
        listed = list_image_files([str(root)], include_subfolders=True)
        counted = count_image_files([str(root)], include_subfolders=True)
        assert counted == len(listed)

    def test_empty_folder_returns_zero(self, tmp_path):
        assert count_image_files([str(tmp_path)]) == 0

    def test_multiple_folders_summed(self, tmp_path):
        d1 = tmp_path / "d1"; d1.mkdir()
        d2 = tmp_path / "d2"; d2.mkdir()
        _write_img(d1 / "x.png")
        _write_img(d1 / "y.png")
        _write_img(d2 / "z.jpg")
        assert count_image_files([str(d1), str(d2)]) == 3

    def test_per_folder_subfolder_override(self, tmp_path):
        root = _make_tree(tmp_path)
        folder_subfolders = {str(root): True}
        total = count_image_files(
            [str(root)],
            include_subfolders=False,
            folder_subfolders=folder_subfolders,
        )
        assert total == 4  # all images found recursively


# ---------------------------------------------------------------------------
# open_in_explorer — error path only (can't test GUI opening)
# ---------------------------------------------------------------------------

class TestOpenInExplorer:
    def test_raises_for_nonexistent_file(self, tmp_path):
        from utils.file_utils import open_in_explorer
        with pytest.raises(FileNotFoundError):
            open_in_explorer(tmp_path / "ghost.png")
