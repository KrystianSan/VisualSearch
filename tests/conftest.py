"""
tests/conftest.py
Shared pytest fixtures for the VisualSearch test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable from any working directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import cv2


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _make_png(path: Path, width: int = 64, height: int = 64,
              color: tuple = (100, 150, 200)) -> Path:
    """Write a solid-colour PNG and return its path."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _make_png_random(path: Path, width: int = 64, height: int = 64,
                     seed: int = 0) -> Path:
    """Write a reproducibly random PNG and return its path."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def img_dir(tmp_path: Path) -> Path:
    """A temporary directory pre-populated with three distinct PNG images."""
    _make_png(tmp_path / "red.png",   color=(0, 0, 200))
    _make_png(tmp_path / "green.png", color=(0, 200, 0))
    _make_png(tmp_path / "blue.png",  color=(200, 0, 0))
    return tmp_path


@pytest.fixture()
def img_pair(tmp_path: Path):
    """A (query, target) pair of identical PNG images."""
    q = _make_png(tmp_path / "query.png",  color=(128, 64, 32))
    t = _make_png(tmp_path / "target.png", color=(128, 64, 32))
    return q, t


@pytest.fixture()
def img_pair_different(tmp_path: Path):
    """A (query, target) pair of clearly different PNG images."""
    q = _make_png(tmp_path / "query.png",  color=(0,   0,   0))
    t = _make_png(tmp_path / "target.png", color=(255, 255, 255))
    return q, t


@pytest.fixture()
def random_unit_vectors() -> np.ndarray:
    """Five L2-normalised 512-dim vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.random((5, 512)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture()
def vector_db_folder(tmp_path: Path, monkeypatch):
    """
    A temporary folder with VectorDatabase's VECTOR_ROOT redirected into tmp_path
    so tests never write to the real vector_db/ directory.
    """
    import config
    fake_root = tmp_path / "vector_db"
    fake_root.mkdir()
    monkeypatch.setattr(config, "VECTOR_ROOT", fake_root)

    # Reload vector_db so it picks up the patched constant
    import importlib
    import core.vector_db
    importlib.reload(core.vector_db)

    folder = tmp_path / "images"
    folder.mkdir()
    yield folder

    # Restore original module state
    importlib.reload(core.vector_db)
