"""
core/algorithms.py
Pure search / comparison functions with no GUI dependencies.
All functions are designed to be called from worker threads.
"""

import hashlib
import logging
import os
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity

log = logging.getLogger(__name__)

from config import (
    QUICK_HASH_CHUNK_SIZE,
    FULL_HASH_CHUNK_SIZE,
    SIFT_CONTRAST_THRESHOLD,
    SIFT_EDGE_THRESHOLD,
    SIFT_MIN_MATCHES,
    SIFT_RATIO_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def calculate_image_hash(image_path) -> str:
    """Full SHA-256 file hash for exact-duplicate detection."""
    hasher = hashlib.sha256()
    with open(image_path, "rb") as fh:
        while chunk := fh.read(FULL_HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_quick_hash(image_path) -> str | None:
    """
    Partial hash of first + middle (+ last for large files) 8 KB chunks.
    Much faster than a full hash; used as a pre-filter.
    """
    file_size = os.path.getsize(image_path)
    hasher = hashlib.sha256()
    try:
        with open(image_path, "rb") as fh:
            hasher.update(fh.read(QUICK_HASH_CHUNK_SIZE))
            if file_size > QUICK_HASH_CHUNK_SIZE * 2:
                fh.seek(file_size // 2)
                hasher.update(fh.read(QUICK_HASH_CHUNK_SIZE))
            if file_size > 1024 * 1024:
                fh.seek(-QUICK_HASH_CHUNK_SIZE, os.SEEK_END)
                hasher.update(fh.read(QUICK_HASH_CHUNK_SIZE))
        return hasher.hexdigest()
    except Exception as exc:
        log.warning("quick_hash: error reading %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Histogram similarity
# ---------------------------------------------------------------------------

def calculate_histogram(image: np.ndarray) -> np.ndarray:
    """Return HSV hue-channel histogram for *image* (BGR input)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Intersection-based histogram similarity in [0, 100]."""
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    total = hist1.sum() + hist2.sum() - intersection
    return float((intersection / total) * 100) if total > 0 else 0.0


def histogram_search(query_path: str, files: list, threshold: float = 0.0):
    """
    CLI-friendly histogram search.
    Prints results to stdout; no GUI interaction.
    """
    query_img = cv2.imdecode(np.fromfile(query_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if query_img is None:
        log.error("histogram_search: cannot read query image: %s", query_path)
        return

    hist1 = calculate_histogram(query_img)
    for file in files:
        if str(file) == query_path:
            continue
        try:
            img = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            hist2 = calculate_histogram(img)
            sim = compare_histograms(hist1, hist2)
            if sim >= threshold:
                log.info("%s  Similarity: %.2f%%", file, sim)
        except Exception as exc:
            log.warning("histogram_search: error processing %s: %s", file, exc)


# ---------------------------------------------------------------------------
# SSIM comparison
# ---------------------------------------------------------------------------

def compare_ssim(path1: str, path2: str) -> float | None:
    """
    Compute SSIM between two same-size images.
    Returns score in [0, 100] or None if images differ in size or cannot be read.
    """
    img1 = cv2.imdecode(np.fromfile(path1, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imdecode(np.fromfile(path2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        return None
    if img1.shape != img2.shape:
        return None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = structural_similarity(gray1, gray2, full=True)
    return score * 100.0


# ---------------------------------------------------------------------------
# SIFT comparison
# ---------------------------------------------------------------------------

def initialize_sift():
    """Return a cv2.SIFT instance, or None on failure."""
    try:
        return cv2.SIFT_create(
            contrastThreshold=SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_EDGE_THRESHOLD,
        )
    except AttributeError:
        try:
            from cv2.xfeatures2d import SIFT_create  # type: ignore
            return SIFT_create()
        except ImportError:
            return None


def sift_similarity(
    query_descriptors: np.ndarray,
    target_path,
    bf_matcher,
    min_matches: int = SIFT_MIN_MATCHES,
    ratio_thresh: float = SIFT_RATIO_THRESHOLD,
) -> float | None:
    """
    Compare query descriptors against an image file.
    Returns similarity in [0, 100] or None if insufficient matches.
    """
    sift = initialize_sift()
    if sift is None:
        return None

    target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        return None

    _, des2 = sift.detectAndCompute(target_img, None)
    if des2 is None or len(des2) < min_matches:
        return None

    matches = bf_matcher.knnMatch(query_descriptors, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good) >= min_matches:
        return (len(good) / len(query_descriptors)) * 100.0
    return None


# ---------------------------------------------------------------------------
# Vector / cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity_batch(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Dot-product similarity for L2-normalised vectors (== cosine sim)."""
    return np.dot(vectors, query)


def score_vector_similarity(raw_sim: float,
                             baseline: float | None = None,
                             exponent: float | None = None) -> float:
    """
    Convert a raw cosine similarity value (0–1) to a display percentage (0–100)
    that is spread across a wider range than a plain multiplication by 100.

    Formula
    -------
    1. Fixed-baseline rescale:
           score = max(0, (raw_sim - baseline) / (1 - baseline))
       Maps [baseline, 1.0] → [0, 1].  Any similarity at or below the baseline
       becomes 0 — these images are considered unrelated.
    2. Power-curve spread:
           score = score ** exponent
       Pushes mid-range values down while leaving high values near 1, creating
       more visual separation between "pretty similar" and "very similar".
    3. Multiply by 100 for the final percentage.

    Both parameters default to the values in config.py so they can be tuned
    without touching this function.
    """
    from config import VECTOR_SIMILARITY_BASELINE, VECTOR_SCORE_EXPONENT
    if baseline is None:
        baseline = VECTOR_SIMILARITY_BASELINE
    if exponent is None:
        exponent = VECTOR_SCORE_EXPONENT
    rescaled = max(0.0, (raw_sim - baseline) / (1.0 - baseline))
    return (rescaled ** exponent) * 100.0
