"""
tests/test_algorithms.py
Unit tests for core/algorithms.py

Covers: hashing, histogram similarity, cosine similarity, score_vector_similarity
(baseline rescaling + power-curve spread), SSIM, and SIFT (skipped when
opencv-contrib-python is absent).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.algorithms import (
    calculate_histogram,
    calculate_image_hash,
    calculate_quick_hash,
    compare_histograms,
    compare_ssim,
    cosine_similarity_batch,
    score_vector_similarity,
    initialize_sift,
)


# ===========================================================================
# Hashing
# ===========================================================================

class TestCalculateImageHash:
    def test_same_file_same_hash(self, img_pair):
        """Identical files produce the same SHA-256 digest."""
        q, t = img_pair
        assert calculate_image_hash(q) == calculate_image_hash(t)

    def test_different_files_different_hash(self, img_pair_different):
        """Clearly different files produce different digests."""
        q, t = img_pair_different
        assert calculate_image_hash(q) != calculate_image_hash(t)

    def test_hash_is_hex_string(self, img_pair):
        q, _ = img_pair
        h = calculate_image_hash(q)
        assert isinstance(h, str)
        assert len(h) == 64          # SHA-256 = 32 bytes = 64 hex chars
        int(h, 16)                   # must be valid hex

    def test_matches_manual_sha256(self, img_pair):
        """Result matches a manually computed SHA-256."""
        q, _ = img_pair
        with open(q, "rb") as fh:
            expected = hashlib.sha256(fh.read()).hexdigest()
        assert calculate_image_hash(q) == expected

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            calculate_image_hash(tmp_path / "no_such_file.png")


class TestCalculateQuickHash:
    def test_returns_string_for_valid_file(self, img_pair):
        q, _ = img_pair
        result = calculate_quick_hash(q)
        assert isinstance(result, str) and len(result) == 64

    def test_same_file_consistent(self, img_pair):
        q, _ = img_pair
        assert calculate_quick_hash(q) == calculate_quick_hash(q)

    def test_identical_files_same_hash(self, img_pair):
        q, t = img_pair
        assert calculate_quick_hash(q) == calculate_quick_hash(t)

    def test_different_files_different_hash(self, img_pair_different):
        q, t = img_pair_different
        assert calculate_quick_hash(q) != calculate_quick_hash(t)

    def test_nonexistent_file_returns_none(self, tmp_path):
        result = calculate_quick_hash(tmp_path / "ghost.png")
        assert result is None


# ===========================================================================
# Histogram similarity
# ===========================================================================

class TestCalculateHistogram:
    def test_returns_ndarray(self, img_pair):
        q, _ = img_pair
        img = cv2.imread(str(q))
        hist = calculate_histogram(img)
        assert isinstance(hist, np.ndarray)
        assert hist.shape == (256, 1)

    def test_histogram_sums_to_pixel_count(self, img_pair):
        """For a solid-colour image every pixel maps to one hue bucket."""
        q, _ = img_pair
        img = cv2.imread(str(q))
        hist = calculate_histogram(img)
        # Solid colour → all pixels in one bucket, total == pixel count
        total_pixels = img.shape[0] * img.shape[1]
        assert int(hist.sum()) == total_pixels


class TestCompareHistograms:
    def test_identical_returns_100(self, img_pair):
        q, _ = img_pair
        img = cv2.imread(str(q))
        hist = calculate_histogram(img)
        score = compare_histograms(hist, hist)
        assert score == pytest.approx(100.0)

    def test_range_0_to_100(self, img_pair_different):
        q, t = img_pair_different
        h1 = calculate_histogram(cv2.imread(str(q)))
        h2 = calculate_histogram(cv2.imread(str(t)))
        score = compare_histograms(h1, h2)
        assert 0.0 <= score <= 100.0

    def test_different_colours_low_score(self, img_pair_different):
        """Black vs white should score very low."""
        q, t = img_pair_different
        h1 = calculate_histogram(cv2.imread(str(q)))
        h2 = calculate_histogram(cv2.imread(str(t)))
        assert compare_histograms(h1, h2) < 50.0

    def test_zero_histogram_returns_zero(self):
        h = np.zeros((256, 1), dtype=np.float32)
        assert compare_histograms(h, h) == 0.0


# ===========================================================================
# Cosine similarity (vectors)
# ===========================================================================

class TestCosineSimilarityBatch:
    def test_identical_unit_vector_gives_one(self):
        v = np.ones(512, dtype=np.float32)
        v /= np.linalg.norm(v)
        result = cosine_similarity_batch(v.reshape(1, -1), v)
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_give_zero(self):
        a = np.zeros(512, dtype=np.float32); a[0] = 1.0
        b = np.zeros(512, dtype=np.float32); b[1] = 1.0
        result = cosine_similarity_batch(a.reshape(1, -1), b)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_batch_shape(self, random_unit_vectors):
        query = random_unit_vectors[0]
        sims = cosine_similarity_batch(random_unit_vectors, query)
        assert sims.shape == (5,)

    def test_self_similarity_is_max(self, random_unit_vectors):
        query = random_unit_vectors[0]
        sims = cosine_similarity_batch(random_unit_vectors, query)
        assert sims[0] == pytest.approx(max(sims), abs=1e-6)

    def test_values_bounded(self, random_unit_vectors):
        query = random_unit_vectors[0]
        sims = cosine_similarity_batch(random_unit_vectors, query)
        assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in sims)


# ===========================================================================
# score_vector_similarity
# ===========================================================================

class TestScoreVectorSimilarity:
    def test_perfect_similarity_gives_100(self):
        assert score_vector_similarity(1.0) == pytest.approx(100.0)

    def test_at_baseline_gives_zero(self):
        # Default baseline is 0.5 — anything at or below maps to 0
        assert score_vector_similarity(0.5) == pytest.approx(0.0)

    def test_below_baseline_clamped_to_zero(self):
        assert score_vector_similarity(0.3) == pytest.approx(0.0)
        assert score_vector_similarity(0.0) == pytest.approx(0.0)

    def test_midpoint_above_baseline(self):
        # raw=0.75 with baseline=0.5, exponent=2.0:
        # rescaled = (0.75 - 0.5) / 0.5 = 0.5  →  0.5^2 * 100 = 25.0
        result = score_vector_similarity(0.75, baseline=0.5, exponent=2.0)
        assert result == pytest.approx(25.0, abs=0.01)

    def test_output_range_zero_to_100(self):
        for raw in [0.0, 0.3, 0.5, 0.7, 0.85, 0.99, 1.0]:
            s = score_vector_similarity(raw)
            assert 0.0 <= s <= 100.0, f"score out of range for raw={raw}: {s}"

    def test_higher_similarity_gives_higher_score(self):
        s1 = score_vector_similarity(0.7)
        s2 = score_vector_similarity(0.85)
        s3 = score_vector_similarity(0.95)
        assert s1 < s2 < s3

    def test_exponent_spreads_scores(self):
        """Higher exponent pushes mid-range values lower."""
        raw = 0.75
        s_linear  = score_vector_similarity(raw, exponent=1.0)
        s_squared = score_vector_similarity(raw, exponent=2.0)
        s_cubic   = score_vector_similarity(raw, exponent=3.0)
        assert s_linear > s_squared > s_cubic

    def test_custom_baseline(self):
        # With baseline=0.8: raw=0.9 → rescaled = (0.9-0.8)/0.2 = 0.5 → 0.5^2*100 = 25
        result = score_vector_similarity(0.9, baseline=0.8, exponent=2.0)
        assert result == pytest.approx(25.0, abs=0.01)

    def test_returns_float(self):
        assert isinstance(score_vector_similarity(0.9), float)


# ===========================================================================
# SSIM
# ===========================================================================

class TestCompareSSIM:
    def test_identical_images_near_100(self, img_pair):
        q, t = img_pair
        score = compare_ssim(str(q), str(t))
        assert score is not None
        assert score == pytest.approx(100.0, abs=1.0)

    def test_different_images_lower_score(self, img_pair_different):
        q, t = img_pair_different
        score = compare_ssim(str(q), str(t))
        # Black vs white is maximally different
        assert score is not None
        assert score < 50.0

    def test_nonexistent_file_returns_none(self, img_pair, tmp_path):
        q, _ = img_pair
        assert compare_ssim(str(q), str(tmp_path / "ghost.png")) is None

    def test_different_size_returns_none(self, tmp_path):
        """SSIM is undefined for images with different dimensions."""
        small = tmp_path / "small.png"
        large = tmp_path / "large.png"
        cv2.imwrite(str(small), np.zeros((32, 32, 3), dtype=np.uint8))
        cv2.imwrite(str(large), np.zeros((64, 64, 3), dtype=np.uint8))
        assert compare_ssim(str(small), str(large)) is None

    def test_score_in_range(self, img_pair_different):
        q, t = img_pair_different
        score = compare_ssim(str(q), str(t))
        assert score is not None
        assert 0.0 <= score <= 100.0


# ===========================================================================
# SIFT  (skipped when opencv-contrib-python unavailable)
# ===========================================================================

@pytest.mark.skipif(initialize_sift() is None, reason="SIFT not available")
class TestSIFT:
    def test_initialize_sift_returns_object(self):
        sift = initialize_sift()
        assert sift is not None

    def test_identical_image_high_similarity(self, tmp_path):
        from core.algorithms import sift_similarity
        # Use a textured image so SIFT finds features
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        path = tmp_path / "textured.png"
        cv2.imwrite(str(path), img)

        sift = initialize_sift()
        _, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        if des is None:
            pytest.skip("No SIFT features found in synthetic image")

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        score = sift_similarity(des, path, bf)
        assert score is not None
        assert score > 0.0

    def test_nonexistent_target_returns_none(self, tmp_path):
        from core.algorithms import sift_similarity
        dummy_des = np.zeros((10, 128), dtype=np.float32)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        result = sift_similarity(dummy_des, tmp_path / "ghost.png", bf)
        assert result is None
