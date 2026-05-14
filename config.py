"""
Application-wide configuration and constants.
Centralizes all magic values, paths, and settings.
"""

from pathlib import Path

# --- Application metadata ---
APP_NAME = "VisualSearch"
APP_VERSION = "2.0.0"
APP_GEOMETRY = "1366x768"
APP_MIN_SIZE = (1000, 720)

# --- Vector database paths ---
# Stored next to config.py regardless of working directory.
# Do NOT commit vector_db/ — it is listed in .gitignore.
# Users regenerate it via Process Folders or: python cli.py index
_PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_ROOT   = _PROJECT_ROOT / "vector_db"
VECTOR_FILE = "vectors.npy"
METADATA_FILE = "metadata.csv"

# --- Supported image extensions ---
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"}

# --- Search defaults ---
DEFAULT_SIMILARITY_THRESHOLD = 50
DEFAULT_SEARCH_MODE = "Vector Similarity"
SEARCH_MODES = [
    "Vector Similarity",
    "Histogram Similarity",
    "Find Duplicates",
    "Duplicate Groups",
    "SSIM Compare",
    "SIFT Compare",
]

# --- Performance ---
MAX_SEARCH_RESULTS = 500
QUICK_HASH_CHUNK_SIZE = 8192          # 8 KB
FULL_HASH_CHUNK_SIZE = 131072         # 128 KB
STATUS_UPDATE_INTERVAL_MS = 100

# --- Default save directory ---
# Uses the user's home directory — works on Windows, macOS, and Linux.
DEFAULT_RESULTS_DIR = Path.home() / "VisualSearchResults"

# --- Vector similarity scoring ---
# Cosine distance ceiling — results with L2 distance above this are rejected
# as clearly unrelated before scoring begins.
# For L2-normalised vectors: dist = sqrt(2*(1-cos_sim)), so 0.5 ≈ cos_sim 0.875.
VECTOR_MAX_DISTANCE         = 0.5

# Fixed baseline for score rescaling.  Any raw cosine similarity at or below
# this value maps to 0 %.  1.0 maps to 100 %.  Keeps the displayed percentage
# meaningful (50 % really means "halfway between unrelated and identical").
VECTOR_SIMILARITY_BASELINE  = 0.5

# Power-curve exponent applied after baseline rescaling to spread scores apart.
# 1.0 = linear (no spread), 2.0 = quadratic (default — pushes midrange down),
# higher values increase spread further.
VECTOR_SCORE_EXPONENT       = 2.0

# --- SIFT parameters ---
SIFT_CONTRAST_THRESHOLD = 0.07
SIFT_EDGE_THRESHOLD = 10
SIFT_MIN_MATCHES = 5
SIFT_RATIO_THRESHOLD = 0.9

# --- Supported languages ---
SUPPORTED_LANGUAGES = ["English", "Spanish", "Polish"]
DEFAULT_LANGUAGE = "English"

# --- UI colors (light/dark tuples) ---
BUTTON_PROCESS_FG    = ("#FFD700", "#FFA500")
BUTTON_PROCESS_HOVER = ("#FFC800", "#FF8C00")
BUTTON_START_FG      = ("#2CC985", "#2FA572")
BUTTON_START_HOVER   = ("#239B6A", "#267A5A")
BUTTON_STOP_FG       = ("#FF4B4B", "#FF3333")
BUTTON_STOP_HOVER    = ("#CC0000", "#B22222")
