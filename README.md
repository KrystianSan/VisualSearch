# VisualSearch

A desktop image similarity search tool built with Python, customtkinter, and PyTorch. Search large image collections using six different algorithms, manage folder indexes, preview results side by side, and save sessions — all from a single window. A full command-line interface is also available for scripting and automation.

---

## What's new in v2.0.0 - beta:

- ### **Redesigned UI**
- ### **Improved UX - info and warning popups, event logging**
- ### **Fixed all major bugs**
- ### **Added results sorting**
- ### **Extended CLI functionality**
- ### **Added tests**

---

## Features

- **Six search modes** — vector similarity, histogram, exact duplicate, duplicate groups, SSIM, and SIFT
- **Resumable vector processing** — indexing picks up from where it left off after interruption or restart
- **Per-folder subfolder control** — enable recursive scanning individually per folder
- **Incomplete index detection** — warns before a vector search if any folder is only partially processed
- **Save / Load sessions** — full state saved as JSON (mode, query image, folders, threshold, all results); export to CSV also available; legacy CSV files still load
- **Missing file warnings** — load warns about folders or result files that no longer exist on disk
- **Sortable results** — all treeview columns are clickable; duplicate groups support sorting by file count and total size
- **Preview panes** — query and selected image shown side by side; duplicate groups show the first two files of the selected group
- **Three languages** — English, Spanish, Polish (all UI strings translated, easily extendable via `i18n.py`)
- **Side-by-side image preview** — With a click of a button you can visually compare two images in a separate, re-sizeable window: one selected by user from the search results and the query image
- **Light / dark theme** — custom CTk navbar follows system appearance automatically
- **Full CLI** — all six search modes available headlessly via `cli.py` with JSON/CSV output
- **Test suite** — 67 unit and integration tests covering `core/` and `utils/`

---

## Project Structure

```
VisualSearch/
├── main.py                    # GUI entry point
├── cli.py                     # Command-line interface (all search modes)
├── config.py                  # All constants, paths, and defaults
├── i18n.py                    # UI string translations (English, Spanish, Polish)
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Development / testing dependencies
├── pytest.ini                 # pytest configuration
│
├── ui/                        # Presentation layer
│   ├── app.py                 # Main window — layout, widgets, event wiring
│   ├── search_controller.py   # Thread management, folder processing, save/load
│   └── spinbox.py             # Custom integer spinbox widget
│
├── search/                    # Search strategy modules (GUI-facing)
│   ├── base.py                # BaseSearch — shared state and thread-safe UI helpers
│   ├── vector_search.py       # ResNet18 cosine similarity
│   ├── histogram_search.py    # HSV histogram intersection
│   ├── duplicate_search.py    # SHA-256 exact-match (single query image)
│   ├── duplicate_groups.py    # Hash-based group clustering (no query needed)
│   ├── ssim_search.py         # Structural Similarity Index
│   └── sift_search.py         # SIFT keypoint matching
│
├── core/                      # Domain logic — no GUI imports
│   ├── algorithms.py          # Hash, histogram, SSIM, SIFT, cosine similarity functions
│   ├── feature_extractor.py   # ResNet18 wrapper — 512-dim L2-normalised vectors
│   └── vector_db.py           # Persistent .npy + .csv storage with atomic writes
│
├── utils/
│   ├── file_utils.py          # Image listing, file counting, OS explorer integration
│   └── image_utils.py         # PIL/OpenCV helpers, canvas-fit resizing
│
├── tests/
│   ├── conftest.py            # Shared fixtures (tmp dirs, synthetic images, vector_db patch)
│   ├── test_algorithms.py     # Hashing, histogram, cosine similarity, scoring, SSIM, SIFT (38 tests)
│   ├── test_vector_db.py      # Save/load, append, prune, atomic writes (27 tests)
│   ├── test_file_utils.py     # Image listing, counting, per-folder overrides (19 tests)
│   └── test_image_utils.py    # PIL/OpenCV loaders, dimensions, geometry (15 tests)
│
└── assets/                    # Icons and static resources (future)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **SIFT search** requires `opencv-contrib-python` instead of `opencv-python`. If you only need the other five modes, the standard `opencv-python` package is sufficient.

> **PyTorch** can be large. Install a CPU-only build if you don't have a GPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Running

### GUI

```bash
python main.py
```

### CLI

```bash
python cli.py --help
```

---

## Search Modes

| Mode | Query image | Requires processing | Description |
|---|---|---|---|
| **Vector Similarity** | ✅ Yes | ✅ Yes | ResNet18 cosine similarity on 512-dim feature vectors. Most accurate for visually similar images. |
| **Histogram Similarity** | ✅ Yes | ❌ No | HSV hue-channel histogram intersection. Fast; good for colour-similar images. |
| **Find Duplicates** | ✅ Yes | ❌ No | SHA-256 exact hash match against a single query image. |
| **Duplicate Groups** | ❌ No | ❌ No | Groups all exact duplicates across selected folders. No query image needed. |
| **SSIM Compare** | ✅ Yes | ❌ No | Structural Similarity Index. Only compares images with identical pixel dimensions to the query. |
| **SIFT Compare** | ✅ Yes | ❌ No | Keypoint feature matching. Good for transformed, rotated, or partially cropped images. |

---

## Vector Processing

Vector Similarity requires folders to be indexed before searching:

1. Add folders in the **Folders** bar at the top
2. Toggle **subfolders** per folder as needed (each row has its own checkbox)
3. Click **▶ Process Folders** — the button turns red and becomes **⏹ Stop Processing**
4. While processing you can still upload a query image
5. Processing resumes automatically from the correct offset if interrupted or the app is restarted

The index is stored in `vector_db/` mirroring your folder structure. Files are re-indexed automatically if their modification time changes. Deleted files are pruned from the index on the next complete run.

You can also build the index from the command line:

```bash
python cli.py index ./photos --recursive
```

---

## Save / Load Results

### Saving

Click **Save Results** to open a save dialog. Two formats are available:

- **`.json`** — full session file including mode, query image path, folders, threshold, timestamp, and the complete result set. Recommended for later reloading.
- **`.csv`** — flat export for use in spreadsheets. Standard results export as `Path, Similarity, Mode, Query Image`. Duplicate groups are flattened with a `Group` column.

The default filename is `{mode}_{timestamp}.json`.

### Loading

Click **Load Results** to open a file. The session state is restored:

- Search mode and similarity threshold are set
- Folders that still exist are re-added to the folder bar
- The query image is reloaded if it still exists on disk
- The treeview is rebuilt in the correct layout (standard or groups)
- A warning dialog lists any folders, query images, or result files that are no longer found on disk

Legacy `.csv` files from older versions still load correctly.

---

## Command-Line Interface

`cli.py` provides headless access to all search modes with consistent options and JSON/CSV output. No GUI is launched.

### Global options

| Flag | Description |
|---|---|
| `-v`, `--verbose` | Enable DEBUG-level logging |

### Subfolder scanning

By default every subcommand scans only the **top-level** of each folder. Pass `-r` / `--recursive` to include all nested subfolders.

```bash
# Top-level only (default)
python cli.py histogram query.jpg ./photos

# Recursive — scans all subfolders
python cli.py histogram query.jpg ./photos --recursive
```

> **Important for vector search:** the index must be built with the same `-r` setting used for searching. If you indexed with `--recursive` but search without it (or vice versa), results will be incomplete. The GUI handles this automatically via per-folder subfolder checkboxes.

### Subcommands

#### `index` — build or update the vector index

```bash
python cli.py index FOLDER [FOLDER ...] [-r] [--no-prune]
```

```bash
python cli.py index ./photos --recursive
python cli.py index ./photos ./archive --recursive
```

#### `vector` — ResNet18 cosine similarity search

```bash
python cli.py vector QUERY_IMAGE FOLDER [FOLDER ...] [-r] [-t THRESHOLD] [--top N] [-o FILE]
```

```bash
python cli.py vector query.jpg ./photos -r -t 75 --top 20
python cli.py vector query.jpg ./photos -r -o results.json
```

#### `histogram` — HSV histogram similarity

```bash
python cli.py histogram QUERY_IMAGE FOLDER [FOLDER ...] [-r] [-t THRESHOLD] [--top N] [-o FILE]
```

```bash
python cli.py histogram query.jpg ./photos -r -t 60 -o results.csv
```

#### `duplicates` — exact duplicate finder (single query)

```bash
python cli.py duplicates QUERY_IMAGE FOLDER [FOLDER ...] [-r] [-o FILE]
```

#### `groups` — find all duplicate groups (no query needed)

```bash
python cli.py groups FOLDER [FOLDER ...] [-r] [-o FILE]
```

```bash
python cli.py groups ./photos ./archive -r -o duplicates.json
```

#### `sift` — SIFT keypoint matching

```bash
python cli.py sift QUERY_IMAGE FOLDER [FOLDER ...] [-r] [-t THRESHOLD] [--top N] [-o FILE]
```

#### `ssim` — Structural Similarity Index

```bash
python cli.py ssim QUERY_IMAGE FOLDER [FOLDER ...] [-r] [-t THRESHOLD] [--top N] [-o FILE]
```

### Output formats

All subcommands print results to stdout by default. Pass `-o FILE` to save instead:

- `-o results.json` — structured JSON
- `-o results.csv` — flat CSV (duplicate groups are flattened with a `Group` column)

---

## Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# With coverage report
pytest --cov=core --cov=utils --cov-report=term-missing
```

**99 tests** across 4 files — all targeting `core/` and `utils/` which have no GUI dependencies:

| File | Tests | What it covers |
|---|---|---|
| `test_algorithms.py` | 38 | SHA-256 hashing, quick hash, histogram similarity, cosine similarity, `score_vector_similarity` (baseline rescaling + power curve), SSIM, SIFT (skipped if unavailable) |
| `test_vector_db.py` | 27 | Save/load round-trip, append, deduplication, delete, prune, count, read metadata, atomic write safety |
| `test_file_utils.py` | 19 | Flat and recursive listing, per-folder subfolder overrides, deduplication, file counting, error handling |
| `test_image_utils.py` | 15 | PIL and OpenCV loaders, dimension reading, canvas-fit geometry |

---

## UI Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Navbar — Settings (language selector), follows system theme     │
├─────────────────────────────────────────────────────────────────┤
│  Folders bar — add/remove/reorder folders, per-folder subfolders │
├──────────────┬──────────────────────────────────────────────────┤
│              │  Results treeview (sortable columns)             │
│  Left        │  Filename ↕ | Full Path ↕ | Similarity ↕        │
│  Sidebar     ├──────────────────────────────────────────────────┤
│              │  Preview strip                                    │
│  • Upload    │  [Query image]      [Selected image]             │
│  • Search    │                                                   │
│  • Process   │                                                   │
│  • Save/Load │                                                   │
├──────────────┴──────────────────────────────────────────────────┤
│  Status bar — progress bar + live processing/search status       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Notes

- **Thread safety** — all heavy work (indexing, searching, hashing) runs on daemon threads; all UI updates go through `root.after(0, ...)` to the main thread
- **No GUI in core** — `core/` and `search/` modules have zero tkinter/customtkinter imports; `search/base.py` communicates back to the UI only via the controller
- **Atomic vector writes** — `vector_db.py` writes to `.tmp.npy` / `.tmp.csv` then renames atomically, preventing corrupt index files on interruption
- **Resume detection** — `VectorDatabase.count_indexed()` reads the existing metadata row count before indexing begins; cached files (mtime match) are skipped without double-counting in the progress bar
- **Vector scoring** — raw cosine similarities are rescaled using a fixed baseline (`VECTOR_SIMILARITY_BASELINE`) and a power-curve exponent (`VECTOR_SCORE_EXPONENT`) to produce visually spread percentage scores. Both constants are in `config.py`. No adaptive threshold — the user's spinbox value is the only filter.
- **Logging** — all modules use Python's `logging` with `getLogger(__name__)`; pass `-v` on the CLI or configure a handler in your own code for full debug output
- **Single source of truth** — all constants, paths, supported extensions, and defaults live in `config.py`

---

## Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.ppm` · `.pgm`

---

## Dependencies

| Package | Purpose |
|---|---|
| `customtkinter` | Modern themed UI widgets |
| `Pillow` | Image loading and display |
| `opencv-python` | Image reading, histograms, SSIM, SIFT |
| `scikit-image` | SSIM computation |
| `numpy` | Vector arithmetic |
| `pandas` | Metadata CSV read/write |
| `torch` + `torchvision` | ResNet18 feature extraction |
| `send2trash` | Safe deletion to system Recycle Bin |
| `darkdetect` | System appearance mode detection |
