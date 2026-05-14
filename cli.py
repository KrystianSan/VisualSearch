"""
cli.py
Command-line interface for VisualSearch.

All search modes that do not require a GUI are available here.
Vector similarity requires folders to have been processed first (via
the 'index' subcommand below).

Usage examples
--------------
# Index a folder of images (builds the vector database)
python cli.py index ./photos --recursive

# Vector similarity search
python cli.py vector query.jpg ./photos --threshold 70 --recursive --top 20

# Histogram similarity search
python cli.py histogram query.jpg ./photos --threshold 60 --recursive

# Find exact duplicates of a single image
python cli.py duplicates query.jpg ./photos --recursive

# Find all duplicate groups across folders (no query image needed)
python cli.py groups ./photos ./more_photos --recursive

# SSIM comparison (same-size images only)
python cli.py ssim query.jpg ./photos --threshold 80 --recursive

# Output results to a file
python cli.py histogram query.jpg ./photos -o results.json
python cli.py histogram query.jpg ./photos -o results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SUPPORTED_EXTENSIONS
from utils.file_utils import list_image_files


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)s  %(message)s",
        level=level,
    )

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_output(results: list[dict], dest: str) -> None:
    """Write results list to JSON or CSV depending on file extension."""
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    if dest.lower().endswith(".csv"):
        if not results:
            log.warning("No results to write.")
            return
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    else:
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
    log.info("Results written to %s", dest)


def _print_results(results: list[dict], limit: int | None = None) -> None:
    shown = results[:limit] if limit else results
    for r in shown:
        parts = [str(v) for v in r.values()]
        print("  ".join(parts))
    if limit and len(results) > limit:
        print(f"  … {len(results) - limit} more (use --top to increase)")


# ---------------------------------------------------------------------------
# Subcommand: index
# ---------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> int:
    """Build or update the vector index for one or more folders."""
    from core.feature_extractor import FeatureExtractor
    from core.vector_db import VectorDatabase
    import numpy as np

    extractor = FeatureExtractor()

    for folder_str in args.folders:
        folder = Path(folder_str)
        if not folder.exists():
            log.error("Folder not found: %s", folder)
            continue

        log.info("Indexing %s (recursive=%s) …", folder, args.recursive)
        pattern = folder.rglob("*") if args.recursive else folder.glob("*")
        existing_meta = VectorDatabase.read_metadata(folder)

        new_vectors, new_meta, current_files = [], [], set()
        skipped = updated = errors = 0

        for entry in pattern:
            if not (entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS):
                continue
            entry_str = entry.as_posix()
            current_files.add(entry_str)
            mtime = entry.stat().st_mtime

            if existing_meta.get(entry_str) == mtime:
                skipped += 1
                continue

            vec = extractor.extract(entry)
            if vec is not None:
                new_vectors.append(vec)
                new_meta.append({"path": entry_str, "mtime": mtime})
                updated += 1
            else:
                errors += 1

        if new_vectors:
            VectorDatabase.save(folder, np.array(new_vectors), new_meta)

        if not args.no_prune and current_files:
            pruned = VectorDatabase.prune(folder, current_files)
            if pruned:
                log.info("Pruned %d stale entries.", pruned)

        log.info(
            "Done: %d extracted, %d skipped (cached), %d errors.",
            updated, skipped, errors,
        )

    return 0


# ---------------------------------------------------------------------------
# Subcommand: vector
# ---------------------------------------------------------------------------

def cmd_vector(args: argparse.Namespace) -> int:
    """Vector similarity search using ResNet18 features."""
    from core.feature_extractor import FeatureExtractor
    from core.vector_db import VectorDatabase
    from core.algorithms import cosine_similarity_batch, score_vector_similarity
    import numpy as np
    import pandas as pd
    from config import METADATA_FILE, VECTOR_MAX_DISTANCE

    query = Path(args.query)
    if not query.exists():
        log.error("Query image not found: %s", query)
        return 1

    extractor = FeatureExtractor()
    log.info("Extracting query vector …")
    query_vec = extractor.extract(query)
    if query_vec is None:
        log.error("Feature extraction failed for %s", query)
        return 1

    query_path = query.resolve()
    files = list_image_files(args.folders, include_subfolders=args.recursive)
    log.info("Scanning %d image(s) across %d folder(s) …", len(files), len(args.folders))

    # Collect vector files
    vec_files = []
    for folder_str in args.folders:
        fp = Path(folder_str)
        if args.recursive:
            for root, _, _ in os.walk(fp):
                vf = VectorDatabase.get_paths(root)[0]
                if vf.exists():
                    vec_files.append(vf)
        else:
            vf = VectorDatabase.get_paths(fp)[0]
            if vf.exists():
                vec_files.append(vf)

    if not vec_files:
        log.error("No vector index found. Run: python cli.py index %s", " ".join(args.folders))
        return 1

    results: list[tuple[str, float]] = []

    for vf in vec_files:
        try:
            vectors = np.load(vf)
            meta_df = pd.read_csv(vf.parent / METADATA_FILE)
            raw_sims = cosine_similarity_batch(vectors, query_vec)
            dists    = np.linalg.norm(vectors - query_vec, axis=1)

            for i, (raw_sim, dist) in enumerate(zip(raw_sims, dists)):
                if Path(meta_df.iloc[i]["path"]).resolve() == query_path:
                    continue
                if dist > VECTOR_MAX_DISTANCE:
                    continue
                score = score_vector_similarity(float(raw_sim))
                if score >= args.threshold:
                    results.append((meta_df.iloc[i]["path"], score))
        except Exception as exc:
            log.warning("Error reading %s: %s", vf, exc)

    results.sort(key=lambda x: x[1], reverse=True)
    if args.top:
        results = results[:args.top]

    out = [{"path": p, "similarity": f"{s:.1f}%"} for p, s in results]
    log.info("Found %d match(es) above threshold %.1f%%.", len(out), args.threshold)

    if args.output:
        _write_output(out, args.output)
    else:
        _print_results(out)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: histogram
# ---------------------------------------------------------------------------

def cmd_histogram(args: argparse.Namespace) -> int:
    """HSV histogram intersection similarity search."""
    import cv2
    import numpy as np
    from core.algorithms import calculate_histogram, compare_histograms

    query = Path(args.query)
    if not query.exists():
        log.error("Query image not found: %s", query)
        return 1

    img = cv2.imdecode(np.fromfile(str(query), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        log.error("Cannot read query image: %s", query)
        return 1

    hist1 = calculate_histogram(img)
    files = list_image_files(args.folders, include_subfolders=args.recursive)
    log.info("Comparing against %d image(s) …", len(files))

    results = []
    for f in files:
        if Path(f).resolve() == query.resolve():
            continue
        try:
            img2 = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img2 is None:
                continue
            sim = compare_histograms(hist1, calculate_histogram(img2))
            if sim >= args.threshold:
                results.append({"path": str(f), "similarity": f"{sim:.2f}%"})
        except Exception as exc:
            log.warning("Error processing %s: %s", f, exc)

    results.sort(key=lambda x: float(x["similarity"].rstrip("%")), reverse=True)
    if args.top:
        results = results[:args.top]

    log.info("Found %d match(es).", len(results))

    if args.output:
        _write_output(results, args.output)
    else:
        _print_results(results)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: duplicates (single query)
# ---------------------------------------------------------------------------

def cmd_duplicates(args: argparse.Namespace) -> int:
    """Find exact duplicates of a single query image using SHA-256."""
    from core.algorithms import calculate_image_hash, calculate_quick_hash

    query = Path(args.query)
    if not query.exists():
        log.error("Query image not found: %s", query)
        return 1

    log.info("Hashing query image …")
    target_hash  = calculate_image_hash(str(query))
    target_quick = calculate_quick_hash(str(query))
    target_size  = query.stat().st_size

    files = list_image_files(args.folders, include_subfolders=args.recursive)
    log.info("Scanning %d image(s) …", len(files))

    results = []
    for f in files:
        if Path(f).resolve() == query.resolve():
            continue
        try:
            if f.stat().st_size != target_size:
                continue
            if calculate_quick_hash(str(f)) != target_quick:
                continue
            if calculate_image_hash(str(f)) == target_hash:
                results.append({"path": str(f), "match": "exact duplicate"})
        except Exception as exc:
            log.warning("Error processing %s: %s", f, exc)

    log.info("Found %d duplicate(s).", len(results))

    if args.output:
        _write_output(results, args.output)
    else:
        _print_results(results)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: groups (all duplicates, no query needed)
# ---------------------------------------------------------------------------

def cmd_groups(args: argparse.Namespace) -> int:
    """Find all duplicate groups across folders using SHA-256 hashing."""
    from core.algorithms import calculate_image_hash, calculate_quick_hash
    from concurrent.futures import ThreadPoolExecutor, as_completed

    files = list_image_files(args.folders, include_subfolders=args.recursive)
    log.info("Hashing %d image(s) …", len(files))

    def _hash(f):
        try:
            return calculate_image_hash(str(f)), f.stat().st_size, str(f)
        except Exception:
            return None, 0, str(f)

    hash_groups: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(_hash, f): f for f in files}
        done = 0
        for future in as_completed(futures):
            h, size, path = future.result()
            done += 1
            if done % 500 == 0:
                log.info("  %d / %d hashed …", done, len(files))
            if h:
                g = hash_groups.setdefault(h, {"files": [], "total_size": 0})
                g["files"].append(path)
                g["total_size"] += size

    groups = sorted(
        [g for g in hash_groups.values() if len(g["files"]) >= 2],
        key=lambda x: x["total_size"],
        reverse=True,
    )
    log.info("Found %d duplicate group(s).", len(groups))

    if args.output and args.output.lower().endswith(".json"):
        out = [
            {
                "files": g["files"],
                "count": len(g["files"]),
                "total_size_mb": round(g["total_size"] / (1024 * 1024), 2),
            }
            for g in groups
        ]
        _write_output(out, args.output)
    elif args.output:
        # CSV — flatten groups
        rows = []
        for i, g in enumerate(groups, 1):
            for path in g["files"]:
                rows.append({"group": i, "path": path,
                             "total_size_mb": round(g["total_size"] / (1024 * 1024), 2)})
        _write_output(rows, args.output)
    else:
        for i, g in enumerate(groups, 1):
            size_mb = g["total_size"] / (1024 * 1024)
            print(f"\nGroup {i}  ({len(g['files'])} files, {size_mb:.2f} MB)")
            for path in g["files"]:
                print(f"  {path}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: sift
# ---------------------------------------------------------------------------

def cmd_sift(args: argparse.Namespace) -> int:
    """SIFT keypoint matching — good for rotated or partially cropped images."""
    import cv2
    from core.algorithms import initialize_sift, sift_similarity
    from config import SIFT_MIN_MATCHES

    query = Path(args.query)
    if not query.exists():
        log.error("Query image not found: %s", query)
        return 1

    sift = initialize_sift()
    if sift is None:
        log.error("SIFT unavailable. Install opencv-contrib-python.")
        return 1

    query_gray = cv2.imread(str(query), cv2.IMREAD_GRAYSCALE)
    if query_gray is None:
        log.error("Cannot read query image: %s", query)
        return 1

    _, des1 = sift.detectAndCompute(query_gray, None)
    if des1 is None or len(des1) < SIFT_MIN_MATCHES:
        log.error("Not enough SIFT features in query image (need %d).", SIFT_MIN_MATCHES)
        return 1

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    files = list_image_files(args.folders, include_subfolders=args.recursive)
    log.info("Comparing against %d image(s) …", len(files))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []

    with ThreadPoolExecutor(max_workers=os.cpu_count() // 2 or 2) as ex:
        futures = {ex.submit(sift_similarity, des1, f, bf): f for f in files}
        for future in as_completed(futures):
            f = futures[future]
            if Path(f).resolve() == query.resolve():
                continue
            try:
                sim = future.result()
                if sim is not None and sim >= args.threshold:
                    results.append({"path": str(f), "similarity": f"{sim:.2f}%"})
            except Exception as exc:
                log.warning("SIFT error on %s: %s", f, exc)

    results.sort(key=lambda x: float(x["similarity"].rstrip("%")), reverse=True)
    if args.top:
        results = results[:args.top]

    log.info("Found %d match(es).", len(results))

    if args.output:
        _write_output(results, args.output)
    else:
        _print_results(results)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: ssim
# ---------------------------------------------------------------------------

def cmd_ssim(args: argparse.Namespace) -> int:
    """SSIM comparison — only matches images with identical pixel dimensions."""
    from core.algorithms import compare_ssim

    query = Path(args.query)
    if not query.exists():
        log.error("Query image not found: %s", query)
        return 1

    files = list_image_files(args.folders, include_subfolders=args.recursive)
    log.info("Comparing against %d image(s) …", len(files))

    results = []
    for f in files:
        if Path(f).resolve() == query.resolve():
            continue
        score = compare_ssim(str(query), str(f))
        if score is not None and score >= args.threshold:
            results.append({"path": str(f), "ssim": f"{score:.2f}%"})

    results.sort(key=lambda x: float(x["ssim"].rstrip("%")), reverse=True)
    if args.top:
        results = results[:args.top]

    log.info("Found %d match(es).", len(results))

    if args.output:
        _write_output(results, args.output)
    else:
        _print_results(results)

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="VisualSearch — command-line image similarity search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # --- Shared arguments factory ---
    def add_folders(p, positional=True, nargs="+"):
        if positional:
            p.add_argument("folders", nargs=nargs, metavar="FOLDER",
                           help="One or more image folders.")
        else:
            p.add_argument("folders", nargs=nargs, metavar="FOLDER")

    def add_query(p):
        p.add_argument("query", metavar="QUERY_IMAGE",
                       help="Path to the query image.")

    def add_common_search(p):
        add_folders(p)
        p.add_argument("-r", "--recursive", action="store_true",
                       help="Scan subfolders recursively. Without this flag only the top-level folder is scanned.")
        p.add_argument("-t", "--threshold", type=float, default=60.0,
                       help="Minimum similarity score 0–100 (default: 60).")
        p.add_argument("--top", type=int, default=None,
                       help="Maximum number of results to show/save.")
        p.add_argument("-o", "--output", metavar="FILE",
                       help="Save results to FILE (.json or .csv). Prints to stdout if omitted.")

    # index
    p_index = sub.add_parser("index", help="Build / update the vector index for folders.")
    add_folders(p_index)
    p_index.add_argument("-r", "--recursive", action="store_true",
                         help="Scan subfolders recursively. Without this flag only the top-level folder is scanned.")
    p_index.add_argument("--no-prune", action="store_true",
                         help="Skip pruning deleted files from the existing index. For faster execution")
    p_index.set_defaults(func=cmd_index)

    # vector
    p_vec = sub.add_parser("vector", help="Vector similarity search (ResNet18).")
    add_query(p_vec)
    add_common_search(p_vec)
    p_vec.set_defaults(func=cmd_vector)

    # histogram
    p_hist = sub.add_parser("histogram", help="HSV histogram similarity search.")
    add_query(p_hist)
    add_common_search(p_hist)
    p_hist.set_defaults(func=cmd_histogram)

    # duplicates
    p_dup = sub.add_parser("duplicates", help="Find exact duplicates of a single image.")
    add_query(p_dup)
    add_folders(p_dup)
    p_dup.add_argument("-r", "--recursive", action="store_true",
                       help="Scan subfolders recursively. Without this flag only the top-level folder is scanned.")
    p_dup.add_argument("-o", "--output", metavar="FILE",
                       help="Save results to FILE (.json or .csv).")
    p_dup.set_defaults(func=cmd_duplicates)

    # groups
    p_grp = sub.add_parser("groups", help="Find all duplicate groups (no query needed).")
    add_folders(p_grp)
    p_grp.add_argument("-r", "--recursive", action="store_true",
                       help="Scan subfolders recursively. Without this flag only the top-level folder is scanned.")
    p_grp.add_argument("-o", "--output", metavar="FILE",
                       help="Save results to FILE (.json or .csv).")
    p_grp.set_defaults(func=cmd_groups)

    # sift
    p_sift = sub.add_parser("sift", help="SIFT keypoint matching (rotated/cropped images).")
    add_query(p_sift)
    add_common_search(p_sift)
    p_sift.set_defaults(func=cmd_sift)

    # ssim
    p_ssim = sub.add_parser("ssim", help="SSIM comparison (identical-size images only).")
    add_query(p_ssim)
    add_common_search(p_ssim)
    p_ssim.set_defaults(func=cmd_ssim)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    sys.exit(args.func(args))
