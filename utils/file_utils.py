"""
utils/file_utils.py
File-system helpers: listing images, opening in the OS file explorer, etc.
No GUI dependencies.
"""

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

from config import SUPPORTED_EXTENSIONS


def list_image_files(folders: list, include_subfolders: bool = False,
                     folder_subfolders: dict | None = None) -> list[Path]:
    """
    Return an ordered list of image file paths from the given folders.

    folder_subfolders, if provided, maps folder path string → bool and
    overrides include_subfolders on a per-folder basis.
    """
    files: list[Path] = []
    processed: set[Path] = set()

    resolved_folders = [Path(f).resolve() for f in folders]

    for folder, orig_str in zip(resolved_folders, folders):
        if any(folder.is_relative_to(p) for p in processed):
            continue
        processed.add(folder)

        # Per-folder subfolder flag takes priority over the global default
        if folder_subfolders is not None:
            inc_sub = folder_subfolders.get(orig_str, include_subfolders)
        else:
            inc_sub = include_subfolders

        try:
            if inc_sub:
                stack = [folder]
                while stack:
                    current = stack.pop()
                    # Sort: dirs first (depth-first), then files
                    entries = sorted(
                        current.iterdir(),
                        key=lambda x: (x.is_file(), x.name),
                        reverse=True,
                    )
                    for entry in entries:
                        resolved = entry.resolve()
                        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                            files.append(resolved)
                        elif entry.is_dir():
                            stack.append(resolved)
            else:
                for entry in folder.iterdir():
                    if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                        files.append(entry.resolve())

        except PermissionError as exc:
            log.warning("list_image_files: permission denied: %s – %s", folder, exc)
        except Exception as exc:
            log.warning("list_image_files: error processing %s: %s", folder, exc)

    return files


def open_in_explorer(file_path) -> None:
    """
    Open the parent folder of *file_path* in the OS file manager,
    highlight the file, and bring the window to the foreground.
    """
    import subprocess
    resolved = Path(file_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    if os.name == "nt":
        # ShellExecuteW triggers Explorer as a foreground window — subprocess
        # would open it in the background because it inherits our process's
        # lower foreground priority.
        import ctypes
        ctypes.windll.shell32.ShellExecuteW(
            None,               # parent HWND
            "open",             # verb
            "explorer.exe",     # program
            f"/select,{resolved}",  # arguments — no extra quoting needed here
            None,               # working directory
            1,                  # SW_SHOWNORMAL — show and bring to front
        )
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", str(resolved)], check=False)
    else:
        subprocess.run(["xdg-open", str(resolved.parent)], check=False)


def count_image_files(folders: list, include_subfolders: bool = False,
                      folder_subfolders: dict | None = None) -> int:
    """Quick file count without building the full list."""
    total = 0
    for folder in folders:
        inc_sub = folder_subfolders.get(str(folder), include_subfolders) \
            if folder_subfolders is not None else include_subfolders
        fp = Path(folder)
        it = fp.rglob("*") if inc_sub else fp.glob("*")
        for entry in it:
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                total += 1
    return total
