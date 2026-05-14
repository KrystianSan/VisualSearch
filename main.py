"""
VisualSearch — entry point.

GUI (default):
    python main.py

CLI (all search modes, indexing, output to file):
    python cli.py --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import customtkinter as ctk
from ui.app import VisualSearch


def main() -> None:
    if len(sys.argv) > 1:
        print(
            "To use the command-line interface run:\n"
            "    python cli.py --help\n\n"
            "Launching GUI …",
            file=sys.stderr,
        )
    root = ctk.CTk()
    VisualSearch(root)
    root.mainloop()


if __name__ == "__main__":
    main()
