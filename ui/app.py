"""
ui/app.py
Main application window — VisualSearch.

Layout
------
┌──────────────────────────────────────────────────────────────┐
│  Folders bar                                                 │
├─────────────┬────────────────────────────────────────────────┤
│             │  Results treeview                              │
│  Left       │  Filename | Path | Similarity                  │
│  Control    ├────────────────────────────────────────────────┤
│  Sidebar    │  Image preview strip                           │
│             │  [Query canvas]       [Selected canvas]        │
├─────────────┴────────────────────────────────────────────────┤
│  Status label + Progress bar                                 │
└──────────────────────────────────────────────────────────────┘
"""

import logging
import os
import threading
import tkinter as tk
from pathlib import Path
from queue import Queue
from tkinter import filedialog, messagebox, ttk, Menu

import customtkinter as ctk
from customtkinter import (
    CTk, CTkFrame, CTkButton, CTkLabel,
    CTkScrollbar, CTkComboBox, CTkCheckBox, IntVar,
)
from PIL import Image
from send2trash import send2trash

log = logging.getLogger(__name__)

from ui.spinbox import CustomSpinbox
from config import (
    APP_NAME, APP_GEOMETRY, APP_MIN_SIZE,
    SEARCH_MODES, DEFAULT_SEARCH_MODE, DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_LANGUAGE,
    BUTTON_PROCESS_FG, BUTTON_PROCESS_HOVER,
    BUTTON_START_FG, BUTTON_START_HOVER,
    BUTTON_STOP_FG, BUTTON_STOP_HOVER,
)
from i18n import TRANSLATIONS, get_text
from core.feature_extractor import FeatureExtractor
from utils.image_utils import fit_image_to_canvas
from utils.file_utils import open_in_explorer
from ui.search_controller import SearchController

# Height of the image-preview strip (pixels)
PREVIEW_HEIGHT = 200
# Fixed sidebar width (pixels)
SIDEBAR_WIDTH = 240


class VisualSearch:
    def __init__(self, root: CTk):
        self.root = root
        self._configure_root()

        # State
        self.added_folders: list[str] = []
        self.folder_subfolders: dict[str, bool] = {}   # path → include subfolders
        self._selected_folder_idx: int | None = None
        self.query_image: Image.Image | None = None
        self.target_image_path: str | None = None
        self.current_language = DEFAULT_LANGUAGE

        ctk.set_default_color_theme("dark-blue")
        self.vector_extractor = FeatureExtractor()

        # Build UI sections
        self._build_menu()               # row 0 — custom CTk navbar
        self._build_folders_bar()        # row 1 — full width
        self._build_sidebar()            # row 2, col 0 — controls
        self._build_main_pane()          # row 2, col 1 — results + preview
        self._build_status_bar()         # row 3 — full width

        # Controllers
        self.search_controller = SearchController(self)

        # Theme watch
        self.last_appearance_mode = ctk.get_appearance_mode()
        self._update_canvas_colors()
        self._start_theme_monitor()

    # ------------------------------------------------------------------ #
    # Root configuration                                                   #
    # ------------------------------------------------------------------ #

    def _configure_root(self):
        self.root.title(APP_NAME)
        self.root.geometry(APP_GEOMETRY)
        self.root.minsize(*APP_MIN_SIZE)
        # col 0 = sidebar (fixed), col 1 = main area (expands)
        self.root.columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0)   # navbar
        self.root.rowconfigure(1, weight=0)   # folders bar
        self.root.rowconfigure(2, weight=1)   # content
        self.root.rowconfigure(3, weight=0)   # status

    # ------------------------------------------------------------------ #
    # Menu                                                                 #
    # ------------------------------------------------------------------ #

    def _build_menu(self):
        """Custom CTk navbar — replaces the native white menubar on Windows."""
        navbar = CTkFrame(self.root, height=28, corner_radius=0)
        navbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        navbar.grid_propagate(False)
        navbar.columnconfigure(0, weight=1)

        t = lambda k: get_text(self.current_language, k)

        # Settings dropdown button
        self._settings_btn = CTkButton(
            navbar,
            text=t("settings"),
            width=80, height=24,
            corner_radius=4,
            fg_color="transparent",
            hover_color=("gray80", "gray30"),
            text_color=("gray10", "gray90"),
            font=ctk.CTkFont(size=12),
            command=self._show_settings_menu,
        )
        self._settings_btn.grid(row=0, column=0, sticky="w", padx=4, pady=2)

    def _show_settings_menu(self):
        """Pop up a native Menu anchored below the Settings button."""
        t = lambda k: get_text(self.current_language, k)
        menu = Menu(self.root, tearoff=0)
        lang_menu = Menu(menu, tearoff=0)
        for lang in TRANSLATIONS:
            lang_menu.add_command(
                label=lang, command=lambda l=lang: self.change_language(l)
            )
        menu.add_cascade(label=t("language"), menu=lang_menu)

        # Apply theme colours to the popup
        mode = ctk.get_appearance_mode()
        is_dark = mode == "Dark"
        bg       = "#2b2b2b" if is_dark else "#f0f0f0"
        fg       = "#ffffff" if is_dark else "#000000"
        act_bg   = "#3d3d3d" if is_dark else "#d0d0d0"
        opts = dict(bg=bg, fg=fg, activebackground=act_bg,
                    activeforeground=fg, relief="flat", borderwidth=0)
        for m in (menu, lang_menu):
            try:
                m.configure(**opts)
            except Exception:
                pass

        # Position below the button
        btn = self._settings_btn
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height()
        menu.tk_popup(x, y)

    # ------------------------------------------------------------------ #
    # Folders bar  (row 0, full width)                                     #
    # ------------------------------------------------------------------ #

    def _build_folders_bar(self):
        bar = CTkFrame(self.root, height=116)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 3))
        bar.grid_propagate(False)
        bar.columnconfigure(1, weight=1)
        bar.rowconfigure(0, weight=1)
        bar.rowconfigure(1, weight=1)

        t = lambda k: get_text(self.current_language, k)

        self.folder_up_button = CTkButton(bar, text=t("folder_up"), width=110,
                                          command=self._move_folder_up)
        self.folder_up_button.grid(row=0, column=0, padx=(6, 4), pady=(6, 2), sticky="ew")

        self.folder_down_button = CTkButton(bar, text=t("folder_down"), width=110,
                                            command=self._move_folder_down)
        self.folder_down_button.grid(row=1, column=0, padx=(6, 4), pady=(2, 6), sticky="ew")

        # Scrollable frame replaces CTkListbox — allows per-row widgets
        self._folders_scroll = ctk.CTkScrollableFrame(bar, height=96, orientation="vertical")
        self._folders_scroll.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=4, pady=6)
        self._folders_scroll.columnconfigure(0, weight=1)

        self.remove_folder_button = CTkButton(bar, text=t("remove_folder"), width=110,
                                              command=self._remove_folder)
        self.remove_folder_button.grid(row=0, column=2, padx=(4, 6), pady=(6, 2), sticky="ew")

        self.add_folder_button = CTkButton(bar, text=t("add_folder"), width=110,
                                           command=self._add_folder)
        self.add_folder_button.grid(row=1, column=2, padx=(4, 6), pady=(2, 6), sticky="ew")

    # ------------------------------------------------------------------ #
    # Left sidebar  (row 1, col 0)                                         #
    # ------------------------------------------------------------------ #

    def _build_sidebar(self):
        sidebar = self.sidebar = CTkFrame(self.root, width=SIDEBAR_WIDTH)
        sidebar.grid(row=2, column=0, sticky="nsew", padx=(6, 3), pady=3)
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        t = lambda k: get_text(self.current_language, k)
        PX = 8   # horizontal padding, used consistently
        PY = 3   # default vertical padding

        def g(widget, r, pady=PY):
            widget.grid(row=r, column=0, sticky="ew", padx=PX, pady=pady)

        row = 0

        # --- Query image ---
        self.upload_image_button = CTkButton(
            sidebar, text=t("upload_image"), command=self._upload_query_image
        )
        g(self.upload_image_button, row, pady=(10, PY)); row += 1

        _separator(sidebar, row); row += 1

        # --- Search mode ---
        self.search_mode_label = CTkLabel(sidebar, text=t("search_mode") + ":", anchor="w")
        self.search_mode_label.grid(row=row, column=0, sticky="w", padx=PX, pady=PY)
        row += 1
        self.search_combobox = CTkComboBox(sidebar, values=SEARCH_MODES, state="readonly")
        self.search_combobox.set(DEFAULT_SEARCH_MODE)
        g(self.search_combobox, row); row += 1

        # --- Similarity threshold ---
        self.similarity_threshold_label = CTkLabel(
            sidebar, text=t("similarity_threshold") + ":", anchor="w")
        self.similarity_threshold_label.grid(row=row, column=0, sticky="w", padx=PX, pady=PY)
        row += 1

        sim_row = CTkFrame(sidebar, fg_color="transparent")
        sim_row.grid(row=row, column=0, sticky="w", padx=PX, pady=PY)
        self.sim = CustomSpinbox(sim_row, width=110, height=30, step_size=1, from_=0, to=100)
        self.sim.grid(row=0, column=0)
        self.sim.set(DEFAULT_SIMILARITY_THRESHOLD)
        CTkLabel(sim_row, text="%").grid(row=0, column=1, padx=(4, 0))
        row += 1

        _separator(sidebar, row); row += 1

        # --- Primary action buttons ---
        self.start_search_button = CTkButton(
            sidebar, text="▶  " + t("start_search"),
            fg_color=BUTTON_START_FG, hover_color=BUTTON_START_HOVER,
            text_color=("white", "white"), corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_search_button_click,
        )
        g(self.start_search_button, row); row += 1

        self.process_button = CTkButton(
            sidebar, text="▶  " + t("process_folders"),
            fg_color=BUTTON_PROCESS_FG, hover_color=BUTTON_PROCESS_HOVER,
            text_color=("black", "white"), corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_process_button_click,
        )
        g(self.process_button, row); row += 1

        _separator(sidebar, row); row += 1

        # --- Secondary actions ---
        self.show_images_button = CTkButton(
            sidebar, text=t("show_full_size"), command=self._show_images
        )
        g(self.show_images_button, row); row += 1

        self.open_in_explorer_button = CTkButton(
            sidebar, text=t("open_in_explorer"), command=self._open_in_explorer
        )
        g(self.open_in_explorer_button, row); row += 1

        _separator(sidebar, row); row += 1

        self.save_button = CTkButton(
            sidebar, text=t("save_results"),
            command=lambda: self.search_controller.save_results()
        )
        g(self.save_button, row); row += 1

        self.load_button = CTkButton(
            sidebar, text=t("load_results"),
            command=lambda: self.search_controller.load_results()
        )
        g(self.load_button, row); row += 1

        _separator(sidebar, row); row += 1

        self.delete_selected_button = CTkButton(
            sidebar, text=t("delete_selected"),
            fg_color=BUTTON_STOP_FG, hover_color=BUTTON_STOP_HOVER,
            text_color=("white", "white"),
            command=self._delete_selected,
        )
        g(self.delete_selected_button, row, pady=(PY, 10)); row += 1

        # Push everything up, let remaining space stay empty at the bottom
        sidebar.rowconfigure(row, weight=1)

    # ------------------------------------------------------------------ #
    # Main pane  (row 1, col 1) — treeview + preview                      #
    # ------------------------------------------------------------------ #

    def _build_main_pane(self):
        pane = CTkFrame(self.root)
        pane.grid(row=2, column=1, sticky="nsew", padx=(3, 6), pady=3)
        pane.columnconfigure(0, weight=1)
        pane.rowconfigure(0, weight=1)         # treeview — equal share
        pane.rowconfigure(1, weight=1)         # preview  — equal share

        self._build_results_pane(pane)
        self._build_preview_strip(pane)

    # ---- Results treeview -------------------------------------------- #

    def _build_results_pane(self, parent):
        outer = CTkFrame(parent)
        outer.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 3))
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # Header row: title + result counter
        header = CTkFrame(outer, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        header.columnconfigure(0, weight=1)

        CTkLabel(header, text="Search Results", font=ctk.CTkFont(size=13, weight="bold"),
                 anchor="w").grid(row=0, column=0, sticky="w")
        self.result_count_label = CTkLabel(header, text="", anchor="e",
                                           text_color=("gray50", "gray60"))
        self.result_count_label.grid(row=0, column=1, sticky="e")

        # Tree + scrollbars live in this frame — kept as instance var for rebuild
        self._tree_frame = CTkFrame(outer, fg_color="transparent")
        self._tree_frame.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 6))
        self._tree_frame.columnconfigure(0, weight=1)
        self._tree_frame.rowconfigure(0, weight=1)

        self._build_standard_treeview()

    def _build_standard_treeview(self):
        """Create (or recreate) the treeview with the standard 3-column layout."""
        self._destroy_treeview()

        self.tree = ttk.Treeview(
            self._tree_frame,
            columns=("filename", "path", "similarity"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("filename",   text="Filename ↕",   anchor="w",
                          command=lambda: self._sort_column("filename", str))
        self.tree.heading("path",       text="Full Path ↕",  anchor="w",
                          command=lambda: self._sort_column("path", str))
        self.tree.heading("similarity", text="Similarity ▼", anchor="center",
                          command=lambda: self._sort_column("similarity", float))
        self._sort_state = {"col": "similarity", "desc": True}  # tracks active sort
        self.tree.column("filename",    width=200, minwidth=120, stretch=False)
        self.tree.column("path",        width=400, minwidth=200, stretch=True)
        self.tree.column("similarity",  width=90,  minwidth=70,  stretch=False, anchor="center")

        self._attach_tree_scrollbars()
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-Button-1>", self._on_double_click)
        self._apply_treeview_style()
        self._update_treeview_headings()

    def build_groups_treeview(self):
        """Create (or recreate) the treeview for duplicate-groups display."""
        self._destroy_treeview()

        self.tree = ttk.Treeview(
            self._tree_frame,
            columns=("File", "Size", "DisplaySize", "Dimensions", "FileCount"),
            show="tree headings",
            selectmode="browse",
        )
        self.tree.column("#0",           width=250, stretch=False)
        self.tree.column("File",         width=300, minwidth=150, stretch=True)
        self.tree.column("Size",         width=0,   stretch=False)   # hidden raw bytes
        self.tree.column("DisplaySize",  width=90,  minwidth=70,  stretch=False, anchor="center")
        self.tree.column("Dimensions",   width=100, minwidth=70,  stretch=False, anchor="center")
        self.tree.column("FileCount",    width=80,  minwidth=60,  stretch=False, anchor="center")
        self.tree.heading("#0",          text="Filename ↕",
                          command=lambda: self._sort_group_children("#0"))
        self.tree.heading("File",        text="File Path ↕",
                          command=lambda: self._sort_group_children("File"))
        self.tree.heading("DisplaySize", text="Size ↕",
                          command=lambda: self._sort_groups_header("Size"))
        self.tree.heading("Dimensions",  text="Dimensions")
        self.tree.heading("FileCount",   text="Files ↕",
                          command=lambda: self._sort_groups_header("FileCount"))
        self.tree.tag_configure("group_header",         background="#2a7a6e",
                                foreground="white", font=("Segoe UI", 9, "bold"))
        self.tree.tag_configure("group_header_selected", background="#1a5a4e",
                                foreground="white", font=("Segoe UI", 9, "bold"))

        self._attach_tree_scrollbars()
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._apply_treeview_style()
        self._groups_sort_desc = True
        self._update_treeview_headings()

    def _destroy_treeview(self):
        """Destroy the existing treeview and its scrollbars if they exist."""
        if hasattr(self, "tree") and self.tree.winfo_exists():
            self.tree.destroy()
        # Remove any leftover scrollbar widgets from the tree_frame
        for w in self._tree_frame.winfo_children():
            w.destroy()

    def _attach_tree_scrollbars(self):
        scroll_y = CTkScrollbar(self._tree_frame, orientation="vertical",
                                command=self.tree.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x = CTkScrollbar(self._tree_frame, orientation="horizontal",
                                command=self.tree.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")

    def _apply_treeview_style(self):
        """Style the treeview to match the CTk theme with improved readability."""
        mode = ctk.get_appearance_mode()
        is_dark = mode == "Dark"

        bg   = self.root._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        fg   = self.root._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        sel  = self.root._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        # Subtle alternating row color
        alt  = "#2b2b2b" if is_dark else "#f0f4f8"
        hdr  = "#1f538d" if is_dark else "#1a5fa8"

        s = ttk.Style()
        s.theme_use("default")
        s.configure("Treeview",
                    background=bg,
                    foreground=fg,
                    fieldbackground=bg,
                    rowheight=24,
                    borderwidth=0,
                    font=("Segoe UI", 9))
        s.configure("Treeview.Heading",
                    background=hdr,
                    foreground="white",
                    relief="flat",
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview",
              background=[("selected", sel)],
              foreground=[("selected", "white")])
        s.map("Treeview.Heading",
              background=[("active", "#1a4a7a")])

        # Alternating row tags
        self.tree.tag_configure("oddrow",  background=bg)
        self.tree.tag_configure("evenrow", background=alt)

        self.root.bind("<<TreeviewSelect>>", lambda e: self.root.focus_set())

    # ---- Image preview strip ----------------------------------------- #

    def _build_preview_strip(self, parent):
        strip = CTkFrame(parent)
        strip.grid(row=1, column=0, sticky="nsew")
        strip.columnconfigure(0, weight=1)
        strip.columnconfigure(2, weight=1)
        strip.rowconfigure(1, weight=1)   # canvases expand, labels stay fixed

        # Row 0 — labels
        self.query_label = CTkLabel(
            strip, text=get_text(self.current_language, "query_image"),
            text_color=("gray50", "gray60"),
            font=ctk.CTkFont(size=10),
        )
        self.query_label.grid(row=0, column=0, sticky="ew", padx=(6, 3), pady=(4, 0))

        self.selected_label = CTkLabel(
            strip, text=get_text(self.current_language, "selected_image"),
            text_color=("gray50", "gray60"),
            font=ctk.CTkFont(size=10),
        )
        self.selected_label.grid(row=0, column=2, sticky="ew", padx=(3, 6), pady=(4, 0))

        # Row 1 — canvases
        self.canvas_uploaded = tk.Canvas(
            strip, bg="gray13", highlightthickness=1, highlightbackground="gray30"
        )
        self.canvas_selected = tk.Canvas(
            strip, bg="gray13", highlightthickness=1, highlightbackground="gray30"
        )
        self.canvas_uploaded.grid(row=1, column=0, sticky="nsew", padx=(6, 3), pady=(2, 6))
        self.canvas_selected.grid(row=1, column=2, sticky="nsew", padx=(3, 6), pady=(2, 6))

        # Divider spans both rows
        CTkFrame(strip, width=2, fg_color=("gray70", "gray40")).grid(
            row=0, column=1, rowspan=2, sticky="ns", pady=6
        )

    # ------------------------------------------------------------------ #
    # Process button toggle                                                #
    # ------------------------------------------------------------------ #

    def _on_search_button_click(self):
        if self.search_controller.stop_flag.is_set() or \
                (self.search_controller.search_thread and
                 self.search_controller.search_thread.is_alive()):
            self.search_controller.stop_search()
        else:
            self.search_controller.run_search()

    def _on_process_button_click(self):
        if self.search_controller.processing_active:
            self.search_controller.stop_processing()
        else:
            self.search_controller.process_folders()

    # ------------------------------------------------------------------ #
    # Status bar  (row 2, full width)                                      #
    # ------------------------------------------------------------------ #

    def _build_status_bar(self):
        bar = CTkFrame(self.root)
        bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(3, 6))
        bar.columnconfigure(0, weight=1)

        self.status = tk.StringVar(value="Ready")
        self.status_bar = CTkLabel(bar, textvariable=self.status, anchor="w",
                                   font=ctk.CTkFont(size=11))
        self.status_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))

        self.progress = ttk.Progressbar(bar, orient="horizontal", mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        ttk.Style().configure("TProgressbar", thickness=6)

    # ------------------------------------------------------------------ #
    # Column sorting                                                        #
    # ------------------------------------------------------------------ #

    def _update_treeview_headings(self, language: str = None) -> None:
        """Update treeview column headings to the current language."""
        if language is None:
            language = self.current_language
        t = lambda k: get_text(language, k)
        try:
            if self._is_groups_treeview():
                # Determine active header sort arrow
                hs = getattr(self, "_groups_header_sort", {})
                h_col, h_desc = hs.get("col", ""), hs.get("desc", True)
                h_arrow = "▼" if h_desc else "▲"
                # Determine active child sort arrow
                cs = getattr(self, "_groups_children_sort", {})
                c_col, c_desc = cs.get("col", ""), cs.get("desc", False)
                c_arrow = "▼" if c_desc else "▲"

                self.tree.heading("#0", text=t("col_filename") +
                                  (" " + c_arrow if c_col == "#0" else " ↕"))
                self.tree.heading("File", text=t("col_file_path") +
                                  (" " + c_arrow if c_col == "File" else " ↕"))
                self.tree.heading("DisplaySize", text=t("col_size") +
                                  (" " + h_arrow if h_col == "Size" else " ↕"))
                self.tree.heading("Dimensions",  text=t("col_dimensions"))
                self.tree.heading("FileCount", text=t("col_files") +
                                  (" " + h_arrow if h_col == "FileCount" else " ↕"))
            else:
                state = getattr(self, "_sort_state", {})
                active = state.get("col", "")
                desc   = state.get("desc", True)
                arrow  = "▼" if desc else "▲"
                self.tree.heading("filename", anchor="w",
                    text=t("col_filename")   + (" " + arrow if active == "filename"   else " ↕"))
                self.tree.heading("path", anchor="w",
                    text=t("col_path")       + (" " + arrow if active == "path"       else " ↕"))
                self.tree.heading("similarity", anchor="center",
                    text=t("col_similarity") + (" " + arrow if active == "similarity" else " ↕"))
        except tk.TclError:
            pass  # tree not yet built or destroyed
        """Generic column sort for the standard treeview. Toggles direction."""
        items = self.tree.get_children("")
        if not items:
            return

        state = getattr(self, "_sort_state", {"col": col, "desc": True})
        # Toggle direction if same column, else start descending
        desc = not state["desc"] if state["col"] == col else True
        self._sort_state = {"col": col, "desc": desc}

        def _key(item):
            val = self.tree.set(item, col)
            if cast is float:
                try:
                    return float(str(val).rstrip("%").strip())
                except ValueError:
                    return 0.0
            return str(val).lower()

        sorted_items = sorted(items, key=_key, reverse=desc)
        for idx, item in enumerate(sorted_items):
            self.tree.move(item, "", idx)
        self._restripe()

        # Reset all heading labels then set arrow on active column
        self._update_treeview_headings()

    def _sort_groups_header(self, col: str) -> None:
        """Sort top-level group headers by file count or total size."""
        groups = self.tree.get_children("")
        if not groups:
            return

        state = getattr(self, "_groups_header_sort", {"col": col, "desc": True})
        desc = not state["desc"] if state["col"] == col else True
        self._groups_header_sort = {"col": col, "desc": desc}

        def _key(item):
            try:
                return int(self.tree.set(item, col))
            except (ValueError, tk.TclError):
                return 0

        sorted_groups = sorted(groups, key=_key, reverse=desc)
        for idx, group in enumerate(sorted_groups):
            self.tree.move(group, "", idx)

        arrow = "▼" if desc else "▲"
        # Reset both sortable header columns, then mark the active one
        self._update_treeview_headings()

    def _sort_group_children(self, col: str) -> None:
        """Sort child rows within every group by filename (#0 text) or path (File)."""
        groups = self.tree.get_children("")
        if not groups:
            return

        state = getattr(self, "_groups_children_sort", {"col": col, "desc": False})
        desc = not state["desc"] if state["col"] == col else False
        self._groups_children_sort = {"col": col, "desc": desc}

        for group in groups:
            children = self.tree.get_children(group)
            if not children:
                continue

            def _key(item, c=col):
                if c == "#0":
                    return self.tree.item(item, "text").lower()
                return str(self.tree.set(item, c)).lower()

            sorted_children = sorted(children, key=_key, reverse=desc)
            for idx, child in enumerate(sorted_children):
                self.tree.move(child, group, idx)
                # Reapply alternating row tags
                tag = "evenrow" if idx % 2 == 0 else "oddrow"
                self.tree.item(child, tags=(tag,))

        self._update_treeview_headings()

    def _restripe(self):
        """Reapply alternating row tags after any reorder."""
        for idx, item in enumerate(self.tree.get_children("")):
            tag = "evenrow" if idx % 2 == 0 else "oddrow"
            self.tree.item(item, tags=(tag,))

    # ------------------------------------------------------------------ #
    # Button state management                                              #
    # ------------------------------------------------------------------ #

    def set_searching(self):
        """Switch search button to Stop (green), disable everything else."""
        for btn in self._search_sensitive_buttons():
            btn.configure(state=tk.DISABLED)
        self.sim.set_state(tk.DISABLED)
        self._set_folder_checkboxes_state(tk.DISABLED)
        # Toggle to Stop — red like stop processing
        self.start_search_button.configure(
            state=tk.NORMAL,
            text="⏹  " + get_text(self.current_language, "stop_search"),
            fg_color=BUTTON_STOP_FG,
            hover_color=BUTTON_STOP_HOVER,
        )

    def set_idle(self):
        """Restore search button to Start, re-enable everything."""
        for btn in self._search_sensitive_buttons():
            btn.configure(state=tk.NORMAL)
        self.sim.set_state(tk.NORMAL)
        self._set_folder_checkboxes_state(tk.NORMAL)
        self.start_search_button.configure(
            text="▶  " + get_text(self.current_language, "start_search"),
            fg_color=BUTTON_START_FG,
            hover_color=BUTTON_START_HOVER,
        )

    def set_process_button_running(self):
        """Called by the controller when processing starts — lock UI."""
        for btn in self._processing_sensitive_buttons():
            btn.configure(state=tk.DISABLED)
        self._set_folder_checkboxes_state(tk.DISABLED)
        self.process_button.configure(
            text="⏹  " + get_text(self.current_language, "stop_processing"),
            fg_color=BUTTON_STOP_FG,
            hover_color=BUTTON_STOP_HOVER,
            text_color=("white", "white"),
        )

    def set_process_button_idle(self):
        """Called by the controller when processing finishes or is stopped."""
        for btn in self._processing_sensitive_buttons():
            btn.configure(state=tk.NORMAL)
        self._set_folder_checkboxes_state(tk.NORMAL)
        self.process_button.configure(
            text="▶  " + get_text(self.current_language, "process_folders"),
            fg_color=BUTTON_PROCESS_FG,
            hover_color=BUTTON_PROCESS_HOVER,
            text_color=("black", "white"),
        )

    def _set_folder_checkboxes_state(self, state: str):
        """Enable or disable the subfolder checkboxes in every folder row."""
        for row_frame in self._folders_scroll.winfo_children():
            for widget in row_frame.winfo_children():
                if isinstance(widget, CTkCheckBox):
                    widget.configure(state=state)

    def _search_sensitive_buttons(self):
        """Buttons disabled while a search runs (excludes the toggle itself)."""
        return [
            self.upload_image_button,
            self.search_combobox,
            self.process_button,
            self.save_button,
            self.load_button,
            self.delete_selected_button,
            self.add_folder_button,
            self.remove_folder_button,
            self.folder_up_button,
            self.folder_down_button,
        ]

    def _processing_sensitive_buttons(self):
        """Buttons disabled while folder processing runs (excludes the toggle itself)."""
        return [
            self.search_combobox,
            self.start_search_button,
            self.save_button,
            self.load_button,
            self.delete_selected_button,
            self.add_folder_button,
            self.remove_folder_button,
            self.folder_up_button,
            self.folder_down_button,
        ]

    # ------------------------------------------------------------------ #
    # Folder management                                                    #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Folder management                                                    #
    # ------------------------------------------------------------------ #

    def _rebuild_folder_rows(self):
        """Destroy and recreate all rows in the folders scrollable frame."""
        for w in self._folders_scroll.winfo_children():
            w.destroy()

        is_dark = ctk.get_appearance_mode() == "Dark"
        sel_color   = "#1f538d" if is_dark else "#1a5fa8"
        norm_color  = "transparent"

        for idx, path in enumerate(self.added_folders):
            row_frame = CTkFrame(self._folders_scroll, fg_color=
                sel_color if idx == self._selected_folder_idx else norm_color,
                corner_radius=4)
            row_frame.grid(row=idx, column=0, sticky="ew", padx=2, pady=1)
            row_frame.columnconfigure(0, weight=1)

            label = CTkLabel(row_frame, text=path, anchor="w",
                             font=ctk.CTkFont(size=11))
            label.grid(row=0, column=0, sticky="ew", padx=(6, 4), pady=2)

            var = IntVar(value=1 if self.folder_subfolders.get(path, False) else 0)
            cb = CTkCheckBox(row_frame, text=get_text(self.current_language, "subfolders"), variable=var,
                             onvalue=1, offvalue=0, width=20,
                             font=ctk.CTkFont(size=11),
                             command=lambda p=path, v=var: self._toggle_subfolder(p, v))
            cb.grid(row=0, column=1, padx=(0, 6), pady=2)

            # Click anywhere on the row to select it
            for widget in (row_frame, label):
                widget.bind("<Button-1>", lambda e, i=idx: self._select_folder_row(i))

    def _select_folder_row(self, idx: int):
        self._selected_folder_idx = idx
        self._rebuild_folder_rows()

    def _toggle_subfolder(self, path: str, var: IntVar):
        self.folder_subfolders[path] = bool(var.get())

    def _add_folder(self):
        path = filedialog.askdirectory()
        if not path:
            return
        if path in self.added_folders:
            messagebox.showinfo("Info", "Folder already added.")
            return
        self.added_folders.append(path)
        self.folder_subfolders[path] = False
        self._selected_folder_idx = len(self.added_folders) - 1
        self._rebuild_folder_rows()

    def _remove_folder(self):
        idx = self._selected_folder_idx
        if idx is None or idx >= len(self.added_folders):
            return
        path = self.added_folders.pop(idx)
        self.folder_subfolders.pop(path, None)
        self._selected_folder_idx = min(idx, len(self.added_folders) - 1) if self.added_folders else None
        self._rebuild_folder_rows()

    def _move_folder_up(self):
        idx = self._selected_folder_idx
        if idx is None or idx <= 0:
            return
        self.added_folders.insert(idx - 1, self.added_folders.pop(idx))
        self._selected_folder_idx = idx - 1
        self._rebuild_folder_rows()

    def _move_folder_down(self):
        idx = self._selected_folder_idx
        if idx is None or idx >= len(self.added_folders) - 1:
            return
        self.added_folders.insert(idx + 1, self.added_folders.pop(idx))
        self._selected_folder_idx = idx + 1
        self._rebuild_folder_rows()

    # ------------------------------------------------------------------ #
    # Image upload                                                         #
    # ------------------------------------------------------------------ #

    def _upload_query_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.ppm *.pgm")]
        )
        if path:
            self.target_image_path = path
            self.query_image = Image.open(path)
            name = os.path.basename(path)
            self.status.set(f"Query image: {name}")
            self.query_label.configure(
                text=get_text(self.current_language, "query_prefix") + f": {name}"
            )
            self._display_on_canvas(self.query_image, self.canvas_uploaded)
        elif not self.target_image_path:
            messagebox.showinfo("Info", "No query image selected.")

    # ------------------------------------------------------------------ #
    # Canvas helpers                                                       #
    # ------------------------------------------------------------------ #

    def _display_uploaded(self, image: Image.Image):
        self._display_on_canvas(image, self.canvas_uploaded)

    def _display_on_canvas(self, image: Image.Image, canvas: tk.Canvas):
        canvas.original_image = image
        canvas.unbind("<Configure>")
        canvas.bind("<Configure>", lambda e, c=canvas: self._handle_canvas_resize(canvas=c))
        self._handle_canvas_resize(canvas=canvas)

    def _handle_canvas_resize(self, event=None, canvas: tk.Canvas = None):
        target = canvas or (event.widget if event else None)
        if target is None or not hasattr(target, "original_image"):
            return
        w, h = target.winfo_width(), target.winfo_height()
        if w <= 1 or h <= 1:
            return
        try:
            photo, x, y = fit_image_to_canvas(target.original_image, w, h)
            target.delete("all")
            target.create_image(x, y, anchor=tk.NW, image=photo)
            target.image = photo
        except Exception as exc:
            log.debug("canvas resize: %s", exc)

    # ------------------------------------------------------------------ #
    # Tree selection → image preview                                       #
    # ------------------------------------------------------------------ #

    def _on_double_click(self, event):
        """Open full-size view on row double-click, ignore heading double-clicks."""
        region = self.tree.identify_region(event.x, event.y)
        if region == "heading":
            return
        self._show_images()

    def _on_tree_select(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        mode = self.search_combobox.get()

        if mode == "Duplicate Groups":
            # Reset all group headers to default color, highlight selected one
            for top in self.tree.get_children():
                tags = list(self.tree.item(top)["tags"])
                if "group_header_selected" in tags:
                    tags = ["group_header" if t == "group_header_selected" else t for t in tags]
                    self.tree.item(top, tags=tags)

            # Find which header to highlight
            header = item if self.tree.get_children(item) else self.tree.parent(item)
            if header:
                tags = list(self.tree.item(header)["tags"])
                tags = ["group_header_selected" if t == "group_header" else t for t in tags]
                self.tree.item(header, tags=tags)

            children = self.tree.get_children(item)
            if children:
                self._try_display(self.tree.item(children[0])["values"][0], self.canvas_uploaded)
                if len(children) > 1:
                    self._try_display(self.tree.item(children[1])["values"][0], self.canvas_selected)
            else:
                parent = self.tree.parent(item)
                if parent:
                    siblings = self.tree.get_children(parent)
                    if siblings:
                        self._try_display(self.tree.item(siblings[0])["values"][0], self.canvas_uploaded)
                self._try_display(self.tree.item(item)["values"][0], self.canvas_selected)
        else:
            # values = (filename, path, similarity)
            values = self.tree.item(item)["values"]
            path = self._path_from_values(values)
            if not path:
                return
            self._try_display(str(path), self.canvas_selected)
            name = os.path.basename(str(path))
            self.selected_label.configure(
                text=get_text(self.current_language, "selected_prefix") + f": {name}"
            )

    def _try_display(self, raw_path: str, canvas: tk.Canvas):
        path = Path(str(raw_path).split(" Similarity:")[0].strip('"'))
        if path.exists():
            try:
                self._display_on_canvas(Image.open(path), canvas)
            except Exception as exc:
                log.warning("display: %s", exc)

    # ------------------------------------------------------------------ #
    # Result count helper (called by search modules after inserting rows)  #
    # ------------------------------------------------------------------ #

    def update_result_count(self):
        n = len(self.tree.get_children())
        self.result_count_label.configure(
            text=f"{n} result{'s' if n != 1 else ''}" if n else ""
        )

    # ------------------------------------------------------------------ #
    # Path extraction helper                                               #
    # ------------------------------------------------------------------ #

    def _get_selected_path(self) -> Path | None:
        """
        Return the file path for the currently selected treeview row, or None.

        Handles three cases:
          - Standard results:        values = (filename, path, similarity)  → index 1
          - Duplicate group child:   values = (path, size, display, dims)   → index 0
          - Duplicate group header:  values = ("", "", "", "")              → skip, use first child
        """
        selected = self.tree.selection()
        if not selected:
            return None
        item = selected[0]
        values = self.tree.item(item)["values"]

        if self._is_groups_treeview():
            children = self.tree.get_children(item)
            if children:
                values = self.tree.item(children[0])["values"]
            raw = self._path_from_values(values)
        else:
            raw = self._path_from_values(values)

        if not raw:
            return None
        p = Path(str(raw))
        return p if p.exists() else None

    # ------------------------------------------------------------------ #
    # Show full size                                                        #
    # ------------------------------------------------------------------ #

    def _show_images(self):
        selected_path = self._get_selected_path()
        if selected_path is None:
            messagebox.showinfo("Info", "No valid file selected.")
            return

        target_path = Path(self.target_image_path) if self.target_image_path else selected_path

        win = tk.Toplevel(self.root)
        win.title("Full Size Comparison")
        win.geometry("1100x650")
        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)
        win.rowconfigure(0, weight=0)
        win.rowconfigure(1, weight=1)

        # Store canvases in explicit order: [left=query, right=selected]
        canvases = []
        for col, (label_text, img_path) in enumerate([
            ("Query Image", target_path),
            ("Selected Image", selected_path),
        ]):
            CTkLabel(win, text=label_text,
                     font=ctk.CTkFont(size=11, weight="bold")).grid(
                row=0, column=col, pady=(8, 2))
            c = tk.Canvas(win, bg="gray10", highlightthickness=0)
            c.grid(row=1, column=col, sticky="nsew",
                   padx=(8 if col == 0 else 4, 4 if col == 0 else 8), pady=(0, 8))
            try:
                c.original_image = Image.open(img_path)
            except Exception:
                pass
            canvases.append(c)

        def on_resize(_=None):
            for c in canvases:
                if hasattr(c, "original_image"):
                    w, h = c.winfo_width(), c.winfo_height()
                    if w > 1 and h > 1:
                        try:
                            photo, x, y = fit_image_to_canvas(c.original_image, w, h)
                            c.delete("all")
                            c.create_image(x, y, anchor=tk.NW, image=photo)
                            c.image = photo
                        except Exception:
                            pass

        win.bind("<Configure>", on_resize)
        win.after(120, on_resize)

    # ------------------------------------------------------------------ #
    # Open in Explorer                                                      #
    # ------------------------------------------------------------------ #

    def _open_in_explorer(self):
        path = self._get_selected_path()
        if path is None:
            messagebox.showinfo("Info", "No valid file selected.")
            return
        try:
            open_in_explorer(str(path))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ------------------------------------------------------------------ #
    # Delete selected                                                       #
    # ------------------------------------------------------------------ #

    def _is_groups_treeview(self) -> bool:
        """Return True when the current treeview is in duplicate-groups mode."""
        try:
            return list(self.tree["columns"]) == ["File", "Size", "DisplaySize", "Dimensions", "FileCount"]
        except Exception:
            return False

    def _path_from_values(self, values: tuple) -> str | None:
        """Extract the file path from a treeview row's values tuple."""
        if not values:
            return None
        if self._is_groups_treeview():
            return str(values[0])          # (path, size_bytes, size_kb, dims)
        return str(values[1]) if len(values) >= 3 else str(values[0])  # (filename, path, similarity)

    def _delete_selected(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "No items selected.")
            return

        files_to_delete: list[tuple] = []
        for item in selected_items:
            children = self.tree.get_children(item)
            if children:
                # Group header selected — keep the first file, queue the rest
                for child in children[1:]:
                    v = self.tree.item(child)["values"]
                    path = self._path_from_values(v)
                    if path:
                        files_to_delete.append((child, path))
            else:
                v = self.tree.item(item)["values"]
                path = self._path_from_values(v)
                if path:
                    files_to_delete.append((item, path))

        if not files_to_delete:
            return
        if not messagebox.askyesno("Confirm",
                                   f"Move {len(files_to_delete)} file(s) to Recycle Bin?"):
            return

        q: Queue = Queue()

        def worker():
            for item_id, path in files_to_delete:
                try:
                    send2trash(str(path))
                    q.put(("ok", item_id))
                except Exception as exc:
                    q.put(("err", str(exc)))
            q.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            while not q.empty():
                kind, data = q.get()
                if kind == "ok":
                    if self.tree.exists(data):
                        self.tree.delete(data)
                elif kind == "err":
                    messagebox.showerror("Deletion Error", data)
                elif kind == "done":
                    self.update_result_count()
                    return
            self.root.after(100, poll)

        self.root.after(100, poll)

    # ------------------------------------------------------------------ #
    # Theme                                                                 #
    # ------------------------------------------------------------------ #

    def _update_canvas_colors(self):
        mode = ctk.get_appearance_mode()
        bg = "gray10" if mode == "Dark" else "gray85"
        hl = "gray30" if mode == "Dark" else "gray60"
        for c in (self.canvas_uploaded, self.canvas_selected):
            c.configure(bg=bg, highlightbackground=hl)

    # Keep update_canvas_colors as public alias (called externally)
    def update_canvas_colors(self):
        self._update_canvas_colors()

    def _start_theme_monitor(self):
        mode = ctk.get_appearance_mode()
        if mode != self.last_appearance_mode:
            self._update_canvas_colors()
            self._apply_treeview_style()
            self.last_appearance_mode = mode
        self.root.after(500, self._start_theme_monitor)

    # ------------------------------------------------------------------ #
    # Language                                                              #
    # ------------------------------------------------------------------ #

    def change_language(self, language: str):
        self.current_language = language
        t = lambda k: get_text(language, k)

        # Navbar
        self._settings_btn.configure(text=t("settings"))

        # Folder bar
        self.add_folder_button.configure(text=t("add_folder"))
        self.remove_folder_button.configure(text=t("remove_folder"))
        self.folder_up_button.configure(text=t("folder_up"))
        self.folder_down_button.configure(text=t("folder_down"))

        # Sidebar labels
        self.search_mode_label.configure(text=t("search_mode") + ":")
        self.similarity_threshold_label.configure(text=t("similarity_threshold") + ":")

        # Sidebar buttons
        if not self.search_controller.processing_active:
            self.process_button.configure(text="▶  " + t("process_folders"))
        self.save_button.configure(text=t("save_results"))
        self.load_button.configure(text=t("load_results"))
        self.upload_image_button.configure(text=t("upload_image"))
        self.start_search_button.configure(text="▶  " + t("start_search"))
        self.delete_selected_button.configure(text=t("delete_selected"))
        self.show_images_button.configure(text=t("show_full_size"))
        self.open_in_explorer_button.configure(text=t("open_in_explorer"))

        # Preview strip labels — reset to base text; filename shown when active
        self.query_label.configure(text=t("query_image"))
        self.selected_label.configure(text=t("selected_image"))

        # Subfolder checkboxes in folder rows
        for row_frame in self._folders_scroll.winfo_children():
            for widget in row_frame.winfo_children():
                if isinstance(widget, CTkCheckBox):
                    widget.configure(text=t("subfolders"))

        # Treeview column headings
        self._update_treeview_headings(language)


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def _separator(parent, row: int):
    """Thin horizontal divider line."""
    CTkFrame(parent, height=1, fg_color=("gray70", "gray35")).grid(
        row=row, column=0, sticky="ew", padx=8, pady=4
    )
