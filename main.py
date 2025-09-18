import tkinter as tk
import os, sys
import hashlib
import cv2
from pathlib import Path
import time
from tkinter.ttk import *
from tkinter import filedialog, messagebox, ttk, simpledialog, Menu
import pandas as pd
import threading
import numpy as np
import ttkthemes
from CTkListbox import CTkListbox
from PIL import Image, ImageTk, UnidentifiedImageError
from skimage.metrics import structural_similarity
import csv
import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkButton, CTkLabel, CTkEntry, CTkScrollbar, CTkComboBox, CTkCheckBox, StringVar, IntVar

import threading
from queue import Queue
from send2trash import send2trash

import darkdetect

from concurrent.futures import ThreadPoolExecutor, as_completed


import CustomSpinbox

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights



def calculate_image_hash(image_path):
    """Full file hash for final verification"""
    hasher = hashlib.sha256()
    with open(image_path, 'rb') as f:
        while chunk := f.read(131072):  # 128KB chunks
            hasher.update(chunk)
    return hasher.hexdigest()

def calculate_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    return h_hist

def compare_histograms(hist1, hist2):
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    similarity = (intersection / (hist1.sum() + hist2.sum() - intersection)) * 100
    return similarity

VECTOR_ROOT = Path("vector_db")
VECTOR_FILE = "vectors.npy"
METADATA_FILE = "metadata.csv"


class FeatureExtractor:
    def __init__(self):
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=self.weights)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Output: 512-dim vectors
        self.model.eval()
        self.transform = self.weights.transforms()

    def extract(self, image_path):
        try:
            img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                features = self.model(img_tensor)

            features = features.squeeze().numpy()
            return features / np.linalg.norm(features)  # L2-normalize
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            return None

class ImSearch:
    def __init__(self, root):
        self.subfolders = IntVar()
        self.time = IntVar()
        self.stop_search_flag = threading.Event()
        self.search_thread = None

        self.root = root
        self.root.title("ImSearch")
        self.root.geometry("1366x768")
        root.minsize(1000, 720)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)

        #style = ttkthemes.ThemedStyle()  # do this

        #style.theme_use('breeze')

        # if darkdetect.theme() == "Dark":
        #     customtkinter.set_appearance_mode("Light")
        # else:
        #customtkinter.set_appearance_mode(darkdetect.theme())
        ctk.set_default_color_theme("dark-blue")


        self.folder_count = 0
        self.folder_path = None
        self.query_image = None
        self.target_image_path = None
        self.include_subfolders = None
        self.files_list = []
        self.added_folders = []
        self.analyzed_files_count = 0

        self.quick_hash_cache = {}

        self.vector_extractor = FeatureExtractor()
        self.vectors = []

        self.feature_cache = {}  # {file: (kp, des)}
        self.cache_version = "1.0"

        #self.sift = self._initialize_sift()

        self.current_language = "English"

        self.languages = {
            "English": {
                "add_folder": "Add Folder",
                "remove_folder": "Remove Folder",
                "folder_up": "Priority ▲",
                "folder_down": "Priority ▼",
                "upload_image": "Upload Query Image",
                "search_mode": "Choose search mode",
                "similarity_threshold": "Similarity threshold",
                "delete_selected": "Delete Selected",
                "search_subfolders": "Search subfolders",
                "language": "Language",
                "start_search": "Start search",
                "stop_search": "Stop search"
            },
            "Spanish": {
                "add_folder": "Agregar Carpeta",
                "remove_folder": "Eliminar Carpeta",
                "folder_up": "Prioridad ▲",
                "folder_down": "Prioridad ▼",
                "upload_image": "Subir Imagen de Consulta",
                "search_mode": "Elija el modo de búsqueda",
                "similarity_threshold": "Umbral de similitud",
                "delete_selected": "Eliminar Seleccionado",
                "search_subfolders": "Buscar en subcarpetas",
                "language": "Idioma",
                "start_search": "Iniciar búsqueda",
                "stop_search": "Detener la búsqueda"
            },
            "Polish": {
                "add_folder": "Dodaj folder",
                "remove_folder": "Usuń folder",
                "folder_up": "Priorytet ▲",
                "folder_down": "Priorytet ▼",
                "upload_image": "Załaduj zdjęcie",
                "search_mode": "Tryb wyszukiwania",
                "similarity_threshold": "Próg podobieństwa",
                "delete_selected": "Usuń wybrane",
                "search_subfolders": "Szukaj w podfolderach",
                "language": "Język",
                "start_search": "Szukaj",
                "stop_search": "Zatrzymaj wyszukiwanie"
            }
            # Additional languages can be added here.
        }

        menubar = Menu(root)
        root.config(menu=menubar)

        settings_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        #menubar.add_cascade(label="Help", menu=)
        #menubar.add_cascade(label="About", menu=)

        language_menu = Menu(settings_menu, tearoff=0)
        for language in self.languages.keys():
            language_menu.add_command(label=language, command=lambda lang=language: self.change_language(lang))
        settings_menu.add_cascade(label=self.languages[self.current_language]["language"], menu=language_menu)
        #settings_menu.add_cascade(label=)


        folders_frame = CTkFrame(root)


        folders_frame.grid(row=0, column=0, sticky="news", columnspan=3)
        folders_frame.rowconfigure(0, weight=1)
        folders_frame.rowconfigure(1, weight=1)
        folders_frame.rowconfigure(2, weight=0)

        folders_frame.columnconfigure(0, weight=1)
        folders_frame.columnconfigure(1, weight=5)
        folders_frame.columnconfigure(2, weight=1)


        search_frame = CTkFrame(root)
        search_frame.grid(row=1, column=1, sticky="news")
        search_frame.grid_rowconfigure(1, weight=0)
        search_frame.grid_columnconfigure(0, weight=1)



        results_frame = CTkFrame(root)
        results_frame.grid(row=2, column=0, sticky="news", columnspan=3)

        # results_frame.rowconfigure(0, weight=1)
        # results_frame.columnconfigure(0, weight=1)

        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=0)  # No expansion for scrollbar
        results_frame.rowconfigure(0, weight=1)


        self.canvas_uploaded = tk.Canvas(root, bg="gray13", highlightthickness=2, highlightbackground="gray28")
        self.canvas_selected = tk.Canvas(root, bg="gray13", highlightthickness=2, highlightbackground="gray28")

        self.last_appearance_mode = ctk.get_appearance_mode()
        self.update_canvas_colors()
        self.start_theme_monitor()


        self.canvas_uploaded.grid(row=1, column=0, sticky="news", padx=6, pady=6)
        self.canvas_selected.grid(row=1, column=2, sticky="news", padx=6, pady=6)

        self.folders_listbox = CTkListbox(folders_frame, border_width=2)#, selectmode=tk.SINGLE)
        #self.folders_listbox.place(x=166, y=10, height=80, width=1034)
        self.folders_listbox.grid(row=0, column=1, sticky="nsew", padx=5, pady=5, rowspan=2)

        # folder button
        self.add_folder_button = CTkButton(folders_frame, text=self.languages[self.current_language]["add_folder"],
                                            command=self.add_folder)
        #self.add_folder_button.place(x=1212, y=50, height=40, width=140)
        self.add_folder_button.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        self.remove_folder_button = CTkButton(folders_frame, text=self.languages[self.current_language]["remove_folder"],
                                               command=self.remove_folder)
        #self.remove_folder_button.place(x=1212, y=10, height=40, width=140)
        self.remove_folder_button.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        self.folder_up_button = CTkButton(folders_frame, text=self.languages[self.current_language]["folder_up"],
                                           command=self.move_up)
        #self.folder_up_button.place(x=14, y=10, height=40, width=140)
        self.folder_up_button.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.folder_down_button = CTkButton(folders_frame, text=self.languages[self.current_language]["folder_down"],
                                             command=self.move_down)
        #self.folder_down_button.place(x=14, y=50, height=40, width=140)
        self.folder_down_button.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # select image button
        self.upload_image_button = CTkButton(search_frame, text=self.languages[self.current_language]["upload_image"], command=self.upload_query_image)
        #self.upload_image_button.place(x=496, y=122, height=52, width=374)

        #self.sim.place(x=650, y=282, height=20, width=32)
        #tk.Label(search_frame, text="%", fg="black").place(x=682, y=283, height=20, width=10)
        #tk.Label(search_frame, text="%", fg="black").pack()
        #tk.Label(search_frame, text="%", fg="black").grid()

        self.search_combobox = CTkComboBox(search_frame, values=["Vector Similarity", "Histogram Similarity", "Find Duplicates", "Duplicate Groups", "SSIM Compare", "SIFT Compare"], state="readonly")
        #self.search_combobox.place(x=650, y=192, height=34, width=220)
        #self.search_combobox.pack()
        #self.search_combobox.grid()

        #self.search_combobox.current(0)

        #self.start_search_button = CTkButton(search_frame, text=self.languages[self.current_language]["start_search"], command=self.run_search)
        #self.start_search_button.place(x=710, y=280, height=34, width=160)
        #self.start_search_button.pack()
        #self.start_search_button.grid()

        #self.stop_search_button = CTkButton(search_frame, text=self.languages[self.current_language]["stop_search"], command=self.stop_search)
        #self.stop_search_button.place(x=710, y=324, height=34, width=160)
        #self.stop_search_button.pack()
        #self.stop_search_button.grid()

        self.default_columns = ("path", "similarity")
        self.default_headings = {
            "path": "Image path",
            "similarity": "Similarity (%)"
        }

        self.subfolder_button = CTkCheckBox(search_frame, text=self.languages[self.current_language]["search_subfolders"], variable=self.subfolders,
                                                onvalue=1, offvalue=0)
        #self.subfolder_button.place(x=650, y=243)
        #self.subfolder_button.pack()
        #self.subfolder_button.grid()
        self.process_button = ctk.CTkButton(
            search_frame,
            text="Process Folders",
            fg_color=("#FFD700", "#FFA500"),  # Yellow/Orange (light/dark)
            hover_color=("#FFC800", "#FF8C00"),
            text_color=("black", "white"),  # Dark text in light mode, white in dark
            corner_radius=8,
            command=self.process_folders
        )

        # Start Search Button (Green)
        self.start_search_button = ctk.CTkButton(
            search_frame,
            text="Start Search",
            fg_color=("#2CC985", "#2FA572"),  # Green theme
            hover_color=("#239B6A", "#267A5A"),
            text_color=("white", "white"),
            corner_radius=8,
            command=self.run_search
        )

        # Stop Search Button (Red)
        self.stop_search_button = ctk.CTkButton(
            search_frame,
            text="Stop Search",
            fg_color=("#FF4B4B", "#FF3333"),  # Red theme
            hover_color=("#CC0000", "#B22222"),
            text_color=("white", "white"),
            corner_radius=8,
            command=self.stop_search
        )
        self.stop_search_button.configure(state=tk.DISABLED)

        #tree_frame = Frame(root)
        #tree_frame.pack(side="right", fill="y", pady=1, padx=1)
        # self.tree.heading("#0", text="Duplicate Group")
        # self.tree.heading("File Path", text="File Path")
        # self.tree = ttk.Treeview(
        #     results_frame,
        #     columns=("File 1", "File 2"),
        #     show="tree headings",  # For group nodes, use show="tree headings"
        #     height=10
        # )
        # self.tree.heading("File 1", text="Primary File")
        # self.tree.heading("File 2", text="Duplicate File")\
        # self.tree = ttk.Treeview(
        #     results_frame,
        #     columns=("File 1", "File 2"),
        #     show="headings",  # For group nodes, use show="tree headings"
        #     height=8
        # )
        # self.tree.heading("File 1", text="Primary File")
        # self.tree.heading("File 2", text="Duplicate File")


        # self.tree = ttk.Treeview(results_frame, columns=("path", "similarity"),
        #                          show="headings", height=8, selectmode="browse")
        # self.tree.heading("path", text="Image path")
        # self.tree.heading("similarity", text="Similarity (%)")
        # self.tree.column("path", width=400)
        # self.tree.column("similarity", width=100)
        #
        # verscrlbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        # self.tree.configure(yscrollcommand=verscrlbar.set)
        #
        # self.tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        # verscrlbar.grid(row=0, column=1, sticky="ns", pady=5)

        self.tree_container = CTkFrame(results_frame)
        self.tree_container.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.tree_container, columns=("path", "similarity"), show="headings",
                                 selectmode="browse")
        verscrlbar = CTkScrollbar(self.tree_container,
                                   orientation="vertical",
                                   command=self.tree.yview)
        verscrlbar.place(in_=self.tree,                   # Relative to Treeview
                         relx=1.0,                         # Right edge of Treeview
                         x=-8,                            # Move left 20px
                         rely=.08,                         # Start at vertical center
                         relheight=.92,                    # Half of Treeview height
                         anchor="n")
        #verscrlbar.pack(side="right", fill="y", pady=1, padx=1)
        self.tree.configure(yscrollcommand=verscrlbar.set)
        self.tree.heading("path", text="Image path")
        self.tree.heading("similarity", text="Similarity (%)")
        self.tree.column("path")
        self.tree.column("similarity")
        self.tree.pack(padx=6, pady=6, fill=tk.BOTH)
        self.tree.bind("<<TreeviewSelect>>", self.display_selected)


        #self.tree = ttk.Treeview(results_frame, columns=("path", "similarity"), show="headings", height=8, selectmode="browse")
        # verscrlbar = ttk.Scrollbar(results_frame,
        #                            orient="vertical",
        #                            command=self.tree.yview)
        #verscrlbar.pack(side="right", fill="y", pady=1, padx=1)
        #self.tree.configure(yscrollcommand=verscrlbar.set)
        # self.tree.heading("path", text="Image path")
        # self.tree.heading("similarity", text="Similarity (%)")
        # self.tree.column("path")
        # self.tree.column("similarity")
        #self.tree.place(x=0, y=0, height=100, width=1340)
        #self.tree.pack(padx=10, pady=10, fill=tk.BOTH)
        #tree_frame.place(x=14, y=640, height=100, width=1340)
        #tree_frame.grid
        #self.tree.bind("<<TreeviewSelect>>", self.display_selected)

        bg_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        text_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        selected_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])

        treestyle = ttk.Style()
        treestyle.theme_use('default')
        treestyle.configure("Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color,
                            borderwidth=0)
        treestyle.map('Treeview', background=[('selected', bg_color)], foreground=[('selected', selected_color)])
        root.bind("<<TreeviewSelect>>", lambda event: root.focus_set())

        # Status Bar with Progress Bar
        self.status = tk.StringVar()
        self.status.set("Ready")
        self.status_bar = CTkLabel(results_frame, textvariable=self.status,
                                   # relief=tk.SUNKEN,
                                   anchor='w')
        #self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5)
        self.status_bar.pack(fill=tk.BOTH)

        #Style.configure('TProgressbar', thickness=10, pbarrelief='flat')

        self.progress = ttk.Progressbar(results_frame, orient="horizontal", #style='TProgressbar',
                                       # width=100,
                                       mode="determinate")
        #self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.progress.pack(fill=tk.BOTH)

        # Save and Load Buttons
        self.save_button = CTkButton(search_frame, text="Save Results", command=self.save_results)
        #self.save_button.place(x=708, y=500, height=40, width=100)
        #self.save_button.pack()

        self.load_button = CTkButton(search_frame, text="Load Results", command=self.load_results)
        #self.load_button.place(x=604, y=500, height=40, width=100)
        #self.load_button.pack()

        self.show_images_button = CTkButton(search_frame, text="Show in full size", command=self.show_images)
        #self.show_images_button.place(x=496, y=440, height=40, width=374)
        #self.show_images_button.pack()

        self.open_in_explorer_button = CTkButton(search_frame, text="Open in Explorer", command=self.open_in_explorer)
        #self.open_in_explorer_button.place(x=540, y=584, height=34, width=120)
        #self.open_in_explorer_button.pack()
        #self.open_in_explorer_button.grid()

        self.delete_selected_button = CTkButton(search_frame, text=self.languages[self.current_language]["delete_selected"], command=self.delete_selected)
        #self.delete_selected_button.place(x=700, y=584, height=34, width=120)
        #self.delete_selected_button.pack()
        self.delete_queue = Queue()
        self.deletion_thread = None

        search_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        search_frame.columnconfigure(1, weight=1)
        search_frame.rowconfigure(8, weight=1)  # For expanding space

        # Row 0: Query Image
        self.upload_image_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Row 1: Search Mode
        self.search_mode_label = CTkLabel(search_frame, text=self.languages[self.current_language]["search_mode"] + ":")
        self.search_mode_label.grid(row=1, column=0, sticky="w", padx=(2, 5))

        self.search_combobox.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        self.search_combobox.set("Vector Similarity")

        # Row 2: Similarity Threshold
        self.similarity_threshold_label = CTkLabel(search_frame, text=self.languages[self.current_language]["similarity_threshold"] + ":")
        self.similarity_threshold_label.grid(row=2, column=0, sticky="w", padx=(2, 5), pady=2)

        # Create a container frame for spinbox and percentage label
        sim_frame = CTkFrame(search_frame, fg_color=bg_color)
        sim_frame.grid(row=2, column=1, sticky="ew", padx=2, pady=2)

        # Configure columns in the sim_frame
        sim_frame.columnconfigure(0, weight=0)  # Don't expand spinbox column
        sim_frame.columnconfigure(1, weight=0)  # Fixed width for percentage

        self.sim = CustomSpinbox.CustomSpinbox(
            sim_frame,
            width=102,  # Width in pixels (adjust as needed)
            height=32,  # Height in pixels
            step_size=1,
            from_=0,
            to=100
        )
        self.sim.grid(row=0, column=0, sticky="e", padx=(0, 2))
        self.sim.set(50)  # Set initial value to 50 instead of delete/insert

        CTkLabel(sim_frame, text="%").grid(row=0, column=1, sticky="w", padx=(2, 0))

        # Row 3: Subfolders Checkbutton
        self.subfolder_button.grid(row=3, column=0, columnspan=2, sticky="w", padx=2, pady=5)

        # Row 4: Search Buttons
        self.start_search_button.grid(row=4, column=0, sticky="ew", padx=2, pady=2)
        self.stop_search_button.grid(row=4, column=1, sticky="ew", padx=2, pady=2)

        # Row 5: File Operations
        self.show_images_button.grid(row=7, column=0, sticky="ew", padx=2, pady=2)
        self.open_in_explorer_button.grid(row=7, column=1, sticky="ew", padx=2, pady=2)

        # Row 6: Additional Actions
        self.process_button.grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        self.save_button.grid(row=6, column=1, sticky="ew", padx=2, pady=2)
        self.load_button.grid(row=6, column=0, sticky="ew", padx=2, pady=2)
        self.delete_selected_button.grid(row=8, columnspan=2, sticky="ew", padx=2, pady=2)

        # Configure column weights
        search_frame.columnconfigure(0, weight=1)
        search_frame.columnconfigure(1, weight=1)

        #self.search_combobox.configure(width=20)
        #self.subfolder_button.configure(padding=5)

        search_frame.grid_columnconfigure(0, weight=1, minsize=160)
        # row = 0
        # self.upload_image_button.grid(row=row, column=0, sticky="ew", pady=2);
        # row += 1
        # self.search_combobox.grid(row=row, column=0, sticky="ew", pady=2);
        # row += 1
        # self.subfolder_button.grid(row=row, column=0, sticky="w", pady=2);
        # row += 1
        # self.similarity_threshold_label.grid(row=row, column=0, sticky="w", pady=2)
        # self.sim.grid(row=row, column=0, sticky="e", pady=2);
        # row += 1
        # self.start_search_button.grid(row=row, column=0, sticky="ew", pady=2);
        # row += 1
        # self.stop_search_button.grid(row=row, column=0, sticky="ew", pady=2);
        # row += 1
        # self.reprocess_button.grid(row=row, column=0, sticky="ew", pady=2);
        # row += 1
        # self.open_in_explorer_button.grid(row=row, column=0, sticky="ew", pady=2);
        # row += 1
        # self.delete_selected_button.grid(row=row, column=0, sticky="ew", pady=2);
        # row += 1


        # Add processing control variables
        self.processing_flag = threading.Event()
        self.current_processing_thread = None

    #     root.bind("<Configure>", self._on_window_resize)
    #
    # def _on_window_resize(self, event):
    #     """Handle window resizing to maintain minimum dimensions"""
    #     if event.widget == self.root:
    #         # Enforce minimum size
    #         if event.width < 1000:
    #             self.root.geometry(f"1000x{event.height}")
    #         if event.height < 600:
    #             self.root.geometry(f"{event.width}x600")

        # self.tree.column("#0", width=200, stretch=tk.NO)
        # self.tree.column("File", width=300)
        # self.tree.column("Size", width=100, anchor=tk.E)
        # self.tree.column("Dimensions", width=100, anchor=tk.CENTER)
        # self.tree.column("Hash", width=150)

    def update_canvas_colors(self):
        """Update canvas colors based on current system theme"""
        current_mode = ctk.get_appearance_mode()

        if current_mode == "Dark":
            bg_color = "gray90"
            highlight_color = "gray70"

        else:  # Light mode
            bg_color = "gray13"
            highlight_color = "gray28"

        # Update canvas colors
        self.canvas_uploaded.configure(
            bg=bg_color,
            highlightbackground=highlight_color
        )
        self.canvas_selected.configure(
            bg=bg_color,
            highlightbackground=highlight_color
        )

    def start_theme_monitor(self):
        """Check for theme changes every 500ms"""
        current_mode = ctk.get_appearance_mode()
        if current_mode != self.last_appearance_mode:
            self.update_canvas_colors()
            self.last_appearance_mode = current_mode
        self.root.after(500, self.start_theme_monitor)

    def reset_ui(self):
        self.progress["value"] = 0
        self.stop_search_flag.clear()

    def upload_query_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.ppm *.pgm")])
        if file_path:
            self.target_image_path = file_path
            self.status.set(f"Selected Target Image: {os.path.basename(file_path)}")
            self.query_image = Image.open(file_path, 'r')
            self.display_uploaded(self.query_image)
        elif self.target_image_path:
            pass
        else:
            self.status.set(f"Selected Target Image: None")
            messagebox.showinfo("Info", "No target image selected")

    def add_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        # Simple duplicate check (exact match only)
        if folder_path in self.added_folders:
            messagebox.showinfo("Info", "This exact folder path is already added")
            return

        self.added_folders.append(folder_path)
        self.folders_listbox.insert(tk.END, folder_path)
        self.folder_count += 1

    def remove_folder(self):
        selected_folder_name = self.folders_listbox.get(self.folders_listbox.curselection())
        selected_folder_index = self.folders_listbox.curselection()
        if selected_folder_name:
            self.folders_listbox.delete(selected_folder_index)
            self.folder_path = None
            self.added_folders.remove(selected_folder_name)

    def move_up(self):
        idx = self.folders_listbox.curselection()
        #and selected_index[0]
        if idx > 0:
            self.folders_listbox.move_up(idx)
            # # Swap in listbox
            # self.folders_listbox.insert(idx - 1, self.folders_listbox.get(idx))
            # self.folders_listbox.delete(idx + 1)
            # # Swap in data storage
            self.added_folders.insert(idx - 1, self.added_folders.pop(idx))
            # # Maintain selection
            # self.folders_listbox.selection_clear(0, tk.END)
            # self.folders_listbox.selection_set(idx - 1)

    def move_down(self):
        idx = self.folders_listbox.curselection()
        #and selected_index[0]
        if idx < self.folders_listbox.size() - 1:
            self.folders_listbox.move_down(idx)
            # # Swap in listbox
            # self.folders_listbox.insert(idx + 2, self.folders_listbox.get(idx))
            # self.folders_listbox.delete(idx)
            # # Swap in data storage
            self.added_folders.insert(idx + 1, self.added_folders.pop(idx))
            # # Maintain selection
            # self.folders_listbox.selection_clear(0, tk.END)
            # self.folders_listbox.selection_set(idx + 1)

    def stop_search(self):
        """Stop the current search and reset state"""
        self.stop_search_flag.set()
        self.stop_search_button.configure(state=tk.DISABLED)

        # Reset UI elements
        self.progress["value"] = 0
        self.status.set("Search stopped by user")

        # Clear any partial results
        if self.search_combobox.get() == "Duplicate Groups":
            # Only clear if we're in group mode
            self.tree.delete(*self.tree.get_children())

    # def delete_selected(self):
    #     selected_item = self.tree.selection()
    #     if not selected_item:
    #         messagebox.showinfo("Info", "No image selected")
    #         return
    #
    #     selected_file = self.tree.item(selected_item)['values'][0]
    #     confirmation = messagebox.askyesno("Confirm",
    #                                        f"Do you really want to delete {selected_file}? The file will be deleted from disk")
    #
    #     if confirmation:
    #         try:
    #             os.remove(selected_file)
    #             self.tree.delete(selected_item)
    #             messagebox.showinfo("Info", f"Image {selected_file} has been deleted.")
    #         except Exception as e:
    #             messagebox.showerror("Error", f"Failed to delete image: {str(e)}")

    def delete_selected(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "No items selected")
            return

        # Collect files to delete and groups to process
        files_to_delete = []
        groups_to_process = []

        for item in selected_items:
            # Check if this is a group header
            if self.tree.get_children(item):
                groups_to_process.append(item)
            else:
                # Regular file item
                file_path = self.tree.item(item)['values'][0]
                files_to_delete.append((item, file_path))

        # Process group headers
        for group_item in groups_to_process:
            children = self.tree.get_children(group_item)
            if not children:
                continue

            # Keep first child (skip deletion)
            for child in children[1:]:
                file_path = self.tree.item(child)['values'][0]
                files_to_delete.append((child, file_path))

        if not files_to_delete:
            messagebox.showinfo("Info", "No files to delete")
            return

        confirmation = messagebox.askyesno(
            "Confirm",
            f"Move {len(files_to_delete)} files to Recycle Bin?\n"
            "Files can be restored from Recycle Bin if needed."
        )

        if confirmation:
            # Start background deletion thread
            self.deletion_thread = threading.Thread(
                target=self._process_deletions,
                args=(files_to_delete,),
                daemon=True
            )
            self.deletion_thread.start()

            # Start monitoring the queue
            self._monitor_deletion_queue()

    def _process_deletions(self, delete_list):
        """Background thread: Handle actual file operations"""
        for item, file_path in delete_list:
            try:
                # Universal Recycle Bin handling
                send2trash(file_path)
                self.delete_queue.put(('success', item, file_path))
            except Exception as e:
                self.delete_queue.put(('error', item, f"{file_path}: {str(e)}"))

        self.delete_queue.put(('done', None, None))

    def _monitor_deletion_queue(self):
        """Main thread: Process deletion results from queue"""
        group_updates = {}

        while not self.delete_queue.empty():
            result_type, item, data = self.delete_queue.get()

            if result_type == 'success':
                # Find parent group if this was a child item
                parent = self.tree.parent(item)
                if parent:
                    # Track group updates
                    group_updates[parent] = group_updates.get(parent, 0) + 1
                self.tree.delete(item)
            elif result_type == 'error':
                messagebox.showerror("Deletion Error", data)
            elif result_type == 'done':
                # Update group headers after all deletions
                for group_item, deleted_count in group_updates.items():
                    if self.tree.exists(group_item):  # Check if group still exists
                        children = self.tree.get_children(group_item)
                        if children:
                            # Get new group size from remaining children
                            total_size = sum(self.tree.item(child)['values'][1] for child in children)
                            group_size_mb = total_size / (1024 * 1024)

                            # Update group header text
                            self.tree.item(
                                group_item,
                                text=f"Group - {group_size_mb:.2f} MB ({len(children)} files)"
                            )
                        else:
                            # Remove empty group
                            self.tree.delete(group_item)

            self.delete_queue.task_done()
            self.root.update_idletasks()

        # Check again after 100ms if not done
        if self.deletion_thread.is_alive():
            self.root.after(100, self._monitor_deletion_queue)

    def handle_canvas_resize(self, event=None, canvas=None):
        """Handle resizing of images in canvases while maintaining aspect ratio"""
        if event is not None:
            canvas = event.widget
        elif canvas is None:
            return

        if not hasattr(canvas, 'original_image'):
            return

        try:
            # Get current canvas dimensions
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Skip if canvas is too small
            if canvas_width <= 1 or canvas_height <= 1:
                return

            original_image = canvas.original_image
            orig_width, orig_height = original_image.size

            # Calculate aspect ratio-preserving dimensions
            ratio = min(canvas_width / orig_width,
                        canvas_height / orig_height)
            new_width = max(1, int(orig_width * ratio))
            new_height = max(1, int(orig_height * ratio))

            # Resize with high-quality filter
            resized_image = original_image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )
            img_tk = ImageTk.PhotoImage(resized_image)

            # Update canvas display
            canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
            canvas.image = img_tk  # Maintain reference

        except Exception as e:
            print(f"Resize error: {e}")

    def display_uploaded(self, image):
        width_factor = self.canvas_uploaded.winfo_width() / image.width
        height_factor = self.canvas_uploaded.winfo_height() / image.height
        scale_factor = min(width_factor, height_factor)
        resized_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)))
        img = ImageTk.PhotoImage(resized_image)
        x_position = (self.canvas_uploaded.winfo_width() - resized_image.width) // 2
        y_position = (self.canvas_uploaded.winfo_height() - resized_image.height) // 2
        self.canvas_uploaded.delete("all")
        self.canvas_uploaded.create_image(x_position, y_position, anchor=tk.NW, image=img)
        self.canvas_uploaded.image = img
        self.canvas_uploaded.original_image = image
        self.canvas_uploaded.unbind("<Configure>")
        self.canvas_uploaded.bind("<Configure>", self.handle_canvas_resize)
        self.handle_canvas_resize(canvas=self.canvas_uploaded)

    def display_selected(self, event=None):
        """Display selected image with dynamic resizing (now handles event parameter)"""
        selected_items = self.tree.selection()
        if not selected_items:
            return

        selected_item = selected_items[0]
        current_mode = self.search_combobox.get()

        # Clear canvases while preserving original uploaded image
        self.canvas_selected.delete("all")

        def display_image(file_path, canvas):
            """Helper to display image with dynamic resizing capability"""
            try:
                image = Image.open(file_path)
                # Store original image and setup resize handling
                canvas.original_image = image
                canvas.unbind("<Configure>")
                canvas.bind("<Configure>", self.handle_canvas_resize)
                self.handle_canvas_resize(canvas=canvas)
            except Exception as e:
                print(f"Error displaying {file_path}: {str(e)}")

        if current_mode == "Duplicate Groups":
            children = self.tree.get_children(selected_item)
            if children:
                file_paths = []
                for child in self.tree.get_children(selected_item):
                    child_file = self.tree.item(child)["values"][0].split(" Similarity:")[0].strip('\"')
                    if Path(child_file).exists():
                        file_paths.append(child_file)
                if file_paths:
                    display_image(file_paths[0], self.canvas_uploaded)
                    if len(file_paths) > 1:
                        display_image(file_paths[1], self.canvas_selected)
            else:
                parent = self.tree.parent(selected_item)
                if parent:
                    group_files = []
                    for child in self.tree.get_children(parent):
                        child_file = self.tree.item(child)["values"][0].split(" Similarity:")[0].strip('\"')
                        if Path(child_file).exists():
                            group_files.append(child_file)
                    if group_files:
                        display_image(group_files[0], self.canvas_uploaded)
                file_path = self.tree.item(selected_item)["values"][0].split(" Similarity:")[0].strip('\"')
                if Path(file_path).exists():
                    display_image(file_path, self.canvas_selected)
        else:
            file_path = self.tree.item(selected_item)['values'][0].split(" Similarity:")[0].strip('\"')
            if Path(file_path).exists():
                display_image(file_path, self.canvas_selected)

    def list_files(self, folders=None, include_subfolders=False):
        """Process folders in listbox order, prioritizing subfolders of earlier entries"""
        files = []
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"}
        processed_paths = set()

        # Convert to resolved Path objects
        folders = [Path(f).resolve() for f in folders]

        # Process in listbox order while filtering subpaths
        for folder in folders:
            # Skip if already processed as subfolder of previous entry
            if any(folder.is_relative_to(p) for p in processed_paths):
                continue

            processed_paths.add(folder)

            try:
                if include_subfolders:
                    # Depth-first search to prioritize subfolders of current folder
                    dir_stack = [folder]
                    while dir_stack:
                        current_dir = dir_stack.pop()
                        entries = sorted(current_dir.iterdir(), key=lambda x: (x.is_file(), x.name), reverse=True)

                        for entry in entries:
                            entry_path = entry.resolve()
                            if entry.is_file() and entry.suffix.lower() in valid_ext:
                                files.append(entry_path)
                            elif entry.is_dir():
                                # Add subdirectories to stack first (depth-first)
                                dir_stack.append(entry_path)
                else:
                    # Process top-level files
                    for entry in folder.iterdir():
                        if entry.is_file() and entry.suffix.lower() in valid_ext:
                            files.append(entry.resolve())

            except Exception as e:
                print(f"Error processing {folder}: {e}")

        return files

    def run_search(self):
        # Check if a search is already running
        # Reset search state before starting new search
        self.stop_search_flag.clear()
        self.stop_search_button.configure(state=tk.NORMAL)
        self.progress["value"] = 0
        self.status.set("Starting search...")

        # Check if a search is already running
        if self.search_thread and self.search_thread.is_alive():
            messagebox.showinfo("Info", "A search is already in progress. Please wait or stop the current search.")
            return
        if self.search_combobox.get() != "Duplicate Groups":
            self.reset_treeview()

        self.tree.delete(*self.tree.get_children())
        self.tree.configure(show="headings")

        # Existing condition checks remain unchanged
        has_folders = bool(self.added_folders)

        if self.search_combobox.get() == "Duplicate Groups" and not has_folders:
            tk.messagebox.showinfo("Info", "Please add search folders.")
        elif not self.query_image and has_folders and self.search_combobox.get() != "Duplicate Groups":
            tk.messagebox.showinfo("Info", "No file selected. Upload query image to start the search")
        elif not self.query_image and not has_folders and self.search_combobox.get() != "Duplicate Groups":
            tk.messagebox.showinfo("Info", "Please add folders and select a query image.")
        elif self.query_image and not has_folders:
            tk.messagebox.showinfo("Info", "Please add search folders.")
        else:
            self.tree.delete(*self.tree.get_children())
            include_subfolders = self.subfolders.get() == 1

            # Get files from all added folders once
            self.files_list = self.list_files(self.added_folders, include_subfolders)
            total_files = len(self.files_list)
            self.progress["maximum"] = total_files
            search_type = self.search_combobox.get()
            self.stop_search_button.configure(state=tk.NORMAL)

            # Assign all search threads to self.search_thread
            if search_type == "Vector Similarity":
                self.search_thread = threading.Thread(target=self.vector_search)
                self.search_thread.start()
            elif search_type == "Find Duplicates":
                self.search_thread = threading.Thread(target=self.search_duplicates)
                self.search_thread.start()
            elif search_type == "Histogram Similarity":
                image_uploaded = cv2.imdecode(np.fromfile(self.target_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                hist1 = calculate_histogram(image_uploaded)
                self.search_thread = threading.Thread(target=self.search_histogram,
                                                      args=(self.files_list, hist1))
                self.search_thread.start()
            elif search_type == "Duplicate Groups":
                self.search_thread = threading.Thread(target=self.duplicate_groups)
                self.search_thread.start()
            elif search_type == "SSIM Compare":
                self.search_thread = threading.Thread(target=self.ssim_compare,
                                                      args=self.files_list)
                self.search_thread.start()
            elif search_type == "SIFT Compare":
                self.search_thread = threading.Thread(target=self.sift_compare)
                self.search_thread.start()

    def _reset_search_ui(self):
        """Reset UI elements after search completes or stops"""
        self.stop_search_button.configure(state=tk.DISABLED)
        self.progress["value"] = 0
        self.search_thread = None
        self.stop_search_flag.clear()

    def reset_treeview(self):
        """Reset treeview to default configuration"""
        # Clear existing columns
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
            self.tree.column(col, width=0, stretch=False)

        # Reset to default columns
        self.tree.configure(columns=self.default_columns, show="headings")

        # Configure default headings
        for col in self.default_columns:
            self.tree.heading(col, text=self.default_headings[col])

        # Set column widths
        self.tree.column("path", width=400)
        self.tree.column("similarity", width=100)

        # Rebind selection event
        self.tree.bind("<<TreeviewSelect>>", self.display_selected)

    def get_vector_path(self, folder_path):
        """Get standardized vector storage path for a folder"""
        folder_path = Path(folder_path).resolve()
        rel_path = folder_path.relative_to(folder_path.anchor)
        vector_dir = VECTOR_ROOT / rel_path
        vector_dir.mkdir(parents=True, exist_ok=True)
        return vector_dir / VECTOR_FILE, vector_dir / METADATA_FILE

    def process_folder(self, folder_path, include_subfolders=False):
        folder_path = Path(folder_path)
        vector_path, meta_path = self.get_vector_path(folder_path)
        vector_path.parent.mkdir(parents=True, exist_ok=True)

        # Read metadata
        existing_meta = []
        existing_dict = {}
        if meta_path.exists():
            try:
                meta_df = pd.read_csv(meta_path)
                existing_meta = meta_df.to_dict('records')
                existing_dict = {row['path']: row['mtime'] for row in existing_meta}
            except Exception as e:
                print(f"Error reading metadata: {e}")
                meta_path.unlink(missing_ok=True)

        # Collect all image files first
        search_pattern = folder_path.rglob('*') if include_subfolders else folder_path.glob('*')
        image_files = []
        for entry in search_pattern:
            if entry.is_file() and entry.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                image_files.append(entry)

        # Process files
        new_vectors = []
        new_meta = []
        processed_count = 0

        for entry in image_files:
            if self.stop_search_flag.is_set():
                break

            try:
                entry_str = entry.as_posix()
                current_mtime = entry.stat().st_mtime

                # Skip unchanged files
                if existing_dict.get(entry_str) == current_mtime:
                    # Still count as processed
                    processed_count += 1
                    # Update progress for skipped files
                    self.root.after(0, self._update_file_progress)
                    continue

                # Process image
                vector = self.vector_extractor.extract(entry)
                if vector is not None:
                    new_vectors.append(vector)
                    new_meta.append({'path': entry_str, 'mtime': current_mtime})

                processed_count += 1
                # Update progress for each processed file
                self.root.after(0, self._update_file_progress)
            except Exception as e:
                print(f"Error processing {entry}: {str(e)}")
                processed_count += 1
                self.root.after(0, self._update_file_progress)
                continue

        # Update vectors and metadata
        if new_vectors:
            try:
                # Save vectors
                if vector_path.exists():
                    existing_vectors = np.load(vector_path, mmap_mode='r')
                    updated_vectors = np.vstack([existing_vectors, new_vectors])
                    del existing_vectors
                else:
                    updated_vectors = np.array(new_vectors)
                np.save(vector_path, updated_vectors)

                # Update metadata
                new_paths = {item['path'] for item in new_meta}
                filtered_existing = [row for row in existing_meta if row['path'] not in new_paths]
                updated_meta_df = pd.DataFrame(filtered_existing + new_meta)
                updated_meta_df.to_csv(meta_path, index=False)

            except Exception as e:
                print(f"Error saving data: {str(e)}")
                vector_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return 0

        return processed_count

    def _process_all_folders(self, folders, include_subfolders):
        """Background thread logic for reprocessing"""
        # Initialize progress tracking variables
        self.total_files = self._count_image_files(folders, include_subfolders)
        self.processed_files = 0
        self.start_time = time.time()  # Store start time as instance variable
        self.processing_active = True  # Flag to control status updates

        # Start status update thread
        status_thread = threading.Thread(target=self._update_status_during_processing)
        status_thread.daemon = True
        status_thread.start()

        self.root.after(0, self._init_progress_bar, self.total_files)

        # Delete existing vector files in all folders
        all_folders = set()
        for folder in folders:
            folder = Path(folder)
            if include_subfolders:
                # Collect all subdirectories
                for entry in folder.rglob('*'):
                    if entry.is_dir():
                        all_folders.add(entry)
            else:
                all_folders.add(folder)

        for folder in all_folders:
            vec_path, meta_path = self.get_vector_path(folder)
            if vec_path.exists():
                try:
                    os.remove(vec_path)
                except:
                    pass
            if meta_path.exists():
                try:
                    os.remove(meta_path)
                except:
                    pass

        # Process each top-level folder
        for folder in folders:
            if self.processing_flag.is_set():
                break
            # ACTUALLY CALL process_folder HERE
            self.process_folder(folder, include_subfolders)

        # Processing complete
        self.processing_active = False
        status_thread.join(1.0)  # Give status thread a moment to finish

        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m {secs:.1f}s"

        # Completion message with time information
        self.root.after(0, lambda: [
            self.status.set(f"Processing completed in {time_str} - {self.processed_files} files processed"),
            messagebox.showinfo("Info", "All folders processed with current model")
        ])

    def _count_image_files(self, folders, include_subfolders):
        """Count all image files in folders"""
        total = 0
        for folder in folders:
            folder = Path(folder)  # Convert to Path object
            if include_subfolders:
                it = folder.rglob('*')
            else:
                it = folder.glob('*')
            for entry in it:
                if entry.is_file() and entry.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    total += 1
        return total

    def _init_progress_bar(self, total_files):
        """Initialize progress bar with file count"""
        self.progress["maximum"] = total_files
        self.progress["value"] = 0
        self.status.set(f"Processing 0/{total_files} files")

    def _update_status_during_processing(self):
        """Thread to continuously update status during processing"""
        while self.processing_active:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(elapsed, 60)
            time_str = f"{int(mins)}m {secs:.1f}s"

            # Calculate files per second
            fps = self.processed_files / elapsed if elapsed > 0 else 0

            # Update status in main thread
            status_text = (f"Processing: {self.processed_files}/{self.total_files} files "
                           f"({time_str}, {fps:.1f} files/sec)")
            self.root.after(0, self.status.set, status_text)

            time.sleep(0.1)

    def _update_file_progress(self):
        """Update progress for each processed file"""
        self.processed_files += 1
        self.progress["value"] = self.processed_files

        # Update UI periodically to prevent freezing
        if self.processed_files % 10 == 0:
            self.root.update_idletasks()

    def process_folders(self):
        """Force reprocessing of all folders in a background thread"""
        if self.current_processing_thread and self.current_processing_thread.is_alive():
            self.processing_flag.set()
            self.current_processing_thread.join(timeout=5)

        # Retrieve Tkinter data in the main thread
        count = self.folders_listbox.size()
        folders=[]
        for i in range(count):
            folders.append(self.folders_listbox.get(i))
        include_subfolders = self.subfolders.get() == 1

        self.processing_flag.clear()
        self.current_processing_thread = threading.Thread(
            target=self._process_all_folders,
            args=(folders, include_subfolders),
            daemon=True
        )
        self.current_processing_thread.start()
        #elif inna_metoda_indeksowania()

    # def _process_all_folders(self, folders, include_subfolders):
    #     """Background thread logic for reprocessing"""
    #     # Delete existing files first
    #     all_folders = set()
    #     for folder in folders:
    #         if include_subfolders:
    #             for root, dirs, _ in os.walk(folder):
    #                 all_folders.add(root)
    #         else:
    #             all_folders.add(folder)
    #
    #     # Delete existing vector/meta files
    #     for folder in all_folders:
    #         vec_path, meta_path = self.get_vector_path(folder)
    #         if vec_path.exists():
    #             os.remove(vec_path)
    #         if meta_path.exists():
    #             os.remove(meta_path)
    #
    #     # Now reprocess (reuse the existing processing logic)
    #     self.process_all_folders(folders, include_subfolders)
    #
    #     # Show completion message in the main thread
    #     self.root.after(0, lambda: messagebox.showinfo(
    #         "Info", "All folders processed with current model"
    #     ))

    def _update_progress_max(self, total):
        """Thread-safe progress max setup"""
        self.progress["value"] = total
        self.status.set("Initializing folder processing...")

    def _update_progress(self, current, total):
        """Thread-safe progress update"""
        self.progress["value"] = current
        self.status.set(f"Processed {current}/{total} files")

    def _get_folder_structure(self, folders, include_subfolders):
        """Get ordered list of folders with hierarchy"""
        ordered_folders = []
        for folder in folders:
            if include_subfolders:
                ordered_folders.extend([str(p) for p in Path(folder).rglob('')
                                        if p.is_dir()])
            else:
                ordered_folders.append(str(folder))
        return ordered_folders

    def vector_search(self):
        """Entry point for vector similarity search"""
        try:
            self.tree.delete(*self.tree.get_children())
            query_vector = self.vector_extractor.extract(Path(self.target_image_path))

            if query_vector is None:
                messagebox.showerror("Error", "Feature extraction failed")
                return

            # Get user threshold
            user_threshold = int(self.sim.get())  # Capture user threshold

            # Get search parameters
            folders = [self.folders_listbox.get(i) for i in range(self.folders_listbox.size())]
            include_subfolders = self.subfolders.get() == 1
            vector_files = []

            # Build ordered vector file list
            for folder in folders:
                folder_path = Path(folder)
                if include_subfolders:
                    for root, _, _ in os.walk(folder_path):
                        vec_file = self.get_vector_path(root)[0]
                        if vec_file.exists():
                            vector_files.append(vec_file)
                else:
                    vec_file = self.get_vector_path(folder_path)[0]
                    if vec_file.exists():
                        vector_files.append(vec_file)

            if not vector_files:
                messagebox.showinfo("Info", "Process folders first")
                return

            # Configure and start search thread
            self.progress["maximum"] = len(vector_files)
            self.search_thread = threading.Thread(
                target=self._vector_search_thread,
                args=(vector_files, query_vector, user_threshold),  # Pass user threshold
                daemon=True
            )
            self.stop_search_button.configure(state=tk.NORMAL)
            self.search_thread.start()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # def _vector_search_thread(self, vector_files, query_vector):
    #     """Background thread for vector processing with integrated progress/completion"""
    #     try:
    #         # Validate vector dimensions
    #         if query_vector.shape[0] != 512:
    #             self.root.after(0, lambda: messagebox.showerror(
    #                 "Error", "Query vector dimension mismatch (expected 512)"))
    #             return
    #
    #         results = []
    #         start_time = time.time()
    #
    #         for idx, vec_file in enumerate(vector_files, 1):
    #             if self.stop_search_flag.is_set():
    #                 break
    #
    #             # Update progress directly in main thread
    #             self.root.after(0,
    #                             lambda current_idx=idx, current_file=vec_file: [
    #                                 self.progress.config(value=current_idx),
    #                                 self.status.set(
    #                                     f"Searching {current_file.parent} ({current_idx}/{len(vector_files)})")
    #                             ]
    #                             )
    #
    #             try:
    #                 # Load vectors and metadata
    #                 vectors = np.load(vec_file, mmap_mode='r')
    #                 meta_file = vec_file.parent / METADATA_FILE
    #                 meta_df = pd.read_csv(meta_file)
    #
    #                 # Calculate similarities
    #                 similarities = np.dot(vectors, query_vector)
    #                 euclidean_dists = np.linalg.norm(vectors - query_vector, axis=1)
    #
    #                 # Apply similarity threshold
    #                 for i, (sim, dist) in enumerate(zip(similarities, euclidean_dists)):
    #                     sim_percent = sim * 100
    #                     if sim_percent >= int(self.sim.get()):
    #                         results.append((meta_df.iloc[i]['path'], sim_percent))
    #
    #             except Exception as e:
    #                 print(f"Error processing {vec_file}: {str(e)}")
    #
    #         # Finalize results in main thread
    #         elapsed = time.time() - start_time
    #         self.root.after(0, lambda: (
    #             self.tree.delete(*self.tree.get_children()),
    #             [self.tree.insert("", tk.END, values=(path, f"{similarity:.2f}"))
    #              for path, similarity in sorted(results, key=lambda x: -x[1])],
    #             self.status.set(f"Found {len(results)} matches in {elapsed:.2f}s"),
    #             self.progress.__setitem__("value", 0),
    #             self.stop_search_button.configure(state=tk.DISABLED)
    #         ))
    #
    #     except Exception as e:
    #         self.root.after(0, lambda: messagebox.showerror("Search Error", str(e)))

    def _vector_search_thread(self, vector_files, query_vector, user_threshold):
        """Threaded vector search with ordered processing and scaled similarity"""
        try:
            # Validate vector dimensions first
            if query_vector.shape[0] != 512:
                self.root.after(0, messagebox.showerror,
                                "Error", "Query vector dimension mismatch (expected 512)")
                return

            # Get absolute path of query image
            query_image_path = Path(self.target_image_path).resolve()
            results = []
            start_time = time.time()
            total_files = len(vector_files)
            min_similarity = float('inf')  # Track min similarity for scaling
            max_similarity = float('-inf')  # Track max similarity for scaling

            for idx, vec_file in enumerate(vector_files, 1):
                if self.stop_search_flag.is_set():
                    break

                # Update progress in main thread
                self.root.after(0, self._update_vector_progress,
                                idx, total_files, vec_file.parent)

                # Process current vector file
                try:
                    vectors = np.load(vec_file, mmap_mode='r')
                    meta_file = vec_file.parent / METADATA_FILE
                    meta_df = pd.read_csv(meta_file)

                    # Calculate similarities
                    similarities = np.dot(vectors, query_vector)
                    euclidean_dists = np.linalg.norm(vectors - query_vector, axis=1)

                    # Apply thresholds
                    for i, (sim, dist) in enumerate(zip(similarities, euclidean_dists)):
                        # Skip the query image itself
                        current_path = Path(meta_df.iloc[i]['path'])
                        if current_path == query_image_path:
                            continue

                        sim_percent = sim * 100

                        # Track min/max for scaling
                        if sim_percent > user_threshold:
                            if sim_percent < min_similarity:
                                min_similarity = sim_percent
                            if sim_percent > max_similarity:
                                max_similarity = sim_percent

                        if (sim_percent >= max(user_threshold, 70) and
                                dist <= 0.5 and
                                sim_percent >= self._calculate_adaptive_threshold(similarities)):
                            results.append((meta_df.iloc[i]['path'], sim_percent))

                except Exception as e:
                    print(f"Error processing {vec_file}: {str(e)}")

            # Apply scaling to results
            scaled_results = []
            if max_similarity > min_similarity:  # Avoid division by zero
                for path, sim in results:
                    # Scale similarity from [min_similarity, max_similarity] to [0, 100]
                    scaled_sim = 100 * (sim - min_similarity) / (max_similarity - min_similarity)
                    scaled_results.append((path, scaled_sim))
            else:
                scaled_results = results  # Use raw values if no variation

            # Finalize in main thread
            elapsed = time.time() - start_time
            self.root.after(0, self._complete_vector_search, scaled_results, elapsed)

        except Exception as e:
            self.root.after(0, messagebox.showerror,
                            "Search Error", str(e))
        finally:
            # Ensure UI is reset even if thread crashes
            self.root.after(0, self._reset_search_ui)

    def _update_vector_progress(self, current, total, folder):
        """Thread-safe progress update for vector search"""
        self.progress["value"] = current
        self.status.set(f"Searching {folder} ({current}/{total})")

    def _complete_vector_search(self, results, elapsed_time):
        """Finalize search in main thread with scaled results"""
        self.tree.delete(*self.tree.get_children())
        max_results = min(500, len(results))  # Limit to 500 results

        # Sort by scaled similarity
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:max_results]

        for path, scaled_sim in sorted_results:
            self.tree.insert("", tk.END, values=(path, f"{scaled_sim:.2f}%"))

        self.status.set(f"Found {len(results)} matches in {elapsed_time:.2f}s")
        self.progress["value"] = 0
        self.stop_search_button.configure(state=tk.DISABLED)
        self._reset_search_ui()

    def _calculate_adaptive_threshold(self, similarities):
        """Calculate dynamic threshold based on similarity distribution"""
        similarities = np.array(similarities)
        if len(similarities) == 0:
            return 0

        # Use 90th percentile as baseline
        threshold = np.percentile(similarities, 90) * 100
        return max(threshold, 70)  # Minimum 70% threshold

    def search_histogram(self, files, hist1):
        start_time = time.time()
        analyzed_files_count = 0
        files_found = 0

        for count, file in enumerate(files, start=1):
            if self.stop_search_flag.is_set():
                self.status.set(f"Search stopped by user. {analyzed_files_count} files analyzed")
                break
            self.status.set(f"Analyzing files ({count}/{len(files)}) - {files_found} matches found")
            self.root.update_idletasks()
            analyzed_files_count += 1
            if file != self.target_image_path:
                image_queued = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
                hist2 = calculate_histogram(image_queued)
                similarity = compare_histograms(hist1, hist2)
                if similarity >= int(self.sim.get()):
                    self.tree.insert("", tk.END, values=(file, f"{similarity:.2f}"))
                    files_found += 1
            self.progress["value"] = count
            self.root.update_idletasks()
        else:
            self.status.set(f"Completed. {analyzed_files_count} files analyzed")

        end_time = time.time()
        elapsed_time = end_time - start_time
        if analyzed_files_count != 0:
            self.status.set(f"Completed in {elapsed_time:.2f} seconds, {analyzed_files_count} files analyzed, {files_found} similar images found")
            self.progress["value"] = 0
        else:
            messagebox.showinfo("Error", f"No image files found in selected folders. Try ticking the \"Search subfolders\" option")
            self.progress["value"] = 0

        self.reset_ui()
        self.stop_search_button.configure(state=tk.DISABLED)

        self.search_thread = None

    def calculate_quick_hash(self, image_path):
        """Fast partial hash of first and middle 8KB chunks"""
        chunk_size = 8192  # 8KB chunks
        file_size = os.path.getsize(image_path)
        hasher = hashlib.sha256()

        try:
            with open(image_path, 'rb') as f:
                # First chunk
                hasher.update(f.read(chunk_size))

                # Middle chunk
                if file_size > chunk_size * 2:
                    f.seek(file_size // 2)
                    hasher.update(f.read(chunk_size))

                # Last chunk for files > 1MB
                if file_size > 1024 * 1024:
                    f.seek(-chunk_size, os.SEEK_END)
                    hasher.update(f.read(chunk_size))

            return hasher.hexdigest()
        except Exception as e:
            print(f"Error reading {image_path}: {str(e)}")
            return None

    def search_duplicates(self):
        """Find all duplicates of the target image using parallel processing"""
        try:
            self.tree.delete(*self.tree.get_children())
        except Exception as e:
            messagebox.showerror("Error", str(e))

        self.status.set("Initializing duplicate search...")

        include_subfolders = self.subfolders.get() == 1
        # Use added_folders instead of folder_path
        files = self.list_files(self.added_folders, include_subfolders)
        self.progress["maximum"] = len(files)

        # Pre-calculate target hash once
        target_hash = calculate_image_hash(self.target_image_path)

        self.search_thread = threading.Thread(target=self._search_duplicates_thread,
                                              args=(files, target_hash))
        self.search_thread.start()

    def _search_duplicates_thread(self, files, source_hash):
        """Modified thread with quick checksum filtering"""
        start_time = time.time()
        analyzed_files_count = 0

        # Precompute target quick hash
        target_quick_hash = self.calculate_quick_hash(self.target_image_path)
        target_size = os.path.getsize(self.target_image_path)

        for count, file in enumerate(files, start=1):
            if self.stop_search_flag.is_set():
                self.status.set(f"Search stopped. Analyzed {analyzed_files_count} files")
                break

            try:
                # First check: File size comparison
                file_size = os.path.getsize(file)
                if file_size != target_size:
                    continue

                # Second check: Quick hash comparison
                file_quick_hash = self.calculate_quick_hash(file)
                if file_quick_hash != target_quick_hash:
                    continue

                # Final check: Full hash comparison
                analyzed_files_count += 1
                self.status.set(f"Analyzing files ({count}/{len(files)})")
                file_hash = calculate_image_hash(file)

                if file_hash == source_hash and self.target_image_path != file:
                    self.tree.insert("", tk.END, values=(file, "Duplicate"))

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
            finally:
                # Ensure UI is reset even if thread crashes
                self.root.after(0, self._reset_search_ui)

            self.progress["value"] = count
            self.root.update_idletasks()

        elapsed_time = time.time() - start_time
        self.status.set(f"Found {self.tree.get_children().__len__()} duplicates in {elapsed_time:.1f}s")
        self.progress["value"] = 0
        self.stop_search_button.configure(state=tk.DISABLED)

        self.search_thread = None

    def duplicate_groups(self):
        """Find all duplicate groups in the dataset using hash grouping"""
        if not self.added_folders:
            tk.messagebox.showinfo("Info", "Please select a folder.")
            return

        self.status.set("Searching for duplicate groups...")
        self.progress['value'] = 0

        # self.tree.tag_configure('even_group', background='#f0f0f0')
        # self.tree.tag_configure('odd_group', background='white')

        # Thread management now handled in run_search
        self._configure_treeview()
        self._duplicate_groups_thread()

    def _duplicate_groups_thread(self):
        """Threaded duplicate group search with sorting and group metrics"""
        start_time = time.time()
        self.root.after(0, lambda: self.tree.delete(*self.tree.get_children()))
        self.root.after(0, self._configure_treeview)

        include_subfolders = self.subfolders.get() == 1
        files = self.list_files(self.added_folders, include_subfolders)
        self.root.after(0, lambda: self.progress.configure(maximum=len(files)))

        # Phase 1: Group files by hash
        hash_groups = {}
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(self._process_file, file): file for file in files}
            for idx, future in enumerate(as_completed(futures), 1):
                if self.stop_search_flag.is_set():
                    break
                file_hash, file_size, file_path = future.result()
                if file_hash:
                    hash_groups.setdefault(file_hash, {'files': [], 'total_size': 0})
                    hash_groups[file_hash]['files'].append((file_path, file_size))
                    hash_groups[file_hash]['total_size'] += file_size

                # Fix: Capture current idx value in local variable
                current_idx = idx
                total_files = len(files)
                self.root.after(0, lambda: self._update_progress(current_idx, total_files))

        # Phase 2: Prepare groups for display
        sorted_groups = sorted(
            [group for group in hash_groups.values() if len(group['files']) >= 2],
            key=lambda x: x['total_size'],
            reverse=True
        )

        # Calculate elapsed time for status message
        elapsed_time = time.time() - start_time
        status_message = f"Found {len(sorted_groups)} duplicate groups in {elapsed_time:.2f}s"

        # Execute GUI updates in main thread
        self.root.after(0, lambda: self._finalize_duplicate_search(sorted_groups, status_message))

    def _finalize_duplicate_search(self, sorted_groups, status_message):
        """Finalize GUI updates after duplicate search"""
        # Insert groups into Treeview hierarchically
        for group_idx, group in enumerate(sorted_groups, 1):
            # Convert group total size to MB
            group_size_mb = group['total_size'] / (1024 * 1024)

            # Create the formatted group header text
            group_header = f"Group {group_idx} - {group_size_mb:.2f} MB ({len(group['files'])} files)"

            parent = self.tree.insert(
                "", "end",
                text=group_header,  # Use the formatted text here
                values=("", "", "", ""),  # Empty values for all columns
                tags=('group_header',),
                open=True
            )

            for file_idx, (file_path, file_size) in enumerate(sorted(group['files']), 1):
                try:
                    with Image.open(file_path) as img:
                        dimensions = f"{img.width}x{img.height}"
                except:
                    dimensions = "N/A"

                file_size_kb = file_size / 1024
                self.tree.insert(
                    parent, "end",
                    text=os.path.basename(file_path),  # Show filename in tree column
                    values=(
                        file_path,  # Full path in hidden column
                        file_size,  # Original byte value in hidden column
                        f"{file_size_kb:.2f} KB",
                        dimensions
                    ),
                    tags=(f'{"even" if file_idx % 2 else "odd"}_row',)
                )

        # Update status and reset UI
        self.status.set(status_message)
        self.progress["value"] = 0
        self.stop_search_button.configure(state=tk.DISABLED)
        self.search_thread = None
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        # Ensure UI is reset even if thread crashes
        self.root.after(0, self._reset_search_ui)

    def _configure_treeview(self):
        """Configure treeview columns with proper formatting"""
        # Show the tree column by setting show="tree headings"
        self.tree.configure(show="tree headings")  # ADD THIS LINE

        self.tree["columns"] = ("File", "Size", "DisplaySize", "Dimensions")
        self.tree.column("#0", width=300, stretch=tk.NO)
        self.tree.column("File", width=300)
        self.tree.column("Size", width=0, stretch=tk.NO)  # Hidden raw size
        self.tree.column("DisplaySize", width=100, anchor="center")
        self.tree.column("Dimensions", width=100, anchor="center")

        self.tree.heading("#0", text="Filename")
        self.tree.heading("File", text="File Path")
        self.tree.heading("DisplaySize", text="Size")
        self.tree.heading("Dimensions", text="Dimensions")

        self.tree.tag_configure('group_header', background='#3a7ebf',
                                foreground='white', font=('Helvetica', 10, 'bold'))

    def _human_readable_size(self, size_bytes):
        """Convert bytes to human-readable format without external dependencies"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    # def _sort_tree(self, column):
    #     """Sort tree items by column"""
    #     converter = {
    #         "Size": float,
    #         "DisplaySize": lambda x: float(x.rstrip(' BKMGT')),
    #         "Dimensions": lambda x: tuple(map(int, x.split('x'))),
    #         "Width": int,
    #         "Height": int
    #     }
    #
    #     # Get current sort order and reverse it
    #     reverse = self.tree.heading(column)["direction"] == "asc"
    #     self.tree.heading(column, command=lambda: self._sort_tree(column))
    #
    #     # Sort the items
    #     items = [(self.tree.set(child, column), child)
    #              for child in self.tree.get_children('')]
    #     items.sort(key=lambda x: converter.get(column, str)(x[0]), reverse=reverse)
    #
    #     for index, (_, child) in enumerate(items):
    #         self.tree.move(child, '', index)
    #
    #     # Update heading arrow
    #     self.tree.heading(column, direction="desc" if reverse else "asc")

    def _sort_by_column(self, column):
        """Sort groups by total size or files by individual size"""
        if column == "Size":
            items = [(self.tree.set(child, "Size"), child)
                     for child in self.tree.get_children('')]
            items.sort(key=lambda x: float(x[0]), reverse=True)

            for index, (_, child) in enumerate(items):
                self.tree.move(child, '', index)

    def _on_tree_select(self, event):
        """Handle selection for image preview in duplicate groups mode"""
        selected = self.tree.selection()
        if not selected:
            return

        item = self.tree.item(selected[0])
        # For group headers, we want to show images without changing selection
        if self.search_combobox.get() == "Duplicate Groups" and self.tree.get_children(selected[0]):
            self.display_selected()
        else:
            # Regular selection handling
            if self.tree.parent(selected[0]):  # Child item
                file_path = item["values"][0]
            else:  # Group header - get first child
                children = self.tree.get_children(selected[0])
                if children:
                    file_path = self.tree.item(children[0])["values"][0]
                else:
                    return
            self.display_selected()

    def _process_file(self, file_path):
        """Thread-safe file processing with error handling"""
        try:
            file_hash = calculate_image_hash(file_path)
            file_size = os.path.getsize(file_path)
            return (file_hash, file_size, file_path)
        except Exception as e:
            return (None, 0, file_path)

    # def _on_tree_select(self, event):
    #     """Handle group header clicks to show first image"""
    #     selected = self.tree.selection()
    #     if not selected:
    #         return
    #
    #     item = self.tree.item(selected[0])
    #
    #     # Check if it's a group header (has children)
    #     children = self.tree.get_children(selected[0])
    #     if children:
    #         # Get first child's file path
    #         first_child = self.tree.item(children[0])
    #         file_path = first_child['values'][0]
    #         self.display_selected(file_path)
    #     else:
    #         # Regular file item
    #         self.display_selected()

    def _initialize_sift(self):
        """Initialize SIFT detector with version checking"""
        try:
            # Try modern SIFT (OpenCV ≥ 4.4.0)
            return cv2.SIFT_create()
        except AttributeError:
            try:
                # Fallback to legacy SIFT (OpenCV 3.x)
                from cv2.xfeatures2d import SIFT_create
                return SIFT_create()
            except ImportError:
                messagebox.showerror("SIFT Error",
                                     "SIFT requires opencv-contrib-python")
                return None
        except Exception as e:
            messagebox.showerror("SIFT Error", f"Failed to initialize SIFT: {str(e)}")
            return None

    def sift_compare(self):
        """Entry point for SIFT-based similarity search"""
        try:
            self.tree.delete(*self.tree.get_children())
            self.stop_search_flag.clear()  # Reset stop flag when starting new search
        except Exception as e:
            messagebox.showerror("Error", str(e))

        self.status.set("Initializing SIFT search...")

        include_subfolders = self.subfolders.get() == 1
        files = self.list_files(self.added_folders, include_subfolders)
        self.progress["maximum"] = len(files)

        # Limit thread count to reduce CPU load (adjust max_workers as needed)
        max_workers = max(2, os.cpu_count() // 2)  # Reduced from full CPU count
        self.search_thread = threading.Thread(target=self._sift_compare_thread,
                                              args=(files, max_workers))
        self.search_thread.start()

    def _sift_compare_thread(self, files, max_workers):
        processed_count = 0
        files_found = 0  # Track matching files
        start_time = time.time()

        sift = cv2.SIFT_create(contrastThreshold=0.07, edgeThreshold=10)
        query_img = cv2.imread(self.target_image_path, cv2.IMREAD_GRAYSCALE)
        kp1, des1 = sift.detectAndCompute(query_img, None)

        if des1 is None or len(des1) < 10:
            self.status.set("No features found in query image")
            return

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        MIN_MATCHES = 5
        RATIO_THRESH = 0.9
        total_files = len(files)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in files:
                if self.stop_search_flag.is_set():
                    break
                futures.append(executor.submit(
                    self._process_sift_file,
                    file, bf, des1, MIN_MATCHES, RATIO_THRESH
                ))

            for future in as_completed(futures):
                if self.stop_search_flag.is_set():
                    for f in futures:
                        f.cancel()
                    break

                processed_count += 1
                self.progress["value"] = processed_count

                try:
                    result = future.result()
                    if result:
                        files_found += 1
                        # Update GUI in main thread
                        self.tree.insert("", tk.END, values=result)
                except Exception as e:
                    continue

                self.status.set(
                    f"Analyzing files ({processed_count}/{total_files}) - "
                    f"{files_found} matches found"
                )
                self.root.update_idletasks()


        elapsed_time = time.time() - start_time
        stop_status = "stopped" if self.stop_search_flag.is_set() else "completed"
        status_message = (
            f"{stop_status.capitalize()} in {elapsed_time:.2f}s - "
            f"{processed_count} files analyzed, "
            f"{files_found} matches found"
        )
        self.status.set(status_message)
        self.progress["value"] = 0
        self.stop_search_button.configure(state=tk.DISABLED)
        self.stop_search_flag.clear()
        self.search_thread = None
        # Ensure UI is reset even if thread crashes
        self.root.after(0, self._reset_search_ui)

    def _process_sift_file(self, file, bf, des1, min_matches, ratio_thresh):
        if self.stop_search_flag.is_set():
            return None

        try:
            target_img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if target_img is None:
                return None

            # Check stop flag before heavy computation
            if self.stop_search_flag.is_set():
                return None

            sift = cv2.SIFT_create(contrastThreshold=0.07, edgeThreshold=10)
            kp2, des2 = sift.detectAndCompute(target_img, None)

            if des2 is None or len(des2) < min_matches:
                return None

            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)
                if self.stop_search_flag.is_set():
                    return None

            if len(good) >= min_matches:
                similarity = len(good) / len(des1)
                if similarity * 100 >= int(self.sim.get()):
                    # Return data instead of modifying GUI here
                    return (str(file), f"{similarity * 100:.2f}")

            return None  # Explicit return if no match

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            return None

    def ssim_compare(self, *files):
        start_time = time.time()
        analyzed_files_count = 0
        files_found = 0
        total_files = len(files)
        for count, file in enumerate(files, start=1):
            analyzed_files_count += 1
            if self.stop_search_flag.is_set():
                self.status.set(f"Search stopped by user. {analyzed_files_count} files analyzed")
                break

            self.status.set(f"Analyzing files ({analyzed_files_count}/{total_files}) - "
                    f"{files_found} matches found")
            self.root.update_idletasks()

            if Image.open(self.target_image_path).width == Image.open(file).width and Image.open(self.target_image_path).height == Image.open(file).height:
                image = cv2.imdecode(np.fromfile(self.target_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                compare = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                # Convert images to grayscale
                first_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                second_gray = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

                # Compute SSIM between two images
                score, diff = structural_similarity(first_gray, second_gray, full=True)
                if score*100 >= int(self.sim.get()):
                    self.tree.insert("", tk.END, values=(file, f"{score*100:.2f}"))
                    files_found += 1

                # The diff image contains the actual image differences between the two images
                # and is represented as a floating point data type so we must convert the array
                # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
                diff = (diff * 255).astype("uint8")

                # Threshold the difference image, followed by finding contours to
                # obtain the regions that differ between the two images
                thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                # Highlight differences
                #mask = np.zeros(image.shape, dtype='uint8')
                #filled = file.copy()

                # for c in contours:
                #     area = cv2.contourArea(c)
                #     if area > 100:
                #         x, y, w, h = cv2.boundingRect(c)
                #         cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                #         cv2.rectangle(compare, (x, y), (x + w, y + h), (36, 255, 12), 2)
                #         cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                        #cv2.drawContours(filled, [c], 0, (0, 255, 0), -1)

                # if source_hash == calculate_image_hash(file) and self.target_image_path != file:
                #     self.tree.insert("", tk.END, values=(file, "Duplicate"))
                self.progress["value"] = count
                self.root.update_idletasks()
        else:
            self.status.set(f"Completed. {analyzed_files_count} files analyzed")

        end_time = time.time()
        elapsed_time = end_time - start_time
        if analyzed_files_count != 0:
            self.status.set(f"Completed in {elapsed_time:.2f} seconds, {analyzed_files_count} files analyzed, {files_found} similar images found")
            self.progress["value"] = 0
        else:
            messagebox.showinfo("Error", f"No image files found in selected folders. Try ticking the \"Search subfolders\" option")
            self.progress["value"] = 0

        self.reset_ui()
        self.stop_search_button.configure(state=tk.DISABLED)

        self.search_thread = None

    def save_results(self):
        if not self.tree.get_children():
            messagebox.showinfo("Info", "No search results to save.")
            return

        default_save_dir = "C:/ImSearchResults"

        if not os.path.exists(default_save_dir):
            os.makedirs(default_save_dir)

        count = 0
        for file in os.listdir(default_save_dir):
            if file.startswith("result_") and file.endswith(".csv"):
                count += 1

        proposed_filename = f"result_{count + 1}.csv"

        file_path = filedialog.asksaveasfilename(
            initialdir=default_save_dir,
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=proposed_filename
        )

        if not file_path:
            return

        similarity_threshold = self.sim.get()
        search_subfolders = "Yes" if self.subfolders.get() == 1 else "No"

        # save to file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Target Image", self.target_image_path])
            writer.writerow(["Similarity Threshold (%)", similarity_threshold])
            writer.writerow(["Search Subfolders", search_subfolders])
            writer.writerow([])
            writer.writerow(["Name", "Similarity"])
            for item in self.tree.get_children():
                file_name, similarity = self.tree.item(item, "values")
                writer.writerow([file_name, similarity])

        messagebox.showinfo("Info", f"Results saved to {file_path}")

    def load_results(self):
        file_path = filedialog.askopenfilename(
            title="Load Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir="C:/ImSearchResults"
        )

        if not file_path:
            return

        with open(file_path, mode='r') as file:
            reader = csv.reader(file)

            self.tree.delete(*self.tree.get_children())

            target_image_path = None
            similarity_threshold = None
            search_subfolders = None

            for row in reader:
                if not row:  # empty rows
                    continue

                if row[0] == "Target Image":
                    target_image_path = row[1]
                    continue
                elif row[0] == "Similarity Threshold (%)":
                    similarity_threshold = row[1]
                    continue
                elif row[0] == "Search Subfolders":
                    search_subfolders = row[1]
                    continue
                elif row[0] == "Name" and row[1] == "Similarity":
                    continue  # skip header

                if len(row) == 2:
                    file_name, similarity = row
                    self.tree.insert("", tk.END, values=(file_name, similarity))

            if target_image_path:
                self.target_image_path = target_image_path
                self.query_image = Image.open(target_image_path, 'r')
                self.display_uploaded(self.query_image)

            if similarity_threshold:
                self.sim.delete(0, tk.END)
                self.sim.insert(0, similarity_threshold)

            if search_subfolders:
                self.subfolders.set(1 if search_subfolders == "Yes" else 0)

            messagebox.showinfo("Info", f"Results loaded from {file_path}")

    def show_images(self):
        selected_item = self.tree.selection()
        if selected_item:
            selected_file_path = self.tree.item(selected_item)['values'][0]
            pos = selected_file_path.find(" Similarity:")
            if pos > -1:
                selected_file_path = selected_file_path[:pos]

            if self.search_combobox == "Duplicate Groups" and self.target_image_path is None:
                target_image_pil = Image.open(self.tree.item(selected_item)['values'][0])
            else:
                target_image_pil = Image.open(self.target_image_path)
            selected_image_pil = Image.open(selected_file_path)

            new_window = tk.Toplevel(self.root)
            new_window.title("Full Size Images")

            # Create frames
            frame_uploaded = tk.Frame(new_window)
            frame_selected = tk.Frame(new_window)
            frame_uploaded.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            frame_selected.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create canvases
            canvas_target = tk.Canvas(frame_uploaded, bg="white", relief=tk.SUNKEN)
            canvas_selected = tk.Canvas(frame_selected, bg="white", relief=tk.SUNKEN)
            canvas_target.pack(fill=tk.BOTH, expand=True)
            canvas_selected.pack(fill=tk.BOTH, expand=True)

            # Store original images
            canvas_target.original_image = target_image_pil
            canvas_selected.original_image = selected_image_pil

            def on_resize(event):
                for canvas, frame in [(canvas_target, frame_uploaded),
                                      (canvas_selected, frame_selected)]:
                    if not hasattr(canvas, 'original_image'):
                        continue

                    original_image = canvas.original_image
                    available_width = frame.winfo_width()
                    available_height = frame.winfo_height()

                    # Skip if frame has no visible area
                    if available_width <= 1 or available_height <= 1:
                        continue

                    # Calculate aspect ratio-preserving dimensions
                    orig_width, orig_height = original_image.size
                    ratio = min(
                        available_width / orig_width,
                        available_height / orig_height
                    )
                    new_width = int(orig_width * ratio)
                    new_height = int(orig_height * ratio)

                    # Ensure minimum dimensions of 1 pixel
                    new_width = max(1, new_width)
                    new_height = max(1, new_height)

                    try:
                        resized_image = original_image.resize(
                            (new_width, new_height),
                            Image.Resampling.LANCZOS
                        )
                        photo = ImageTk.PhotoImage(resized_image)

                        canvas.delete("all")
                        canvas.create_image(
                            (available_width - new_width) // 2,
                            (available_height - new_height) // 2,
                            anchor=tk.NW,
                            image=photo
                        )
                        canvas.image = photo
                    except Exception as e:
                        print(f"Resize error: {e}")

            new_window.bind("<Configure>", on_resize)
            new_window.after(100, lambda: on_resize(None))


    def open_in_explorer(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showinfo("Info", "No image selected")
            return
        selected_file = self.tree.item(selected_item)['values'][0]
        if " Similarity:" in selected_file:
            selected_file = selected_file.split(" Similarity:")[0]
        selected_file = Path(selected_file).resolve()
        if selected_file.exists():
            if os.name == 'nt':  # Windows
                os.system(f'explorer /select,"{selected_file}"')
            elif os.name == 'posix':
                # macOS
                try:
                    os.system(f'open -R "{selected_file}"')
                except:
                    # Linux
                    os.system(f'xdg-open "{selected_file.parent}"')
        else:
            messagebox.showinfo("Info", f"File {selected_file} does not exist")

    def change_language(self, language):
        """Change application language and update UI text."""
        self.current_language = language

        # Update all UI elements with new language text
        self.add_folder_button.configure(text=self.languages[language]["add_folder"])
        self.remove_folder_button.configure(text=self.languages[language]["remove_folder"])
        self.folder_up_button.configure(text=self.languages[language]["folder_up"])
        self.folder_down_button.configure(text=self.languages[language]["folder_down"])
        self.upload_image_button.configure(text=self.languages[language]["upload_image"])
        self.search_mode_label.configure(text=self.languages[language]["search_mode"] + ":")  # Add colon
        self.similarity_threshold_label.configure(
            text=self.languages[language]["similarity_threshold"] + ":")  # Add colon
        self.delete_selected_button.configure(text=self.languages[language]["delete_selected"])
        self.subfolder_button.configure(text=self.languages[language]["search_subfolders"])
        self.start_search_button.configure(text=self.languages[language]["start_search"])
        self.stop_search_button.configure(text=self.languages[language]["stop_search"])

        # Update language menu label WITHOUT recreating the entire menu
        menubar = self.root.winfo_toplevel().nametowidget("!menu")  # Get existing menu
        settings_menu = menubar.nametowidget(menubar.entrycget(0, "menu"))  # Get "Settings" menu

        # Update the "Language" cascade label
        settings_menu.entryconfig(0, label=self.languages[language]["language"])

        # Update individual language menu items (optional, if you want translated language names)
        language_menu = settings_menu.nametowidget(settings_menu.entrycget(0, "menu"))
        for i, lang in enumerate(self.languages.keys()):
            language_menu.entryconfig(i, label=self.languages[lang]["language_name"])


#run from terminal, experimental
def findSimilar5(self, img_path, folder_path, method):
    try:
        image1 = cv2.imread(Path(img_path))
        hist1 = calculate_histogram(image1)
    except Exception as e:
        tk.messagebox.showerror("Error", f"Unable to read source image: {e}")
        return
    files = self.listFiles(Path(folder_path))
    # if method == 1:
    #     self.search_duplicates()
    # elif method == 2:
    #     self.search_histogram()
    # elif method == 3:
    #     self.ssim_compare()
    # elif method == 4:
    #     self.duplicate_pairs()
    for file in files:
        if file != img_path:  # Exclude the source image itself
            try:
                image2 = cv2.imread(str(file))
                hist2 = calculate_histogram(image2)
                similarity = self.compare_histogram(hist1, hist2)
                if similarity >= 1:
                    print(f"{file} Similarity: {similarity:.2f}%")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

# class Folder_frame(Frame):
#     def __init__(self, parent):
#         super().__init__(parent)
#
#         self.pack(pady=10)


if __name__ == "__main__":
    match len(sys.argv):
        case 1:
            root = ctk.CTk()
            app = ImSearch(root)
            root.mainloop()
        case 4:
            img = sys.argv[1]
            path = sys.argv[2]
            method = sys.argv[3]
            if method=="1":findSimilar5(img, path, method)
        case _:print("Error\n Correct command main.py [query image_path] [folder_search path] [search_method]")