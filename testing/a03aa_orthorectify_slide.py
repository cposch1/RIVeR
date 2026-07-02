#!/usr/bin/env python3
"""
RIVeR-ICE Orthorectification — Interactive GUI

Features:
- Strictly load FRAMES_DIR/_frame_paths.parquet (fast, no implicit re-scan).
- Optional refresh: --refresh-index flag or “Refresh Index” button to rebuild index.
- Three selectors (Listboxes): Camera, Date, Time
  * Background green for items that are already processed (see below).
- “Load Frame” → shows the frame and lets you click 4 GCPs in order.
- “Reset Points” → clears points and immediately allows new clicks (no reload needed).
- “Save & Transform” → all-in-one:
    1) Save GCP image coordinates (CSV + annotated PNG)
    2) Compute and save real-world distances (CSV)
    3) Run orthorectification and save orthorectified PNG
    4) Save transformation matrix JSON
    5) Pop up the orthorectified image (Matplotlib window)
    6) Print save paths to console (no pop-up notifications)
    7) Automatically reset points after saving to avoid mismatches
- Coloring logic:
  - Time item (HHMMSS) → green if transform JSON exists for (camera, date, time)
  - Date item (YYYYMMDD) → green if all times for that date are transformed
  - Camera item → green if all dates & times for that camera are transformed

Environment:
    source ~/RIVeR/testing/setup.sh
    conda activate velo
    python ~/RIVeR/testing/orthorectify.py
    # or to rebuild index first:
    python ~/RIVeR/testing/orthorectify.py --refresh-index
    # or to see river.config logs:
    python ~/RIVeR/testing/orthorectify.py --verbose
"""

from __future__ import annotations

# ---- Quiet-by-default logging (must be before importing river.config) ----
import os as _os
import sys as _sys
import logging as _logging

_verbose = ("--verbose" in _sys.argv) or (_os.environ.get("ORTHO_GUI_VERBOSE") == "1")
_logging.basicConfig(level=_logging.INFO if _verbose else _logging.WARNING)
_logging.getLogger("river.config").setLevel(_logging.INFO if _verbose else _logging.WARNING)

from river.config import *  # noqa: F401,F403

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import os  # used in main() for env check
import numpy as np
import pandas as pd

import rasterio
from rasterio.transform import from_bounds

from skimage import exposure
import cv2

import matplotlib
matplotlib.use("TkAgg")  # embed in Tkinter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

import tkinter as tk
from tkinter import ttk, messagebox


# ---------------------------
# Frames index helpers
# ---------------------------

def collect_frame_paths(frame_dir: Path) -> pd.DataFrame:
    """
    Scan frames directory structured like:
        frame_dir / camera / date / time_hhmmss / *.jpg

    Returns DataFrame: camera, date_yyyymmdd, time_hhmmss, frame_path
    """
    frame_dir = frame_dir.resolve()
    rows = []
    if not frame_dir.exists():
        return pd.DataFrame(columns=["camera", "date_yyyymmdd", "time_hhmmss", "frame_path"])

    for camera_dir in frame_dir.iterdir():
        if not camera_dir.is_dir():
            continue
        camera = camera_dir.name

        for date_dir in camera_dir.iterdir():
            if not date_dir.is_dir():
                continue
            date = date_dir.name  # e.g., "20250426"

            for time_dir in date_dir.iterdir():
                if not time_dir.is_dir():
                    continue
                time_hhmmss = time_dir.name  # already "HHMMSS"

                # Collect all JPG frames
                for jpg in time_dir.glob("*.jpg"):
                    rows.append({
                        "camera": camera,
                        "date_yyyymmdd": date,
                        "time_hhmmss": time_hhmmss,
                        "frame_path": str(jpg.resolve())
                    })

    df = pd.DataFrame(rows, columns=["camera", "date_yyyymmdd", "time_hhmmss", "frame_path"])
    return df


def build_and_save_frames_index(frames_root: Path) -> pd.DataFrame:
    """
    Scan FRAMES_DIR and (re)write _frame_paths.parquet + _frame_paths.csv.
    """
    frames_root = frames_root.resolve()
    df = collect_frame_paths(frames_root)

    parquet_path = frames_root / "_frame_paths.parquet"
    csv_path = frames_root / "_frame_paths.csv"

    # Try Parquet; engine might be missing: ignore on failure
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    # Always write CSV
    df.to_csv(csv_path, index=False)
    return df


def load_frames_index_only(frames_root: Path) -> pd.DataFrame:
    """
    Load FRAMES_DIR/_frame_paths.parquet exclusively.
    Do NOT rescan the filesystem here.
    """
    frames_root = frames_root.resolve()
    parquet_path = frames_root / "_frame_paths.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Frames index not found: {parquet_path}\n"
            f"Run frame extraction (which writes the index) or refresh the index manually."
        )

    try:
        return pd.read_parquet(parquet_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read frames index at {parquet_path}: {e}")



# ---------------------------
# Orthorectification helpers
# ---------------------------

def calc_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calc_distances_world(gcp_cam: str) -> Dict[str, float]:
    point_coords_world = load_gcps_real(gcp_cam)  # from RIVeR
    distances = {
        'd12': calc_dist(point_coords_world['point1'], point_coords_world['point2']),
        'd23': calc_dist(point_coords_world['point2'], point_coords_world['point3']),
        'd34': calc_dist(point_coords_world['point3'], point_coords_world['point4']),
        'd41': calc_dist(point_coords_world['point4'], point_coords_world['point1']),
        'd13': calc_dist(point_coords_world['point1'], point_coords_world['point3']),
        'd24': calc_dist(point_coords_world['point2'], point_coords_world['point4']),
    }
    return distances


def transform_json_path(cam: str, date: str, time_: str) -> Path:
    """Path to the transformation JSON for this (camera, date, time)."""
    return rect_dir / cam / f"{cam}_transform_{date}_{time_}.json"


def global_extent_path(cam: str, absolute_mode: bool) -> Path:

    if absolute_mode:
        return rect_dir / cam / f"{cam}_global_extent_epsg32623.json"

    return rect_dir / cam / f"{cam}_global_extent_local.json"


# ---------------------------
# GEOTIFF export helpers
# ---------------------------


def save_geotiff(
    img: np.ndarray,
    extent,
    output_path: Path,
    epsg: int = 32623
):
    """
    Save orthorectified raster as GeoTIFF.

    Parameters
    ----------
    img : ndarray
        Orthorectified image.

    extent : tuple/list
        [xmin, xmax, ymin, ymax]
        Real-world coordinates in EPSG:32623.

    output_path : Path
        Output GeoTIFF path.

    epsg : int
        CRS EPSG code.
    """

    xmin, xmax, ymin, ymax = extent

    height, width = img.shape[:2]

    transform = from_bounds(
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if img.ndim == 2:

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=img.dtype,
            crs=f"EPSG:{epsg}",
            transform=transform,
            compress="lzw"
        ) as dst:

            dst.write(img, 1)

    else:

        bands = img.shape[2]

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=img.dtype,
            crs=f"EPSG:{epsg}",
            transform=transform,
            compress="lzw"
        ) as dst:

            for i in range(bands):
                dst.write(img[:, :, i], i + 1)


# ---------------------------
# Image extent helpers
# ---------------------------

def crop_image_to_extent(img, source_extent, target_extent):

    sxmin, sxmax, symin, symax = source_extent
    txmin, txmax, tymin, tymax = target_extent

    h, w = img.shape[:2]

    col0 = int((txmin - sxmin) / (sxmax - sxmin) * w)
    col1 = int((txmax - sxmin) / (sxmax - sxmin) * w)

    row1 = int((symax - tymin) / (symax - symin) * h)
    row0 = int((symax - tymax) / (symax - symin) * h)

    col0 = max(0, col0)
    col1 = min(w, col1)

    row0 = max(0, row0)
    row1 = min(h, row1)

    cropped_img = img[row0:row1, col0:col1]

    # compute TRUE extent from actual pixel indices

    new_xmin = sxmin + (col0 / w) * (sxmax - sxmin)
    new_xmax = sxmin + (col1 / w) * (sxmax - sxmin)

    new_ymax = symax - (row0 / h) * (symax - symin)
    new_ymin = symax - (row1 / h) * (symax - symin)

    cropped_extent = (
        new_xmin,
        new_xmax,
        new_ymin,
        new_ymax
    )

    return cropped_img, cropped_extent


# ---------------------------
# GUI App
# ---------------------------

class OrthoApp:
    def __init__(self, master: tk.Tk, df_frames: pd.DataFrame, frames_root: Path, xlim=None, ylim=None):
        self.master = master
        self.df_frames = df_frames.copy()
        self.frames_root = frames_root

        self.custom_xlim = xlim
        self.custom_ylim = ylim

        self.global_extent: Optional[Tuple[float, float, float, float]] = None

        self.master.title("RIVeR-ICE Orthorectification")
        self.master.geometry("1280x820")

        # State
        self.selected_camera: Optional[str] = None
        self.selected_date: Optional[str] = None
        self.selected_time: Optional[str] = None

        self.points: List[Tuple[int, int]] = []
        self._mpl_cid = None
        self.current_img = None
        self.current_frame_path: Optional[Path] = None
        self._transformation = None
        self._absolute_mode = False
        self.display_img = None

        self.base_points: Optional[List[Tuple[int, int]]] = None
        self.point_offsets = [0, 0, 0, 0, 0]
        self.sliders = []

        # Build UI
        self._build_ui()
        self._populate_cameras()

        
    def _use_manual_extent(self):
            return (
                self.custom_xlim is not None
                and self.custom_ylim is not None
            )


    # ---- UI ----

    def _build_ui(self):
        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        # Left: Camera list
        lf_cam = ttk.LabelFrame(top, text="Camera")
        lf_cam.pack(side=tk.LEFT, padx=(0, 8), pady=4, fill=tk.Y)
        self.lb_camera = tk.Listbox(lf_cam, height=10, exportselection=False)
        self.lb_cam_scroll = ttk.Scrollbar(lf_cam, orient="vertical", command=self.lb_camera.yview)
        self.lb_camera.configure(yscrollcommand=self.lb_cam_scroll.set, width=28)
        self.lb_camera.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lb_cam_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_camera.bind("<<ListboxSelect>>", self._on_camera_selected)

        # Middle: Date list
        lf_date = ttk.LabelFrame(top, text="Date (YYYYMMDD)")
        lf_date.pack(side=tk.LEFT, padx=(0, 8), pady=4, fill=tk.Y)
        self.lb_date = tk.Listbox(lf_date, height=10, exportselection=False)
        self.lb_date_scroll = ttk.Scrollbar(lf_date, orient="vertical", command=self.lb_date.yview)
        self.lb_date.configure(yscrollcommand=self.lb_date_scroll.set, width=16)
        self.lb_date.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lb_date_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_date.bind("<<ListboxSelect>>", self._on_date_selected)

        # Right: Time list
        lf_time = ttk.LabelFrame(top, text="Time (HHMMSS)")
        lf_time.pack(side=tk.LEFT, padx=(0, 8), pady=4, fill=tk.Y)
        self.lb_time = tk.Listbox(lf_time, height=10, exportselection=False)
        self.lb_time_scroll = ttk.Scrollbar(lf_time, orient="vertical", command=self.lb_time.yview)
        self.lb_time.configure(yscrollcommand=self.lb_time_scroll.set, width=12)
        self.lb_time.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lb_time_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_time.bind("<<ListboxSelect>>", self._on_time_selected)

        # Right-most: GCP sliders
        slider_frame = ttk.LabelFrame(
            top,
            text="Adjust GCPs (Vertical)"
        )
        slider_frame.pack(
            side=tk.LEFT,
            padx=(0, 8),
            pady=4,
            fill=tk.Y
        )
        self.sliders = []
        for i in range(5):
        
            row = ttk.Frame(slider_frame)
            row.pack(
                fill=tk.X,
                padx=4,
                pady=2
            )
            ttk.Label(
                row,
                text=f"P{i+1}"
            ).pack(
                side=tk.LEFT
            )
            s = tk.Scale(
                row,
                from_=-200,
                to=200,
                orient=tk.HORIZONTAL,
                length=180,
                resolution=1,
                command=lambda val, idx=i:
                    self._on_slider_move(idx, val),
                state=tk.DISABLED
            )
            s.pack(
                side=tk.LEFT,
                padx=5
            )
            self.sliders.append(s)

        # Buttons
        btns = ttk.Frame(self.master)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 8))

        self.btn_load = ttk.Button(btns, text="Load Frame", command=self.load_frame_selection)
        self.btn_load.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_reset = ttk.Button(btns, text="Reset Points", command=self.reset_points, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.LEFT, padx=(0, 8))

        # Single action button that saves GCPs, distances, runs transform, saves outputs, and shows image
        self.btn_save_transform = ttk.Button(btns, text="Save & Transform", command=self.save_and_transform, state=tk.DISABLED)
        self.btn_save_transform.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_refresh = ttk.Button(btns, text="Refresh Index", command=self._refresh_index)
        self.btn_refresh.pack(side=tk.LEFT, padx=(0, 8))

        # Local/absolute coord check
        self.use_absolute_coords = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            btns,
            text="Use EPSG:32623\n(enables GEOTIF export)",
            variable=self.use_absolute_coords
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Display buffer
        ttk.Label(btns, text="Image display and export extent\nbuffer around GCPs (m):").pack(side=tk.LEFT, padx=(15, 2))
        
        self.buffer_var = tk.DoubleVar(value=20.0)
        
        self.buffer_entry = ttk.Entry(
            btns,
            textvariable=self.buffer_var,
            width=6
        )
        
        self.buffer_entry.pack(side=tk.LEFT)
        
        
        # Display clipper
        ttk.Label(btns, text="GEOTIF export extent\nbuffer around GCPs (m):").pack(side=tk.LEFT, padx=(15, 2))
        
        self.clip_var = tk.DoubleVar(value=20.0)
        
        self.clip_entry = ttk.Entry(
            btns,
            textvariable=self.clip_var,
            width=6
        )
        
        self.clip_entry.pack(side=tk.LEFT)

        # Buffer/clipper availability
        if self._use_manual_extent():
            self.buffer_entry.configure(state="disabled")
            self.clip_entry.configure(state="disabled")

        # Contrast enhancement
        ttk.Label(btns, text="CLAHE\n(contrast enhancement)").pack(side=tk.LEFT, padx=(15, 2))
        self.clahe_var = tk.IntVar(value=0)
        
        self.clahe_slider = tk.Scale(
            btns,
            from_=0,
            to=5,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.clahe_var,
            command=self._update_contrast,
            length=150
        )
        self.clahe_slider.pack(side=tk.LEFT)


        # Fan line toggling
        self.show_fans = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            btns,
            text="Show fan lines",
            variable=self.show_fans,
            command=self._apply_offsets_and_draw
        ).pack(side=tk.LEFT, padx=(15,0))

        # Status line
        self.status = tk.StringVar(value="Select Camera, Date, Time; then Load Frame.")
        ttk.Label(self.master, textvariable=self.status).pack(side=tk.TOP, anchor="w", padx=8)

        
        # Matplotlib Figure embedded in Tk
        self.fig: Figure = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # ---- Population & coloring ----

    def _populate_cameras(self):
        self.lb_camera.delete(0, tk.END)
        cams = sorted(self.df_frames["camera"].unique())
        for i, cam in enumerate(cams):
            self.lb_camera.insert(tk.END, cam)
            if self._camera_all_done(cam):
                self.lb_camera.itemconfig(i, {'bg': '#d0f0d0'})  # light green
        # Reset selections
        self.selected_camera = None
        self.selected_date = None
        self.selected_time = None

        self.lb_date.delete(0, tk.END)
        self.lb_time.delete(0, tk.END)

        if cams:
            self.lb_camera.selection_set(0)
            self._on_camera_selected()

    def _populate_dates(self):
        self.lb_date.delete(0, tk.END)
        if not self.selected_camera:
            return
        dates = sorted(self.df_frames.loc[self.df_frames["camera"] == self.selected_camera, "date_yyyymmdd"].unique())
        for i, date in enumerate(dates):
            self.lb_date.insert(tk.END, date)
            if self._date_all_done(self.selected_camera, date):
                self.lb_date.itemconfig(i, {'bg': '#d0f0d0'})  # light green

        self.selected_date = None
        self.lb_time.delete(0, tk.END)
        if dates:
            self.lb_date.selection_set(0)
            self._on_date_selected()

    def _populate_times(self):
        self.lb_time.delete(0, tk.END)
        if not (self.selected_camera and self.selected_date):
            return
        times = sorted(self.df_frames[
            (self.df_frames["camera"] == self.selected_camera) &
            (self.df_frames["date_yyyymmdd"] == self.selected_date)
        ]["time_hhmmss"].unique())
        for i, t in enumerate(times):
            self.lb_time.insert(tk.END, t)
            if self._transform_exists(self.selected_camera, self.selected_date, t):
                self.lb_time.itemconfig(i, {'bg': '#d0f0d0'})  # light green
        self.selected_time = None
        if times:
            self.lb_time.selection_set(0)
            self._on_time_selected()

    # ---- Completion checks ----

    def _transform_exists(self, cam: str, date: str, time_: str) -> bool:
        return transform_json_path(cam, date, time_).exists()

    def _date_all_done(self, cam: str, date: str) -> bool:
        df = self.df_frames
        times = set(df[(df["camera"] == cam) & (df["date_yyyymmdd"] == date)]["time_hhmmss"].unique())
        if not times:
            return False
        for t in times:
            if not self._transform_exists(cam, date, t):
                return False
        return True

    def _camera_all_done(self, cam: str) -> bool:
        df = self.df_frames
        dates = set(df[df["camera"] == cam]["date_yyyymmdd"].unique())
        if not dates:
            return False
        for d in dates:
            if not self._date_all_done(cam, d):
                return False
        return True

    # ---- Selection handlers ----

    def _clear_points_state(self, due_to_selection_change: bool):
        """Clear selected points; optionally also clear image & clicks when selection changes."""
        self.points = []
        self.btn_save_transform["state"] = tk.DISABLED
        self._disable_clicks()
        self.base_points = None
        self.point_offsets = [0, 0, 0, 0, 0]
        
        for s in getattr(self, "sliders", []):
            s.set(0)
            s["state"] = tk.DISABLED
        if due_to_selection_change:
            # Clear the displayed image to force reloading the correct one for new selection
            self.current_img = None
            self.current_frame_path = None
            self.ax.axis('off')
            self.canvas.draw_idle()
            self.status.set("Selection changed. Click 'Load Frame' to continue.")
        else:
            # Keep current image and allow immediate re-picking
            if self.current_img is not None:
                self._draw_image()
                self._enable_clicks()
                self.status.set("Points reset. Click the 4 GCPs in order.")

    def _on_camera_selected(self, *args):
        sel = self.lb_camera.curselection()
        if not sel:
            self.selected_camera = None
            return
            
        self.selected_camera = self.lb_camera.get(sel[0])
        self.global_extent = None

        self._populate_dates()
        self._clear_points_state(due_to_selection_change=True)

    def _on_date_selected(self, *args):
        sel = self.lb_date.curselection()
        if not sel:
            self.selected_date = None
            return
        self.selected_date = self.lb_date.get(sel[0])
        self._populate_times()
        self._clear_points_state(due_to_selection_change=True)

    def _on_time_selected(self, *args):
        sel = self.lb_time.curselection()
        if not sel:
            self.selected_time = None
            return
        self.selected_time = self.lb_time.get(sel[0])
        self._clear_points_state(due_to_selection_change=True)

    # ---- Slider handlers ----
    def _try_load_previous_gcps(self):
        cam = self.selected_camera
    
        gcps_cam_dir = gcps_dir / cam
        if not gcps_cam_dir.exists():
            return
    
        files = sorted(
            gcps_cam_dir.glob(
                f"{cam}_gcps_img_*.csv"
            )
        )
        
        if not files:
            return
        
        current_dt = pd.to_datetime(
            f"{self.selected_date}{self.selected_time}",
            format="%Y%m%d%H%M%S"
        )
        
        candidate_files = []
        
        for f in files:
        
            try:
        
                stem = f.stem
        
                parts = stem.split("_")
        
                file_date = parts[-2]
                file_time = parts[-1]
        
                file_dt = pd.to_datetime(
                    f"{file_date}{file_time}",
                    format="%Y%m%d%H%M%S"
                )
        
                if file_dt < current_dt:
        
                    candidate_files.append(
                        (
                            file_dt,
                            f
                        )
                    )
        
            except Exception:
                continue
        
        if not candidate_files:
        
            print(
                "[INFO] No previous GCP solution found"
            )
        
            return
        
        latest_file = max(
            candidate_files,
            key=lambda x: x[0]
        )[1]
        
        print(
            f"[INFO] Using previous GCPs: "
            f"{latest_file.name}"
        )
    
        try:
            pts = []
    
            with latest_file.open("r") as f:
                reader = csv.reader(f)
            
                for row in reader:
                    pts.append(
                        (
                            int(float(row[0])),
                            int(float(row[1]))
                        )
                    )
            
            stake_file = (
                gcps_dir
                / cam
                / f"{cam}_stake_point_{file_date}_{file_time}.csv"
            )
            
            if stake_file.exists():
            
                with stake_file.open("r") as f:
            
                    row = next(csv.reader(f))
            
                    pts.append(
                        (
                            int(float(row[0])),
                            int(float(row[1]))
                        )
                    )
            
            print(f"Loaded from: {latest_file}")
            print(f"Raw file had {len(pts)} points")
            print(pts)

    
            if len(pts) in (4, 5):
                
                
                if len(pts) == 4:
                
                    print(
                        "[INFO] No stake point found in previous solution."
                    )

                print(f"Loaded {len(pts)} points")
                print(pts)
                self.base_points = pts
                print("Reference GCPs loaded")
                self.point_offsets = [0, 0, 0, 0, 0]
    
                for i, s in enumerate(self.sliders):

                    s.set(0)
                
                    if i < len(self.base_points):
                        s["state"] = tk.NORMAL
                    else:
                        s["state"] = tk.DISABLED
    
                self._apply_offsets_and_draw()
    
                self._disable_clicks()
    
                self.status.set(
                    "Loaded previous GCPs. Adjust each point with sliders."
                )
    
        except Exception as e:
            print(f"[WARN] Failed to load previous GCPs: {e}")

    
    
    def _on_slider_move(self, idx, val):
    
        if self.base_points is None:
            return
    
        if idx >= len(self.base_points):
            return
    
        self.point_offsets[idx] = int(float(val))
    
        self._apply_offsets_and_draw()



    def _apply_offsets_and_draw(self):
        
        if self.base_points is None or self.current_img is None:
            return
        print("base_points:", len(self.base_points))
        print("point_offsets:", len(self.point_offsets))
        print(self.point_offsets)
   
        # Save current zoom/pan state
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
    
        n = min(
            len(self.base_points),
            len(self.point_offsets)
        )
        
        self.points = [
            (
                self.base_points[i][0],
                self.base_points[i][1] + self.point_offsets[i]
            )
            for i in range(n)
        ]
    
        self._draw_image()
    
        # Restore zoom/pan
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
    
        for i, (x, y) in enumerate(self.points, start=1):
    
            color = 'black' if i == 1 else 'black'
    
            self.ax.plot(
                x,
                y,
                'o',
                color=color,
                markersize=3
            )
    
            
            label = "P5" if i == 5 else str(i)
            
            self.ax.text(
                x,
                y,
                label,
                color=color,
                fontsize=8,
                ha='left',
                va='bottom'
            )
    
        self.canvas.draw_idle()
    
        self._disable_clicks()
        self.btn_save_transform["state"] = tk.NORMAL


    # Reference fans handler

    def _draw_gcp_fans(self):

        if self.base_points is None:
            return
    
        x1, y1 = self.base_points[0]
        x2, y2 = self.base_points[1]
        x3, y3 = self.base_points[2]
        x4, y4 = self.base_points[3]

    
        offsets = np.arange(
            -100,
            101,
            10
        )
    
        for dy in offsets:
            self.ax.plot(
                [x2, x1],
                [y2, y1 + dy],
                color="0.6",      # light grey
                alpha=0.3,
                linewidth=1
            )
        for dy in offsets:
        
            self.ax.plot(
                [x3, x4],
                [y3, y4 + dy],
                color="0.6",      # light grey
                alpha=0.3,
                linewidth=1
            )

        if len(self.point_offsets) >= 4:
            self.ax.plot(
            [x2, x1],
            [
                y2 + self.point_offsets[1],
                y1 + self.point_offsets[0]
            ],
            color="0.6",
            alpha=0.6,
            linewidth=1.5
        )
        
        self.ax.plot(
            [x3, x4],
            [
                y3 + self.point_offsets[2],
                y4 + self.point_offsets[3]
            ],
            color="0.6",
            alpha=0.6,
            linewidth=1.5
        )

    def _draw_gcp_lines(self):

        if len(self.points) < 4:
            return
    
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        x3, y3 = self.points[2]
        x4, y4 = self.points[3]
    
        self.ax.plot(
            [x1, x2],
            [y1, y2],
            color="0.4",
            linewidth=1.5
        )
    
        self.ax.plot(
            [x4, x3],
            [y4, y3],
            color="0.4",
            linewidth=1.5
        )


    # ---- Core actions ----

    def load_frame_selection(self):
        if not (self.selected_camera and self.selected_date and self.selected_time):
            messagebox.showwarning("Selection", "Please select Camera, Date, and Time.")
            return

        try:
            frame, frame_rgb, frame_path = load_frame(self.df_frames, self.selected_camera,
                                                      self.selected_date, self.selected_time)
        except Exception as e:
            messagebox.showerror("Load Frame", f"Failed to load frame: {e}")
            return

        self.current_frame_path = Path(frame_path)
        try:
            self.current_img = mpimg.imread(str(self.current_frame_path))
            self.display_img = self.current_img.copy()
        except Exception as e:
            messagebox.showerror("Read Image", f"Failed to read image from path:\n{self.current_frame_path}\n\n{e}")
            return

        # Show image & prepare click capture
        self._draw_image()
        self._enable_clicks()
        self.points = []
        self._try_load_previous_gcps()
        self.status.set(f"Loaded: {self.selected_camera} {self.selected_date} {self.selected_time}. Click 4 points in order.")
        self.btn_reset["state"] = tk.NORMAL
        self.btn_save_transform["state"] = tk.DISABLED  # enable after 4 points
        self.toolbar.zoom()

    def _draw_image(self):
        self.ax.clear()
        self.ax.axis("off")
        if self.current_img is not None:
            img_to_show = (
                self.display_img
                if self.display_img is not None
                else self.current_img
            )
            self.ax.imshow(img_to_show)

            self._draw_gcp_lines()
            if self.show_fans.get():
                self._draw_gcp_fans()

            self.ax.set_title(
                f"Select GCPs:\n"
                f"1) left upstream\n2) right upstream\n3) right downstream\n4) left downstream\n\n"
                f"{self.current_frame_path}"
            )
        self.canvas.draw_idle()

    
    def _update_contrast(self, value=None):

        if self.current_img is None:
            return
    
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
    
        img = self.current_img.astype(np.float32)
    
        if img.max() > 1:
            img = img / 255.0
    
        # --------------------------------------------------
        # Estimate valid range only from image center
        # --------------------------------------------------
    
        h, w = img.shape[:2]
    
        center = img[
            int(0.33*h):int(0.66*h),
            int(0.25*w):int(0.75*w)
        ]
    
        # optional: ignore almost-black pixels
        valid = center[center > 0.02]
    
        if valid.size > 0:
    
            lo = np.percentile(valid, 1)
            hi = np.percentile(valid, 99)
    
            img = (img - lo) / (hi - lo)
            img = np.clip(img, 0, 1)
    
        strength = self.clahe_var.get()

        
        # No enhancement at all
        if strength == 0:
        
            self.display_img = self.current_img
        
            self._draw_image()

            for i, (x, y) in enumerate(self.points, start=1):
                color = 'black'
                self.ax.plot(
                    x,
                    y,
                    'o',
                    color=color,
                    markersize=3
                )
                self.ax.text(
                    x,
                    y,
                    str(i),
                    color=color,
                    fontsize=8,
                    ha='left',
                    va='bottom'
                )
        
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        
            self.canvas.draw_idle()
        
            return


        if strength > 0:
        
            clip_limit_map = {
                1: 0.005,
                2: 0.01,
                3: 0.02,
                4: 0.05,
                5: 0.10,
            }
        
            img = exposure.equalize_adapthist(
                img,
                clip_limit=clip_limit_map[strength]
            )
    
        self.display_img = img
    
        self._draw_image()

        # Re-draw existing GCPs
        for i, (x, y) in enumerate(self.points, start=1):
        
            color = 'black'
        
            self.ax.plot(
                x,
                y,
                'o',
                color=color,
                markersize=3
            )
        
            
            label = "P5" if i == 5 else str(i)
            
            self.ax.text(
                x,
                y,
                label,
                color=color,
                fontsize=8,
                ha='left',
                va='bottom'
            )
    
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
    
        self.canvas.draw_idle()



    def _enable_clicks(self):
        if self._mpl_cid:
            self.canvas.mpl_disconnect(self._mpl_cid)
        self._mpl_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _disable_clicks(self):
        if self._mpl_cid:
            self.canvas.mpl_disconnect(self._mpl_cid)
            self._mpl_cid = None

    def reset_points(self):
        if self.current_img is None:
            return
    
        self._clear_points_state(
            due_to_selection_change=False
        )
    
        self.base_points = None
        self.point_offsets = [0, 0, 0, 0, 0]
    
        for s in self.sliders:
            for i, s in enumerate(self.sliders):
                s.set(0)
            
                if i < len(self.base_points):
                    s["state"] = tk.NORMAL
                else:
                    s["state"] = tk.DISABLED

    def _on_click(self, event):
        if hasattr(self, "toolbar") and self.toolbar.mode != "":
            return
        if self.current_img is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        n = len(self.points) + 1

        # Register and draw
        self.points.append((x, y))
        if n == 1:
            self.ax.plot(x, y, 'o', color='black', markersize=3)  # red for point 1
            self.ax.text(x, y, "1", color='black', fontsize=8, ha='left', va='bottom')
        else:
            self.ax.plot(x, y, 'o', color='black', markersize=3)  # blue for 2-4
            self.ax.text(x, y, f"{n}", color='black', fontsize=8, ha='left', va='bottom')
        self.canvas.draw_idle()

        if n == 4:
            self.status.set(
                "4 GCPs selected. Click stake point not affected by surface hydrology."
            )
        
            self.btn_save_transform["state"] = tk.NORMAL

        if n == 5:
            self._disable_clicks()
        
            self.status.set(
                "Stake point not affected by surface hydrology.\nAll required points selected."
            )

    def save_and_transform(self):
        """Single action: save GCPs (CSV+PNG), distances CSV, run transform, save outputs, show ortho image."""
        
        if len(self.points) < 4:
            messagebox.showwarning(
                "GCPs",
                "Please select at least the 4 orthorectification GCPs."
            )
            return

        if not (self.selected_camera and self.selected_date and self.selected_time):
            messagebox.showwarning("Selection", "Please select Camera, Date, and Time.")
            return

        cam = self.selected_camera
        date = self.selected_date
        time_ = self.selected_time

       
        # Check if transformation already exists
        transf_file = transform_json_path(cam, date, time_)
        
        if transf_file.exists():
            choice = messagebox.askyesno(
                "Overwrite?",
                f"Transformation already exists for:\n"
                f"{cam} {date} {time_}\n\n"
                f"Do you want to overwrite it?"
            )
            if not choice:
                print("[Info] Operation cancelled by user (no overwrite).")
                return
            else:
                # ✅ HARD overwrite: remove all existing outputs first
        
                try:
                    # JSON
                    transf_file.unlink(missing_ok=True)
        
                    # Orthorectified PNG (with overlays)
                    ortho_dir = rect_dir / cam / "orthorectification_imgs"
                    ortho_img = ortho_dir / f"{cam}_orthorect_{date}_{time_}.png"
                    ortho_img.unlink(missing_ok=True)
        
                    # Clean ortho image
                    ortho_clean_dir = rect_dir / cam / "ortho_imgs"
                    ortho_clean = ortho_clean_dir / f"{cam}_orthoimg_{date}_{time_}.png"
                    ortho_clean.unlink(missing_ok=True)

        
                except Exception as e:
                    print(f"[WARN] Failed to clean previous outputs: {e}")


        # ---- Save GCP image coordinates
        try:
            gcps_img_file = gcps_dir / cam / f"{cam}_gcps_img_{date}_{time_}.csv"
            gcps_img_file.parent.mkdir(parents=True, exist_ok=True)
            with gcps_img_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.points)

            gcps_img_dir = gcps_dir / cam / "gcps_imgs" 
            gcps_img_dir.mkdir(parents=True, exist_ok=True)
            gcps_img_png = gcps_img_dir / f"{cam}_gcps_img_{date}_{time_}.png"
            # Save the current annotated figure
            self.fig.savefig(str(gcps_img_png))

            print(f"[Saved] GCP image coords CSV: {gcps_img_file}")
            print(f"[Saved] GCP annotated PNG:   {gcps_img_png}")
        except Exception as e:
            messagebox.showerror("Save GCPs", f"Failed to save GCP coordinates:\n{e}")
            return

        # ---- Compute & save distances
        try:
            distances = calc_distances_world(cam)
            distances_print = {k: round(v, 2) for k, v in distances.items()}
            print("[Info] GCPs real-world distances:", distances_print)

            gcps_dist_file = gcps_dir / cam / f"{cam}_gcps_dist.csv"
            gcps_dist_file.parent.mkdir(parents=True, exist_ok=True)
            with gcps_dist_file.open("w", newline="") as f:
                writer = csv.writer(f)
                for _, value in distances_print.items():
                    writer.writerow([value])
            print(f"[Saved] Distances CSV: {gcps_dist_file}")
        except Exception as e:
            messagebox.showerror("Distances", f"Failed to compute/save distances:\n{e}")
            return

        # ---- Run transform
        try:
            gcps_img_file = (
                gcps_dir
                / cam
                / f"{cam}_gcps_img_{date}_{time_}.csv"
            )

            # Save GCPs
            with gcps_img_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.points[:4])

            # Save metric point
            if len(self.points) >= 5:
                metric_file = (
                    gcps_dir
                    / cam
                    / f"{cam}_stake_point_{date}_{time_}.csv"
                )
                with metric_file.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.points[4])
            
            transformation = transform(
                self.df_frames,
                cam,
                date,
                time_,
                absolute_coords=self.use_absolute_coords.get()
            )

            self._transformation = transformation
            self._absolute_mode = self.use_absolute_coords.get()
           
           
            print("Extent:")
            print(transformation["extent"])
            
            print("Transformation matrix:")
            print(np.array(transformation["transformation_matrix"]))


        except Exception as e:
            messagebox.showerror("Transform", f"Transformation failed:\n{e}")
            return

        # ---- Visualize + save orthorectified image and matrix
        try:
            self._visualize_and_save(self._transformation, cam, date, time_)
        except Exception as e:
            messagebox.showerror("Visualization", f"Failed to visualize/save orthorectification:\n{e}")
            return

        # Update coloring for this time/date/camera as done
        self._populate_times()
        self._populate_dates()
        self._populate_cameras()

        # Auto-reset points after Save & Transform to avoid mismatches
        self._clear_points_state(due_to_selection_change=False)
        self.status.set("Saved & transformed successfully. Points reset.")

    def _visualize_and_save(self, transformation: dict, cam: str, date: str, time_: str):
        # Prepare original image
        img = self.current_img
        if img is None and self.current_frame_path is not None:
            img = mpimg.imread(str(self.current_frame_path))
        if img is None:
            raise RuntimeError("No image to visualize.")

        # Load saved or current points
        try:
            points_dict = load_gcps_img(cam, date, time_)  # from RIVeR
            x1_pix, y1_pix = points_dict['point1']
            x2_pix, y2_pix = points_dict['point2']
            x3_pix, y3_pix = points_dict['point3']
            x4_pix, y4_pix = points_dict['point4']
        except Exception:
            if len(self.points) < 4:
                messagebox.showwarning(
                    "GCPs",
                    "Please select at least the 4 orthorectification GCPs."
                )
                return

            (x1_pix, y1_pix), (x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix) = self.points

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Original image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Original Image')

        # Draw lines in original
        ax1.plot([x1_pix, x2_pix], [y1_pix, y2_pix], color='#6CD4FF', linewidth=2)
        ax1.plot([x2_pix, x3_pix], [y2_pix, y3_pix], color='#62C655', linewidth=2)
        ax1.plot([x3_pix, x4_pix], [y3_pix, y4_pix], color='#ED6B57', linewidth=2)
        ax1.plot([x4_pix, x1_pix], [y4_pix, y1_pix], color='#F5BF61', linewidth=2)
        ax1.plot([x1_pix, x3_pix], [y1_pix, y3_pix], color='#CC4BC2', linewidth=2)
        ax1.plot([x2_pix, x4_pix], [y2_pix, y4_pix], color='#7765E3', linewidth=2)

        # Points
        ax1.plot(x1_pix, y1_pix, 'o', color='black', markersize=3)
        ax1.text(x1_pix, y1_pix, "1", color='black', fontsize=8, ha='left', va='bottom')
        ax1.plot([x2_pix, x3_pix, x4_pix], [y2_pix, y3_pix, y4_pix], 'o', color='black', markersize=3)
        pts = [(x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix)]
        [ax1.text(x, y, str(i), color='black', fontsize=8, ha='left', va='bottom') for i, (x, y) in enumerate(pts, start=2)]

        
        # Orthorectified image with overlay
        if 'transformed_img' in transformation and 'extent' in transformation:
        
            p = global_extent_path(cam, self._absolute_mode)
            p.parent.mkdir(parents=True, exist_ok=True)
        
            xmin, xmax, ymin, ymax = transformation["extent"]
        
            buffer_m = self.buffer_var.get()
            clip_m = self.clip_var.get()
        
            # --------------------------------------------------
            # CASE 1: Manual extent provided via CLI
            # --------------------------------------------------
            if self._use_manual_extent():
        
                display_extent = (
                    self.custom_xlim[0],
                    self.custom_xlim[1],
                    self.custom_ylim[0],
                    self.custom_ylim[1]
                )
        
                clip_extent = display_extent
        
                self.global_extent = display_extent
        
            # --------------------------------------------------
            # CASE 2: EPSG mode
            # --------------------------------------------------
            elif self._absolute_mode:
        
                gcps = load_gcps_real(cam)
        
                xs = [coord[0] for coord in gcps.values()]
                ys = [coord[1] for coord in gcps.values()]
        
                display_extent = (
                    min(xs) - buffer_m,
                    max(xs) + buffer_m,
                    min(ys) - buffer_m,
                    max(ys) + buffer_m
                )
        
                clip_extent = (
                    min(xs) - clip_m,
                    max(xs) + clip_m,
                    min(ys) - clip_m,
                    max(ys) + clip_m
                )
        
                self.global_extent = display_extent
        
            # --------------------------------------------------
            # CASE 3: Local coordinates
            # --------------------------------------------------
            else:
                rw_points = []
            
                for x, y in [
                    (x1_pix, y1_pix),
                    (x2_pix, y2_pix),
                    (x3_pix, y3_pix),
                    (x4_pix, y4_pix)
                ]:
                    rw = transform_pixel_to_real_world(
                        x,
                        y,
                        transformation["transformation_matrix"]
                    )
                    rw_points.append(rw)
            
                rw_points = np.array(rw_points)
            
                xs = rw_points[:, 0]
                ys = rw_points[:, 1]
            
                display_extent = (
                    np.min(xs) - buffer_m,
                    np.max(xs) + buffer_m,
                    np.min(ys) - buffer_m,
                    np.max(ys) + buffer_m
                )
            
                clip_extent = None
                self.global_extent = display_extent
                
        
            # Save global extent once
            if not p.exists():
                with p.open("w") as f:
                    json.dump(display_extent, f, indent=2)
                
                print(f"[Saved] Global extent: {p}")

        
            # ----- draw raster in its TRUE world position -----
            ax2.imshow(
                transformation['transformed_img'],
                extent=transformation['extent']
            )
        
            # ----- lock the map frame -----
            
            # X limits
            
            if (
                not getattr(self, "_absolute_mode", False)
                and self.custom_xlim is not None
            ):

                ax2.set_xlim(self.custom_xlim[0], self.custom_xlim[1])
            else:
                ax2.set_xlim(display_extent[0], display_extent[1])

            # Y limits
            
            if (
                not getattr(self, "_absolute_mode", False)
                and self.custom_ylim is not None
            ):

                ax2.set_ylim(self.custom_ylim[0], self.custom_ylim[1])
            else:
                ax2.set_ylim(display_extent[2], display_extent[3])

            ax2.set_aspect("equal", adjustable="box")

            
            # Scale bar (use the LOCKED display extent)
            map_width = display_extent[1] - display_extent[0]
            magnitude = 10 ** np.floor(np.log10(map_width * 0.2))
            scale_length = np.round(map_width * 0.2 / magnitude) * magnitude
            scale_length_rounded = int(scale_length) if scale_length < 10 else scale_length
            
            margin = (display_extent[1] - display_extent[0]) * 0.05
            bar_height = (display_extent[3] - display_extent[2]) * 0.015
            x_pos = display_extent[1] - margin - scale_length_rounded
            y_pos = display_extent[2] + margin

            rect = Rectangle((x_pos, y_pos), scale_length_rounded, bar_height,
                             fc='white', ec='black')
            ax2.add_patch(rect)
            ax2.text(x_pos + scale_length_rounded/2, y_pos + 2*bar_height,
                     f'{int(scale_length_rounded)} m',
                     ha='center', va='bottom', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, pad=2))

            # Convert pixel GCPs to world coords to overlay
            rw_points = []
            for x, y in [(x1_pix, y1_pix), (x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix)]:
                rw = transform_pixel_to_real_world(x, y, transformation['transformation_matrix'])
                rw_points.append(rw)
            rw_points = np.array(rw_points)
            x1_rw, y1_rw = rw_points[0]
            x2_rw, y2_rw = rw_points[1]
            x3_rw, y3_rw = rw_points[2]
            x4_rw, y4_rw = rw_points[3]

            # Lines
            ax2.plot([x1_rw, x2_rw], [y1_rw, y2_rw], color='#6CD4FF', linewidth=2)
            ax2.plot([x2_rw, x3_rw], [y2_rw, y3_rw], color='#62C655', linewidth=2)
            ax2.plot([x3_rw, x4_rw], [y3_rw, y4_rw], color='#ED6B57', linewidth=2)
            ax2.plot([x4_rw, x1_rw], [y4_rw, y1_rw], color='#F5BF61', linewidth=2)
            ax2.plot([x1_rw, x3_rw], [y1_rw, y3_rw], color='#CC4BC2', linewidth=2)
            ax2.plot([x2_rw, x4_rw], [y2_rw, y4_rw], color='#7765E3', linewidth=2)

            # Points
            ax2.plot(x1_rw, y1_rw, 'o', color='black', markersize=3)
            ax2.text(x1_rw, y1_rw, "1", color='black', fontsize=8, ha='left', va='bottom')
            ax2.plot([x2_rw, x3_rw, x4_rw], [y2_rw, y3_rw, y4_rw], 'o', color='black', markersize=3)
            pts2 = [(x2_rw, y2_rw), (x3_rw, y3_rw), (x4_rw, y4_rw)]
            [ax2.text(x, y, str(i), color='black', fontsize=8, ha='left', va='bottom') for i, (x, y) in enumerate(pts2, start=2)]

            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Orthorectified Image')

        fig.tight_layout()
        print("ABSOLUTE MODE:", self._absolute_mode)
        print("GLOBAL EXTENT:", self.global_extent)
        print("DISPLAY EXTENT:", display_extent)
        print("TRANSFORMATION EXTENT:", transformation["extent"])
        print("DISPLAY EXTENT:", display_extent)
        print("CLIP EXTENT:", clip_extent)

              
        # Save orthorectification image
        ortho_dir = rect_dir / cam / "orthorectification_imgs"
        ortho_dir.mkdir(parents=True, exist_ok=True)
        
        ortho_img = ortho_dir / f"{cam}_orthorect_{date}_{time_}.png"
        fig.savefig(str(ortho_img),
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                    transparent=True)

        print(f"[Saved] Orthorectified PNG: {ortho_img}")

        # Show the figure (as requested)
        plt.show(block=False)

        # ---- Save orthorectified image only (no overlays) ----
        fig_clean, ax_clean = plt.subplots(figsize=(6, 4))
    
        ax_clean.imshow(
            transformation['transformed_img'],
            extent=transformation['extent']
        )
    
        # Apply same display limits
        
        if (
            not getattr(self, "_absolute_mode", False)
            and self.custom_xlim is not None
        ):

            ax_clean.set_xlim(self.custom_xlim[0], self.custom_xlim[1])
        else:
            ax_clean.set_xlim(display_extent[0], display_extent[1])
    
        
        if (
            not getattr(self, "_absolute_mode", False)
            and self.custom_ylim is not None
        ):

            ax_clean.set_ylim(self.custom_ylim[0], self.custom_ylim[1])
        else:
            ax_clean.set_ylim(display_extent[2], display_extent[3])
    
        ax_clean.set_aspect("equal", adjustable="box")
        #ax_clean.axis("off")  # ✅ removes axes, labels, ticks
    

        # Save clean image
        ortho_clean_dir = rect_dir / cam / "ortho_imgs"
        ortho_clean_dir.mkdir(parents=True, exist_ok=True)
        
        ortho_clean = ortho_clean_dir / f"{cam}_orthoimg_{date}_{time_}.png"

        fig_clean.savefig(str(ortho_clean),
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                    transparent=True)
    
        plt.close(fig_clean)  # prevent extra window

    
        print(f"[Saved] Clean ortho image: {ortho_clean}")


        # Save geotiff
        ortho_tif_dir = rect_dir / cam / "ortho_tifs"
        ortho_tif_dir.mkdir(parents=True, exist_ok=True)
        
        ortho_tif = ortho_tif_dir / f"{cam}_orthotif_{date}_{time_}.tif"
        
        # Local coordinates -> NO GEOTIFF
        if not self._absolute_mode:
        
            print("[Info] Local coordinate mode: GeoTIFF export skipped.")
        
        # EPSG coordinates -> export cropped GeoTIFF
        else:
        
            cropped_img, cropped_extent = crop_image_to_extent(
                transformation["transformed_img"],
                transformation["extent"],
                clip_extent
            )
        
            save_geotiff(
                cropped_img,
                cropped_extent,
                ortho_tif,
                epsg=32623
            )
        
            print(f"[Saved] GeoTIFF: {ortho_tif}")
            print("Requested clip extent:", clip_extent)
            print("Actual crop extent:", cropped_extent)
        

        # Save transformation matrix JSON
        try:
            transformation_matrix = transformation['transformation_matrix']
            transf_file = transform_json_path(cam, date, time_)
            transf_file.parent.mkdir(parents=True, exist_ok=True)
            with transf_file.open("w") as f:
                json.dump(transformation_matrix, f, indent=1)
            print(f"[Saved] Transformation JSON: {transf_file}")
        except Exception as e:
            # If saving matrix fails, we still showed the image and saved PNG
            print(f"[WARN] Failed to save transformation matrix: {e}")

    def _refresh_index(self):
        try:
            df = build_and_save_frames_index(self.frames_root)
            if df.empty:
                messagebox.showwarning("Refresh", f"No frames found under: {self.frames_root}")
                return
            self.df_frames = df
            self._populate_cameras()
            print("[Info] Frames index rebuilt successfully.")
        except Exception as e:
            messagebox.showerror("Refresh", f"Failed to refresh frames index:\n{e}")


# ---------------------------
# Entry point
# ---------------------------

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="orthorectify",
        description="Interactive GCP selection and orthorectification GUI for RIVeR."
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Show INFO logs from river.config during this run.")
    parser.add_argument("--refresh-index", action="store_true",
                        help="Rescan FRAMES_DIR to rebuild _frame_paths.parquet and _frame_paths.csv before launching the GUI.")
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
    )
    
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
    )
  
    args = parser.parse_args(argv)

    frames_root = frames_dir

    if args.refresh_index:
        df_frames = build_and_save_frames_index(frames_root)
        if df_frames.empty:
            tk.Tk().withdraw()
            messagebox.showerror("Frames", f"No frames found under FRAMES_DIR: {frames_root}")
            return 3
    else:
        try:
            df_frames = load_frames_index_only(frames_root)
        except Exception as e:
            tk.Tk().withdraw()
            messagebox.showerror("Frames Index", str(e))
            return 3

    # Launch GUI
    root = tk.Tk()
    app = OrthoApp(root, df_frames, frames_root, xlim=args.xlim,ylim=args.ylim)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv[1:]))