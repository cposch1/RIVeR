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
# --------------------------------------------------------------------------

# ✅ Bring in your RIVeR config & functions (unchanged behavior)
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

import matplotlib
matplotlib.use("TkAgg")  # embed in Tkinter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

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


def _fix_windows_path(p: str) -> str:
    """Convert Git Bash path (/c/...) to Windows path (C:/...)."""
    if p and len(p) > 2 and p[0] == "/" and p[2] == "/":
        drive = p[1].upper()
        return f"{drive}:/{p[3:]}"
    return p


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


# ---------------------------
# Orthorectification display helpers
# ---------------------------

def global_extent_path(cam: str) -> Path:
    """
    Path where the per-camera global orthorectification extent is stored.
    """
    return rect_dir / cam / f"{cam}_global_extent.json"


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

        # Build UI
        self._build_ui()
        self._populate_cameras()

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

        # Status line
        self.status = tk.StringVar(value="Select Camera, Date, Time; then Load Frame.")
        ttk.Label(self.master, textvariable=self.status).pack(side=tk.TOP, anchor="w", padx=8)

        # Matplotlib Figure embedded in Tk
        self.fig: Figure = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        if due_to_selection_change:
            # Clear the displayed image to force reloading the correct one for new selection
            self.current_img = None
            self.current_frame_path = None
            self.ax.clear()
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

        p = global_extent_path(self.selected_camera)
        
        if self.custom_xlim is not None and self.custom_ylim is not None:
            # ✅ Deterministic manual extent
            self.global_extent = (
                self.custom_xlim[0],
                self.custom_xlim[1],
                self.custom_ylim[0],
                self.custom_ylim[1]
            )
        
        elif p.exists():
            try:
                with p.open("r") as f:
                    self.global_extent = tuple(json.load(f))
            except Exception:
                self.global_extent = None
        else:
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
        except Exception as e:
            messagebox.showerror("Read Image", f"Failed to read image from path:\n{self.current_frame_path}\n\n{e}")
            return

        # Show image & prepare click capture
        self._draw_image()
        self._enable_clicks()
        self.points = []
        self.status.set(f"Loaded: {self.selected_camera} {self.selected_date} {self.selected_time}. Click 4 points in order.")
        self.btn_reset["state"] = tk.NORMAL
        self.btn_save_transform["state"] = tk.DISABLED  # enable after 4 points

    def _draw_image(self):
        self.ax.clear()
        self.ax.axis("off")
        if self.current_img is not None:
            self.ax.imshow(self.current_img)
            self.ax.set_title(
                f"Select GCPs:\n"
                f"1) left upstream\n2) right upstream\n3) right downstream\n4) left downstream\n\n"
                f"{self.current_frame_path}"
            )
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
        """Clear points and immediately allow new clicks without reloading."""
        if self.current_img is None:
            return
        self._clear_points_state(due_to_selection_change=False)

    def _on_click(self, event):
        if self.current_img is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        n = len(self.points) + 1

        # Register and draw
        self.points.append((x, y))
        if n == 1:
            self.ax.plot(x, y, 'o', color='#ED6B57', markersize=3)  # red for point 1
            self.ax.text(x, y, "1", color='#ED6B57', fontsize=8, ha='left', va='bottom')
        else:
            self.ax.plot(x, y, 'o', color='#6CD4FF', markersize=3)  # blue for 2-4
            self.ax.text(x, y, f"{n}", color='#6CD4FF', fontsize=8, ha='left', va='bottom')
        self.canvas.draw_idle()

        if n == 4:
            # Done selecting
            self._disable_clicks()
            print("4 points collected:", self.points)
            self.status.set("4 points collected. Click 'Save & Transform' to proceed.")
            self.btn_save_transform["state"] = tk.NORMAL

    def save_and_transform(self):
        """Single action: save GCPs (CSV+PNG), distances CSV, run transform, save outputs, show ortho image."""
        if len(self.points) != 4:
            messagebox.showwarning("GCPs", "Please click exactly 4 points.")
            return
        if not (self.selected_camera and self.selected_date and self.selected_time):
            messagebox.showwarning("Selection", "Please select Camera, Date, and Time.")
            return

        cam = self.selected_camera
        date = self.selected_date
        time_ = self.selected_time

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
            transformation = transform(self.df_frames, cam, date, time_)  # from RIVeR
            self._transformation = transformation
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

        # ✅ Auto-reset points after Save & Transform to avoid mismatches
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
            if len(self.points) != 4:
                raise RuntimeError("Need 4 GCPs (save or select).")
            (x1_pix, y1_pix), (x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix) = self.points

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Original image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Original Image')

        # Draw lines in original (✅ removed accidental extra blue line)
        ax1.plot([x1_pix, x2_pix], [y1_pix, y2_pix], color='#6CD4FF', linewidth=2)
        ax1.plot([x2_pix, x3_pix], [y2_pix, y3_pix], color='#62C655', linewidth=2)
        ax1.plot([x3_pix, x4_pix], [y3_pix, y4_pix], color='#ED6B57', linewidth=2)
        ax1.plot([x4_pix, x1_pix], [y4_pix, y1_pix], color='#F5BF61', linewidth=2)
        ax1.plot([x1_pix, x3_pix], [y1_pix, y3_pix], color='#CC4BC2', linewidth=2)
        ax1.plot([x2_pix, x4_pix], [y2_pix, y4_pix], color='#7765E3', linewidth=2)

        # Points
        ax1.plot(x1_pix, y1_pix, 'o', color='#ED6B57', markersize=3)
        ax1.text(x1_pix, y1_pix, "1", color='#ED6B57', fontsize=8, ha='left', va='bottom')
        ax1.plot([x2_pix, x3_pix, x4_pix], [y2_pix, y3_pix, y4_pix], 'o', color='#6CD4FF', markersize=3)
        pts = [(x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix)]
        [ax1.text(x, y, str(i), color='#6CD4FF', fontsize=8, ha='left', va='bottom') for i, (x, y) in enumerate(pts, start=2)]

        # Orthorectified image with overlay
        
        if 'transformed_img' in transformation and 'extent' in transformation:

            p = global_extent_path(cam)

            # ✅ Make sure directory always exists BEFORE any write attempt
            p.parent.mkdir(parents=True, exist_ok=True)
            
            # ✅ Case 1: manual extent (from CLI)
            if self.custom_xlim is not None and self.custom_ylim is not None:
                self.global_extent = (
                    self.custom_xlim[0],
                    self.custom_xlim[1],
                    self.custom_ylim[0],
                    self.custom_ylim[1]
                )
            
            # ✅ Case 2: compute from first transformation
            elif self.global_extent is None:
                self.global_extent = tuple(transformation['extent'])
            
            # ✅ Save ONCE (prevents overwriting)
            if not p.exists():
                with p.open("w") as f:
                    json.dump(self.global_extent, f, indent=2)
                print(f"[Saved] Global extent: {p}")
            
            display_extent = self.global_extent
        
            # ----- draw raster in its TRUE world position -----
            ax2.imshow(
                transformation['transformed_img'],
                extent=transformation['extent']
            )
        
            # ----- lock the map frame -----
            
            # X limits
            if self.custom_xlim is not None:
                ax2.set_xlim(self.custom_xlim[0], self.custom_xlim[1])
            else:
                ax2.set_xlim(display_extent[0], display_extent[1])

            # Y limits
            if self.custom_ylim is not None:
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

            # Lines (✅ 2–4 diagonal fixed with correct coords and color)
            ax2.plot([x1_rw, x2_rw], [y1_rw, y2_rw], color='#6CD4FF', linewidth=2)
            ax2.plot([x2_rw, x3_rw], [y2_rw, y3_rw], color='#62C655', linewidth=2)
            ax2.plot([x3_rw, x4_rw], [y3_rw, y4_rw], color='#ED6B57', linewidth=2)
            ax2.plot([x4_rw, x1_rw], [y4_rw, y1_rw], color='#F5BF61', linewidth=2)
            ax2.plot([x1_rw, x3_rw], [y1_rw, y3_rw], color='#CC4BC2', linewidth=2)
            ax2.plot([x2_rw, x4_rw], [y2_rw, y4_rw], color='#7765E3', linewidth=2)

            # Points
            ax2.plot(x1_rw, y1_rw, 'o', color='#ED6B57', markersize=3)
            ax2.text(x1_rw, y1_rw, "1", color='#ED6B57', fontsize=8, ha='left', va='bottom')
            ax2.plot([x2_rw, x3_rw, x4_rw], [y2_rw, y3_rw, y4_rw], 'o', color='#6CD4FF', markersize=3)
            pts2 = [(x2_rw, y2_rw), (x3_rw, y3_rw), (x4_rw, y4_rw)]
            [ax2.text(x, y, str(i), color='#6CD4FF', fontsize=8, ha='left', va='bottom') for i, (x, y) in enumerate(pts2, start=2)]

            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Orthorectified Image')

        fig.tight_layout()

        # Save orthorectification image
        
        ortho_dir = rect_dir / cam / "orthorectification_imgs"
        ortho_dir.mkdir(parents=True, exist_ok=True)
        
        ortho_img = ortho_dir / f"{cam}_orthorect_{date}_{time_}.png"
        fig.savefig(str(ortho_img))

        print(f"[Saved] Orthorectified PNG: {ortho_img}")

        # Show the figure (as requested)
        plt.show()

        # ---- Save orthorectified image only (no overlays) ----
        fig_clean, ax_clean = plt.subplots(figsize=(6, 4))
    
        ax_clean.imshow(
            transformation['transformed_img'],
            extent=transformation['extent']
        )
    
        # Apply same display limits
        if self.custom_xlim is not None:
            ax_clean.set_xlim(self.custom_xlim[0], self.custom_xlim[1])
        else:
            ax_clean.set_xlim(display_extent[0], display_extent[1])
    
        if self.custom_ylim is not None:
            ax_clean.set_ylim(self.custom_ylim[0], self.custom_ylim[1])
        else:
            ax_clean.set_ylim(display_extent[2], display_extent[3])
    
        ax_clean.set_aspect("equal", adjustable="box")
        #ax_clean.axis("off")  # ✅ removes axes, labels, ticks
    
        
        ortho_clean_dir = rect_dir / cam / "ortho_imgs"
        ortho_clean_dir.mkdir(parents=True, exist_ok=True)
        
        ortho_clean = ortho_clean_dir / f"{cam}_orthoimg_{date}_{time_}.png"

        fig_clean.savefig(str(ortho_clean), bbox_inches="tight", pad_inches=0)
    
        plt.close(fig_clean)  # prevent extra window
    
        print(f"[Saved] Clean ortho image: {ortho_clean}")

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
        default=[-20, 50],
        help="Manually set x-axis limits for display (default: -20 50)"
    )
    
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=[-10, 20],
        help="Manually set y-axis limits for display (default: -10 20)"
    )
    args = parser.parse_args(argv)

    # Prefer env var; fallback to frames_dir imported from river.config
    if "FRAMES_DIR" in os.environ:
        frames_env = _fix_windows_path(os.environ["FRAMES_DIR"])
        frames_root = Path(frames_env)

    else:
        try:
            frames_root = Path(frames_dir)
        except NameError:
            tk.Tk().withdraw()
            messagebox.showerror("Config", "FRAMES_DIR not set and frames_dir not available from river.config.")
            return 2

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