#!/usr/bin/env python3
"""
RIVeR-ICE — Cross-Section Selection GUI

Purpose:
- Select river cross-sections from orthorectified images
- Two-click selection (left bank → right bank)
- Save coordinates for bathymetry & velocity analysis

Inputs:
- FRAMES_DIR/_frame_paths.parquet
- GCP image coordinates: gcps/<cam>/_modelling/output (--gcps modelled,
  default) or gcps/<cam> (--gcps manual); same timestamp, else that date's
  first file (as in a03b --dyn)

Coordinates:
- local GCP system (default) or EPSG:32623 with --abs-coords.
  Cross-sections stay parallel to the GCP 1->2 baseline.

Outputs:
- <cam>_xs_coord_<date>_<time>.csv
- <cam>_xs_<date>_<time>.png
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
import signal
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, messagebox


import matplotlib.image as mpimg

from river.core.coordinate_transform import (
    transform_real_world_to_pixel
)


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Select river cross-sections from orthorectified images"
)
parser.add_argument("--verbose", action="store_true",
                    help="Enable verbose logging")
parser.add_argument(
    "--xlim",
    nargs=2,
    type=float,
    metavar=("XMIN", "XMAX"),
    default=None,
    help="Manually set x-axis limits for display (default local: -20 50; "
         "with --abs-coords: GCP extent +- --buffer)"
)
parser.add_argument(
    "--ylim",
    nargs=2,
    type=float,
    metavar=("YMIN", "YMAX"),
    default=None,
    help="Manually set y-axis limits for display (default local: -10 20; "
         "with --abs-coords: GCP extent +- --buffer)"
)
parser.add_argument(
    "--abs-coords",
    action="store_true",
    help="Work in absolute EPSG:32623 (UTM) coordinates instead of the local "
         "GCP system."
)
parser.add_argument(
    "--buffer",
    type=float,
    default=20.0,
    help="With --abs-coords: display buffer around the GCP extent (m, default: 20)"
)
parser.add_argument(
    "--gcps",
    choices=["modelled", "manual"],
    default="modelled",
    help="GCP image coordinates to use: modelled (b02 output in "
         "gcps/<camera>/_modelling/output, default) or manual picks "
         "(gcps/<camera>). Per scene: same timestamp, else the first file "
         "of that date (as in a03b_orthorectify_auto.py --dyn)"
)

args = parser.parse_args()

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
import logging
logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
logging.getLogger("river.config").setLevel(
    logging.INFO if args.verbose else logging.WARNING
)

from river.config import *


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def xs_coord_path(cam, date, time_):
    return bathy_dir / cam / f"{cam}_xs_coord_{date}_{time_}.csv"


def xs_img_path(cam, date, time_, view="ortho"):
    out_dir_img = bathy_dir / cam / "xs_imgs" / view
    out_dir_img.mkdir(parents=True, exist_ok=True)
    return out_dir_img / f"{cam}_xs_{date}_{time_}.png"


def transform_json_path(cam, date, time_):
    return (
        rect_dir
        / cam
        / "transforms"
        / f"{cam}_transform_{date}_{time_}.json"
    )


def ortho_img_path(cam, date, time_):
    return (
        rect_dir
        / cam
        / "ortho_imgs"
        / f"{cam}_orthoimg_{date}_{time_}.png"
    )


# ------------------------------------------------------------
# Coordinate spaces (local GCP system vs absolute EPSG:32623)
# ------------------------------------------------------------
# The local system has GCP 1 at (0, 0) and GCP 2 on the +x axis. The absolute
# system is the same frame rotated/translated so GCP 1 and 2 sit at their
# real coordinates (see oblique_view_transformation_matrix), so points convert
# between the two (to within a few cm: the local system uses the rounded
# distances in <cam>_gcps_dist.csv).
SPACE = "abs" if args.abs_coords else "local"

# Cross-section look (clicked, slid and reloaded alike): line plus small,
# semi-transparent bank markers so the bank stays visible below
XS_LINE_COLOR = "#F5BF61"
XS_DOT_SIZE = 2.5
XS_DOT_ALPHA = 0.5
SPACE_LABEL = {"abs": "EPSG:32623", "local": "local"}


def baseline_frame(cam, space):
    """Origin (GCP 1) and unit vectors along / across the GCP 1->2 baseline."""
    if space == "local":
        return np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    real = load_gcps_real(cam)
    p1 = np.array(real["point1"], dtype=float)
    p2 = np.array(real["point2"], dtype=float)
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    return p1, u, np.array([-u[1], u[0]])


def to_st(p, frame):
    """Point -> (s along baseline, t across baseline), metres from GCP 1."""
    o, u, n = frame
    d = np.asarray(p, dtype=float) - o
    return float(d @ u), float(d @ n)


def from_st(s, t, frame):
    o, u, n = frame
    p = o + s * u + t * n
    return float(p[0]), float(p[1])


def point_space(x, y):
    """UTM values are far outside anything a local system reaches."""
    return "abs" if max(abs(x), abs(y)) > 1e4 else "local"


def convert_points(pts, cam, to_space):
    """Convert points between the local and absolute system (via s/t)."""
    out = []
    for x, y in pts:
        src = point_space(x, y)
        if src == to_space:
            out.append((x, y))
            continue
        s, t = to_st((x, y), baseline_frame(cam, src))
        out.append(from_st(s, t, baseline_frame(cam, to_space)))
    return out


def gcp_folder(cam):
    # --gcps modelled: b02 output; --gcps manual: hand-picked GCPs
    if args.gcps == "modelled":
        return gcps_dir / cam / "_modelling" / "output"
    return gcps_dir / cam


def find_gcp_file(cam, date, time_):
    """
    GCP image coordinate CSV for a scene, same rule as a03b --dyn: the file
    with the same timestamp, else the first file of that date. None if the
    date has no file.
    """
    folder = gcp_folder(cam)
    exact = folder / f"{cam}_gcps_img_{date}_{time_}.csv"
    if exact.exists():
        return exact
    same_date = sorted(folder.glob(f"{cam}_gcps_img_{date}_*.csv"))
    return same_date[0] if same_date else None


def gcps_available(cam, date, time_):
    # a scene can be loaded as soon as a GCP image coordinate CSV exists for it
    return find_gcp_file(cam, date, time_) is not None


def xs_exists(cam, date, time_):
    p = bathy_dir / cam / f"{cam}_xs_coord_{date}_{time_}.csv"
    return p.exists()

    
def find_reference_frame(row):
    return Path(row["frame_path"]).parent / "0000000000.jpg"

def find_previous_xs(cam, date, time_):

    current_ts = pd.to_datetime(
        f"{date}{time_}",
        format="%Y%m%d%H%M%S"
    )

    xs_candidates = []

    cam_dir = bathy_dir / cam

    if not cam_dir.exists():
        return None

    for xs_file in cam_dir.glob(f"{cam}_xs_coord_*.csv"):

        try:
            stem = xs_file.stem

            parts = stem.split("_")

            file_date = parts[-2]
            file_time = parts[-1]

            ts = pd.to_datetime(
                f"{file_date}{file_time}",
                format="%Y%m%d%H%M%S"
            )

            if ts < current_ts:
                xs_candidates.append((ts, xs_file))

        except Exception:
            continue

    if not xs_candidates:
        return None

    xs_candidates.sort(key=lambda x: x[0])

    return xs_candidates[-1][1]


# ------------------------------------------------------------
# GUI App
# ------------------------------------------------------------
class CrossSectionApp:
    def __init__(self, master, df_frames, xlim=None, ylim=None):
        self.master = master
        self.df_frames = df_frames

        
        self.custom_xlim = xlim
        self.custom_ylim = ylim


        self.selected_camera = None
        self.selected_date = None
        self.selected_time = None

        self.points_rw: List[Tuple[float, float]] = []
        self.frame = None  # baseline frame of the loaded scene (see to_st)

        self.xs_line = None
        self.xs_dots = None
        self.cid_move = None
        self.preview_line = None

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = self.fig.canvas

        self._build_ui()

        # HANDLE WINDOW CLOSE PROPERLY
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        self._populate_cameras()

    # ---------------- UI ----------------
    def _build_ui(self):
        self.master.title(
            f"RIVeR-ICE — Cross-Section Selection "
            f"(GCPs: {args.gcps}, coordinates: {SPACE_LABEL[SPACE]})"
        )
        self.master.geometry("1200x800")

        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X)

        self.lb_cam = tk.Listbox(top, width=20, exportselection=False)
        self.lb_date = tk.Listbox(top, width=12, exportselection=False)
        self.lb_time = tk.Listbox(top, width=10, exportselection=False)

        self.lb_cam.pack(side=tk.LEFT)
        self.lb_date.pack(side=tk.LEFT)
        self.lb_time.pack(side=tk.LEFT)

        self.lb_cam.bind("<<ListboxSelect>>", self._on_cam)
        self.lb_date.bind("<<ListboxSelect>>", self._on_date)
        self.lb_time.bind("<<ListboxSelect>>", self._on_time)

        btns = ttk.Frame(self.master)
        btns.pack(side=tk.TOP, anchor="w")

        slider_frame = ttk.Frame(self.master)
        slider_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Sliders move the banks along the GCP 1->2 baseline (= X in local)
        bank_unit = "X" if SPACE == "local" else "(m along GCP 1->2)"

        ttk.Label(
            slider_frame,
            text=f"Left bank {bank_unit}"
        ).pack(side=tk.LEFT)
        
        self.left_slider = tk.Scale(
            slider_frame,
            orient=tk.HORIZONTAL,
            length=300,
            resolution=0.05,
            command=self._update_xs_from_sliders
        )
        
        self.left_slider.pack(side=tk.LEFT)
        
        
        ttk.Label(
            slider_frame,
            text=f"Right bank {bank_unit}"
        ).pack(side=tk.LEFT)
        
        self.right_slider = tk.Scale(
            slider_frame,
            orient=tk.HORIZONTAL,
            length=300,
            resolution=0.05,
            command=self._update_xs_from_sliders
        )
        
        self.right_slider.pack(side=tk.LEFT)
        
        self.left_slider.config(state=tk.DISABLED)
        self.right_slider.config(state=tk.DISABLED)

        ttk.Button(btns, text="Load Ortho", command=self.load_ortho).pack(side=tk.LEFT)
        ttk.Button(btns, text="Reset Points", command=self.reset_points).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save Cross-Section", command=self.save_xs).pack(side=tk.LEFT)

        self.fig_canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
            self.fig, master=self.master
        )
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------------- Population ----------------
    def _populate_cameras(self):
        self.lb_cam.delete(0, tk.END)
        cams = sorted(self.df_frames["camera"].unique())
        for c in cams:
            self.lb_cam.insert(tk.END, c)

    def _on_cam(self, *_):
        sel = self.lb_cam.curselection()
        if not sel:
            return
        self.selected_camera = self.lb_cam.get(sel[0])

        self.lb_date.delete(0, tk.END)
        dates = sorted(
            self.df_frames[self.df_frames.camera == self.selected_camera]["date_yyyymmdd"].unique()
        )
        for d in dates:
            self.lb_date.insert(tk.END, d)

    def _on_date(self, *_):
        sel = self.lb_date.curselection()
        if not sel:
            return
        self.selected_date = self.lb_date.get(sel[0])

        self.lb_time.delete(0, tk.END)

        times = sorted(
            self.df_frames[
                (self.df_frames.camera == self.selected_camera) &
                (self.df_frames.date_yyyymmdd == self.selected_date)
            ]["time_hhmmss"].unique()
        )

        for i, t in enumerate(times):
            self.lb_time.insert(tk.END, t)

            if gcps_available(self.selected_camera, self.selected_date, t):
                self.lb_time.itemconfig(i, {'fg': 'black'})
            else:
                self.lb_time.itemconfig(i, {'fg': 'grey'})

            if xs_exists(self.selected_camera, self.selected_date, t):
                self.lb_time.itemconfig(i, {'bg': '#d0f0d0'})

    def _on_time(self, *_):
        sel = self.lb_time.curselection()
        if not sel:
            return
        self.selected_time = self.lb_time.get(sel[0])

    def _on_move(self, event):
        if len(self.points_rw) != 1:
            return
    
        if event.inaxes is not self.ax:
            return
    
        # preview stays parallel to the GCP 1->2 baseline (= horizontal in local)
        x1, y1 = self.points_rw[0]
        _, t1 = to_st((x1, y1), self.frame)
        s2, _ = to_st((event.xdata, event.ydata), self.frame)
        x2, y2 = from_st(s2, t1, self.frame)

        if self.preview_line is not None:
            self.preview_line.remove()

        self.preview_line, = self.ax.plot(
            [x1, x2],
            [y1, y2],
            "--",
            color="cyan",
            linewidth=1
        )

        self.fig_canvas.draw_idle()

    def _update_xs_from_sliders(self, value=None):
        if getattr(self, "_loading_sliders", False):
            return
        if len(self.points_rw) != 2:
            return
    
        # sliders = bank positions along the baseline; keep the offset of bank 1
        s1 = self.left_slider.get()
        s2 = self.right_slider.get()

        _, t = to_st(self.points_rw[0], self.frame)

        self.points_rw = [
            from_st(s1, t, self.frame),
            from_st(s2, t, self.frame)
        ]
        (x1, y1), (x2, y2) = self.points_rw


        self._draw_xs(self.points_rw)

        self.fig_canvas.draw_idle()

    def _draw_xs(self, points):
        """Draw the cross-section (line + bank markers), replacing the old one."""
        for artist in (self.xs_line, self.xs_dots):
            if artist is not None:
                artist.remove()
        self.xs_line = self.xs_dots = None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        if len(points) == 2:
            self.xs_line, = self.ax.plot(
                xs, ys,
                color=XS_LINE_COLOR,
                linewidth=2
            )

        if points:
            self.xs_dots, = self.ax.plot(
                xs, ys,
                "o",
                linestyle="none",
                color="red",
                alpha=XS_DOT_ALPHA,
                markersize=XS_DOT_SIZE
            )

    def _view_limits(self, cam):
        """Display limits: --xlim/--ylim if given, else the mode's default."""
        if SPACE == "abs":
            # like a03b: GCP extent +- buffer, in EPSG:32623
            real = load_gcps_real(cam)
            xs = [c[0] for c in real.values()]
            ys = [c[1] for c in real.values()]
            b = args.buffer
            default_x = (min(xs) - b, max(xs) + b)
            default_y = (min(ys) - b, max(ys) + b)
        else:
            default_x, default_y = (-20, 50), (-10, 20)

        xlim = tuple(self.custom_xlim) if self.custom_xlim else default_x
        ylim = tuple(self.custom_ylim) if self.custom_ylim else default_y
        return xlim, ylim

    # ---------------- Core ----------------
    def load_ortho(self):

        self.points_rw.clear()
        self.left_slider.config(state=tk.DISABLED)
        self.right_slider.config(state=tk.DISABLED)



        if not (self.selected_camera and self.selected_date and self.selected_time):
            messagebox.showwarning("Selection", "Select camera, date, time")
            return

        gcp_file = find_gcp_file(
            self.selected_camera,
            self.selected_date,
            self.selected_time
        )

        if gcp_file is None:
            hint = (
                "Model GCPs for this date in b02, "
                "or start with --gcps manual."
                if args.gcps == "modelled" else
                "Pick GCPs for this date with a03aa, "
                "or start with --gcps modelled."
            )
            messagebox.showerror(
                "Missing GCPs",
                f"No {args.gcps} GCP file for {self.selected_date} in:\n"
                f"{gcp_folder(self.selected_camera)}\n\n{hint}"
            )
            return

        # the GCP file may belong to another time of that date: take its
        # timestamp for the GCPs, but always the selected scene's frame
        gcp_date, gcp_time = gcp_file.stem.split("_")[-2:]
        print(f"[INFO] GCPs: {gcp_file.name}")

        try:
            xlim, ylim = self._view_limits(self.selected_camera)
            display_extent = (xlim[0], xlim[1], ylim[0], ylim[1])

            _, _, frame_path = load_frame(
                self.df_frames,
                self.selected_camera,
                self.selected_date,
                self.selected_time
            )

            trans = transform(
                gcp_cam=self.selected_camera,
                gcp_date=gcp_date,
                gcp_time=gcp_time,
                frame_path=frame_path,
                absolute_coords=args.abs_coords,
                extent_override=display_extent,
                gcp_dir=gcp_file.parent,
            )

            self.trans = trans
            self.frame = baseline_frame(self.selected_camera, SPACE)

        except Exception as e:
            messagebox.showerror("Transform failed", str(e))
            return

        extent = trans["extent"]

        self.ax.clear()
        self.xs_line = None
        self.xs_dots = None
        self.preview_line = None

        self.ax.imshow(
            trans["transformed_img"],
            extent=trans["extent"]
        )

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect("equal")

        if SPACE == "abs":
            self.ax.set_xlabel("Easting (m, EPSG:32623)")
            self.ax.set_ylabel("Northing (m, EPSG:32623)")
            self.ax.ticklabel_format(useOffset=False, style="plain")
        else:
            self.ax.set_xlabel("X (m, local)")
            self.ax.set_ylabel("Y (m, local)")

        print("Transform extent:", trans["extent"])
        print("Current xlim:", xlim)
        print("Current ylim:", ylim)
        print("Image shape:", trans["transformed_img"].shape)



        
        self.ax.set_title("Click LEFT bank then RIGHT bank")

        self.cid = self.canvas.mpl_connect("button_press_event", self._on_click)

        self.cid_move = self.canvas.mpl_connect(
            "motion_notify_event",
            self._on_move
        )

        prev_xs = find_previous_xs(
            self.selected_camera,
            self.selected_date,
            self.selected_time
        )
        
        if prev_xs is not None:
        
            try:
        
                pts = []
        
                with prev_xs.open() as f:
                    for r in csv.reader(f):
                        pts.append(
                            (float(r[0]), float(r[1]))
                        )
        
                if len(pts) == 2:

                    # a previous XS saved in the other coordinate space is
                    # converted (same baseline frame, within a few cm)
                    src_space = point_space(*pts[0])
                    pts = convert_points(pts, self.selected_camera, SPACE)
                    if src_space != SPACE:
                        print(
                            f"[INFO] Previous XS converted from "
                            f"{SPACE_LABEL[src_space]} to {SPACE_LABEL[SPACE]}"
                        )

                    self.points_rw = pts

                    (x1, y1), (x2, y2) = pts

                    self._draw_xs(pts)

                    # slider values = bank positions along the baseline
                    s1, _ = to_st((x1, y1), self.frame)
                    s2, _ = to_st((x2, y2), self.frame)

                    self.left_slider.config(state=tk.NORMAL)
                    self.right_slider.config(state=tk.NORMAL)

                    self.left_slider.configure(
                        from_=s1 - 10,
                        to=s1 + 10
                    )

                    self.right_slider.configure(
                        from_=s2 - 10,
                        to=s2 + 10
                    )

                    self._loading_sliders = True

                    self.left_slider.set(s1)
                    self.right_slider.set(s2)
                    
                    self._loading_sliders = False
        
                    if self.cid_move is not None:
                        self.canvas.mpl_disconnect(self.cid_move)
        
                    if hasattr(self, "cid"):
                        self.canvas.mpl_disconnect(self.cid)
        
                    print(
                        f"[INFO] Loaded previous XS: {prev_xs.name}"
                    )
        
            except Exception as e:
        
                print(
                    f"[WARN] Could not load previous XS: {e}"
                )
        
        self.fig_canvas.draw_idle()

    def _on_click(self, event):

        if event.inaxes is not self.ax:
            return
    
        # first click
        if len(self.points_rw) == 0:
    
            self.points_rw.append(
                (event.xdata, event.ydata)
            )

            self._draw_xs(self.points_rw)
    
        # second click
        elif len(self.points_rw) == 1:
    
            # second bank on the line through bank 1, parallel to the
            # GCP 1->2 baseline (= same y in local coordinates)
            x1, y1 = self.points_rw[0]
            _, t1 = to_st((x1, y1), self.frame)
            s2, _ = to_st((event.xdata, event.ydata), self.frame)

            x2, y2 = from_st(s2, t1, self.frame)

            self.points_rw.append((x2, y2))
    
            if self.preview_line is not None:
                self.preview_line.remove()
                self.preview_line = None

            self._draw_xs(self.points_rw)
    
            self.canvas.mpl_disconnect(self.cid)
    
            if self.cid_move is not None:
                self.canvas.mpl_disconnect(self.cid_move)
    
        self.fig_canvas.draw_idle()

    def reset_points(self):
        self.points_rw.clear()
    
        if self.preview_line is not None:
            self.preview_line.remove()
            self.preview_line = None

        self._draw_xs([])

        self.load_ortho()

    def save_xs(self):
        if len(self.points_rw) != 2:
            messagebox.showwarning("Cross-section", "Select exactly two points")
            return

        p = xs_coord_path(self.selected_camera, self.selected_date, self.selected_time)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Oblique imagery
        row = self.df_frames[
            (self.df_frames.camera == self.selected_camera)
            & (self.df_frames.date_yyyymmdd == self.selected_date)
            & (self.df_frames.time_hhmmss == self.selected_time)
        ].iloc[0]
        
        frame_path = find_reference_frame(row)
        
        if not hasattr(self, "trans"):
            messagebox.showerror(
                "Missing transform",
                "Please load the orthorectified image first."
            )
            return
              
        T = np.array(self.trans["transformation_matrix"])
        
        (x1, y1), (x2, y2) = self.points_rw
        
        left_px = transform_real_world_to_pixel(x1, y1, T)
        right_px = transform_real_world_to_pixel(x2, y2, T)

        
        # Check overwrite
        if p.exists():
            choice = messagebox.askyesno(
                "Overwrite?",
                f"Cross-section already exists for:\n"
                f"{self.selected_camera} {self.selected_date} {self.selected_time}\n\n"
                f"Do you want to overwrite it?"
            )
            if not choice:
                print("[Info] Operation cancelled (no overwrite).")
                return


        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            for x, y in self.points_rw:
                writer.writerow([f"{x:.15f}", f"{y:.15f}"])

        img_p = xs_img_path(
            self.selected_camera,
            self.selected_date,
            self.selected_time,
            view="ortho"
        )
        
        self.fig.savefig(img_p, dpi=300)
        
        print("[DEBUG] Saving oblique image")
        print(frame_path)


        if frame_path.exists():

            frame = mpimg.imread(frame_path)
        
            fig2, ax2 = plt.subplots(figsize=(12, 8))
        
            ax2.imshow(frame)
        
            ax2.plot(
                [left_px[0], right_px[0]],
                [left_px[1], right_px[1]],
                color=XS_LINE_COLOR,
                linewidth=2
            )
        
            ax2.scatter(
                [left_px[0], right_px[0]],
                [left_px[1], right_px[1]],
                color="red",
                alpha=XS_DOT_ALPHA,
                s=XS_DOT_SIZE ** 2  # scatter size is the area (points^2)
            )
        
            ax2.axis("off")
        
            oblique_img = xs_img_path(
                self.selected_camera,
                self.selected_date,
                self.selected_time,
                view="oblique"
            )
        
            fig2.savefig(
                oblique_img,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0
            )
        
            plt.close(fig2)

        messagebox.showinfo(
            "Saved",
            f"Cross-section saved ({SPACE_LABEL[SPACE]} coordinates):\n{p}"
        )

        self._on_date()

    # CLEAN EXIT HANDLER
    def _on_close(self):
        print("[INFO] Closing GUI...")
        try:
            plt.close('all')
            self.master.quit()
            self.master.destroy()
        finally:
            os._exit(0)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main():
    frames_root = frames_dir
    df_frames = pd.read_parquet(frames_root / "_frame_paths.parquet")

    where = "gcps/<camera>/_modelling/output" if args.gcps == "modelled" else "gcps/<camera>"
    print(f"[INFO] GCPs: {args.gcps} ({where}); scenes without them are grey")
    print(f"[INFO] Coordinates: {SPACE_LABEL[SPACE]}")

    root = tk.Tk()
    app = CrossSectionApp(root, df_frames, xlim=args.xlim, ylim=args.ylim)

    # ENABLE CTRL+C
    def handle_sigint(sig, frame):
        print("\n[INFO] Ctrl+C detected — exiting")
        plt.close('all')
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # keep loop responsive to signals
    def _poll():
        root.after(100, _poll)

    _poll()

    root.mainloop()


if __name__ == "__main__":
    main()
