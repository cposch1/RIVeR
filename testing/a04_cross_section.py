#!/usr/bin/env python3
"""
RIVeR-ICE — Cross-Section Selection GUI

Purpose:
- Select river cross-sections from orthorectified images
- Two-click selection (left bank → right bank)
- Save coordinates for bathymetry & velocity analysis

Inputs:
- FRAMES_DIR/_frame_paths.parquet
- Orthorectified images + transform JSONs

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


def xs_img_path(cam, date, time_):
    out_dir_img = bathy_dir / cam / "xs_imgs"
    out_dir_img.mkdir(parents=True, exist_ok=True)
    return out_dir_img / f"{cam}_xs_{date}_{time_}.png"


def load_global_extent(cam):
    p = rect_dir / cam / f"{cam}_global_extent.json"
    if p.exists():
        with p.open() as f:
            return tuple(json.load(f))
    return None


def gcps_exist(cam, date, time_):
    p = gcps_dir / cam / f"{cam}_gcps_img_{date}_{time_}.csv"
    return p.exists(), p


def transform_exists(cam, date, time_):
    p = rect_dir / cam / f"{cam}_transform_{date}_{time_}.json"
    return p.exists()


def xs_exists(cam, date, time_):
    p = bathy_dir / cam / f"{cam}_xs_coord_{date}_{time_}.csv"
    return p.exists()


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

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = self.fig.canvas

        self._build_ui()

        # ✅ HANDLE WINDOW CLOSE PROPERLY
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        self._populate_cameras()

    # ---------------- UI ----------------
    def _build_ui(self):
        self.master.title("RIVeR-ICE — Cross-Section Selection")
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

            if transform_exists(self.selected_camera, self.selected_date, t):
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

    # ---------------- Core ----------------
    def load_ortho(self):

        self.points_rw.clear()

        if not (self.selected_camera and self.selected_date and self.selected_time):
            messagebox.showwarning("Selection", "Select camera, date, time")
            return

        exists, gcp_path = gcps_exist(
            self.selected_camera,
            self.selected_date,
            self.selected_time
        )

        if not exists:
            messagebox.showerror(
                "Missing GCPs",
                f"No GCP file found:\n{gcp_path}\n\nRun orthorectification first."
            )
            return

        try:
            trans = transform(
                self.df_frames,
                self.selected_camera,
                self.selected_date,
                self.selected_time
            )
        except Exception as e:
            messagebox.showerror("Transform failed", str(e))
            return

        extent = load_global_extent(self.selected_camera) or trans["extent"]

        self.ax.clear()
        self.ax.imshow(trans["transformed_img"], extent=trans["extent"])
        # ✅ Apply custom limits if provided
        if self.custom_xlim is not None:
            self.ax.set_xlim(self.custom_xlim[0], self.custom_xlim[1])
        else:
            self.ax.set_xlim(extent[0], extent[1])
        
        if self.custom_ylim is not None:
            self.ax.set_ylim(self.custom_ylim[0], self.custom_ylim[1])
        else:
            self.ax.set_ylim(extent[2], extent[3])
        
        self.ax.set_aspect("equal")


        # ✅ ADD GRID HERE
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        
        x_ticks = np.arange(np.floor(xmin / 2) * 2, np.ceil(xmax / 2) * 2, 2)
        y_ticks = np.arange(np.floor(ymin / 2) * 2, np.ceil(ymax / 2) * 2, 2)
        
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        
        self.ax.grid(True, color='yellow', alpha=0.5, linewidth=0.5)

        
        self.ax.set_title("Click LEFT bank then RIGHT bank")

        self.cid = self.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig_canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return

        self.points_rw.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, "o", markersize=4)

        if len(self.points_rw) == 2:
            (x1, y1), (x2, y2) = self.points_rw
            self.ax.plot([x1, x2], [y1, y2], linewidth=2)
            self.canvas.mpl_disconnect(self.cid)

        self.fig_canvas.draw_idle()

    def reset_points(self):
        self.points_rw.clear()
        self.load_ortho()

    def save_xs(self):
        if len(self.points_rw) != 2:
            messagebox.showwarning("Cross-section", "Select exactly two points")
            return

        p = xs_coord_path(self.selected_camera, self.selected_date, self.selected_time)
        p.parent.mkdir(parents=True, exist_ok=True)

        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            for x, y in self.points_rw:
                writer.writerow([f"{x:.15f}", f"{y:.15f}"])

        img_p = xs_img_path(self.selected_camera, self.selected_date, self.selected_time)
        self.fig.savefig(img_p, dpi=150)

        messagebox.showinfo("Saved", f"Cross-section saved:\n{p}")

        self._on_date()

    # ✅ CLEAN EXIT HANDLER
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
    frames_root = Path(os.environ["FRAMES_DIR"])
    df_frames = pd.read_parquet(frames_root / "_frame_paths.parquet")

    root = tk.Tk()
    app = CrossSectionApp(root, df_frames, xlim=args.xlim, ylim=args.ylim)

    # ✅ ENABLE CTRL+C
    def handle_sigint(sig, frame):
        print("\n[INFO] Ctrl+C detected — exiting")
        plt.close('all')
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ✅ keep loop responsive to signals
    def _poll():
        root.after(100, _poll)

    _poll()

    root.mainloop()


if __name__ == "__main__":
    main()