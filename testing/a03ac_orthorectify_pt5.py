#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from skimage import exposure

import tkinter as tk
from tkinter import ttk, messagebox

from river.config import *


# ============================================================
# Frame index helpers
# ============================================================

def collect_frame_paths(frame_dir: Path) -> pd.DataFrame:

    rows = []

    for camera_dir in frame_dir.iterdir():

        if not camera_dir.is_dir():
            continue

        camera = camera_dir.name

        for date_dir in camera_dir.iterdir():

            if not date_dir.is_dir():
                continue

            date = date_dir.name

            for time_dir in date_dir.iterdir():

                if not time_dir.is_dir():
                    continue

                time_ = time_dir.name

                for jpg in time_dir.glob("*.jpg"):

                    rows.append(
                        {
                            "camera": camera,
                            "date_yyyymmdd": date,
                            "time_hhmmss": time_,
                            "frame_path": str(jpg.resolve())
                        }
                    )

    return pd.DataFrame(rows)


def build_and_save_frames_index(frames_root: Path):

    df = collect_frame_paths(frames_root)

    try:
        df.to_parquet(
            frames_root / "_frame_paths.parquet",
            index=False
        )
    except Exception:
        pass

    df.to_csv(
        frames_root / "_frame_paths.csv",
        index=False
    )

    return df


def load_frames_index_only(frames_root: Path):

    parquet = frames_root / "_frame_paths.parquet"

    if not parquet.exists():
        raise FileNotFoundError(
            f"Missing index:\n{parquet}"
        )

    return pd.read_parquet(parquet)


# ============================================================
# Stake point helpers
# ============================================================

def stake_point_path(
    cam: str,
    date: str,
    time_: str
):
    return (
        gcps_dir
        / cam
        / f"{cam}_stake_point_{date}_{time_}.csv"
    )


def stake_png_path(
    cam: str,
    date: str,
    time_: str
):
    return (
        gcps_dir
        / cam
        / "stake_point_imgs"
        / f"{cam}_stake_point_{date}_{time_}.png"
    )



# ============================================================
# GUI
# ============================================================

class StakePointApp:

    def __init__(
        self,
        master,
        df_frames,
        frames_root
    ):

        self.master = master
        self.df_frames = df_frames
        self.frames_root = frames_root

        self.selected_camera = None
        self.selected_date = None
        self.selected_time = None

        self.current_img = None
        self.display_img = None
        self.current_frame_path = None

        self.base_point = None
        self.point = None
        self.offset = 0

        self._mpl_cid = None

        self.master.title(
            "Stake Point Selector"
        )

        self.master.geometry(
            "1280x800"
        )

        self._build_ui()
        self._populate_cameras()

    # ----------------------------------------------------

    def _build_ui(self):

        top = ttk.Frame(self.master)
        top.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=6,
            pady=6
        )

        # Camera

        lf_cam = ttk.LabelFrame(
            top,
            text="Camera"
        )

        lf_cam.pack(side=tk.LEFT)

        self.lb_camera = tk.Listbox(
            lf_cam,
            exportselection=False,
            width=25
        )

        self.lb_camera.pack()

        self.lb_camera.bind(
            "<<ListboxSelect>>",
            self._on_camera_selected
        )

        # Date

        lf_date = ttk.LabelFrame(
            top,
            text="Date"
        )

        lf_date.pack(
            side=tk.LEFT,
            padx=10
        )

        self.lb_date = tk.Listbox(
            lf_date,
            exportselection=False
        )

        self.lb_date.pack()

        self.lb_date.bind(
            "<<ListboxSelect>>",
            self._on_date_selected
        )

        # Time

        lf_time = ttk.LabelFrame(
            top,
            text="Time"
        )

        lf_time.pack(side=tk.LEFT)

        self.lb_time = tk.Listbox(
            lf_time,
            exportselection=False
        )

        self.lb_time.pack()

        self.lb_time.bind(
            "<<ListboxSelect>>",
            self._on_time_selected
        )

        # Slider

        slider_frame = ttk.LabelFrame(
            top,
            text="Stake Point Vertical Offset"
        )

        slider_frame.pack(
            side=tk.LEFT,
            padx=20
        )

        self.offset_slider = tk.Scale(
            slider_frame,
            from_=-200,
            to=200,
            orient=tk.HORIZONTAL,
            length=250,
            command=self._on_slider_move,
            state=tk.DISABLED
        )

        self.offset_slider.pack()

        # Buttons

        btns = ttk.Frame(self.master)
        btns.pack(
            fill=tk.X,
            padx=8
        )

        ttk.Button(
            btns,
            text="Load Frame",
            command=self.load_frame
        ).pack(side=tk.LEFT)

        ttk.Button(
            btns,
            text="Save Stake Point",
            command=self.save_stake_point
        ).pack(
            side=tk.LEFT,
            padx=10
        )

        ttk.Button(
            btns,
            text="Refresh Index",
            command=self.refresh_index
        ).pack(side=tk.LEFT)

        ttk.Label(
            btns,
            text="CLAHE"
        ).pack(
            side=tk.LEFT,
            padx=(30,5)
        )

        self.clahe_var = tk.IntVar(value=0)

        self.clahe_slider = tk.Scale(
            btns,
            from_=0,
            to=5,
            orient=tk.HORIZONTAL,
            variable=self.clahe_var,
            command=self._update_contrast
        )

        self.clahe_slider.pack(
            side=tk.LEFT
        )

        # Status

        self.status = tk.StringVar(
            value="Select frame"
        )

        ttk.Label(
            self.master,
            textvariable=self.status
        ).pack(
            anchor="w",
            padx=8
        )

        # Figure

        self.fig = Figure(
            figsize=(8,6),
            dpi=100
        )

        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            self.master
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        self.toolbar = NavigationToolbar2Tk(
            self.canvas,
            self.master
        )

        self.toolbar.update()

    # ----------------------------------------------------
    # population
    # ----------------------------------------------------

    def _stake_exists(self, cam, date, time_):
        return stake_point_path(cam, date, time_).exists()
    
    
    def _date_all_done(self, cam, date):
    
        times = set(
            self.df_frames[
                (self.df_frames.camera == cam)
                &
                (self.df_frames.date_yyyymmdd == date)
            ]["time_hhmmss"].unique()
        )
    
        if not times:
            return False
    
        return all(
            self._stake_exists(cam, date, t)
            for t in times
        )
    
    
    def _camera_all_done(self, cam):
    
        dates = set(
            self.df_frames[
                self.df_frames.camera == cam
            ]["date_yyyymmdd"].unique()
        )
    
        if not dates:
            return False
    
        return all(
            self._date_all_done(cam, d)
            for d in dates
        )

    def _populate_cameras(self):

        self.lb_camera.delete(0, tk.END)
    
        cams = sorted(
            self.df_frames.camera.unique()
        )
    
        for i, cam in enumerate(cams):
    
            self.lb_camera.insert(
                tk.END,
                cam
            )
    
            if self._camera_all_done(cam):
    
                self.lb_camera.itemconfig(
                    i,
                    bg="#d0f0d0"
                )
    
        if cams:
            self.lb_camera.selection_set(0)
            self._on_camera_selected()


    def _populate_dates(self):

        self.lb_date.delete(
            0,
            tk.END
        )
    
        dates = sorted(
            self.df_frames.loc[
                self.df_frames.camera
                == self.selected_camera,
                "date_yyyymmdd"
            ].unique()
        )
    
        for i, d in enumerate(dates):
    
            self.lb_date.insert(
                tk.END,
                d
            )
    
            if self._date_all_done(
                self.selected_camera,
                d
            ):
    
                self.lb_date.itemconfig(
                    i,
                    bg="#d0f0d0"
                )
    
        if dates:
            self.lb_date.selection_set(0)
            self._on_date_selected()

    def _populate_times(self):

        self.lb_time.delete(
            0,
            tk.END
        )

        df = self.df_frames

        times = sorted(
            df[
                (df.camera == self.selected_camera)
                &
                (df.date_yyyymmdd == self.selected_date)
            ]["time_hhmmss"].unique()
        )

        for i, t in enumerate(times):

            self.lb_time.insert(
                tk.END,
                t
            )

            if self._stake_exists(
                self.selected_camera,
                self.selected_date,
                t
            ):
                self.lb_time.itemconfig(
                    i,
                    bg="#d0f0d0"
                )

        if times:
            self.lb_time.selection_set(0)
            self._on_time_selected()

    # ----------------------------------------------------

    def _on_camera_selected(
        self,
        *args
    ):
        s = self.lb_camera.curselection()
        if not s:
            return

        self.selected_camera = self.lb_camera.get(
            s[0]
        )

        self._populate_dates()

    def _on_date_selected(
        self,
        *args
    ):
        s = self.lb_date.curselection()
        if not s:
            return

        self.selected_date = self.lb_date.get(
            s[0]
        )

        self._populate_times()

    def _on_time_selected(
        self,
        *args
    ):
        s = self.lb_time.curselection()
        if not s:
            return

        self.selected_time = self.lb_time.get(
            s[0]
        )

    # ----------------------------------------------------

    def load_frame(self):

        row = self.df_frames[
            (self.df_frames.camera == self.selected_camera)
            &
            (self.df_frames.date_yyyymmdd == self.selected_date)
            &
            (self.df_frames.time_hhmmss == self.selected_time)
        ].iloc[0]

        self.current_frame_path = Path(
            row.frame_path
        )

        self.current_img = mpimg.imread(
            str(self.current_frame_path)
        )

        self.display_img = self.current_img.copy()

        self._draw_image()
        self._enable_clicks()

        self._load_previous_point()

    # ----------------------------------------------------

    def _load_previous_point(self):

        cam = self.selected_camera

        files = sorted(
            (
                gcps_dir / cam
            ).glob(
                f"{cam}_stake_point_*.csv"
            )
        )

        if not files:
            return

        current_dt = pd.to_datetime(
            f"{self.selected_date}{self.selected_time}",
            format="%Y%m%d%H%M%S"
        )

        candidates = []

        for f in files:

            try:

                p = f.stem.split("_")

                dt = pd.to_datetime(
                    f"{p[-2]}{p[-1]}",
                    format="%Y%m%d%H%M%S"
                )

                if dt < current_dt:
                    candidates.append(
                        (dt, f)
                    )

            except:
                pass

        if not candidates:
            return

        latest = max(
            candidates,
            key=lambda x: x[0]
        )[1]

        with latest.open() as f:
            row = next(csv.reader(f))

        self.base_point = (
            int(float(row[0])),
            int(float(row[1]))
        )

        self.point = self.base_point

        self.offset_slider.set(0)
        self.offset_slider["state"] = tk.NORMAL

        self._draw_image()

    # ----------------------------------------------------

    def _on_slider_move(self, value):

        if self.base_point is None:
            return
    
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
    
        x, y = self.base_point
    
        self.point = (
            x,
            y + int(float(value))
        )
    
        self._draw_image()
    
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
    
        self.canvas.draw_idle()

    # ----------------------------------------------------

    def _enable_clicks(self):

        if self._mpl_cid:

            self.canvas.mpl_disconnect(
                self._mpl_cid
            )

        self._mpl_cid = self.canvas.mpl_connect(
            "button_press_event",
            self._on_click
        )

    def _on_click(
        self,
        event
    ):
    
        if self.toolbar.mode != "":
            return
    
        if event.xdata is None:
            return
    
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
    
        self.base_point = (
            int(event.xdata),
            int(event.ydata)
        )
    
        self.point = self.base_point
    
        self.offset_slider["state"] = tk.NORMAL
        self.offset_slider.set(0)
    
        self._draw_image()
    
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
    
        self.canvas.draw_idle()

    # ----------------------------------------------------

    def _draw_image(self):

        self.ax.clear()

        if self.display_img is not None:

            self.ax.imshow(
                self.display_img
            )

        self.ax.axis("off")

        if self.point is not None:

            x, y = self.point

            self.ax.plot(
                x,
                y,
                "ro",
                markersize=3
            )
            
            self.ax.text(
                x,
                y,
                "P5",
                color="red",
                fontsize=6,
                ha="left",
                va="bottom"
            )

        self.canvas.draw_idle()

    # ----------------------------------------------------

    def _update_contrast(self, value=None):

        if self.current_img is None:
            return
    
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
    
        img = self.current_img.astype(
            np.float32
        )
    
        if img.max() > 1:
            img /= 255.0
    
        strength = self.clahe_var.get()
    
        if strength == 0:
    
            self.display_img = self.current_img
    
        else:
    
            clip_limit_map = {
                1: 0.005,
                2: 0.01,
                3: 0.02,
                4: 0.05,
                5: 0.10,
            }
    
            self.display_img = exposure.equalize_adapthist(
                img,
                clip_limit=clip_limit_map[strength]
            )
    
        self._draw_image()
    
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
    
        self.canvas.draw_idle()

    # ----------------------------------------------------

    def save_stake_point(self):

        if self.point is None:

            messagebox.showwarning(
                "Stake Point",
                "Select a stake point first."
            )

            return

        cam = self.selected_camera
        date = self.selected_date
        time_ = self.selected_time

        csv_file = stake_point_path(
            cam,
            date,
            time_
        )

        csv_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with csv_file.open(
            "w",
            newline=""
        ) as f:

            csv.writer(f).writerow(
                self.point
            )

        png_file = stake_png_path(
            cam,
            date,
            time_
        )

        png_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.fig.savefig(
            png_file,
            dpi=300
        )

        print(
            f"[Saved] {csv_file}"
        )

        print(
            f"[Saved] {png_file}"
        )

        
        self._populate_times()
        self._populate_dates()
        self._populate_cameras()


    # ----------------------------------------------------

    def refresh_index(self):

        self.df_frames = build_and_save_frames_index(
            self.frames_root
        )

        self._populate_cameras()


# ============================================================
# main
# ============================================================

def main():

    df_frames = load_frames_index_only(
        frames_dir
    )

    root = tk.Tk()

    StakePointApp(
        root,
        df_frames,
        frames_dir
    )

    root.mainloop()


if __name__ == "__main__":
    main()