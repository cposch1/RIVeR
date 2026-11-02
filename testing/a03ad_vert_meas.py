#!/usr/bin/env python3
"""
a03ad_vert_meas.py — vertical measurements below the modelled surface

In every oblique frame, measure vertical distances from the modelled ice
surface: the thin line through the modelled GCP3 and GCP4 of that frame
(b02 central model, _modelling/output_central), extended over the whole
image (towards the camera through GCP4). Three measurements, in this order:
  - water surface (bank)
  - water surface (GCP)
  - stream bed (GCP)

Per measurement (mouse only):
  - hover: a point follows the surface line under the mouse; the first click
    fixes the surface point
  - hover: the second point follows the mouse straight below it (same image
    x); the second click fixes it and saves the measurement
  - the next measurement of the frame is then selected; clicking again on a
    measurement starts it over, "Redo" clears it, "Delete" removes the
    saved one

Distances are saved in px and in metres. The vertical scale is the emergence
calibration (px per m at the calibration point, a03ac), scaled along the line
by the image scale from the frame's homography (central GCP image positions
<-> real coordinates): points farther from the camera look smaller. The
distances are also given relative to the low / high lines (output_low /
output_high, b02 uncertainty envelope). Water depths (in m):
  - water depth (bank) = stream bed (GCP) - water surface (bank)
  - water depth (GCP)  = stream bed (GCP) - water surface (GCP)

Inputs:
  - FRAMES_DIR/_frame_paths.parquet
  - gcps/<cam>/_modelling/output_central|low|high/<cam>_gcps_img_<date>_<time>.csv
    (same timestamp, else the first file of that date)
  - gcps/<cam>/<cam>_gcps_real.csv
  - gcps/<cam>/<cam>_emergence_calibration.csv

Output (one file per frame):
  - gcps/<cam>/vert_meas/<cam>_vert_meas_<date>_<time>.csv
    rows: water_surface (bank), water_surface_gcp, stream_bed (GCP),
    water_depth_bank, water_depth_gcp (if the measurements exist)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

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
# Settings
# ============================================================

LINE_GCPS = (3, 4)          # the surface line runs through these GCPs (GCP4: closest to the camera)

# measurements in their default order (keys as saved in the files)
KINDS = {
    "water_surface": {"label": "Water surface (bank)", "short": "Wb", "color": "deepskyblue"},
    "water_surface_gcp": {"label": "Water surface (GCP)", "short": "Wg", "color": "royalblue"},
    "stream_bed": {"label": "Stream bed (GCP)", "short": "B", "color": "orange"},
}

# water depth = stream bed - water surface
DEPTHS = {
    "water_depth_bank": {"label": "Water depth (bank)", "water": "water_surface", "bed": "stream_bed"},
    "water_depth_gcp": {"label": "Water depth (GCP)", "water": "water_surface_gcp", "bed": "stream_bed"},
}

LINE_COLOR = "#e7298a"      # central GCP model (magenta, as in b02)

COLUMNS = [
    "kind", "x", "y_surface", "y_point", "dy_px", "px_per_m", "dist_m",
    "y_surface_low", "dist_m_low", "y_surface_high", "dist_m_high",
    "gcp_file", "saved_at",
]


# ============================================================
# Frame index helpers (as in a03ac)
# ============================================================

def collect_frame_paths(frame_dir: Path) -> pd.DataFrame:

    rows = []

    for camera_dir in frame_dir.iterdir():
        if not camera_dir.is_dir():
            continue
        for date_dir in camera_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for time_dir in date_dir.iterdir():
                if not time_dir.is_dir():
                    continue
                for jpg in time_dir.glob("*.jpg"):
                    rows.append({
                        "camera": camera_dir.name,
                        "date_yyyymmdd": date_dir.name,
                        "time_hhmmss": time_dir.name,
                        "frame_path": str(jpg.resolve()),
                    })

    return pd.DataFrame(rows)


def build_and_save_frames_index(frames_root: Path):

    df = collect_frame_paths(frames_root)

    try:
        df.to_parquet(frames_root / "_frame_paths.parquet", index=False)
    except Exception:
        pass

    df.to_csv(frames_root / "_frame_paths.csv", index=False)

    return df


def load_frames_index_only(frames_root: Path):

    parquet = frames_root / "_frame_paths.parquet"

    if not parquet.exists():
        raise FileNotFoundError(f"Missing index:\n{parquet}")

    return pd.read_parquet(parquet)


# ============================================================
# Measurement helpers
# ============================================================

def vert_meas_dir(cam: str) -> Path:
    return gcps_dir / cam / "vert_meas"


def vert_meas_path(cam: str, date: str, time_: str) -> Path:
    return vert_meas_dir(cam) / f"{cam}_vert_meas_{date}_{time_}.csv"


def model_gcp_file(cam: str, date: str, time_: str, gcp_set: str):
    # modelled GCP image coordinates of a b02 set: same timestamp, else the
    # first file of that date (as in a04); None if the date has no file
    folder = gcps_dir / cam / "_modelling" / f"output_{gcp_set}"
    exact = folder / f"{cam}_gcps_img_{date}_{time_}.csv"
    if exact.exists():
        return exact
    same_date = sorted(folder.glob(f"{cam}_gcps_img_{date}_*.csv"))
    return same_date[0] if same_date else None


def read_gcps(path: Path) -> np.ndarray:
    # 4 x 2 array of GCP1-4 (x, y)
    return pd.read_csv(path, header=None).values[:4, :2].astype(float)


def surface_line(gcps: np.ndarray):
    # y on the line through the two LINE_GCPS, for any image x
    (xa, ya), (xb, yb) = gcps[LINE_GCPS[0] - 1], gcps[LINE_GCPS[1] - 1]
    slope = (yb - ya) / (xb - xa)
    return lambda x: ya + slope * (np.asarray(x, dtype=float) - xa)


def image_scale_function(gcps_img: np.ndarray, gcps_real: np.ndarray):
    # image scale across the view (px per m of ground along image x) from the
    # homography between the GCP image positions and their real coordinates
    wm, im = gcps_real.mean(0), gcps_img.mean(0)
    a = []
    for (X, Y), (u, v) in zip(gcps_real - wm, gcps_img - im):
        a += [[X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u], [0, 0, 0, X, Y, 1, -v * X, -v * Y, -v]]
    g = np.linalg.inv(np.linalg.svd(np.array(a))[2][-1].reshape(3, 3))      # image -> ground

    def scale(u, v):
        q = g @ np.array([u - im[0], v - im[1], 1.0])
        d = g[:, 0]
        return 1 / np.hypot((d[0] * q[2] - q[0] * d[2]) / q[2] ** 2, (d[1] * q[2] - q[1] * d[2]) / q[2] ** 2)

    return scale


def load_calibration(cam: str):
    # emergence calibration (a03ac): px per m at its reference point
    f = gcps_dir / cam / f"{cam}_emergence_calibration.csv"
    if not f.exists():
        return None
    row = pd.read_csv(f, dtype={"point": str, "date": str, "time": str}).iloc[0]
    return {"point": row["point"], "px_per_m": float(row["px_per_m"]),
            "x": float(row["x"]), "y_base": float(row["y_base"])}


def vertical_scale_function(gcps_central: np.ndarray, gcps_real: np.ndarray, calib):
    # vertical px per m at an image point: calibration px/m, scaled by the
    # image scale at the point relative to the calibration point
    if calib is None:
        return lambda x, y: np.nan

    scale = image_scale_function(gcps_central, gcps_real)
    if calib["point"].startswith("GCP"):
        ref = gcps_central[int(calib["point"][-1]) - 1]
    else:
        ref = (calib["x"], calib["y_base"])
    s_ref = scale(*ref)

    return lambda x, y: calib["px_per_m"] * scale(x, y) / s_ref


def measurement_row(kind, x, dy, lines, vscale, gcp_file):
    # one saved measurement: dy = point below the central line (px, positive = lower)
    y_s = float(lines["central"](x))
    y_p = y_s + dy
    ppm = float(vscale(x, y_s))
    row = {
        "kind": kind, "x": x, "y_surface": y_s, "y_point": y_p, "dy_px": dy,
        "px_per_m": ppm, "dist_m": dy / ppm,
        "gcp_file": gcp_file, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for s in ["low", "high"]:
        if s in lines:
            y_ss = float(lines[s](x))
            row[f"y_surface_{s}"] = y_ss
            row[f"dist_m_{s}"] = (y_p - y_ss) / ppm
    return row


def depth_rows(rows):
    # water depths (m) for each GCP set, where both measurements exist
    out = {}
    for d, spec in DEPTHS.items():
        if spec["water"] in rows and spec["bed"] in rows:
            w, b = rows[spec["water"]], rows[spec["bed"]]
            out[d] = {"kind": d}
            for c in ["dist_m", "dist_m_low", "dist_m_high"]:
                out[d][c] = b.get(c, np.nan) - w.get(c, np.nan)
    return out


def read_measurements(path: Path):
    # saved measurement rows of a frame: kind -> dict (depths are recomputed)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {r["kind"]: r.to_dict() for _, r in df.iterrows() if r["kind"] in KINDS}


def write_measurements(path: Path, rows):
    # write the measurement rows of a frame (+ water depths); no rows -> no file
    rows = {k: rows[k] for k in KINDS if k in rows}
    if not rows:
        if path.exists():
            path.unlink()
        return
    out = list(rows.values()) + list(depth_rows(rows).values())
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).reindex(columns=COLUMNS).to_csv(path, index=False, float_format="%.4f")


# ============================================================
# GUI
# ============================================================

class VertMeasApp:

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

        # modelled surface lines (central, low, high), vertical scale, central GCPs
        self.lines = {}
        self.gcps_central = None
        self.gcp_file = None
        self.vscale = lambda x, y: np.nan

        # measurements of the frame: x on the line, dy below it (px)
        self.meas = {k: {"x": None, "dy": None} for k in KINDS}
        self.saved = {}

        self._mpl_cids = []
        self._bg = None             # figure background for the hover preview (blitting)
        self.prev_seg = None
        self.prev_pt = None

        self.master.title(
            "Vertical Measurements below the Modelled Surface"
        )

        self.master.geometry(
            "1400x900"
        )

        self._build_ui()
        self._populate_cameras()

    # ----------------------------------------------------

    def _build_ui(self):

        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        # Camera / Date / Time

        lf_cam = ttk.LabelFrame(top, text="Camera")
        lf_cam.pack(side=tk.LEFT)
        self.lb_camera = tk.Listbox(lf_cam, exportselection=False, width=25)
        self.lb_camera.pack()
        self.lb_camera.bind("<<ListboxSelect>>", self._on_camera_selected)

        lf_date = ttk.LabelFrame(top, text="Date")
        lf_date.pack(side=tk.LEFT, padx=10)
        self.lb_date = tk.Listbox(lf_date, exportselection=False)
        self.lb_date.pack()
        self.lb_date.bind("<<ListboxSelect>>", self._on_date_selected)

        lf_time = ttk.LabelFrame(top, text="Time (green: all, yellow: some)")
        lf_time.pack(side=tk.LEFT)
        self.lb_time = tk.Listbox(lf_time, exportselection=False)
        self.lb_time.pack()
        self.lb_time.bind("<<ListboxSelect>>", self._on_time_selected)

        # Measurement

        lf_meas = ttk.LabelFrame(top, text="Measurement")
        lf_meas.pack(side=tk.LEFT, padx=20, anchor="n")

        self.kind_var = tk.StringVar(value=next(iter(KINDS)))

        for k, spec in KINDS.items():
            ttk.Radiobutton(
                lf_meas,
                text=spec["label"],
                value=k,
                variable=self.kind_var,
                command=self._on_kind_selected
            ).pack(anchor="w")

        self.show_env_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            lf_meas,
            text="Show low / high lines",
            variable=self.show_env_var,
            command=self._redraw
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(
            lf_meas,
            text="Hover + click: 1) point on the surface line,\n"
                 "2) point straight below it (saved with this click).\n"
                 "Click again to start a measurement over.",
            justify=tk.LEFT
        ).pack(anchor="w", pady=(8, 0))

        # Buttons

        btns = ttk.Frame(self.master)
        btns.pack(fill=tk.X, padx=8)

        ttk.Button(btns, text="Load Frame", command=self.load_frame).pack(side=tk.LEFT)
        ttk.Button(btns, text="Redo Measurement", command=self.redo_measurement).pack(side=tk.LEFT, padx=10)
        ttk.Button(btns, text="Delete Saved Measurement", command=self.delete_measurement).pack(side=tk.LEFT)
        ttk.Button(btns, text="Refresh Index", command=self.refresh_index).pack(side=tk.LEFT, padx=10)

        ttk.Label(btns, text="CLAHE").pack(side=tk.LEFT, padx=(30, 5))
        self.clahe_var = tk.IntVar(value=0)
        self.clahe_slider = tk.Scale(
            btns,
            from_=0,
            to=5,
            orient=tk.HORIZONTAL,
            variable=self.clahe_var,
            command=self._update_contrast
        )
        self.clahe_slider.pack(side=tk.LEFT)

        # Status + measurement info

        self.status = tk.StringVar(value="Select frame")
        ttk.Label(self.master, textvariable=self.status).pack(anchor="w", padx=8)

        self.info = tk.StringVar(value="")
        ttk.Label(self.master, textvariable=self.info, font=("Consolas", 9), justify=tk.LEFT).pack(anchor="w", padx=8)

        # Figure

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()

    # ----------------------------------------------------
    # population
    # ----------------------------------------------------

    def _n_saved(self, cam, date, time_):
        # number of saved measurements of a frame
        return len(read_measurements(vert_meas_path(cam, date, time_)))

    def _date_all_done(self, cam, date):
        times = self.df_frames[
            (self.df_frames.camera == cam) & (self.df_frames.date_yyyymmdd == date)
        ]["time_hhmmss"].unique()
        return len(times) > 0 and all(self._n_saved(cam, date, t) == len(KINDS) for t in times)

    def _camera_all_done(self, cam):
        dates = self.df_frames[self.df_frames.camera == cam]["date_yyyymmdd"].unique()
        return len(dates) > 0 and all(self._date_all_done(cam, d) for d in dates)

    def _populate_cameras(self):

        self.lb_camera.delete(0, tk.END)
        cams = sorted(self.df_frames.camera.unique())

        for i, cam in enumerate(cams):
            self.lb_camera.insert(tk.END, cam)
            if self._camera_all_done(cam):
                self.lb_camera.itemconfig(i, bg="#d0f0d0")

        if cams:
            self.lb_camera.selection_set(0)
            self._on_camera_selected()

    def _populate_dates(self):

        self.lb_date.delete(0, tk.END)
        dates = sorted(self.df_frames.loc[self.df_frames.camera == self.selected_camera, "date_yyyymmdd"].unique())

        for i, d in enumerate(dates):
            self.lb_date.insert(tk.END, d)
            if self._date_all_done(self.selected_camera, d):
                self.lb_date.itemconfig(i, bg="#d0f0d0")

        if dates:
            sel = dates.index(self.selected_date) if self.selected_date in dates else 0
            self.lb_date.selection_set(sel)
            self.lb_date.see(sel)
            self._on_date_selected()

    def _populate_times(self):

        self.lb_time.delete(0, tk.END)
        df = self.df_frames
        times = sorted(df[(df.camera == self.selected_camera) & (df.date_yyyymmdd == self.selected_date)]["time_hhmmss"].unique())

        for i, t in enumerate(times):
            self.lb_time.insert(tk.END, t)
        self._mark_times()

        if times:
            sel = times.index(self.selected_time) if self.selected_time in times else 0
            self.lb_time.selection_set(sel)
            self._on_time_selected()

    def _mark_times(self):
        # colours of the time list: green = all measurements saved, yellow = some
        for i in range(self.lb_time.size()):
            n = self._n_saved(self.selected_camera, self.selected_date, self.lb_time.get(i))
            self.lb_time.itemconfig(i, bg="#d0f0d0" if n == len(KINDS) else "#f6f0b0" if n else "white")

    # ----------------------------------------------------

    def _on_camera_selected(self, *args):
        s = self.lb_camera.curselection()
        if not s:
            return
        self.selected_camera = self.lb_camera.get(s[0])
        self._populate_dates()

    def _on_date_selected(self, *args):
        s = self.lb_date.curselection()
        if not s:
            return
        self.selected_date = self.lb_date.get(s[0])
        self._populate_times()

    def _on_time_selected(self, *args):
        s = self.lb_time.curselection()
        if not s:
            return
        self.selected_time = self.lb_time.get(s[0])

    # ----------------------------------------------------
    # frame
    # ----------------------------------------------------

    def load_frame(self):

        cam, date, time_ = self.selected_camera, self.selected_date, self.selected_time

        rows = self.df_frames[
            (self.df_frames.camera == cam)
            & (self.df_frames.date_yyyymmdd == date)
            & (self.df_frames.time_hhmmss == time_)
        ]

        if rows.empty:
            messagebox.showwarning("Frame", "Select camera, date and time")
            return

        self.current_frame_path = Path(rows.iloc[0].frame_path)
        self.current_img = mpimg.imread(str(self.current_frame_path))
        self.display_img = self.current_img

        # modelled surface lines of this frame (b02 sets)
        self.lines = {}
        self.gcps_central = None
        self.gcp_file = None

        for s in ["central", "low", "high"]:
            f = model_gcp_file(cam, date, time_, s)
            if f is None:
                continue
            gcps = read_gcps(f)
            self.lines[s] = surface_line(gcps)
            if s == "central":
                self.gcps_central = gcps
                self.gcp_file = f.name

        if "central" not in self.lines:
            self.vscale = lambda x, y: np.nan
            self.status.set(
                f"No central GCP model for {date} in {gcps_dir / cam / '_modelling' / 'output_central'} "
                "-> run the GCP model section in b02"
            )
        else:
            real_file = gcps_dir / cam / f"{cam}_gcps_real.csv"
            calib = load_calibration(cam)
            if real_file.exists():
                real = pd.read_csv(real_file, header=None).values[:4, :2].astype(float)
                self.vscale = vertical_scale_function(self.gcps_central, real, calib)
            else:
                self.vscale = lambda x, y: np.nan
            note = "" if calib is not None and real_file.exists() else " | no calibration or real GCP coordinates: metres not available"
            self.status.set(f"GCPs: {self.gcp_file} (output_central / low / high){note}")

        # saved measurements of this frame only
        self.saved = read_measurements(vert_meas_path(cam, date, time_))
        self.meas = {k: {"x": None, "dy": None} for k in KINDS}
        for k, r in self.saved.items():
            self.meas[k] = {"x": float(r["x"]), "dy": float(r["dy_px"])}

        self.kind_var.set(next((k for k in KINDS if k not in self.saved), next(iter(KINDS))))

        self._draw_image()
        self._enable_clicks()
        self._update_info()

    # ----------------------------------------------------
    # measurements
    # ----------------------------------------------------

    def _active(self):
        return self.meas[self.kind_var.get()]

    def _on_kind_selected(self):
        self._redraw()
        self._update_info()

    def _enable_clicks(self):
        for cid in self._mpl_cids:
            self.canvas.mpl_disconnect(cid)
        self._mpl_cids = [
            self.canvas.mpl_connect("button_press_event", self._on_click),
            self.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.canvas.mpl_connect("draw_event", self._on_draw),
        ]

    def _on_draw(self, event):
        # background without the hover preview, for blitting
        self._bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def _on_motion(self, event):

        if self.prev_pt is None or self._bg is None or self.toolbar.mode != "":
            return

        m = self._active()

        if event.inaxes != self.ax or event.xdata is None:
            self.prev_pt.set_data([], [])
            self.prev_seg.set_data([], [])
        elif m["x"] is None or m["dy"] is not None:
            # first point: follows the surface line under the mouse
            x = float(event.xdata)
            self.prev_pt.set_data([x], [float(self.lines["central"](x))])
            self.prev_seg.set_data([], [])
        else:
            # second point: straight below the surface point, follows the mouse vertically
            x = m["x"]
            y = float(event.ydata)
            self.prev_pt.set_data([x], [y])
            self.prev_seg.set_data([x, x], [float(self.lines["central"](x)), y])
            self._update_info(hover_y=y)

        self.canvas.restore_region(self._bg)
        self.ax.draw_artist(self.prev_seg)
        self.ax.draw_artist(self.prev_pt)
        self.canvas.blit(self.ax.bbox)

    def _on_click(self, event):

        if self.toolbar.mode != "":
            return

        if event.xdata is None or event.inaxes != self.ax:
            return

        if "central" not in self.lines:
            messagebox.showwarning("Surface line", "No modelled surface line for this frame.")
            return

        k = self.kind_var.get()
        m = self.meas[k]

        if m["x"] is None or m["dy"] is not None:
            # first click (or a new start): point on the surface line
            m["x"] = float(event.xdata)
            m["dy"] = None
            self.status.set(f"{KINDS[k]['label']}: surface point at x {m['x']:.0f} -> now click the point straight below")
        else:
            # second click: point straight below (same x) -> saved
            m["dy"] = float(event.ydata) - float(self.lines["central"](m["x"]))
            self._save(k)

        self._redraw()
        self._update_info()

    def _save(self, k):

        row = measurement_row(k, self.meas[k]["x"], self.meas[k]["dy"], self.lines, self.vscale, self.gcp_file)

        path = vert_meas_path(self.selected_camera, self.selected_date, self.selected_time)
        rows = read_measurements(path)
        rows[k] = row
        write_measurements(path, rows)
        self.saved = read_measurements(path)
        print(f"[Saved] {path} ({k})")

        # next: the first measurement of this frame that is not saved yet
        missing = [kk for kk in KINDS if kk not in self.saved]
        if missing:
            self.kind_var.set(missing[0])
            self.status.set(f"Saved {KINDS[k]['label']} -> next: {KINDS[missing[0]]['label']}")
        else:
            self.status.set(f"Saved {KINDS[k]['label']} -> all measurements of this frame saved")

        self._mark_times()

    def redo_measurement(self):
        # clear the points of the active measurement; the next two clicks measure (and save) it again
        k = self.kind_var.get()
        self.meas[k] = {"x": None, "dy": None}
        self.status.set(f"{KINDS[k]['label']}: hover and click the surface point on the line")
        self._redraw()
        self._update_info()

    def delete_measurement(self):

        k = self.kind_var.get()
        path = vert_meas_path(self.selected_camera, self.selected_date, self.selected_time)
        rows = read_measurements(path)

        if k not in rows:
            messagebox.showinfo("Measurement", f"No saved {KINDS[k]['label'].lower()} for this frame.")
            return

        if not messagebox.askyesno("Measurement", f"Delete the saved {KINDS[k]['label'].lower()} of this frame?"):
            return

        del rows[k]
        write_measurements(path, rows)
        self.saved = read_measurements(path)
        self.meas[k] = {"x": None, "dy": None}
        self.status.set(f"Deleted {KINDS[k]['label']} of {self.selected_date} {self.selected_time}")
        self._redraw()
        self._update_info()
        self._mark_times()

    # ----------------------------------------------------

    def _row(self, k):
        m = self.meas[k]
        if m["x"] is None or m["dy"] is None or "central" not in self.lines:
            return None
        return measurement_row(k, m["x"], m["dy"], self.lines, self.vscale, self.gcp_file)

    def _update_info(self, hover_y=None):

        lines = []
        rows = {}

        for k, spec in KINDS.items():
            row = self._row(k)
            m = self.meas[k]
            if row is None and hover_y is not None and k == self.kind_var.get() and m["x"] is not None:
                # second point not set yet: values at the mouse position
                dy = hover_y - float(self.lines["central"](m["x"]))
                h = measurement_row(k, m["x"], dy, self.lines, self.vscale, self.gcp_file)
                lines.append(
                    f"{spec['label']:<21}: x {h['x']:.0f} | {h['dy_px']:.1f} px at {h['px_per_m']:.1f} px/m "
                    f"= {h['dist_m']:.3f} m  [mouse]"
                )
                continue
            if row is None:
                if m["x"] is None:
                    lines.append(f"{spec['label']:<21}: -")
                else:
                    lines.append(f"{spec['label']:<21}: surface point at x {m['x']:.0f}, point below not set")
                continue
            rows[k] = row
            lines.append(
                f"{spec['label']:<21}: x {row['x']:.0f} | line y {row['y_surface']:.1f} -> point y {row['y_point']:.1f} | "
                f"{row['dy_px']:.1f} px at {row['px_per_m']:.1f} px/m = {row['dist_m']:.3f} m "
                f"(low {row.get('dist_m_low', np.nan):.3f}, high {row.get('dist_m_high', np.nan):.3f})"
                + ("  [saved]" if k in self.saved else "")
            )

        for d, r in depth_rows(rows).items():
            lines.append(
                f"{DEPTHS[d]['label']:<21}: {r['dist_m']:.3f} m (low {r['dist_m_low']:.3f}, high {r['dist_m_high']:.3f})"
            )

        self.info.set("\n".join(lines))

    # ----------------------------------------------------
    # drawing
    # ----------------------------------------------------

    def _redraw(self):
        if self.current_img is None:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self._draw_image()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def _draw_image(self):

        self.ax.clear()
        self.prev_seg = self.prev_pt = None

        if self.display_img is not None:
            self.ax.imshow(self.display_img)

        self.ax.axis("off")

        if self.current_img is None or "central" not in self.lines:
            self.canvas.draw_idle()
            return

        width = self.current_img.shape[1]
        xs = np.array([0, width - 1], dtype=float)

        # modelled surface: central line (and low / high lines)
        self.ax.plot(xs, self.lines["central"](xs), color=LINE_COLOR, lw=0.8)
        if self.show_env_var.get():
            for s in ["low", "high"]:
                if s in self.lines:
                    self.ax.plot(xs, self.lines[s](xs), color=LINE_COLOR, lw=0.5, ls=":", alpha=0.7)

        for i in LINE_GCPS:
            x, y = self.gcps_central[i - 1]
            self.ax.plot(x, y, "o", color=LINE_COLOR, markersize=3)
            self.ax.text(x, y, f"GCP{i}", color=LINE_COLOR, fontsize=6, ha="left", va="bottom")

        # measurements of this frame (active one thicker)
        for k, spec in KINDS.items():
            m = self.meas[k]
            if m["x"] is None:
                continue
            active = k == self.kind_var.get()
            x = m["x"]
            y_s = float(self.lines["central"](x))
            self.ax.plot(x, y_s, "o", color=spec["color"], markersize=4 if active else 3)
            if m["dy"] is not None:
                y_p = y_s + m["dy"]
                self.ax.plot([x, x], [y_s, y_p], color=spec["color"], lw=1.4 if active else 0.8)
                self.ax.plot(x, y_p, "o", color=spec["color"], markersize=4 if active else 3)
                self.ax.text(x, y_p, f" {spec['short']}", color=spec["color"], fontsize=7, ha="left", va="center")

        # hover preview of the active measurement (drawn by blitting, not in the normal draw)
        color = KINDS[self.kind_var.get()]["color"]
        self.prev_seg, = self.ax.plot([], [], color=color, lw=1.0, ls="--", animated=True)
        self.prev_pt, = self.ax.plot([], [], "o", color=color, markersize=5, mfc="none", animated=True)

        self.canvas.draw_idle()

    # ----------------------------------------------------

    def _update_contrast(self, value=None):

        if self.current_img is None:
            return

        img = self.current_img.astype(np.float32)
        if img.max() > 1:
            img /= 255.0

        strength = self.clahe_var.get()

        if strength == 0:
            self.display_img = self.current_img
        else:
            clip_limit_map = {1: 0.005, 2: 0.01, 3: 0.02, 4: 0.05, 5: 0.10}
            self.display_img = exposure.equalize_adapthist(img, clip_limit=clip_limit_map[strength])

        self._redraw()

    # ----------------------------------------------------

    def refresh_index(self):
        self.df_frames = build_and_save_frames_index(self.frames_root)
        self._populate_cameras()


# ============================================================
# main
# ============================================================

def main():

    df_frames = load_frames_index_only(frames_dir)

    root = tk.Tk()

    VertMeasApp(root, df_frames, frames_dir)

    root.mainloop()


if __name__ == "__main__":
    main()
