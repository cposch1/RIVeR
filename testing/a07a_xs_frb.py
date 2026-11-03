#!/usr/bin/env python3
"""
a07a_xs_frb.py — cross-section and freeboard in the ortho and the oblique view

Replaces the cross-section picking of a07 and the water surface (bank) measurement of a05 (not the water
depth). Left: the frame orthorectified in the local GCP system (GCP1 origin, local x along GCP1 -> GCP2,
local y across, + = left), computed from the frame with the homography of the selected GCP model.
Right: the oblique frame. GCP models: central (_modelling/output_central, b02) and OLS (_modelling/output,
used for the a07 picks so far); selections are saved per model.

Selection (mouse only, per frame and model; the "Selection" buttons show the next point and jump on after
each click; choose a button to pick that point again):
  1) far bank, the bank whose wall faces the camera: hover + click in the ortho or the oblique view. In the
     ortho view the marker moves along the fixed line (local y = LINE_Y). In the oblique view it moves along
     the fixed line on the modelled surface or, with "previous water line" ticked (top right), along the
     water line of an earlier selection (by default of the day before). Banks are vertical: the bank top is
     the modelled-surface point in the same image column, so the selection is always stored on the
     modelled surface.
  2) water surface at the far bank: hover + click in the oblique view on the guideline of the far bank,
     below the bank top.
  3) close bank, the bank whose wall faces away from the camera: as the far bank, on the camera side of it;
     the same freeboard (m) is shown at the close bank, with the water line.
  A complete selection is saved (again after every change); the views then return to the original view and
  no point is active until one is chosen again under "Selection". Picking the far bank again clears the
  water point. "Redo" clears the selection, "Delete" removes the saved file and its images.
  In the csv files, 1 = far bank and 2 = close bank.
  "display image" (with "previous water line" ticked) shows the chosen earlier frame with its selection
  instead of this one, in the same view and CLAHE; read only (no picking while it is shown).
  Images: one per picked point (view and CLAHE as shown) and one of the complete selection (original view,
  no CLAHE). A new scene starts without CLAHE and in the original view.

Banks are assumed vertical: the width is the same at the bank tops and at the water surface. All values
refer to the modelled surface; freeboard in m with the a05 vertical scale (emergence calibration px/m at
GCP4 x image-scale ratio of the homography).
Keys: hold c = zoom, hold x = pan, v = original view. After a zoom, pan, home, back or forward the other
view follows.

Inputs:
  - FRAMES_DIR/_frame_paths.parquet
  - gcps/<cam>/_modelling/output_central | output/<cam>_gcps_img_<date>_<time>.csv (same timestamp, else the
    first file of that date)
  - gcps/<cam>/<cam>_gcps_real.csv (EPSG:32623), gcps/<cam>/<cam>_emergence_calibration.csv

Output (one file per frame and model):
  - data/xs_frb/<cam>/<model>/<cam>_xs_frb_<date>_<time>.csv: banks in local x/y and EPSG:32623 (E, N),
    image positions of the banks and the water surface, width, freeboard (px, m), how each bank was picked
  - data/xs_frb/<cam>/<model>/imgs/<cam>_xs_frb_<date>_<time>_<far_bank|water|close_bank|complete>.png
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use("TkAgg")
# "c" / "x" / "v": zoom / pan / original view here (matplotlib uses "c" for "back" and "v" for "forward")
matplotlib.rcParams["keymap.back"] = [k for k in matplotlib.rcParams["keymap.back"] if k not in ("c", "x", "v")]
matplotlib.rcParams["keymap.forward"] = [k for k in matplotlib.rcParams["keymap.forward"] if k not in ("c", "x", "v")]

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

MODELS = {"central": "output_central", "OLS": "output"}    # GCP model -> folder in gcps/<cam>/_modelling
LINE_Y = 9.2406             # local y (m) of the fixed line: all a07 picks (OLS model) lie on it
X_RANGE = (-20.0, 30.0)     # ortho extent, local x (m)
Y_RANGE = (-6.0, 18.0)      # ortho extent, local y (m)
ORTHO_RES = 0.02            # m per ortho pixel
ORTHO_VIEW = ((-18.0, 10.0), (-6.0, 18.0))     # default ortho view (local x, local y), fitted to the panel
MIN_GAP = 0.05              # m: the close bank lies at least this far on the camera side of the far bank

STEPS = {"bank1": "Far bank (wall facing the camera)",       # selection order
         "water": "Water surface at far bank",
         "bank2": "Close bank (wall facing away)"}
STEP_KEY = {"bank1": "lx1", "bank2": "lx2", "water": "y3"}
BANK = {"1": "far bank", "2": "close bank"}                 # 1 / 2 in the csv files
IMG_TAG = {"bank1": "far_bank", "water": "water", "bank2": "close_bank"}
SURFACE = "modelled surface"

COL = {"line": "#F5BF61", "bank1": "orangered", "bank2": "gold", "water": "deepskyblue", "gcp": "magenta",
       "hover": "white", "prev": "cyan"}
MARK = dict(ms=4, alpha=0.6)            # markers
GUIDE = dict(ls=":", lw=1.0)            # vertical guidelines
CLAHE_LIMITS = {1: 0.005, 2: 0.01, 3: 0.02, 4: 0.05, 5: 0.10}   # as in a05

xs_frb_dir = data_dir / "xs_frb"
IMG_DPI = 150               # images of the selection (imgs/ next to the csv files)

COLUMNS = [
    "model", "date", "time", "local_y", "local_x1", "local_x2", "width_m", "E1", "N1", "E2", "N2",
    "img_x1", "img_y1", "img_x2", "img_y2", "img_y_water1", "img_y_water2", "freeboard_px",
    "px_per_m_bank1", "px_per_m_bank2", "freeboard_m", "bank1_line", "bank2_line", "gcp_file", "saved_at",
]

EMPTY = {"lx1": None, "lx2": None, "y3": None}      # far bank, close bank (local x), water at the far bank (image y)


# ============================================================
# Frame index helpers (as in a05)
# ============================================================

def load_frames_index_only(frames_root: Path):

    parquet = frames_root / "_frame_paths.parquet"

    if not parquet.exists():
        raise FileNotFoundError(f"Missing index:\n{parquet}")

    return pd.read_parquet(parquet)


# ============================================================
# Files
# ============================================================

def frb_path(cam: str, model: str, date: str, time_: str) -> Path:
    return xs_frb_dir / cam / model / f"{cam}_xs_frb_{date}_{time_}.csv"


def frb_img_path(cam: str, model: str, date: str, time_: str, tag: str) -> Path:
    # image of the selection; tag = bank1 | bank2 | water (view and contrast as shown) or complete
    return xs_frb_dir / cam / model / "imgs" / f"{cam}_xs_frb_{date}_{time_}_{tag}.png"


def saved_frames(cam: str, model: str):
    # (date, time) of all saved selections of a camera and model
    out = []
    for f in (xs_frb_dir / cam / model).glob(f"{cam}_xs_frb_*.csv"):
        d, t = f.stem.split("_")[-2:]
        out.append((d, t))
    return sorted(out)


def model_gcp_file(cam: str, model: str, date: str, time_: str):
    # modelled GCP image coordinates: same timestamp, else the first file of that date (as in a07)
    folder = gcps_dir / cam / "_modelling" / MODELS[model]
    exact = folder / f"{cam}_gcps_img_{date}_{time_}.csv"
    if exact.exists():
        return exact
    same_date = sorted(folder.glob(f"{cam}_gcps_img_{date}_*.csv"))
    return same_date[0] if same_date else None


def read_gcps(path: Path) -> np.ndarray:
    # 4 x 2 array of GCP1-4
    return pd.read_csv(path, header=None).values[:4, :2].astype(float)


def load_calibration(cam: str):
    # emergence calibration (a04): px per m at GCP4
    f = gcps_dir / cam / f"{cam}_emergence_calibration.csv"
    if not f.exists():
        return np.nan
    return float(pd.read_csv(f).iloc[0]["px_per_m"])


def read_picks(path: Path):
    # saved selection of a frame and model (also the first test files with s1 / s2 / y_water1)
    if not path.exists():
        return dict(EMPTY), {}
    r = pd.read_csv(path).iloc[0]
    if "local_x1" in r:
        picks = {"lx1": float(r["local_x1"]), "lx2": float(r["local_x2"]), "y3": float(r["img_y_water1"])}
    else:
        picks = {"lx1": float(r["s1"]), "lx2": float(r["s2"]), "y3": float(r["y_water1"])}
    guides = {k: str(r[f"bank{k}_line"]) for k in ("1", "2") if f"bank{k}_line" in r}
    return picks, guides


def read_water_line(path: Path):
    # water line of a saved selection: image points at the far and the close bank
    if not path.exists():
        return None
    r = pd.read_csv(path).iloc[0]
    if "img_x1" in r:
        return (float(r["img_x1"]), float(r["img_y_water1"])), (float(r["img_x2"]), float(r["img_y_water2"]))
    if "x1" in r:
        return (float(r["x1"]), float(r["y_water1"])), (float(r["x2"]), float(r["y_water2"]))
    return None


def wline_y(wline, x):
    # image y of a water line (straight through its two points) at image x
    (xa, ya), (xb, yb) = wline
    if abs(xb - xa) < 1e-9:
        return ya + 0 * np.asarray(x, float)
    return ya + (yb - ya) / (xb - xa) * (np.asarray(x, float) - xa)


def wline_in_image(wline, shape, n=600):
    # a water line, extended, inside the frame
    x = np.linspace(0, shape[1], n)
    y = wline_y(wline, x)
    ok = (y >= 0) & (y <= shape[0])
    return x[ok], y[ok]


# ============================================================
# Geometry of one frame and GCP model
# ============================================================

class Geometry:
    """Homography local (x, y) <-> image of one frame and GCP model, ortho image and vertical scale."""

    def __init__(self, gcps_img: np.ndarray, real: np.ndarray, k_cal: float):

        # ground -> image homography (DLT on centred coordinates, as in a05)
        wm, im = real.mean(0), gcps_img.mean(0)
        a = []
        for (X, Y), (u, v) in zip(real - wm, gcps_img - im):
            a += [[X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u], [0, 0, 0, X, Y, 1, -v * X, -v * Y, -v]]
        h = np.linalg.svd(np.array(a))[2][-1].reshape(3, 3)

        # local system: GCP1 origin, x along GCP1 -> GCP2, y across (+ = left)
        u12 = (real[1] - real[0]) / np.linalg.norm(real[1] - real[0])
        n12 = np.array([-u12[1], u12[0]])
        self.A = np.array([[u12[0], n12[0], real[0, 0]], [u12[1], n12[1], real[0, 1]], [0.0, 0.0, 1.0]])
        C = np.array([[1.0, 0.0, -wm[0]], [0.0, 1.0, -wm[1]], [0.0, 0.0, 1.0]])
        D = np.array([[1.0, 0.0, im[0]], [0.0, 1.0, im[1]], [0.0, 0.0, 1.0]])
        self.M = D @ h @ C @ self.A                       # (local x, local y, 1) -> image (x, y, w)
        p0 = np.linalg.solve(self.A, np.r_[wm, 1.0])
        if (self.M @ p0)[2] < 0:                          # w > 0 in front of the camera
            self.M = -self.M
        self.Minv = np.linalg.inv(self.M)

        # image -> centred ground, for the a05 image scale
        self.g = np.linalg.inv(h)
        self.im = im
        self.k_cal = k_cal
        self.s_ref = self._scale(*gcps_img[3])

        # the fixed line in the image: straight through two of its points
        (xa, xb), (ya, yb), _ = self.proj([0.0, 20.0], LINE_Y)
        self._line = (xa, ya, (yb - ya) / (xb - xa))

    def proj(self, lx, ly):
        # local -> image: x, y and w (w <= 0: behind the camera)
        lx, ly = np.broadcast_arrays(np.atleast_1d(np.asarray(lx, float)), np.atleast_1d(np.asarray(ly, float)))
        q = self.M @ np.vstack([lx.ravel(), ly.ravel(), np.ones(lx.size)])
        return q[0] / q[2], q[1] / q[2], q[2]

    def to_img(self, lx, ly):
        x, y, _ = self.proj(lx, ly)
        return float(x[0]), float(y[0])

    def to_local(self, x, y):
        # image -> local: x, y and w (w <= 0: not on the surface in front of the camera)
        x, y = np.broadcast_arrays(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)))
        q = self.Minv @ np.vstack([x.ravel(), y.ravel(), np.ones(x.size)])
        return q[0] / q[2], q[1] / q[2], q[2]

    def to_utm(self, lx, ly):
        e, n, _ = self.A @ np.array([lx, ly, 1.0])
        return float(e), float(n)

    def line_y(self, x):
        # image y of the fixed line (modelled surface) at image x
        xa, ya, slope = self._line
        return ya + slope * (np.asarray(x, float) - xa)

    def line_x_local(self, x):
        # local x of the fixed-line point in image column x (the bank top above any point of that column)
        lx, _, _ = self.to_local(x, self.line_y(x))
        return float(lx[0])

    def _scale(self, x, y):
        # image scale across the view (px per m), as a05 image_scale_function
        q = self.g @ np.array([x - self.im[0], y - self.im[1], 1.0])
        d = self.g[:, 0]
        return 1 / np.hypot((d[0] * q[2] - q[0] * d[2]) / q[2] ** 2, (d[1] * q[2] - q[1] * d[2]) / q[2] ** 2)

    def vscale(self, x, y):
        # vertical px per m: calibration px/m x image-scale ratio to GCP4 (as a05)
        return self.k_cal * self._scale(x, y) / self.s_ref

    def ortho(self, img: np.ndarray) -> np.ndarray:
        # ortho image over X_RANGE x Y_RANGE (row 0 = Y_RANGE[1]); black behind the camera
        w = int(round((X_RANGE[1] - X_RANGE[0]) / ORTHO_RES))
        hgt = int(round((Y_RANGE[1] - Y_RANGE[0]) / ORTHO_RES))
        P = np.array([[ORTHO_RES, 0.0, X_RANGE[0] + ORTHO_RES / 2],
                      [0.0, -ORTHO_RES, Y_RANGE[1] - ORTHO_RES / 2],
                      [0.0, 0.0, 1.0]])
        MP = self.M @ P
        out = cv2.warpPerspective(img, MP, (w, hgt), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        cols, rows = np.meshgrid(np.arange(w), np.arange(hgt))
        out[(MP[2, 0] * cols + MP[2, 1] * rows + MP[2, 2]) <= 0] = 0
        return out

    def line_in_image(self, shape, n=600):
        # the fixed line projected into the image (points in front of the camera and inside the frame)
        x, y, w = self.proj(np.linspace(*X_RANGE, n), LINE_Y)
        ok = (w > 0) & (x >= 0) & (x <= shape[1]) & (y >= 0) & (y <= shape[0])
        return x[ok], y[ok]


# ============================================================
# Selection
# ============================================================

def bank_pick(picks, k, lx):
    # bank k (1 = far, 2 = close) along the fixed line; the close bank lies on the camera side of the far bank
    lo, hi = X_RANGE
    if k == "1" and picks["lx2"] is not None:
        lo = max(lo, picks["lx2"] + MIN_GAP)
    if k == "2" and picks["lx1"] is not None:
        hi = min(hi, picks["lx1"] - MIN_GAP)
    return float(np.clip(lx, lo, hi))


def water_pick(geo: Geometry, picks, y):
    # water surface on the guideline of the far bank, below the bank top
    _, y1 = geo.to_img(picks["lx1"], LINE_Y)
    return max(float(y), y1)


def next_step(picks):
    # first point still missing; "" when the selection is complete (no point active)
    return next((s for s in STEPS if picks[STEP_KEY[s]] is None), "")


def selection(geo: Geometry, picks: dict):
    """Image positions and freeboard of a (partial) selection."""
    out = {}
    for k in ("1", "2"):
        if picks[f"lx{k}"] is not None:
            out[f"x{k}"], out[f"y{k}"] = geo.to_img(picks[f"lx{k}"], LINE_Y)
            out[f"v{k}"] = geo.vscale(out[f"x{k}"], out[f"y{k}"])
    if picks["y3"] is not None and "x1" in out:
        out["fb_px"] = picks["y3"] - out["y1"]
        out["fb_m"] = out["fb_px"] / out["v1"]
        if "x2" in out:
            out["yw2"] = out["y2"] + out["fb_m"] * out["v2"]
    return out


def frb_row(model, date, time_, geo, picks, gcp_file, guides):
    lx1, lx2 = picks["lx1"], picks["lx2"]
    sel = selection(geo, picks)
    (e1, n1), (e2, n2) = geo.to_utm(lx1, LINE_Y), geo.to_utm(lx2, LINE_Y)
    return {
        "model": model, "date": date, "time": time_, "local_y": LINE_Y, "local_x1": lx1, "local_x2": lx2,
        "width_m": abs(lx1 - lx2), "E1": e1, "N1": n1, "E2": e2, "N2": n2,
        "img_x1": sel["x1"], "img_y1": sel["y1"], "img_x2": sel["x2"], "img_y2": sel["y2"],
        "img_y_water1": picks["y3"], "img_y_water2": sel["yw2"], "freeboard_px": sel["fb_px"],
        "px_per_m_bank1": sel["v1"], "px_per_m_bank2": sel["v2"], "freeboard_m": sel["fb_m"],
        "bank1_line": guides.get("1", ""), "bank2_line": guides.get("2", ""),
        "gcp_file": gcp_file, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# Views: panel-filling limits; the ortho and the oblique view follow each other
# ============================================================

def fit_to_box(ax, xlim, ylim):
    # expand the limits around their centre so that equal scaling fills the panel (no change of its shape)
    pos = ax.get_position(original=True)
    fw, fh = ax.figure.get_size_inches()
    box = (pos.height * fh) / (pos.width * fw)
    (x0, x1), (y0, y1) = xlim, ylim
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    if dx <= 0 or dy <= 0:
        return tuple(xlim), tuple(ylim)
    if dy / dx < box:
        dy = dx * box
    else:
        dx = dy / box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = (1 if x1 >= x0 else -1), (1 if y1 >= y0 else -1)
    return (cx - sx * dx / 2, cx + sx * dx / 2), (cy - sy * dy / 2, cy + sy * dy / 2)


def _rect(xlim, ylim, n=25):
    (x0, x1), (y0, y1) = sorted(xlim), sorted(ylim)
    t = np.linspace(0, 1, n)
    xs = np.r_[x0 + (x1 - x0) * t, np.full(n, x1), x1 - (x1 - x0) * t, np.full(n, x0)]
    ys = np.r_[np.full(n, y0), y0 + (y1 - y0) * t, np.full(n, y1), y1 - (y1 - y0) * t]
    return xs, ys


def oblique_limits_for(geo: Geometry, xlim, ylim, shape):
    # oblique limits showing the ortho view (local x / y limits)
    x, y, w = geo.proj(*_rect(xlim, ylim))
    ok = w > 0
    if ok.sum() < 4:
        return None
    x0, x1 = np.clip([x[ok].min(), x[ok].max()], 0, shape[1])
    y0, y1 = np.clip([y[ok].min(), y[ok].max()], 0, shape[0])
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    return (x0, x1), (y1, y0)


def ortho_limits_for(geo: Geometry, xlim, ylim):
    # ortho limits showing the oblique view (image limits)
    lx, ly, w = geo.to_local(*_rect(xlim, ylim))
    ok = w > 0
    if ok.sum() < 4:
        return None
    x0, x1 = np.clip([lx[ok].min(), lx[ok].max()], *X_RANGE)
    y0, y1 = np.clip([ly[ok].min(), ly[ok].max()], *Y_RANGE)
    if x1 - x0 < 0.2 or y1 - y0 < 0.2:
        return None
    return (x0, x1), (y0, y1)


def set_view(ax, lim):
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[1])


def follow(geo, ax_src, ax_dst, src_is_ortho, shape):
    # fit the source view to its panel and let the other view follow
    set_view(ax_src, fit_to_box(ax_src, ax_src.get_xlim(), ax_src.get_ylim()))
    if src_is_ortho:
        lim = oblique_limits_for(geo, ax_src.get_xlim(), ax_src.get_ylim(), shape)
    else:
        lim = ortho_limits_for(geo, ax_src.get_xlim(), ax_src.get_ylim())
    if lim is not None:
        set_view(ax_dst, fit_to_box(ax_dst, *lim))


# ============================================================
# Drawing (shared by the GUI and the preview renders)
# ============================================================

def draw_static(ax_o, ax_i, ortho_img, img, geo, gcps_img, picks, model, wline=None):
    """Images, fixed line, previous water line, GCPs and the fixed part of the selection (default views)."""

    ax_o.clear()
    ax_i.clear()

    ax_o.imshow(ortho_img, extent=(X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]), origin="upper")
    ax_o.axhline(LINE_Y, color=COL["line"], lw=1, ls="--")
    ax_o.set_xlabel("local x (m)")
    ax_o.set_ylabel("local y (m)")
    ax_o.set_title(f"ortho ({model} model), fixed line local y = {LINE_Y:.2f} m", fontsize=10)

    ax_i.imshow(img)
    lx, ly = geo.line_in_image(img.shape)
    ax_i.plot(lx, ly, "--", color=COL["line"], lw=1)
    if wline is not None:
        ax_i.plot(*wline_in_image(wline, img.shape), "--", color=COL["prev"], lw=1, alpha=0.8)
    ax_i.plot(gcps_img[:, 0], gcps_img[:, 1], "o", mfc="none", mec=COL["gcp"], ms=6, alpha=0.6)
    ax_i.set_title(f"oblique, fixed line on the modelled surface ({model})"
                   + (", previous water line" if wline is not None else ""), fontsize=10)
    ax_i.axis("off")

    sel = selection(geo, picks)
    for k in ("1", "2"):
        if f"x{k}" not in sel:
            continue
        c = COL[f"bank{k}"]
        ax_o.plot(picks[f"lx{k}"], LINE_Y, "o", color=c, **MARK)
        ax_o.annotate(BANK[k].split()[0], (picks[f"lx{k}"], LINE_Y), (0, 6), textcoords="offset points", color=c,
                      ha="center", fontsize=8)
        ax_i.plot([sel[f"x{k}"]] * 2, [0, img.shape[0]], color=c, **GUIDE)
        ax_i.plot(sel[f"x{k}"], sel[f"y{k}"], "o", color=c, **MARK)
    if "x1" in sel and "x2" in sel:
        ax_o.plot([picks["lx1"], picks["lx2"]], [LINE_Y, LINE_Y], "-", color=COL["line"], lw=1.5)
        ax_i.plot([sel["x1"], sel["x2"]], [sel["y1"], sel["y2"]], "-", color=COL["line"], lw=1.2)
    if "fb_px" in sel:
        ax_i.plot(sel["x1"], picks["y3"], "o", color=COL["water"], **MARK)
        if "yw2" in sel:
            ax_i.plot([sel["x1"], sel["x2"]], [picks["y3"], sel["yw2"]], "-", color=COL["water"], lw=1.2)
            ax_i.plot(sel["x2"], sel["yw2"], "o", color=COL["water"], **MARK)

    # default views: the ortho around the channel, the oblique view showing the same area
    set_view(ax_o, fit_to_box(ax_o, *ORTHO_VIEW))
    lim = oblique_limits_for(geo, ax_o.get_xlim(), ax_o.get_ylim(), img.shape)
    if lim is not None:
        set_view(ax_i, fit_to_box(ax_i, *lim))


def make_preview(ax_o, ax_i, animated=True):
    """Moving parts: markers, guideline, bank line and water line."""
    kw = dict(animated=animated)
    return {
        "bank_o": ax_o.plot([], [], "o", color=COL["hover"], mec="k", **MARK, **kw)[0],
        "line_o": ax_o.plot([], [], "-", color=COL["line"], lw=1.2, **kw)[0],
        "bank_i": ax_i.plot([], [], "o", color=COL["hover"], mec="k", **MARK, **kw)[0],
        "prev_i": ax_i.plot([], [], "o", mfc="none", mec=COL["prev"], **MARK, **kw)[0],
        "guide_i": ax_i.plot([], [], color=COL["hover"], **GUIDE, **kw)[0],
        "line_i": ax_i.plot([], [], "-", color=COL["line"], lw=1.0, **kw)[0],
        "water_i": ax_i.plot([], [], "o", color=COL["water"], **MARK, **kw)[0],
        "wline_i": ax_i.plot([], [], "-", color=COL["water"], lw=1.0, **kw)[0],
    }


def clear_preview(art):
    for a in art.values():
        a.set_data([], [])


def preview_bank(art, geo, picks, k, lx, img_h, wline=None):
    """Bank k at local x lx, in both views with its guideline (and its point on the previous water line);
    bank line and water line."""
    clear_preview(art)
    lx = bank_pick(picks, k, lx)
    p = dict(picks, **{f"lx{k}": lx})
    if k == "1":
        p["y3"] = None                                  # the water point belongs to the old far bank
    sel = selection(geo, p)
    x, y = sel[f"x{k}"], sel[f"y{k}"]
    art["bank_o"].set_data([lx], [LINE_Y])
    art["bank_i"].set_data([x], [y])
    art["guide_i"].set_data([x, x], [0, img_h])
    if wline is not None:
        art["prev_i"].set_data([x], [float(wline_y(wline, x))])
    width = None
    if "x1" in sel and "x2" in sel:
        width = abs(p["lx1"] - p["lx2"])
        art["line_o"].set_data([p["lx1"], p["lx2"]], [LINE_Y, LINE_Y])
        art["line_i"].set_data([sel["x1"], sel["x2"]], [sel["y1"], sel["y2"]])
        if "yw2" in sel:
            art["wline_i"].set_data([sel["x1"], sel["x2"]], [p["y3"], sel["yw2"]])
            art["water_i"].set_data([sel["x2"]], [sel["yw2"]])
    return {"lx": lx, "width_m": width}


def preview_water(art, geo, picks, y, img_h):
    """Water surface under the cursor on the guideline of the far bank (below the bank top); water line."""
    clear_preview(art)
    p = dict(picks, y3=water_pick(geo, picks, y))
    sel = selection(geo, p)
    xs, ys = [sel["x1"]], [p["y3"]]
    if "yw2" in sel:
        xs.append(sel["x2"])
        ys.append(sel["yw2"])
        art["wline_i"].set_data(xs, ys)
    art["water_i"].set_data(xs, ys)
    return {"fb_m": sel["fb_m"], "fb_px": sel["fb_px"]}


# ============================================================
# GUI
# ============================================================

class LinkedToolbar(NavigationToolbar2Tk):
    """Navigation toolbar that reports finished zooms, pans and home / back / forward."""

    def __init__(self, canvas, window, on_view):
        self._on_view = on_view
        super().__init__(canvas, window)

    def release_zoom(self, event):
        super().release_zoom(event)
        self._on_view("zoom")

    def release_pan(self, event):
        super().release_pan(event)
        self._on_view("pan")

    def home(self, *args):
        super().home(*args)
        self._on_view("home")

    def back(self, *args):
        super().back(*args)
        self._on_view("back")

    def forward(self, *args):
        super().forward(*args)
        self._on_view("forward")


class XsFrbApp:

    def __init__(self, master, df_frames):

        self.master = master
        self.df_frames = df_frames

        self.selected_camera = None
        self.selected_date = None
        self.selected_time = None

        self.img = None             # frame as loaded
        self.img_disp = None        # frame with the contrast setting
        self.real = None
        self.k_cal = np.nan
        self.geo, self.gcps, self.gcp_file, self.orthos = {}, {}, {}, {}
        self.picks, self.guides = {}, {}
        self.orthos_raw = {}        # ortho images without contrast enhancement (complete-selection image)
        self._prev_disp = None      # earlier frame shown instead of this one (read only), cached

        # previous water line (bank line in the oblique view)
        self.wline = None
        self._prev_frames = []
        self._prev_sel = None       # (date, time) of the previous selection

        self._mpl_cids = []
        self._bg = None
        self.art = None
        self._lims = None           # view limits after the last drawing / view change
        self._printing = False      # saving an image of the figure (no background capture)

        self.master.title("Cross-Section and Freeboard (ortho + oblique)")
        self.master.geometry("1600x950")
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._populate_cameras()

        # hold c = zoom, hold x = pan, v = original view
        self.master.bind("<KeyPress-c>", lambda e: self._tool_hold("zoom", True))
        self.master.bind("<KeyRelease-c>", lambda e: self._tool_hold("zoom", False))
        self.master.bind("<KeyPress-x>", lambda e: self._tool_hold("pan", True))
        self.master.bind("<KeyRelease-x>", lambda e: self._tool_hold("pan", False))
        self.master.bind("<KeyPress-v>", lambda e: self._reset_view())

    # ----------------------------------------------------

    def _build_ui(self):

        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        lf_cam = ttk.LabelFrame(top, text="Camera")
        lf_cam.pack(side=tk.LEFT)
        self.lb_camera = tk.Listbox(lf_cam, exportselection=False, width=25, height=6)
        self.lb_camera.pack()
        self.lb_camera.bind("<<ListboxSelect>>", self._on_camera_selected)

        lf_date = ttk.LabelFrame(top, text="Date")
        lf_date.pack(side=tk.LEFT, padx=10)
        self.lb_date = tk.Listbox(lf_date, exportselection=False, height=6)
        self.lb_date.pack()
        self.lb_date.bind("<<ListboxSelect>>", self._on_date_selected)

        lf_time = ttk.LabelFrame(top, text="Time (green: both models, yellow: one)")
        lf_time.pack(side=tk.LEFT)
        self.lb_time = tk.Listbox(lf_time, exportselection=False, height=6)
        self.lb_time.pack()
        self.lb_time.bind("<<ListboxSelect>>", self._on_time_selected)

        lf_model = ttk.LabelFrame(top, text="GCP model")
        lf_model.pack(side=tk.LEFT, padx=(20, 10), anchor="n")
        self.model_var = tk.StringVar(value=next(iter(MODELS)))
        for m in MODELS:
            ttk.Radiobutton(lf_model, text=m, value=m, variable=self.model_var,
                            command=self._on_model_selected).pack(anchor="w")

        lf_step = ttk.LabelFrame(top, text="Selection (next point)")
        lf_step.pack(side=tk.LEFT, padx=10, anchor="n")
        self.step_var = tk.StringVar(value=next(iter(STEPS)))
        self.step_buttons = []
        for s, label in STEPS.items():
            b = ttk.Radiobutton(lf_step, text=label, value=s, variable=self.step_var, command=self._on_step_selected)
            b.pack(anchor="w")
            self.step_buttons.append(b)

        # top right: bank line in the oblique view (modelled surface or a previous water line)
        lf_line = ttk.LabelFrame(top, text="Bank line in the oblique view")
        lf_line.pack(side=tk.RIGHT, anchor="n", padx=6)
        self.prev_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf_line, text="previous water line (date, time)", variable=self.prev_on,
                        command=self._on_prev_toggle).pack(anchor="w")
        row = ttk.Frame(lf_line)
        row.pack(anchor="w")
        self.lb_pdate = tk.Listbox(row, exportselection=False, height=5, width=10, state=tk.DISABLED)
        self.lb_pdate.pack(side=tk.LEFT)
        self.lb_pdate.bind("<<ListboxSelect>>", self._on_pdate_selected)
        self.lb_ptime = tk.Listbox(row, exportselection=False, height=5, width=8, state=tk.DISABLED)
        self.lb_ptime.pack(side=tk.LEFT, padx=(4, 0))
        self.lb_ptime.bind("<<ListboxSelect>>", self._on_ptime_selected)
        self.line_info = tk.StringVar(value=SURFACE)
        ttk.Label(lf_line, textvariable=self.line_info).pack(anchor="w")
        self.show_prev = tk.BooleanVar(value=False)
        self.cb_show = ttk.Checkbutton(lf_line, text="display image (read only)", variable=self.show_prev,
                                       command=self._on_show_prev, state=tk.DISABLED)
        self.cb_show.pack(anchor="w")

        ttk.Label(
            top,
            text="Banks: hover + click in the ortho or the oblique view.\n"
                 "Water surface at the far bank: hover + click in the\n"
                 "oblique view on its guideline. Saved when complete;\n"
                 "then choose a point under 'Selection' to change it.\n"
                 "Keys: hold c = zoom, hold x = pan, v = original view",
            justify=tk.LEFT
        ).pack(side=tk.LEFT, padx=15, anchor="n")

        btns = ttk.Frame(self.master)
        btns.pack(fill=tk.X, padx=8)

        ttk.Button(btns, text="Load Frame", command=self.load_frame).pack(side=tk.LEFT)
        self.btn_redo = ttk.Button(btns, text="Redo Selection", command=self.redo)
        self.btn_redo.pack(side=tk.LEFT, padx=10)
        self.btn_delete = ttk.Button(btns, text="Delete Saved Selection", command=self.delete)
        self.btn_delete.pack(side=tk.LEFT)

        lf_clahe = ttk.LabelFrame(btns, text="CLAHE")
        lf_clahe.pack(side=tk.LEFT, padx=(30, 5))
        self.clahe_var = tk.IntVar(value=0)
        for v in range(len(CLAHE_LIMITS) + 1):
            ttk.Radiobutton(lf_clahe, text="off" if v == 0 else str(v), value=v, variable=self.clahe_var,
                            command=self._update_contrast).pack(side=tk.LEFT, padx=3)

        self.status = tk.StringVar(value="Select frame")
        ttk.Label(self.master, textvariable=self.status).pack(anchor="w", padx=8)

        self.info = tk.StringVar(value="")
        ttk.Label(self.master, textvariable=self.info, font=("Consolas", 9), justify=tk.LEFT).pack(anchor="w", padx=8)

        self.fig = Figure(figsize=(15, 6.5), dpi=100)
        self.ax_o = self.fig.add_subplot(121)
        self.ax_i = self.fig.add_subplot(122)
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08, wspace=0.08)

        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = LinkedToolbar(self.canvas, self.master, on_view=lambda kind: self._on_view(kind))
        self.toolbar.update()

    # ----------------------------------------------------
    # population
    # ----------------------------------------------------

    def _n_saved(self, cam, date, time_):
        return sum(frb_path(cam, m, date, time_).exists() for m in MODELS)

    def _populate_cameras(self):
        self.lb_camera.delete(0, tk.END)
        cams = sorted(self.df_frames.camera.unique())
        for cam in cams:
            self.lb_camera.insert(tk.END, cam)
        if cams:
            self.lb_camera.selection_set(0)
            self._on_camera_selected()

    def _populate_dates(self):
        self.lb_date.delete(0, tk.END)
        dates = sorted(self.df_frames.loc[self.df_frames.camera == self.selected_camera, "date_yyyymmdd"].unique())
        for d in dates:
            self.lb_date.insert(tk.END, d)
        if dates:
            sel = dates.index(self.selected_date) if self.selected_date in dates else 0
            self.lb_date.selection_set(sel)
            self.lb_date.see(sel)
            self._on_date_selected()

    def _populate_times(self):
        self.lb_time.delete(0, tk.END)
        df = self.df_frames
        times = sorted(df[(df.camera == self.selected_camera) & (df.date_yyyymmdd == self.selected_date)]["time_hhmmss"].unique())
        for t in times:
            self.lb_time.insert(tk.END, t)
        self._mark_times()
        if times:
            sel = times.index(self.selected_time) if self.selected_time in times else 0
            self.lb_time.selection_set(sel)
            self._on_time_selected()

    def _mark_times(self):
        # green = saved for both models, yellow = for one
        for i in range(self.lb_time.size()):
            n = self._n_saved(self.selected_camera, self.selected_date, self.lb_time.get(i))
            self.lb_time.itemconfig(i, bg="#d0f0d0" if n == len(MODELS) else "#f6f0b0" if n else "white")

    def _on_camera_selected(self, *args):
        s = self.lb_camera.curselection()
        if s:
            self.selected_camera = self.lb_camera.get(s[0])
            self._populate_dates()

    def _on_date_selected(self, *args):
        s = self.lb_date.curselection()
        if s:
            self.selected_date = self.lb_date.get(s[0])
            self._populate_times()

    def _on_time_selected(self, *args):
        s = self.lb_time.curselection()
        if s:
            self.selected_time = self.lb_time.get(s[0])

    # ----------------------------------------------------
    # previous water line (bank line in the oblique view)
    # ----------------------------------------------------

    def _fill(self, lb, items, sel=None):
        # (re)fill a listbox of the previous-water-line selection; greyed out unless ticked
        lb.config(state=tk.NORMAL)
        lb.delete(0, tk.END)
        for it in items:
            lb.insert(tk.END, it)
        if sel in items:
            i = items.index(sel)
            lb.selection_set(i)
            lb.see(i)
        lb.config(state=tk.NORMAL if self.prev_on.get() else tk.DISABLED)

    def _closest_time(self, times):
        return min(times, key=lambda t: abs(int(t) - int(self.selected_time))) if times else None

    def _populate_prev(self):
        # saved selections of this camera and model (except this frame); default: the day before
        cur = (self.selected_date, self.selected_time)
        self._prev_frames = [f for f in saved_frames(self.selected_camera, self._model()) if f != cur]
        dates = sorted({d for d, _ in self._prev_frames})
        if self._prev_sel in self._prev_frames:
            d, t = self._prev_sel
        else:
            before = [d for d in dates if d < self.selected_date]
            d = before[-1] if before else (dates[0] if dates else None)
            t = self._closest_time([tt for dd, tt in self._prev_frames if dd == d])
        self._fill(self.lb_pdate, dates, d)
        self._fill(self.lb_ptime, [tt for dd, tt in self._prev_frames if dd == d], t)
        self._prev_sel = (d, t) if d and t else None
        self._load_prev_line()

    def _load_prev_line(self):
        self.wline = None
        if self.prev_on.get() and self._prev_sel is not None:
            self.wline = read_water_line(frb_path(self.selected_camera, self._model(), *self._prev_sel))
        if not self.prev_on.get():
            self.line_info.set(SURFACE)
        elif self.wline is None:
            self.line_info.set("no earlier water line (modelled surface used)")
        else:
            self.line_info.set(f"water line {self._prev_sel[0]} {self._prev_sel[1]}")

    def _wline(self):
        return self.wline if self.prev_on.get() else None

    def _showing_prev(self):
        return self.prev_on.get() and self.show_prev.get() and self._prev_sel is not None

    def _prev_display(self):
        # the earlier frame with its saved selection, contrast as set (cached)
        d, t = self._prev_sel
        m = self._model()
        key = (d, t, m, self.clahe_var.get())
        if self._prev_disp is None or self._prev_disp["key"] != key:
            df = self.df_frames
            row = df[(df.camera == self.selected_camera) & (df.date_yyyymmdd == d) & (df.time_hhmmss == t)]
            f = model_gcp_file(self.selected_camera, m, d, t)
            if row.empty or f is None:
                return None
            img = self._contrast(mpimg.imread(str(row.iloc[0].frame_path)))
            gcps = read_gcps(f)
            geo = Geometry(gcps, self.real, self.k_cal)
            self._prev_disp = {"key": key, "img": img, "geo": geo, "gcps": gcps, "ortho": geo.ortho(img),
                               "picks": read_picks(frb_path(self.selected_camera, m, d, t))[0]}
        return self._prev_disp

    def _set_editing(self):
        # no picking (and no Redo / Delete) while an earlier frame is shown
        state = tk.DISABLED if self._showing_prev() else tk.NORMAL
        for w in [*self.step_buttons, self.btn_redo, self.btn_delete]:
            w.config(state=state)

    def _on_show_prev(self):
        self._set_editing()
        if self.img is not None:
            self._redraw()

    def _disp_geo(self):
        # geometry of the frame shown
        if self._showing_prev():
            pv = self._prev_display()
            if pv is not None:
                return pv["geo"]
        return self.geo[self._model()]

    def _on_prev_toggle(self):
        state = tk.NORMAL if self.prev_on.get() else tk.DISABLED
        self.lb_pdate.config(state=state)
        self.lb_ptime.config(state=state)
        self.cb_show.config(state=state)
        if not self.prev_on.get():
            self.show_prev.set(False)
        self._set_editing()
        if self.img is not None:
            self._load_prev_line()
            self._redraw()

    def _on_pdate_selected(self, *args):
        s = self.lb_pdate.curselection()
        if not s or self.img is None:
            return
        d = self.lb_pdate.get(s[0])
        times = [tt for dd, tt in self._prev_frames if dd == d]
        t = self._closest_time(times)
        self._fill(self.lb_ptime, times, t)
        self._prev_sel = (d, t) if t else None
        self._load_prev_line()
        self._set_editing()
        self._redraw()

    def _on_ptime_selected(self, *args):
        s, sd = self.lb_ptime.curselection(), self.lb_pdate.curselection()
        if not s or not sd or self.img is None:
            return
        self._prev_sel = (self.lb_pdate.get(sd[0]), self.lb_ptime.get(s[0]))
        self._load_prev_line()
        self._set_editing()
        self._redraw()

    # ----------------------------------------------------
    # frame
    # ----------------------------------------------------

    def load_frame(self):

        cam, date, time_ = self.selected_camera, self.selected_date, self.selected_time
        df = self.df_frames
        rows = df[(df.camera == cam) & (df.date_yyyymmdd == date) & (df.time_hhmmss == time_)]
        if rows.empty:
            messagebox.showwarning("Frame", "Select camera, date and time")
            return

        real_file = gcps_dir / cam / f"{cam}_gcps_real.csv"
        if not real_file.exists():
            messagebox.showwarning("GCPs", f"Missing {real_file}")
            return
        self.real = pd.read_csv(real_file, header=None).values[:4, :2].astype(float)
        self.k_cal = load_calibration(cam)

        self.img = mpimg.imread(str(rows.iloc[0].frame_path))
        self.clahe_var.set(0)                   # new scene: no contrast enhancement
        self._apply_contrast()

        self.geo, self.gcps, self.gcp_file, self.orthos, self.picks, self.guides = {}, {}, {}, {}, {}, {}
        self.orthos_raw = {}
        for m in MODELS:
            f = model_gcp_file(cam, m, date, time_)
            if f is None:
                continue
            self.gcps[m] = read_gcps(f)
            self.gcp_file[m] = f.name
            self.geo[m] = Geometry(self.gcps[m], self.real, self.k_cal)
            self.picks[m], self.guides[m] = read_picks(frb_path(cam, m, date, time_))

        if not self.geo:
            messagebox.showwarning("GCPs", f"No modelled GCPs for {date} ({', '.join(MODELS.values())})")
            return
        if self.model_var.get() not in self.geo:
            self.model_var.set(next(iter(self.geo)))
        self.step_var.set(next_step(self.picks[self._model()]))
        self._prev_sel = None                   # new scene: previous water line of the day before
        self.show_prev.set(False)               # new scene: this frame shown
        self._prev_disp = None
        self._populate_prev()
        self._set_editing()

        self._redraw(keep_view=False)
        self.toolbar.update()                   # new frame: clear the zoom history
        self._enable_events()

    def _contrast(self, img):
        # image with the contrast setting
        v = self.clahe_var.get()
        if v == 0:
            return img
        return (exposure.equalize_adapthist(img, clip_limit=CLAHE_LIMITS[v]) * 255).astype(np.uint8)

    def _apply_contrast(self):
        self.img_disp = self._contrast(self.img)

    def _update_contrast(self, *args):
        if self.img is None:
            return
        self._apply_contrast()
        self.orthos = {}
        self._redraw()

    def _on_model_selected(self):
        if self.img is not None:
            if self.model_var.get() not in self.geo:
                messagebox.showwarning("GCP model", f"No {self.model_var.get()} GCPs for this frame")
                self.model_var.set(next(iter(self.geo)))
            self.step_var.set(next_step(self.picks[self._model()]))
            self._populate_prev()
            self._set_editing()
            self._redraw()

    def _on_step_selected(self):
        if self.img is not None:
            self._update_status()

    # ----------------------------------------------------
    # drawing
    # ----------------------------------------------------

    def _model(self):
        return self.model_var.get()

    def _view(self):
        return (tuple(self.ax_o.get_xlim()), tuple(self.ax_o.get_ylim()),
                tuple(self.ax_i.get_xlim()), tuple(self.ax_i.get_ylim()))

    def _redraw(self, keep_view=True):
        m = self._model()
        view = self._view() if keep_view and self._lims is not None else None
        pv = self._prev_display() if self._showing_prev() else None
        if pv is not None:
            # the earlier frame with its saved selection (read only), view and contrast as set
            draw_static(self.ax_o, self.ax_i, pv["ortho"], pv["img"], pv["geo"], pv["gcps"], pv["picks"], m,
                        wline=self._wline())
            label = f"{self._prev_sel[0]} {self._prev_sel[1]} (read only)"
            self.ax_o.set_title(f"{label}: ortho ({m} model)", fontsize=10, color="crimson")
            self.ax_i.set_title(f"{label}: oblique with its selection", fontsize=10, color="crimson")
        else:
            if m not in self.orthos:
                self.orthos[m] = self.geo[m].ortho(self.img_disp)
            draw_static(self.ax_o, self.ax_i, self.orthos[m], self.img_disp, self.geo[m], self.gcps[m],
                        self.picks[m], m, wline=self._wline())
        if view is not None:
            set_view(self.ax_o, view[:2])
            set_view(self.ax_i, view[2:])
        self.art = make_preview(self.ax_o, self.ax_i, animated=True)
        self.canvas.draw()
        self._lims = self._view()
        self._update_status()

    def _update_status(self, hover=None):
        m = self._model()
        if self._showing_prev():
            self.status.set(f"{self.selected_date} {self.selected_time} | {m} | showing {self._prev_sel[0]} "
                            f"{self._prev_sel[1]} with its selection (read only): untick 'display image' to pick")
            self.info.set("")
            return
        p = self.picks[m]
        step = self.step_var.get()
        saved = frb_path(self.selected_camera, m, self.selected_date, self.selected_time).exists()
        complete = all(v is not None for v in p.values())
        state = ("saved" if saved else "complete") if complete else ("incomplete, saved file kept" if saved else "")
        if step in STEPS:
            where = "ortho or oblique view" if step != "water" else "oblique view, guideline of the far bank"
            need = "  (pick the far bank first)" if step == "water" and p["lx1"] is None else ""
            nxt = f"next: {STEPS[step]} ({where}){need}"
        else:
            nxt = "no point active: choose one under 'Selection' to change it"
        self.status.set(f"{self.selected_date} {self.selected_time} | {m} | GCPs {self.gcp_file[m]} | {nxt}"
                        + (f" | {state}" if state else "") + f" | bank line: {self.line_info.get()}")

        sel = selection(self.geo[m], p)
        parts = []
        if p["lx1"] is not None:
            parts.append(f"far bank local x {p['lx1']:.2f} m")
        if p["lx2"] is not None:
            parts.append(f"close bank local x {p['lx2']:.2f} m")
        if p["lx1"] is not None and p["lx2"] is not None:
            parts.append(f"width {abs(p['lx1'] - p['lx2']):.2f} m")
        if "fb_m" in sel:
            parts.append(f"freeboard {sel['fb_px']:.1f} px = {sel['fb_m']:.3f} m ({sel['v1']:.1f} px/m)")
        if hover:
            if "fb_m" in hover:
                parts.append(f"hover: freeboard {hover['fb_px']:.1f} px = {hover['fb_m']:.3f} m")
            else:
                parts.append(f"hover: local x {hover['lx']:.2f} m"
                             + (f", width {hover['width_m']:.2f} m" if hover["width_m"] is not None else ""))
        self.info.set(" | ".join(parts))

    # ----------------------------------------------------
    # zoom / pan (held keys), original view; the other view follows
    # ----------------------------------------------------

    def _tool_hold(self, tool, down):
        # zoom / pan active while c / x is held down
        if self.img is None:
            return
        mode = {"zoom": "zoom rect", "pan": "pan/zoom"}[tool]
        if (self.toolbar.mode == mode) != down:
            getattr(self.toolbar, tool)()           # switches the tool on / off

    def _reset_view(self):
        # original view: the default ortho view and the oblique view showing the same area
        if self.img is None:
            return
        set_view(self.ax_o, fit_to_box(self.ax_o, *ORTHO_VIEW))
        follow(self._disp_geo(), self.ax_o, self.ax_i, True, self.img_disp.shape)
        self.toolbar.push_current()
        self._lims = self._view()
        self.canvas.draw_idle()

    def _on_view(self, kind):
        # called by the toolbar after a zoom, pan, home, back or forward
        if self.img is None or self._lims is None:
            return
        view = self._view()
        changed_o, changed_i = view[:2] != self._lims[:2], view[2:] != self._lims[2:]
        if changed_o != changed_i:
            if changed_o:
                follow(self._disp_geo(), self.ax_o, self.ax_i, True, self.img_disp.shape)
            else:
                follow(self._disp_geo(), self.ax_i, self.ax_o, False, self.img_disp.shape)
            if kind in ("zoom", "pan"):
                self.toolbar.push_current()
        self._lims = self._view()
        self.canvas.draw_idle()

    # ----------------------------------------------------
    # events
    # ----------------------------------------------------

    def _enable_events(self):
        for cid in self._mpl_cids:
            self.canvas.mpl_disconnect(cid)
        self._mpl_cids = [
            self.canvas.mpl_connect("button_press_event", self._on_click),
            self.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.canvas.mpl_connect("draw_event", self._on_draw),
        ]

    def _on_draw(self, event):
        # background without the moving parts, for blitting
        if not self._printing:
            self._bg = self.canvas.copy_from_bbox(self.fig.bbox)

    def _bank_x(self, event):
        # local x of a bank under the cursor: ortho x, or the bank top in the image column of the cursor
        if event.inaxes is self.ax_o:
            return float(event.xdata)
        if event.inaxes is self.ax_i:
            return self.geo[self._model()].line_x_local(float(event.xdata))
        return None

    def _on_motion(self, event):

        if self.art is None or self._bg is None or self.toolbar.mode != "":
            return

        m = self._model()
        p = self.picks[m]
        step = "" if self._showing_prev() else self.step_var.get()      # no picking while an earlier frame is shown
        h = self.img_disp.shape[0]
        hover = None

        lx = self._bank_x(event) if step in ("bank1", "bank2") and event.xdata is not None else None
        if lx is not None:
            hover = preview_bank(self.art, self.geo[m], p, step[-1], lx, h, self._wline())
        elif step == "water" and event.inaxes is self.ax_i and event.ydata is not None and p["lx1"] is not None:
            hover = preview_water(self.art, self.geo[m], p, float(event.ydata), h)
        else:
            clear_preview(self.art)

        self.canvas.restore_region(self._bg)
        for a in self.art.values():
            a.axes.draw_artist(a)
        self.canvas.blit(self.fig.bbox)
        self._update_status(hover=hover)

    def _on_click(self, event):

        if self.toolbar.mode != "" or event.xdata is None or self.img is None or self._showing_prev():
            return

        m = self._model()
        p = self.picks[m]
        step = self.step_var.get()

        if step in ("bank1", "bank2"):
            lx = self._bank_x(event)
            if lx is None:
                return
            k = step[-1]
            p[f"lx{k}"] = bank_pick(p, k, lx)
            if event.inaxes is self.ax_o:
                self.guides[m][k] = "ortho"
            else:
                self.guides[m][k] = "oblique, " + (f"water line {self._prev_sel[0]} {self._prev_sel[1]}"
                                                   if self._wline() is not None else SURFACE)
            if k == "1":
                p["y3"] = None                          # the water point belongs to the old far bank
        elif step == "water":
            if event.inaxes is not self.ax_i or p["lx1"] is None:
                return
            p["y3"] = water_pick(self.geo[m], p, float(event.ydata))
        else:
            return                                      # no point active (selection complete)

        self._redraw()
        self._image(IMG_TAG[step])                      # this point, view and contrast as shown
        if all(v is not None for v in p.values()):
            self._save()
            self._image_complete()
            self.step_var.set("")                       # no point active until one is chosen again
            self._reset_view()
        else:
            self.step_var.set(next_step(p))
        self._update_status()

    # ----------------------------------------------------
    # save / images / redo / delete
    # ----------------------------------------------------

    def _save(self):
        m = self._model()
        path = frb_path(self.selected_camera, m, self.selected_date, self.selected_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = frb_row(m, self.selected_date, self.selected_time, self.geo[m], self.picks[m], self.gcp_file[m],
                      self.guides[m])
        pd.DataFrame([row], columns=COLUMNS).to_csv(path, index=False)
        print(f"[Saved] {path}")
        self._mark_times()

    def _title(self, tag):
        m = self._model()
        p = self.picks[m]
        sel = selection(self.geo[m], p)
        parts = [f"{self.selected_camera} {self.selected_date} {self.selected_time}", m, tag.replace("_", " ")]
        if p["lx1"] is not None and p["lx2"] is not None:
            parts.append(f"width {abs(p['lx1'] - p['lx2']):.2f} m")
        if "fb_m" in sel:
            parts.append(f"freeboard {sel['fb_m']:.3f} m")
        if self._wline() is not None:
            parts.append(f"bank line: {self.line_info.get()}")
        return " | ".join(parts)

    def _image(self, tag):
        # the figure as shown (view and contrast), after picking the point `tag`
        path = frb_img_path(self.selected_camera, self._model(), self.selected_date, self.selected_time, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        title = self.fig.text(0.99, 0.005, self._title(tag), ha="right", va="bottom", fontsize=10)
        self._printing = True
        try:
            self.fig.savefig(path, dpi=IMG_DPI)
        finally:
            self._printing = False
            title.remove()
            self.canvas.draw()
        print(f"[Image] {path}")

    def _image_complete(self):
        # complete selection: both views in the original view, without contrast enhancement
        m = self._model()
        if m not in self.orthos_raw:
            self.orthos_raw[m] = self.geo[m].ortho(self.img)
        fig = Figure(figsize=self.fig.get_size_inches(), dpi=100)
        ax_o, ax_i = fig.add_subplot(121), fig.add_subplot(122)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08, wspace=0.08)
        draw_static(ax_o, ax_i, self.orthos_raw[m], self.img, self.geo[m], self.gcps[m], self.picks[m], m,
                    wline=self._wline())
        fig.text(0.5, 0.995, self._title("complete"), ha="center", va="top", fontsize=10)
        path = frb_img_path(self.selected_camera, m, self.selected_date, self.selected_time, "complete")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=IMG_DPI)
        print(f"[Image] {path}")

    def redo(self):
        if self.img is None:
            return
        self.picks[self._model()] = dict(EMPTY)
        self.guides[self._model()] = {}
        self.step_var.set(next(iter(STEPS)))
        self._redraw()

    def delete(self):
        if self.img is None:
            return
        m = self._model()
        path = frb_path(self.selected_camera, m, self.selected_date, self.selected_time)
        if not path.exists():
            messagebox.showinfo("Delete", f"No saved selection ({m}) for this frame")
            return
        if messagebox.askyesno("Delete", f"Delete the saved selection ({m}) and its images?\n{path.name}"):
            path.unlink()
            for tag in [*IMG_TAG.values(), "complete", "bank1", "bank2"]:      # incl. the first image names
                frb_img_path(self.selected_camera, m, self.selected_date, self.selected_time, tag).unlink(missing_ok=True)
            self.picks[m] = dict(EMPTY)
            self.guides[m] = {}
            self.step_var.set(next(iter(STEPS)))
            self._mark_times()
            self._redraw()

    def _on_close(self):
        self.master.quit()
        self.master.destroy()


def main():
    df_frames = load_frames_index_only(frames_dir)
    root = tk.Tk()
    XsFrbApp(root, df_frames)
    root.mainloop()


if __name__ == "__main__":
    main()
