# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Vertical measurements below the modelled surface (a05)
#
# Per-frame measurements of `a05_vert_meas.py` (`gcps/<cam>/vert_meas/<cam>_vert_meas_<date>_<time>.csv`):
# - **water surface (bank)**, **water surface (GCP)** and **stream bed (GCP)**: vertical distance below the
#   modelled surface (line through the modelled GCP3 and GCP4, central GCP model of b02), in m with the local
#   vertical scale along the line; also relative to the low / high lines (b02 uncertainty envelope)
# - **water depth (bank)** = stream bed − water surface (bank), **water depth (GCP)** = stream bed − water surface (GCP)
#   (frames with both; computed here from the measurements, as in a05)
#
# Combined with the surface lowering of the central GCP model (GCP4, b02) for levels relative to the surface on
# 2025-07-01, with the camera-site air temperature (PDH) and, where cross-sections exist, the channel width (a07).

# %%
from river.config import gcps_dir, bathy_dir, aws_dir
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress

# %% [markdown]
# ## Load data

# %%
# ==================================================
# USER SETTINGS
# ==================================================
cam = "ilh-cam1-pt"
cam_dir = gcps_dir / cam
vert_dir = cam_dir / "vert_meas"                  # a05 output
xs_dir = bathy_dir / cam                          # a07 cross-sections (channel width)
t0 = pd.Timestamp("2025-07-01 12:00")             # reference: calibration image / first stake frame

KINDS = ["water_surface", "water_surface_gcp", "stream_bed"]
DEPTHS = {"water_depth_bank": "water_surface", "water_depth_gcp": "water_surface_gcp"}    # depth = stream bed − water surface
LABEL = {"water_surface": "water surface (bank)", "water_surface_gcp": "water surface (GCP)", "stream_bed": "stream bed (GCP)",
         "water_depth_bank": "water depth (bank)", "water_depth_gcp": "water depth (GCP)"}
COL = {"water_surface": "deepskyblue", "water_surface_gcp": "royalblue", "stream_bed": "orange",
       "water_depth_bank": "deepskyblue", "water_depth_gcp": "royalblue", "surface": "#e7298a"}


def p_text(p):
    return "p < 0.01" if p < 0.01 else f"p = {p:.2f}"


# ==================================================
# LOAD MEASUREMENTS (one file per frame)
# ==================================================
records = []

for f in sorted(vert_dir.glob(f"{cam}_vert_meas_*.csv")):
    m = re.search(r"(\d{8})_(\d{6})", f.name)
    df_f = pd.read_csv(f)
    df_f["timestamp"] = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
    records.append(df_f)

df_long = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
print(f"{len(records)} frames with measurements in {vert_dir}")

# wide: one row per frame; columns (variable, kind)
if not df_long.empty:
    df_long = df_long[df_long["kind"].isin(KINDS)]        # measurements; the water depths are computed below
    df_v = pd.concat(
        {c: df_long.pivot_table(index="timestamp", columns="kind", values=c, aggfunc="first", dropna=False)
         for c in ["dist_m", "dist_m_low", "dist_m_high", "x", "dy_px", "px_per_m"]},
        axis=1
    ).sort_index()
    # water depths = stream bed − water surface (central, low and high lines)
    for d, w in DEPTHS.items():
        if {w, "stream_bed"} <= set(df_v["dist_m"].columns):
            for c in ["dist_m", "dist_m_low", "dist_m_high"]:
                df_v[(c, d)] = df_v[(c, "stream_bed")] - df_v[(c, w)]
    print(df_v["dist_m"].notna().sum().reindex(KINDS + list(DEPTHS)).dropna().astype(int).rename("frames").to_string())
else:
    df_v = pd.DataFrame()
    print("No measurements yet -> measure with a05_vert_meas.py")

# %%
# ==================================================
# SURFACE LOWERING (central GCP model, GCP4) AND CAMERA-SITE AIR TEMPERATURE
# ==================================================
calib = pd.read_csv(cam_dir / f"{cam}_emergence_calibration.csv", dtype={"point": str, "date": str, "time": str}).iloc[0]
pxm = float(calib["px_per_m"])

env = pd.read_csv(cam_dir / "_modelling" / "gcp_model_envelope.csv", parse_dates=["timestamp"])
env4 = env[env["gcp_id"] == 4].set_index("timestamp").sort_index()
surf = pd.DataFrame({
    "L_m": (env4["y_central"] - env4["y_central"].loc[t0]) / pxm,     # surface lowering since t0 (positive = lower)
    "L_m_lo": env4["dy_p16"] / pxm,                                   # ±1σ (dy_* relative to the first stake frame = t0)
    "L_m_hi": env4["dy_p84"] / pxm,
})

# camera-site logger: 10-min values labelled at the interval end -> hourly means at the hour start
logger = pd.read_csv(aws_dir / "hydro_ilh_Table1.dat", skiprows=[0, 2, 3], na_values=["NAN"],
                     parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")
t_cam = logger["AirTemp_C"].shift(-10, freq="min").resample("1h").mean().interpolate(limit=3)
pdh_h = t_cam.clip(lower=0)                                          # positive degree hours per hour (°C·h)
pdh_24h = pdh_h.rolling("24h", closed="left").sum()                  # PDH of the 24 h before t (°C·h)
cum_pdh = pdh_h.loc[t0:].cumsum().shift(1, fill_value=0) / 24        # PDH since t0 before t (°C·d)

print(f"Calibration: {calib['point']} {calib['emergence_m']} m on {calib['date']} -> {pxm:.1f} px/m")

if not df_v.empty:
    near = dict(method="nearest", tolerance=pd.Timedelta("13h"))
    df_v[("L_m", "")] = surf["L_m"].reindex(df_v.index, **near)
    df_v[("PDH_24h", "")] = pdh_24h.reindex(df_v.index, **near)
    df_v[("cum_PDH", "")] = cum_pdh.reindex(df_v.index, **near)

# %% [markdown]
# ## Overview

# %%
if not df_v.empty:
    complete = df_v["dist_m"].reindex(columns=KINDS).notna().all(axis=1)
    print(f"Frames: {len(df_v)} ({df_v.index.min():%Y-%m-%d} -> {df_v.index.max():%Y-%m-%d}), "
          f"with all {len(KINDS)} measurements: {complete.sum()}")
    summary = df_v["dist_m"].reindex(columns=KINDS + list(DEPTHS)).describe().T[["count", "mean", "min", "max"]]
    display(summary.round(3)) if "display" in globals() else print(summary.round(3))

    # where along the surface line the measurements were made (QC)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for k in KINDS:
        if k in df_v["x"].columns:
            axes[0].plot(df_v.index, df_v["x"][k], "o", color=COL[k], label=LABEL[k])
            axes[1].plot(df_v.index, df_v["px_per_m"][k], "o", color=COL[k], label=LABEL[k])
    axes[0].set_ylabel("Image x of the measurement (px)")
    axes[1].set_ylabel("Local vertical scale (px per m)")
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Distances below the modelled surface and water depth

# %%
if not df_v.empty:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    for k in KINDS:
        if k not in df_v["dist_m"].columns:
            continue
        d = df_v[[("dist_m", k), ("dist_m_low", k), ("dist_m_high", k)]].droplevel(1, axis=1).dropna(subset=["dist_m"])
        ax.fill_between(d.index, d["dist_m_low"], d["dist_m_high"], color=COL[k], alpha=0.2, lw=0)
        ax.plot(d.index, d["dist_m"], color=COL[k], lw=1.5, label=f"{LABEL[k]} (band: low / high GCP model)")
    ax.invert_yaxis()
    ax.set_ylabel("Below the modelled surface (m)")
    ax.set_title("Vertical distance below the modelled surface")

    ax = axes[1]
    for k in DEPTHS:
        if k not in df_v["dist_m"].columns:
            continue
        d = df_v[[("dist_m", k), ("dist_m_low", k), ("dist_m_high", k)]].droplevel(1, axis=1).dropna(subset=["dist_m"])
        ax.fill_between(d.index, d["dist_m_low"], d["dist_m_high"], color=COL[k], alpha=0.2, lw=0)
        ax.plot(d.index, d["dist_m"], color=COL[k], lw=1.5, label=f"{LABEL[k]} (band: low / high GCP model)")
    ax.set_ylabel("Water depth (m)")
    ax.set_title("Water depth = stream bed (GCP) − water surface (bank / GCP)")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Levels relative to the surface on 2025-07-01
#
# Level = −(surface lowering since 2025-07-01 + distance below the modelled surface). The surface lowering is the
# central GCP model at GCP4 (b02), i.e. the ambient surface is assumed to lower uniformly along the line.

# %%
if not df_v.empty:
    lev = pd.DataFrame({k: -(df_v[("L_m", "")] + df_v["dist_m"][k]) for k in KINDS if k in df_v["dist_m"].columns})
    s = surf.loc[t0:df_v.index.max() + pd.Timedelta("1D")]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(s.index, -s["L_m_lo"], -s["L_m_hi"], color=COL["surface"], alpha=0.15, lw=0)
    ax.plot(s.index, -s["L_m"], color=COL["surface"], lw=1.6, label="modelled surface (GCP4 central model, band ±1σ)")
    for k in lev:
        ok = lev[k].notna()
        ax.plot(lev.index[ok], lev[k][ok], color=COL[k], lw=1.5, label=LABEL[k])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("Level relative to the surface on 2025-07-01 (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # linear trends over the measured period (m per day)
    trend_rows = []
    days = (lev.index - lev.index.min()) / pd.Timedelta("1D")
    for name, ser in [("surface", -df_v[("L_m", "")])] + [(k, lev[k]) for k in lev]:
        ok = ser.notna().values
        if ok.sum() < 3:
            continue
        fit = linregress(days[ok], ser.values[ok])
        trend_rows.append({"level": name, "m_per_day": fit.slope, "r": fit.rvalue, "p": fit.pvalue, "n": int(ok.sum())})
    trends = pd.DataFrame(trend_rows).set_index("level")
    display(trends.round(4)) if "display" in globals() else print(trends.round(4))

# %% [markdown]
# ## Relation to air temperature (camera-site logger)

# %%
panels = []

if not df_v.empty:
    # water depths vs PDH of the 24 h before the frame; stream bed vs cumulative PDH
    panels = [(df_v[("PDH_24h", "")], df_v["dist_m"][k], "PDH of the 24 h before the frame (°C)", "Water depth (m)", LABEL[k])
              for k in DEPTHS if k in df_v["dist_m"].columns]
    if "stream_bed" in df_v["dist_m"].columns:
        panels.append((df_v[("cum_PDH", "")], df_v["dist_m"]["stream_bed"], "Cumulative PDH since 2025-07-01 (°C·d)",
                       "Below the modelled surface (m)", LABEL["stream_bed"]))

if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), squeeze=False)

    for ax, (x, y, xlabel, ylabel, title) in zip(axes[0], panels):
        ok = x.notna() & y.notna()
        sc = ax.scatter(x[ok], y[ok], c=mdates.date2num(df_v.index[ok]), cmap="viridis")
        if ok.sum() >= 3:
            fit = linregress(x[ok], y[ok])
            xx = np.array([x[ok].min(), x[ok].max()])
            ax.plot(xx, fit.intercept + fit.slope * xx, color="k", lw=1,
                    label=f"r = {fit.rvalue:.2f}, {p_text(fit.pvalue)}, n = {ok.sum()}")
            ax.legend(loc="upper left")
        fig.colorbar(sc, ax=ax, label="Date", format=mdates.DateFormatter("%m-%d"))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Relation to channel width (a07 cross-sections, same frames)

# %%
widths = {}

for f in xs_dir.glob(f"{cam}_xs_coord_*.csv"):
    m = re.search(r"(\d{8})_(\d{6})", f.name)
    pts = pd.read_csv(f, header=None).values[:2, :2].astype(float)
    widths[pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")] = float(np.hypot(*(pts[1] - pts[0])))

width = pd.Series(widths, dtype=float).sort_index()
print(f"{len(width)} cross-sections in {xs_dir}")

for k in DEPTHS:
    if df_v.empty or k not in df_v["dist_m"].columns or width.empty:
        continue
    dw = pd.DataFrame({"depth_m": df_v["dist_m"][k], "width_m": width.reindex(df_v.index)}).dropna()
    print(f"Frames with {LABEL[k]} and width: {len(dw)}")

    if len(dw) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        sc = ax.scatter(dw["width_m"], dw["depth_m"], c=mdates.date2num(dw.index), cmap="viridis")
        fit = linregress(dw["width_m"], dw["depth_m"])
        xx = np.array([dw["width_m"].min(), dw["width_m"].max()])
        ax.plot(xx, fit.intercept + fit.slope * xx, color="k", lw=1,
                label=f"r = {fit.rvalue:.2f}, {p_text(fit.pvalue)}, n = {len(dw)}")
        fig.colorbar(sc, ax=ax, label="Date", format=mdates.DateFormatter("%m-%d"))
        ax.set_title(LABEL[k])
        ax.set_xlabel("Channel width (m)")
        ax.set_ylabel("Water depth (m)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(dw.index, dw["width_m"] / dw["depth_m"], color="k", lw=1.2)
        ax.set_ylabel("Width / depth")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
