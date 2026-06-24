#!/usr/bin/env python3
"""
RIVeR-ICE — Bathymetry Generator

Purpose:
- Compute bathymetric profiles from cross-sections and transformations
- Generate per-scene bathymetry CSV, JSON, and visualization

Inputs:
- FRAMES_DIR/_frame_paths.parquet
- Cross-sections: <cam>_xs_coord_<date>_<time>.csv
- Transform JSONs: <cam>_transform_<date>_<time>.json

Outputs:
- Bathymetry CSV: <cam>_bath_<date>_<time>.csv
- Cross-section JSON: <cam>_xs_<date>_<time>.json
- Visualization PNG: bath_imgs/<cam>_bath_<date>_<time>.png

Parameters:
  --water-level / --water-lvl   Default: 0.2
  --num-points / --num-pts      Default: 15
  --left-st                     Default: 0
  --alpha                       Default: 1
  --verbose                     Show river.config logs
"""

from __future__ import annotations

# ✅ Quiet logging BEFORE import
import sys as _sys
import os as _os
import logging as _logging

_verbose = ("--verbose" in _sys.argv)
_logging.basicConfig(level=_logging.INFO if _verbose else _logging.WARNING)
_logging.getLogger("river.config").setLevel(
    _logging.INFO if _verbose else _logging.WARNING
)

import argparse
import json
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from river.config import *
from river.core.coordinate_transform import transform_real_world_to_pixel


# ---------------------------
# Helpers
# ---------------------------

def load_frames_index():
    return pd.read_parquet(frames_dir / "_frame_paths.parquet")


def find_reference_frame(row):
    return Path(row["frame_path"]).parent / "0000000000.jpg"


def xs_coord_path(cam, date, time_):
    return bathy_dir / cam / f"{cam}_xs_coord_{date}_{time_}.csv"


def transform_path(cam, date, time_):
    return rect_dir / cam / f"{cam}_transform_{date}_{time_}.json"


def xs_json_path(cam, date, time_):
    return bathy_dir / cam / f"{cam}_xs_{date}_{time_}.json"


# ---------------------------
# Per-camera XS length
# ---------------------------

def get_max_xs_length_per_camera():
    max_len_dict = {}

    for cam_dir in bathy_dir.glob("*"):
        if not cam_dir.is_dir():
            continue

        cam = cam_dir.name
        max_len = 0.0

        for xs_file in cam_dir.glob("*_xs_coord_*.csv"):
            try:
                pts = []
                with xs_file.open() as f:
                    for r in csv.reader(f):
                        if not r:
                            continue
                        pts.append((float(r[0]), float(r[1])))

                if len(pts) != 2:
                    continue

                (x1, y1), (x2, y2) = pts
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                max_len = max(max_len, length)

            except Exception:
                continue

        max_len_dict[cam] = max_len

    return max_len_dict


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--water-lvl", type=float, default=0.2, help="Water level/depth (default: 0.2 m)")
    parser.add_argument("--num-pts", type=int, default=15, help="Number points aalong cross-section (default: 15)")
    parser.add_argument("--left-st", type=float, default=0, help="Offset of first point from left bank (default: 0 m)")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha parameter (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    lvl = args.water_lvl
    npts = args.num_pts
    left_st = args.left_st
    alpha = args.alpha

    df = load_frames_index()
    df_unique = df.drop_duplicates(
        ["camera", "date_yyyymmdd", "time_hhmmss"]
    )

    max_xs_length_dict = get_max_xs_length_per_camera()

    for cam, val in max_xs_length_dict.items():
        print(f"[INFO] {cam} max XS length: {val:.2f} m")

    #Overwrite decision
    all_existing_outputs = []
    
    for _, row in df_unique.iterrows():
        cam = row["camera"]
        date = row["date_yyyymmdd"]
        time_ = row["time_hhmmss"]
    
        bath_file = bathy_dir / cam / f"{cam}_bath_{date}_{time_}.csv"
        json_file = xs_json_path(cam, date, time_)
        out_img = bathy_dir / cam / "bath_imgs" / f"{cam}_bath_{date}_{time_}.png"
    
        if bath_file.exists() or json_file.exists() or out_img.exists():
            all_existing_outputs.append((cam, date, time_))
            break  # ✅ one is enough to trigger prompt
    
    
    if all_existing_outputs:
        print("Some outputs already exist.")
    
        choice = input("Overwrite ALL existing outputs? (y/n): ").strip().lower()
    
        if choice == "y":
            overwrite_mode = True
            append_mode = False
        else:
            choice2 = input("Append (skip existing outputs)? (y/n): ").strip().lower()
            if choice2 == "y":
                overwrite_mode = False
                append_mode = True
            else:
                print("Aborted: no processing done.")
                return
    else:
        overwrite_mode = True
        append_mode = False

    for _, row in df_unique.iterrows():
        cam = row["camera"]
        date = row["date_yyyymmdd"]
        time_ = row["time_hhmmss"]

        try:
            xs_file = xs_coord_path(cam, date, time_)
            tf_file = transform_path(cam, date, time_)

            if not xs_file.exists() or not tf_file.exists():
                continue

            print(f"[INFO] {cam} {date} {time_}")

            out_dir = bathy_dir / cam
            out_dir.mkdir(parents=True, exist_ok=True)

            bath_file = out_dir / f"{cam}_bath_{date}_{time_}.csv"
            json_file = xs_json_path(cam, date, time_)
            out_img = bathy_dir / cam / "bath_imgs" / f"{cam}_bath_{date}_{time_}.png"


            # -------------------------
            # Load cross-section
            # -------------------------
            pts = []
            with xs_file.open() as f:
                for r in csv.reader(f):
                    if not r:
                        continue
                    pts.append((float(r[0]), float(r[1])))

            if len(pts) != 2:
                continue

            (x_le, y_le), (x_ri, y_ri) = pts

            # -------------------------
            # Load transform
            # -------------------------
            with tf_file.open() as f:
                T = np.array(json.load(f))

            length = np.sqrt((x_ri - x_le)**2 + (y_ri - y_le)**2)
            xs = np.linspace(0, length, npts)

            xL, xR = 0, length
            xM = length / 2

            def bath(x):
                return lvl * (x - xM)**2 / ((xL - xM) * (xR - xM)) * (-1)

            bath_points = [(float(x), float(bath(x))) for x in xs]

            if overwrite_mode or (append_mode and not bath_file.exists()):
                with bath_file.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["d", "h"])
                    w.writerows(bath_points)

            left_px = transform_real_world_to_pixel(x_le, y_le, T)
            right_px = transform_real_world_to_pixel(x_ri, y_ri, T)

            xsections = {
                "section1": {
                    "east_l": x_le,
                    "north_l": y_le,
                    "east_r": x_ri,
                    "north_r": y_ri,
                    "level": lvl,
                    "num_stations": npts,
                    "alpha": alpha,
                    "bath": str(bath_file),
                    "left_station": left_st,
                    "xl": float(left_px[0]),
                    "yl": float(left_px[1]),
                    "xr": float(right_px[0]),
                    "yr": float(right_px[1]),
                    "rw_length": float(length)
                }
            }

            if overwrite_mode or (append_mode and not json_file.exists()):
                with json_file.open("w") as f:
                    json.dump(xsections, f, indent=2)

            frame_path = find_reference_frame(row)
            if not frame_path.exists():
                continue

            frame = mpimg.imread(frame_path)

            stations = np.array([p[0] for p in bath_points])
            stages = np.round([p[1] for p in bath_points], 6)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            ax1.imshow(frame)
            ax1.plot([left_px[0], right_px[0]],
                     [left_px[1], right_px[1]],
                     color='#F5BF61', linewidth=2)

            ax1.axis("off")

            # Bathymetry plotting
            ax2.plot(stations, stages, 'k-', linewidth=2)
            
            ax2.axhline(y=lvl, color='#6CD4FF', linestyle='--')
            
            ax2.fill_between(
                stations,
                stages,
                lvl,
                where=(stages <= lvl + 1e-10),
                color='#6CD4FF',
                alpha=0.3
            )
            
            # Scaling
            max_len = max_xs_length_dict.get(cam, length)
            margin = 0.05 * max_len
            
            ax2.set_xlim(-margin, max_len + margin)
            ax2.set_xlabel("Distance from left bank (m)")
            ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Bathymetry Profile")
            ax2.grid(True)

            fig.tight_layout()

            if overwrite_mode or (append_mode and not out_img.exists()):
                out_img.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_img, dpi=300, bbox_inches="tight", pad_inches=0)

            plt.close(fig)

            if not overwrite_mode and append_mode:
                print("[INFO] Append mode: skipping existing outputs")

        except Exception as e:
            print(f"[FAIL] {cam} {date} {time_}: {e}")


if __name__ == "__main__":
    main()