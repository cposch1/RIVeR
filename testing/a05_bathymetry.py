#!/usr/bin/env python3
from __future__ import annotations

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
    return pd.read_parquet(Path(frames_dir) / "_frame_paths.parquet")


def find_reference_frame(row):
    return Path(row["frame_path"]).parent / "0000000000.jpg"


def xs_coord_path(cam, date, time_):
    return bathy_dir / cam / f"{cam}_xs_coord_{date}_{time_}.csv"


def transform_path(cam, date, time_):
    return rect_dir / cam / f"{cam}_transform_{date}_{time_}.json"


# ✅ NEW: JSON path
def xs_json_path(cam, date, time_):
    return bathy_dir / cam / f"{cam}_xs_{date}_{time_}.json"


# ✅ NEW: per-camera max XS length
def get_max_xs_length_per_camera():
    max_len_dict = {}

    for cam_dir in (bathy_dir).glob("*"):
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

                if length > max_len:
                    max_len = length

            except Exception:
                continue

        max_len_dict[cam] = max_len

    return max_len_dict


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-level", type=float, required=True)
    parser.add_argument("--num-points", type=int, default=15)

    args = parser.parse_args()
    lvl = args.water_level
    npts = args.num_points

    df = load_frames_index()
    df_unique = df.drop_duplicates(
        ["camera", "date_yyyymmdd", "time_hhmmss"]
    )

    # ✅ compute per-camera limits
    max_xs_length_dict = get_max_xs_length_per_camera()

    for cam, val in max_xs_length_dict.items():
        print(f"[INFO] {cam} max XS length: {val:.2f} m")

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

            # -------------------------
            # Distance
            # -------------------------
            length = np.sqrt((x_ri - x_le)**2 + (y_ri - y_le)**2)

            xs = np.linspace(0, length, npts)

            xL = 0
            xR = length
            xM = length / 2

            def bath(x):
                return lvl * (x - xM)**2 / ((xL - xM)*(xR - xM)) * (-1)

            bath_points = [(float(x), float(bath(x))) for x in xs]

            # -------------------------
            # Save bathymetry CSV
            # -------------------------
            out_dir = bathy_dir / cam
            out_dir.mkdir(parents=True, exist_ok=True)

            bath_file = out_dir / f"{cam}_bath_{date}_{time_}.csv"

            with bath_file.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["d", "h"])
                w.writerows(bath_points)

            # -------------------------
            # ✅ COMPUTE PIXEL COORDS
            # -------------------------
            left_px = transform_real_world_to_pixel(x_le, y_le, T)
            right_px = transform_real_world_to_pixel(x_ri, y_ri, T)

            # -------------------------
            # ✅ CREATE JSON
            # -------------------------
            xsections = {
                "section1": {
                    "east_l": x_le,
                    "north_l": y_le,
                    "east_r": x_ri,
                    "north_r": y_ri,
                    "level": lvl,
                    "num_stations": npts,
                    "alpha": 1,
                    "bath": str(bath_file),
                    "left_station": 2.0,
                    "xl": float(left_px[0]),
                    "yl": float(left_px[1]),
                    "xr": float(right_px[0]),
                    "yr": float(right_px[1]),
                    "rw_length": float(length)
                }
            }

            json_file = xs_json_path(cam, date, time_)

            with json_file.open("w") as f:
                json.dump(xsections, f, indent=2)

            # -------------------------
            # Visualization
            # -------------------------
            frame = mpimg.imread(find_reference_frame(row))

            stations = np.array([p[0] for p in bath_points])
            stages = np.array([p[1] for p in bath_points])
            stages = np.round(stages, 6)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # ---- LEFT: image ----
            ax1.imshow(frame)
            ax1.plot([left_px[0], right_px[0]],
                     [left_px[1], right_px[1]],
                     color='#F5BF61', linewidth=2)
            ax1.plot(left_px[0], left_px[1], 'o', color='#ED6B57', markersize=4)
            ax1.plot(right_px[0], right_px[1], 'o', color='#62C655', markersize=4)

            ax1.set_title("Cross-Section Location")
            ax1.axis("off")

            # ---- RIGHT: bathymetry ----
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

            max_xs_length = max_xs_length_dict.get(cam, length)
            margin = 0.05 * max_xs_length

            ax2.set_xlim(-margin, max_xs_length + margin)
            ax2.set_xlabel("Distance from left bank (m)")
            ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Bathymetry Profile")
            ax2.grid(True)

            fig.tight_layout()

            # -------------------------
            # Save image
            # -------------------------
            out_dir_img = bathy_dir / cam / "bath_imgs"
            out_dir_img.mkdir(parents=True, exist_ok=True)

            out_img = out_dir_img / f"{cam}_bath_{date}_{time_}.png"

            fig.savefig(out_img, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        except Exception as e:
            print(f"[FAIL] {cam} {date} {time_}: {e}")


if __name__ == "__main__":
    main()