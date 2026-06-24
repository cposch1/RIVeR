#!/usr/bin/env python3
from __future__ import annotations

import os
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from river.config import *
from river.core.define_roi_masks import (
    recommend_height_roi,
    create_mask_and_bbox
)
from river.core.piv_pipeline import run_test, run_analyze_all
from river.core.coordinate_transform import transform_real_world_to_pixel


# ---------------------------
# Helpers
# ---------------------------

def load_frames_index():
    return pd.read_parquet(frames_dir / "_frame_paths.parquet")


def frame_dir_from_row(row):
    p = Path(row["frame_path"])
    if not p.is_absolute():
        p = frames_dir / p
    return p.parent


def ref_frame_path(row):
    p = Path(row["frame_path"])
    if not p.is_absolute():
        p = frames_dir / p
    return p.parent / "0000000000.jpg"


def xs_coord_path(cam, date, time_):
    return bathy_dir / cam / f"{cam}_xs_coord_{date}_{time_}.csv"


def transform_path(cam, date, time_):
    return rect_dir / cam / f"{cam}_transform_{date}_{time_}.json"


def build_xsection(cam, date, time_, T, num_stations):

    pts = []
    with xs_coord_path(cam, date, time_).open() as f:
        for r in csv.reader(f):
            if not r:
                continue
            pts.append((float(r[0]), float(r[1])))

    (x_le, y_le), (x_ri, y_ri) = pts

    length = np.sqrt((x_ri - x_le)**2 + (y_ri - y_le)**2)

    xl, yl = transform_real_world_to_pixel(x_le, y_le, T)
    xr, yr = transform_real_world_to_pixel(x_ri, y_ri, T)

    return {
        "section1": {
            "east_l": x_le,
            "north_l": y_le,
            "east_r": x_ri,
            "north_r": y_ri,
            "xl": xl,
            "yl": yl,
            "xr": xr,
            "yr": yr,
            "rw_length": length,
            "num_stations": num_stations
        }
    }


# ---------------------------
# Main
# ---------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--ia1", type=int, default=128)
    parser.add_argument("--ia2", type=int, default=64)
    parser.add_argument("--num-stations", type=int, default=15)

    args = parser.parse_args()

    df = load_frames_index()
    df_unique = df.drop_duplicates(["camera", "date_yyyymmdd", "time_hhmmss"])

    for _, row in df_unique.iterrows():

        cam = row["camera"]
        date = row["date_yyyymmdd"]
        time_ = row["time_hhmmss"]

        xs_file = xs_coord_path(cam, date, time_)
        tf_file = transform_path(cam, date, time_)

        if not xs_file.exists() or not tf_file.exists():
            continue

        print(f"[INFO] Running {cam} {date} {time_}")

        try:
            # -------------------------
            # Load transform
            # -------------------------
            with tf_file.open() as f:
                T = np.array(json.load(f))

            xsections = build_xsection(
                cam, date, time_,
                T,
                args.num_stations
            )

            frame_dir = frame_dir_from_row(row)

            # -------------------------
            # Load reference frame
            # -------------------------
            frame_path = ref_frame_path(row)

            if not frame_path.exists():
                print(f"[WARN] Missing frame: {frame_path}")
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"[WARN] Failed to load image: {frame_path}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -------------------------
            # ROI + MASK
            # -------------------------
            height_roi = recommend_height_roi(
                xsections,
                args.ia1,
                T
            )

            mask, bbox = create_mask_and_bbox(
                frame,
                xsections,
                T,
                height_roi
            )

            if np.sum(mask) < 2000:
                print("[WARN] mask too small")
                continue

            # -------------------------
            # FRAME CHECK (NOW BOTH MODES)
            # -------------------------
            frames = sorted(frame_dir.glob("*.jpg"))

            if len(frames) < 2:
                print("[WARN] Not enough frames for PIV")
                continue

            # -------------------------
            # DEBUG MASK IMAGE
            # -------------------------
            mask_dir = piv_dir / cam / "mask_imgs"
            mask_dir.mkdir(parents=True, exist_ok=True)

            overlay = np.zeros(frame_rgb.shape[:2])
            overlay_mask = np.ones_like(overlay) * 0.5
            overlay_mask[mask == 1] = 0

            fig = plt.figure(figsize=(12, 8))
            plt.imshow(frame_rgb)
            plt.imshow(overlay, alpha=overlay_mask, cmap='gray')

            rect = plt.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2,
                edgecolor='#6CD4FF',
                facecolor='none',
                linestyle='--'
            )
            plt.gca().add_patch(rect)

            xs = xsections["section1"]

            plt.plot([xs["xl"], xs["xr"]],
                     [xs["yl"], xs["yr"]],
                     color='#F5BF61', linewidth=2)

            plt.axis("off")

            mask_img = mask_dir / f"{cam}_mask_{date}_{time_}.png"
            plt.savefig(mask_img, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()

            # -------------------------
            # TEST MODE
            # -------------------------
            if args.mode == "test":

                img1, img2 = frames[0], frames[1]

                piv = run_test(
                    image_1=img1,
                    image_2=img2,
                    mask=mask,
                    bbox=bbox,
                    interrogation_area_1=args.ia1,
                    interrogation_area_2=args.ia2
                )

                frame1 = cv2.imread(str(img1))
                frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(12, 8))
                plt.imshow(frame1_rgb)

                x = np.array(piv['x']).reshape(piv['shape'])
                y = np.array(piv['y']).reshape(piv['shape'])
                u = np.array(piv['u']).reshape(piv['shape'])
                v = np.array(piv['v']).reshape(piv['shape'])

                plt.imshow(overlay, alpha=overlay_mask, cmap='gray')
                plt.quiver(x, y, u, -v, color='blue')

                plt.axis('off')
                plt.tight_layout()

                out_img_dir = piv_dir / cam / "piv_test_imgs"
                out_img_dir.mkdir(parents=True, exist_ok=True)

                plt.savefig(out_img_dir / f"{cam}_piv_test_{date}_{time_}.png",
                            dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close()

            # -------------------------
            # FULL MODE (NOW SAFE)
            # -------------------------
            else:

                try:
                    piv = run_analyze_all(
                        frame_dir,
                        mask=mask,
                        bbox=bbox,
                        interrogation_area_1=args.ia1,
                        interrogation_area_2=args.ia2
                    )
                except Exception as e:
                    print("[WARN] PIV full mode failed — skipping")
                    continue

                x = np.array(piv['x'])
                y = np.array(piv['y'])
                u = np.array(piv['u_median'])
                v = np.array(piv['v_median'])

                plt.figure(figsize=(12, 8))
                plt.imshow(frame_rgb)
                plt.imshow(overlay, alpha=overlay_mask, cmap='gray')

                plt.quiver(x, y, u, -v, color='blue')

                plt.axis('off')

                img_dir = piv_dir / cam / "piv_imgs"
                img_dir.mkdir(parents=True, exist_ok=True)

                plt.savefig(img_dir / f"{cam}_piv_{date}_{time_}.png",
                            dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close()

                data_dir = piv_dir / cam
                data_dir.mkdir(parents=True, exist_ok=True)

                with open(data_dir / f"{cam}_piv_{date}_{time_}.json", "w") as f:
                    json.dump(piv, f, indent=2)

                print(f"[SAVED] {cam} {date} {time_}")

        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[WARN] Skipping {cam} {date} {time_}")
            continue


if __name__ == "__main__":
    main()