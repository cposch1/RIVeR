#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

from river.config import *  # noqa


# ---------------------------
# Helpers
# ---------------------------

def parse_json_filename(json_path: Path):
    m = re.match(r"(.*?)_transform_(\d{8})_(\d{6})\.json", json_path.name)
    if not m:
        raise ValueError(f"Invalid JSON name: {json_path}")
    return m.group(1), m.group(2), m.group(3)


def load_frames_index(frames_root: Path):
    return pd.read_parquet(frames_root / "_frame_paths.parquet")

def _fix_windows_path(p: str) -> str:
    """Convert Git Bash path (/c/...) → Windows path (C:/...)."""
    if p and len(p) > 2 and p[0] == "/" and p[2] == "/":
        drive = p[1].upper()
        return f"{drive}:/{p[3:]}"
    return p


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    json_path = Path(_fix_windows_path(args.json))
    cam, ref_date, ref_time = parse_json_filename(json_path)

    print(f"[INFO] Camera: {cam}")

    
    frames_root = Path(_fix_windows_path(str(frames_dir)))
    df = load_frames_index(frames_root)

    df = df[df["camera"] == cam]

    df_unique = df.drop_duplicates(
        subset=["camera", "date_yyyymmdd", "time_hhmmss"]
    ).sort_values(["date_yyyymmdd", "time_hhmmss"])

    print(f"[INFO] Processing {len(df_unique)} scenes")

    # ✅ GUI defaults
    xlim = (-20, 50)
    ylim = (-10, 20)

    # ✅ output dirs
    full_dir = rect_dir / cam / "orthorectification_imgs_auto"
    clean_dir = rect_dir / cam / "ortho_imgs_auto"
    full_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # ✅ reference GCPs
    gcp_ref = gcps_dir / cam / f"{cam}_gcps_img_{ref_date}_{ref_time}.csv"

    for _, row in df_unique.iterrows():
        date = row["date_yyyymmdd"]
        time_ = row["time_hhmmss"]

        try:
            # ---------- GCP reuse ----------
            gcp_target = gcps_dir / cam / f"{cam}_gcps_img_{date}_{time_}.csv"
            gcp_target.parent.mkdir(parents=True, exist_ok=True)
            if not gcp_target.exists():
                shutil.copy(gcp_ref, gcp_target)

            # ---------- TRANSFORM ----------
            transformation = transform(df, cam, date, time_)

            # ---------- SAVE JSON ----------
            transf_file = rect_dir / cam / f"{cam}_transform_{date}_{time_}.json"
            with transf_file.open("w") as f:
                json.dump(transformation["transformation_matrix"], f, indent=1)

            # ---------- LOAD FRAME ----------
            
            frame_path_fixed = Path(_fix_windows_path(str(row["frame_path"])))
            time_dir = frame_path_fixed.parent

            img = mpimg.imread(time_dir / "0000000000.jpg")

            # ---------- LOAD GCPs ----------
            pts = load_gcps_img(cam, date, time_)
            (x1, y1) = pts["point1"]
            (x2, y2) = pts["point2"]
            (x3, y3) = pts["point3"]
            (x4, y4) = pts["point4"]

            # ---------- PIX → WORLD ----------
            rw = []
            for x, y in [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]:
                rw.append(transform_pixel_to_real_world(x, y, transformation["transformation_matrix"]))
            rw = np.array(rw)

            xw, yw = rw[:,0], rw[:,1]

            # =====================================================
            # ✅ FULL (GUI IDENTICAL)
            # =====================================================
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))

            # ORIGINAL
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title('Original Image')

            ax1.plot([x1,x2],[y1,y2], color='#6CD4FF', linewidth=2)
            ax1.plot([x2,x3],[y2,y3], color='#62C655', linewidth=2)
            ax1.plot([x3,x4],[y3,y4], color='#ED6B57', linewidth=2)
            ax1.plot([x4,x1],[y4,y1], color='#F5BF61', linewidth=2)
            ax1.plot([x1,x3],[y1,y3], color='#CC4BC2', linewidth=2)
            ax1.plot([x2,x4],[y2,y4], color='#7765E3', linewidth=2)

            ax1.plot(x1, y1, 'o', color='#ED6B57', markersize=3)
            ax1.text(x1,y1,"1",color='#ED6B57',fontsize=8)

            for i,(x,y) in enumerate([(x2,y2),(x3,y3),(x4,y4)],start=2):
                ax1.plot(x,y,'o',color='#6CD4FF',markersize=3)
                ax1.text(x,y,str(i),color='#6CD4FF',fontsize=8)

            # ORTHO
            ax2.imshow(
                transformation["transformed_img"],
                extent=transformation["extent"]
            )

            ax2.set_xlim(*xlim)
            ax2.set_ylim(*ylim)
            ax2.set_aspect("equal")

            ax2.plot([xw[0],xw[1]],[yw[0],yw[1]], color='#6CD4FF', linewidth=2)
            ax2.plot([xw[1],xw[2]],[yw[1],yw[2]], color='#62C655', linewidth=2)
            ax2.plot([xw[2],xw[3]],[yw[2],yw[3]], color='#ED6B57', linewidth=2)
            ax2.plot([xw[3],xw[0]],[yw[3],yw[0]], color='#F5BF61', linewidth=2)
            ax2.plot([xw[0],xw[2]],[yw[0],yw[2]], color='#CC4BC2', linewidth=2)
            ax2.plot([xw[1],xw[3]],[yw[1],yw[3]], color='#7765E3', linewidth=2)

            ax2.plot(xw[0], yw[0], 'o', color='#ED6B57', markersize=3)
            ax2.text(xw[0], yw[0], "1", color='#ED6B57', fontsize=8)

            for i in range(1,4):
                ax2.plot(xw[i], yw[i], 'o', color='#6CD4FF', markersize=3)
                ax2.text(xw[i], yw[i], str(i+1), color='#6CD4FF', fontsize=8)

            # ✅ SCALE BAR (GUI identical)
            display_extent = (*xlim, *ylim)
            map_width = display_extent[1] - display_extent[0]
            magnitude = 10 ** np.floor(np.log10(map_width * 0.2))
            scale_length = np.round(map_width * 0.2 / magnitude) * magnitude
            scale_length_rounded = int(scale_length) if scale_length < 10 else scale_length

            margin = map_width * 0.05
            bar_height = (display_extent[3] - display_extent[2]) * 0.015

            x_pos = display_extent[1] - margin - scale_length_rounded
            y_pos = display_extent[2] + margin

            rect = Rectangle((x_pos, y_pos), scale_length_rounded, bar_height,
                             fc='white', ec='black')
            ax2.add_patch(rect)

            ax2.text(x_pos + scale_length_rounded/2,
                     y_pos + 2*bar_height,
                     f'{int(scale_length_rounded)} m',
                     ha='center', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.7, pad=2))

            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Orthorectified Image')

            fig.tight_layout()

            fig.savefig(
                full_dir / f"{cam}_orthorect_{date}_{time_}.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True
            )
            plt.close(fig)

            # =====================================================
            # ✅ CLEAN IMAGE (BIGGER + AXES)
            # =====================================================
            fig2, ax = plt.subplots(figsize=(10,6))

            ax.imshow(transformation["transformed_img"],
                      extent=transformation["extent"])

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            fig2.savefig(
                clean_dir / f"{cam}_orthoimg_{date}_{time_}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True
            )
            plt.close(fig2)

            print(f"[OK] {date} {time_}")

        except Exception as e:
            print(f"[FAIL] {date} {time_}: {e}")


if __name__ == "__main__":
    main()