#!/usr/bin/env python3
"""
orthorectify_auto.py — Batch orthorectification using a reference transformation

Description:
  - Processes all scenes for a given camera using a reference transformation JSON.
  - Reuses reference GCP image coordinates for all scenes.
  - Runs orthorectification and saves:
      * Full annotated orthorectified images
      * Clean orthorectified images
      * Transformation JSONs per scene

Inputs:
  --json PATH
      Path to a reference transformation JSON file:
      <camera>_transform_<YYYYMMDD>_<HHMMSS>.json

Requirements:
  - FRAMES_DIR/_frame_paths.parquet must exist
  - GCP real-world coordinates must exist
  - Reference GCP image coordinates must exist

Outputs:
  - Orthorectified images:
      rect_dir/<camera>/orthorectification_imgs_auto/
      rect_dir/<camera>/ortho_imgs_auto/
  - Transformation JSONs:
      rect_dir/<camera>/<camera>_transform_<date>_<time>.json

Behavior:
  - Quiet by default (suppresses river.config logs)
  - Use --verbose to enable detailed logging

Example:
  python orthorectify_auto.py --json path/to/cam_transform_20250707_120000.json
  python orthorectify_auto.py --json ... --verbose
"""

from __future__ import annotations

import sys as _sys
import os as _os
import logging as _logging

_verbose = ("--verbose" in _sys.argv) or (_os.environ.get("ORTHO_AUTO_VERBOSE") == "1")
_logging.basicConfig(level=_logging.INFO if _verbose else _logging.WARNING)
_logging.getLogger("river.config").setLevel(_logging.INFO if _verbose else _logging.WARNING)


import argparse
import json
import re
import shutil
from pathlib import Path

import rasterio
from rasterio.transform import from_bounds

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


# ---------------------------
# GEOTIFF export helpers
# ---------------------------


def save_geotiff(
    img: np.ndarray,
    extent,
    output_path: Path,
    epsg: int = 32623
):
    """
    Save orthorectified raster as GeoTIFF.

    Parameters
    ----------
    img : ndarray
        Orthorectified image.

    extent : tuple/list
        [xmin, xmax, ymin, ymax]
        Real-world coordinates in EPSG:32623.

    output_path : Path
        Output GeoTIFF path.

    epsg : int
        CRS EPSG code.
    """

    xmin, xmax, ymin, ymax = extent

    height, width = img.shape[:2]

    transform = from_bounds(
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if img.ndim == 2:

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=img.dtype,
            crs=f"EPSG:{epsg}",
            transform=transform,
            compress="lzw"
        ) as dst:

            dst.write(img, 1)

    else:

        bands = img.shape[2]

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=img.dtype,
            crs=f"EPSG:{epsg}",
            transform=transform,
            compress="lzw"
        ) as dst:

            for i in range(bands):
                dst.write(img[:, :, i], i + 1)


# ---------------------------
# Image extent helpers
# ---------------------------

def crop_image_to_extent(img, source_extent, target_extent):

    sxmin, sxmax, symin, symax = source_extent
    txmin, txmax, tymin, tymax = target_extent

    h, w = img.shape[:2]

    col0 = int((txmin - sxmin) / (sxmax - sxmin) * w)
    col1 = int((txmax - sxmin) / (sxmax - sxmin) * w)

    row1 = int((symax - tymin) / (symax - symin) * h)
    row0 = int((symax - tymax) / (symax - symin) * h)

    col0 = max(0, col0)
    col1 = min(w, col1)

    row0 = max(0, row0)
    row1 = min(h, row1)

    cropped_img = img[row0:row1, col0:col1]

    # compute TRUE extent from actual pixel indices

    new_xmin = sxmin + (col0 / w) * (sxmax - sxmin)
    new_xmax = sxmin + (col1 / w) * (sxmax - sxmin)

    new_ymax = symax - (row0 / h) * (symax - symin)
    new_ymin = symax - (row1 / h) * (symax - symin)

    cropped_extent = (
        new_xmin,
        new_xmax,
        new_ymin,
        new_ymax
    )

    return cropped_img, cropped_extent


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--abs-coords",action="store_true",help="Use EPSG:32623 coordinates and export GeoTIFFs")
    parser.add_argument("--buffer",type=float,default=20.0,help="Buffer around GCP extent (m)")
    parser.add_argument("--dyn",action="store_true",help="Process daily transformations dynamically.")
    parser.add_argument("--time",type=str,help="Only process scenes with this HHMMSS timestamp")
    parser.add_argument("--overwrite",action="store_true",help="Overwrite existing outputs")


    args = parser.parse_args()

    json_path = Path(args.json)
    cam, ref_date, ref_time = parse_json_filename(json_path)
    abs_mod = args.abs_coords
    buffer_m = args.buffer

    print(f"[INFO] Camera: {cam}")

    
    frames_root = frames_dir
    df = load_frames_index(frames_root)

    df = df[df["camera"] == cam]

    df_unique = df.drop_duplicates(
        subset=["camera", "date_yyyymmdd", "time_hhmmss"]
    ).sort_values(["date_yyyymmdd", "time_hhmmss"])

    # Optional time filter
    if args.time is not None:
    
        df_unique = df_unique[
            df_unique["time_hhmmss"]
            .astype(str)
            .str.zfill(6)
            == args.time
        ]
    
        print(
            f"[INFO] Filtering to time {args.time}: "
            f"{len(df_unique)} scenes"
        )

    print(f"[INFO] Processing {len(df_unique)} scenes")

    gcp_files = sorted((gcps_dir / cam).glob(f"{cam}_gcps_img_*.csv"))
    available_gcps = {}
    for f in gcp_files:
    
        stem = f.stem
        parts = stem.split("_")
    
        date = parts[-2]
        time_ = parts[-1]
    
        available_gcps[date] = time_

    # Extents handling
    if abs_mod:
        gcps = load_gcps_real(cam)
    
        xs = [coord[0] for coord in gcps.values()]
        ys = [coord[1] for coord in gcps.values()]
    
        display_extent = (
            min(xs) - buffer_m,
            max(xs) + buffer_m,
            min(ys) - buffer_m,
            max(ys) + buffer_m
        )
    else:
        xlim = (-20, 50)
        ylim = (-10, 20)
        display_extent = (
            xlim[0],
            xlim[1],
            ylim[0],
            ylim[1]
        )
    clip_extent = display_extent

    # ✅ output dirs
    full_dir = rect_dir / cam / "auto_orthorectification_imgs"
    clean_dir = rect_dir / cam / "auto_ortho_imgs"
    tif_dir = rect_dir / cam / "auto_ortho_tifs"
    full_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    tif_dir.mkdir(parents=True, exist_ok=True)


    # ✅ reference GCPs
    gcp_ref = gcps_dir / cam / f"{cam}_gcps_img_{ref_date}_{ref_time}.csv"  
    processed = []
    skipped = []

    for _, row in df_unique.iterrows():

        date = row["date_yyyymmdd"]
        time_ = row["time_hhmmss"]
    
        # ----------------------------------------------------------
        # Skip already processed scenes unless --overwrite is used
        # ----------------------------------------------------------
        full_png = (
            full_dir
            / f"{cam}_orthorect_{date}_{time_}.png"
        )
    
        clean_png = (
            clean_dir
            / f"{cam}_orthoimg_{date}_{time_}.png"
        )
    
        transf_file = (
            rect_dir
            / cam
            / f"{cam}_transform_{date}_{time_}.json"
        )
    
        outputs_exist = (
            full_png.exists()
            and clean_png.exists()
            and transf_file.exists()
        )
        
        if abs_mod:
            ortho_tif = (
                tif_dir
                / f"{cam}_orthotif_{date}_{time_}.tif"
            )
        
            outputs_exist = (
                outputs_exist
                and ortho_tif.exists()
            )
    
        if outputs_exist and not args.overwrite:
    
            skipped.append(
                (
                    date,
                    time_,
                    "already_processed"
                )
            )
    
            print(
                f"[SKIP] {date} {time_} "
                "(already processed)"
            )
    
            continue
    
        try:
    
            # ---------- GCP reuse ----------
            gcp_target = (
                gcps_dir
                / cam
                / f"{cam}_gcps_img_{date}_{time_}.csv"
            )
    
            gcp_target.parent.mkdir(
                parents=True,
                exist_ok=True
            )
    
            # ---------- GCP handling ----------
            if args.dyn:
    
                # Skip dates with no available GCP file
                if date not in available_gcps:
    
                    skipped.append(
                        (
                            date,
                            time_,
                            "no_gcps_for_date"
                        )
                    )
    
                    print(
                        f"[SKIP] {date} {time_} "
                        f"(no GCP file available for this date)"
                    )
    
                    continue
    
                # Use the reference GCP time for that date
                reference_time = available_gcps[date]
    
                source_gcp = (
                    gcps_dir
                    / cam
                    / f"{cam}_gcps_img_{date}_{reference_time}.csv"
                )
    
                # Create timestamp-specific GCP file
                if (
                    (not gcp_target.exists() or args.overwrite)
                    and source_gcp != gcp_target
                ):
                    shutil.copy(
                        source_gcp,
                        gcp_target
                    )
    
            else:
    
                # Use single reference GCP file
                if (
                    (not gcp_target.exists() or args.overwrite)
                    and source_gcp != gcp_target
                ):
                    shutil.copy(
                        source_gcp,
                        gcp_target
                    )

            # ---------- TRANSFORM ----------
            transformation = transform(df, cam, date, time_, absolute_coords=abs_mod)

            # ---------- SAVE JSON ----------
            with transf_file.open("w") as f:
                json.dump(transformation["transformation_matrix"], f, indent=1)

            # ---------- LOAD FRAME ----------
            frame_path = Path(row["frame_path"])
            img = mpimg.imread(frame_path)

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
            ax1.text(x1, y1, "1", color='#ED6B57', fontsize=8)
            
            for i, (x, y) in enumerate(
                [(x2, y2), (x3, y3), (x4, y4)],
                start=2
            ):
                ax1.plot(
                    x,
                    y,
                    'o',
                    color='#6CD4FF',
                    markersize=3
                )
            
                ax1.text(
                    x,
                    y,
                    str(i),
                    color='#6CD4FF',
                    fontsize=8
                )

            # ORTHO
            ax2.imshow(
                transformation["transformed_img"],
                extent=transformation["extent"]
            )


            if abs_mod:
                ax2.set_xlim(
                    display_extent[0],
                    display_extent[1]
                )
                
                ax2.set_ylim(
                    display_extent[2],
                    display_extent[3]
                )

            else:
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

            if abs_mod:
                ax.set_xlim(
                    display_extent[0],
                    display_extent[1]
                )
                
                ax.set_ylim(
                    display_extent[2],
                    display_extent[3]
                )

            else:
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

            if abs_mod:
                ortho_tif = (
                    tif_dir
                    / f"{cam}_orthotif_{date}_{time_}.tif"
                )
            
                cropped_img, cropped_extent = crop_image_to_extent(
                    transformation["transformed_img"],
                    transformation["extent"],
                    clip_extent
                )
            
                save_geotiff(
                    cropped_img,
                    cropped_extent,
                    ortho_tif,
                    epsg=32623
                )
            
                print(
                    f"[Saved] GeoTIFF: {ortho_tif}"
                )

            processed.append(
                (
                    date,
                    time_
                )
            )
            print(f"[OK] {date} {time_}")

        except Exception as e:
            skipped.append(
                (
                    date,
                    time_,
                    str(e)
                )
            )
            print(f"[FAIL] {date} {time_}: {e}")

    log_file = rect_dir / cam / "auto_log.txt"
    with log_file.open("w") as f:
    
        f.write("AUTO ORTHORECTIFICATION LOG\n")
        f.write("=" * 70 + "\n\n")
    
        f.write(f"Camera: {cam}\n")
        f.write(f"Total scenes found: {len(df_unique)}\n")
        f.write(f"Available img_gcps files: {len(available_gcps)}\n")
        f.write(f"Processed scenes: {len(processed)}\n")
        f.write(f"Skipped/failed scenes: {len(skipped)}\n\n")
    
        f.write("AVAILABLE IMG_GCPS FILES\n")
        f.write("-" * 70 + "\n")
    
        for d, t in sorted(available_gcps.items()):
            f.write(
                f"{d} -> reference GCP time {t}\n"
            )
    
        f.write("\n\nPROCESSED SCENES\n")
        f.write("-" * 70 + "\n")
    
        for d, t in processed:
            f.write(f"{d} {t}\n")
    
        f.write("\n\nSKIPPED / FAILED SCENES\n")
        f.write("-" * 70 + "\n")
    
        for d, t, reason in skipped:
            f.write(f"{d} {t} : {reason}\n")
    
    print(f"[INFO] Log written: {log_file}")

if __name__ == "__main__":
    main()