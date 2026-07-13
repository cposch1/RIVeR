#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from river.config import *


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_centerline(cam):

    p = gcps_dir / cam / f"{cam}_centerline.csv"

    df = pd.read_csv(
        p,
        header=None
    )

    return (
        float(df.iloc[0, 0]),
        float(df.iloc[0, 1]),
        float(df.iloc[1, 0]),
        float(df.iloc[1, 1]),
    )


def find_reference_frame(row):

    return Path(row["frame_path"]).parent / "0000000000.jpg"


def find_video(camera, date, time_):

    pattern = f"{camera}_{date}-{time_}-*.mp4"

    matches = list(
        video_dir.glob(
            f"{camera}/**/{pattern}"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No video found for {camera} {date} {time_}"
        )

    return matches[0]


def load_video_meta(camera):

    meta_file = video_dir / camera / f"_{camera}_meta.csv"

    return pd.read_csv(meta_file)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def create_centerline_video(
    camera,
    date,
    time_
):

    print(
        f"\n[INFO] {camera} {date} {time_}"
    )

    # --------------------------------------------------------
    # Load frames dataframe
    # --------------------------------------------------------

    df_frames = pd.read_parquet(
        frames_dir / "_frame_paths.parquet"
    )

    row = df_frames[
        (df_frames.camera == camera)
        & (df_frames.date_yyyymmdd.astype(str) == str(date))
        & (df_frames.time_hhmmss.astype(str) == str(time_))
    ].iloc[0]

    frame_path = find_reference_frame(row)

    video_path = find_video(
        camera,
        date,
        time_
    )

    print("Frame :", frame_path)
    print("Video :", video_path)

    # --------------------------------------------------------
    # Video metadata
    # --------------------------------------------------------

    meta = load_video_meta(camera)

    meta_row = meta[
        (meta.date_yyyymmdd.astype(str) == str(date))
        & (meta.time_hhmmss.astype(str) == str(time_))
    ].iloc[0]

    fps = int(meta_row["fps"])

    res = str(meta_row["resolution"])

    width = int(res.split("x")[0])
    height = int(res.split("x")[1])

    print(
        f"[INFO] fps={fps}, "
        f"resolution={width}x{height}"
    )

    # --------------------------------------------------------
    # Centerline
    # --------------------------------------------------------

    (
        x1_rw,
        y1_rw,
        x2_rw,
        y2_rw
    ) = load_centerline(camera)

    print("\nCENTERLINE FROM FILE")
    print(x1_rw, y1_rw)
    print(x2_rw, y2_rw)

    print(
        gcps_dir / camera / f"{camera}_centerline.csv"
    )
    
    trans = transform(
        df_frames,
        camera,
        date,
        time_,
        absolute_coords=True
    )

    T = np.array(
        trans["transformation_matrix"]
    )

    p1 = transform_real_world_to_pixel(
        x1_rw,
        y1_rw,
        T
    )

    p2 = transform_real_world_to_pixel(
        x2_rw,
        y2_rw,
        T
    )
    print(p1)
    print(p2)

    # --------------------------------------------------------
    # Extend centerline
    # --------------------------------------------------------

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    length = np.hypot(dx, dy)

    p1_ext = p1
    p2_ext = p2

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix="river_centerline_"
        )
    )

    intro_png = tmp_dir / "centerline.png"

    # --------------------------------------------------------
    # Create first frame image
    # --------------------------------------------------------

    frame = mpimg.imread(frame_path)

    dpi = 100

    fig, ax = plt.subplots(
        figsize=(width / dpi, height / dpi),
        dpi=dpi
    )

    ax.imshow(frame)

    ax.plot(
        [p1_ext[0], p2_ext[0]],
        [p1_ext[1], p2_ext[1]],
        "--",
        color="grey",
        alpha=0.5,
        linewidth=3
    )

    ax.axis("off")

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1
    )

    fig.savefig(
        intro_png,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Single frame video
    # --------------------------------------------------------

    intro_mp4 = tmp_dir / "intro.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", str(intro_png),
            "-frames:v", "1",
            "-r", str(fps),
            "-vf",
            (
                f"scale={width}:{height},"
                "pad=ceil(iw/2)*2:ceil(ih/2)*2"
            ),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(intro_mp4)
        ],
        check=True
    )

    # --------------------------------------------------------
    # Concat list
    # --------------------------------------------------------

    concat_txt = tmp_dir / "concat.txt"

    with concat_txt.open("w") as f:

        f.write(
            f"file '{intro_mp4.as_posix()}'\n"
        )

        f.write(
            f"file '{video_path.as_posix()}'\n"
        )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    out_root = video_dir / "_center_add"

    rel_video = video_path.relative_to(
        video_dir
    )

    out_video = (
        out_root /
        rel_video.parent /
        f"{video_path.stem}_centerline.mp4"
    )

    out_video.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "[INFO] Writing:",
        out_video
    )

    # --------------------------------------------------------
    # Final video
    # --------------------------------------------------------

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "16",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_video)
        ],
        check=True
    )

    shutil.rmtree(
        tmp_dir,
        ignore_errors=True
    )

    print(
        f"[DONE] {out_video}"
    )

    print("CENTERLINE WORLD")
    print(x1_rw, y1_rw)
    print(x2_rw, y2_rw)
    
    print("CENTERLINE PIXEL")
    print(p1)
    print(p2)


# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":

    create_centerline_video(
        camera="ilh-cam1-pt",
        date="20250804",
        time_="140000"
    )