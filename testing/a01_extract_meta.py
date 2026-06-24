#!/usr/bin/env python3
"""
a01_extract_meta.py — scan VIDEO_DIR (from environment) and create per-camera metadata CSVs.

Environment:
  - Requires VIDEO_DIR to be exported (e.g., via:  source setup.sh)
  - The script will scan VIDEO_DIR recursively for video files.

Filename convention expected:
  <camera>_<YYYYMMDD>-<HHMMSS>-<HHMMSS>.<ext>
Example:
  CAM1_20250117-093012-103015.mp4

Outputs (per camera, written under VIDEO_DIR/<camera>/):
  - _<camera>_meta.csv       — per-video metadata
  - _<camera>_error_log.csv  — files that failed to probe, with error messages

Dependencies:
  - Python 3.8+
  - OpenCV for Python (cv2):  pip install opencv-python
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logging.getLogger("river.config").setLevel(logging.ERROR)
from river.config import *

import cv2


# ---------- Config ----------

DEFAULT_EXTS = (".mp4", ".avi", ".mkv", ".mov")

# Regex for "<camera>_<YYYYMMDD>-<HHMMSS>-<HHMMSS>"
STEM_RE = re.compile(
    r'^(?P<camera>[^_]+)_(?P<date>\d{8})-(?P<start>\d{6})-(?P<end>\d{6})$',
    re.IGNORECASE
)


# ---------- Helpers ----------

def _fmt_duration_hhmmss(total_seconds: Optional[float]) -> str:
    """Format seconds as hhmmss (always include hours, zero-padded). Empty string if None."""
    if total_seconds is None:
        return ""
    secs = int(round(total_seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}{m:02d}{s:02d}"


def _safe_duration_via_ratio(video_path: Path) -> Optional[float]:
    """
    Fallback duration: try seeking to end and reading timestamp in msec.
    Returns seconds or None (backend dependent).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            return pos_msec / 1000.0
        return None
    finally:
        cap.release()


def check_video_info(video_path: Path) -> dict:
    """
    Extract: duration, total_frames, fps, resolution, size_gb, bitrate_mbps
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    finally:
        cap.release()

    size_bytes = video_path.stat().st_size
    size_gb = size_bytes / (1024 ** 3)

    duration_s = None
    if total_frames > 0 and fps and fps > 0:
        duration_s = total_frames / fps

    if not duration_s:
        fallback_s = _safe_duration_via_ratio(video_path)
        if fallback_s and fallback_s > 0:
            duration_s = fallback_s

    bitrate_mbps = (size_bytes * 8 / duration_s / 1e6) if duration_s else None

    return {
        "duration_hhmmss": _fmt_duration_hhmmss(duration_s),
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "size_gb": round(size_gb, 2),
        "bitrate_mbps": round(bitrate_mbps, 3) if bitrate_mbps else None,
    }


# ---------- Discovery ----------

class DiscoveredItem(Tuple[Path, str, str, str, str]):
    """(path, camera, date, start, end)"""
    pass


def discover_videos(
    root: Path,
    suffixes: Iterable[str]
) -> Tuple[List[DiscoveredItem], int, set[str]]:
    """
    Walk 'root' and collect items that match suffix and naming convention.

    Returns:
      - items: list of (path, camera, date, start, end) for matched stems
      - total_video_files: number of files with allowed suffixes (matched + unmatched stems)
      - cameras: set of camera names discovered from matched stems
    """
    suffixes_set = {s.lower() for s in suffixes}
    items: List[DiscoveredItem] = []
    total_video_files = 0
    cameras: set[str] = set()

    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.stem.startswith("._"):  # ignore macOS dot-underscore artifacts
            continue
        if path.suffix.lower() not in suffixes_set:
            continue

        total_video_files += 1

        session_dir = path.parent
        camera_dir = path.parent.parent
        
        m = STEM_RE.match(path.stem)
        if not m:
            continue
        
        camera = m.group("camera")
        date = m.group("date")
        start = m.group("start")
        end = m.group("end")

        items.append((path, camera, date, start, end))
        cameras.add(camera)

        if camera != camera_dir.name:
            continue

    return items, total_video_files, cameras


# ---------- Processing + CSV writing ----------

def process_items_and_write_csvs(
    videos_root: Path,
    items: List[DiscoveredItem],
    overwrite: bool = True,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Process the discovered items list and write per-camera CSVs.
    """
    rows_per_camera: Dict[str, list[dict]] = {}
    errors_per_camera: Dict[str, list[dict]] = {}

    for path, camera, date, start_time, _end in items:
        try:
            info = check_video_info(path)
            row = {
                "date_yyyymmdd": date,
                "time_hhmmss": start_time,
                "duration_hhmmss": info["duration_hhmmss"],
                "total_frames": info["total_frames"],
                "fps": info["fps"],
                "resolution": info["resolution"],
                "bitrate_mbps": info["bitrate_mbps"],
                "size_gb": info["size_gb"],
                "path": str(path.resolve()),
            }
            rows_per_camera.setdefault(camera, []).append(row)
        except Exception as e:
            try:
                fallback_size_gb = round(path.stat().st_size / (1024 ** 3), 2)
            except Exception:
                fallback_size_gb = None

            errors_per_camera.setdefault(camera, []).append({
                "date_yyyymmdd": date,
                "time_hhmmss": start_time,
                "duration_hhmmss": "",
                "total_frames": "",
                "fps": "",
                "resolution": "",
                "bitrate_mbps": "",
                "size_gb": fallback_size_gb,
                "path": str(path.resolve()),
                "error_message": str(e),
            })

    camera_to_csv: Dict[str, Path] = {}
    for camera, rows in rows_per_camera.items():
        rows.sort(key=lambda r: (r["date_yyyymmdd"], r["time_hhmmss"]))
        csv_path = videos_root / camera / f"_{camera}_meta.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        camera_to_csv[camera] = csv_path

        mode = "w" if overwrite or not csv_path.exists() else "a"
        write_header = (mode == "w")

        with csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "date_yyyymmdd",
                    "time_hhmmss",
                    "duration_hhmmss",
                    "total_frames",
                    "fps",
                    "resolution",
                    "bitrate_mbps",
                    "size_gb",
                    "path",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    err_to_csv: Dict[str, Path] = {}
    for camera, rows in errors_per_camera.items():
        rows.sort(key=lambda r: (r["date_yyyymmdd"], r["time_hhmmss"]))
        err_csv_path = videos_root / camera / f"_{camera}_error_log.csv"
        err_csv_path.parent.mkdir(parents=True, exist_ok=True)
        err_to_csv[camera] = err_csv_path

        mode = "w" if overwrite or not err_csv_path.exists() else "a"
        write_header = (mode == "w")

        with err_csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "date_yyyymmdd",
                    "time_hhmmss",
                    "duration_hhmmss",
                    "total_frames",
                    "fps",
                    "resolution",
                    "bitrate_mbps",
                    "size_gb",
                    "path",
                    "error_message"
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    return camera_to_csv, err_to_csv


def format_dict_as_lines(d: Dict[str, Path]) -> str:
    if not d:
        return "  (none)"
    return "\n".join(f"  {cam} -> {path}" for cam, path in sorted(d.items()))


# ---------- CLI ----------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="extract_meta",
        description="Generate per-camera metadata/error CSVs from VIDEO_DIR (exported in environment).",
    )
    p.add_argument("--cam")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", default=True)
    mode.add_argument("--append", action="store_true")
    p.add_argument("--verbose", action="store_true")  # ✅ ADDED
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    # Suppress config logging unless verbose
    if not args.verbose:
        logging.getLogger("river.config").setLevel(logging.ERROR)

    video_dir_env = video_dir
    videos_root = Path(video_dir_env)

    overwrite = not args.append
    items, total_video_files, cameras = discover_videos(videos_root, args.ext)

    # Camera filter
    if args.cam:
        if args.cam not in cameras:
            cam_list = ", ".join(sorted(cameras)) if cameras else "(none discovered)"
            print(
                f"ERROR: Camera '{args.cam}' was not found in VIDEO_DIR.\n"
                f"       Available cameras from matched filenames: {cam_list}",
                file=sys.stderr,
            )
            return 4
    
        # filter items
        items = [t for t in items if t[1] == args.cam]
    
        if not items:
            print(
                f"ERROR: No videos matched for camera '{args.cam}' after filtering.",
                file=sys.stderr,
            )
            return 5

    # Overwrite function
    if overwrite:
        existing_outputs = []
        cams_to_check = [args.cam] if args.cam else cameras
        for cam in cams_to_check:
            meta_path = videos_root / cam / f"_{cam}_meta.csv"
            err_path = videos_root / cam / f"_{cam}_error_log.csv"

            if meta_path.exists():
                existing_outputs.append(meta_path)
            if err_path.exists():
                existing_outputs.append(err_path)

        if existing_outputs:
            print("The following files already exist and would be overwritten:")
            for p in existing_outputs:
                print(f"  {p}")

            choice = input("Overwrite? (y/n): ").strip().lower()
            if choice != "y":
                print("Aborted: no files were written.")
                return 0

    camera_csvs, err_csvs = process_items_and_write_csvs(
        videos_root,
        items,
        overwrite,
    )

    print("Following metadata files were created/updated:")
    print(format_dict_as_lines(camera_csvs))
    print("\nFollowing error log files were created/updated:")
    print(format_dict_as_lines(err_csvs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))