#!/usr/bin/env python3
"""
extract_meta.py — scan VIDEO_DIR (from environment) and create per-camera metadata CSVs.

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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2  # pip install opencv-python


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

        session_dir = path.parent           # 20250723-000000-235900
        camera_dir = path.parent.parent     # ilhh
        
        m = STEM_RE.match(path.stem)        # ✅ match FILENAME
        if not m:
            continue
        
        camera = m.group("camera")
        date = m.group("date")
        start = m.group("start")
        end = m.group("end")

        items.append((path, camera, date, start, end))
        cameras.add(camera)

        if camera != camera_dir.name:
            continue  # or log a warning

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
            # Populate error row with safe fallbacks
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

    # Write meta CSVs
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

    # Write error CSVs
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
    p.add_argument(
        "--camera",
        help="Only process a single camera (name must match the filename prefix before the first '_').",
    )
    p.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_EXTS),
        help="File extensions to include (case-insensitive). Example: --ext .mp4 .mkv",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing CSVs (default).",
    )
    mode.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSVs instead of overwriting.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    video_dir_env = os.environ.get("VIDEO_DIR")
    if not video_dir_env:
        print(
            "ERROR: VIDEO_DIR is not set. Did you run 'source setup.sh' in this shell?",
            file=sys.stderr,
        )
        return 2

    videos_root = Path(video_dir_env).expanduser().resolve()
    if not videos_root.exists() or not videos_root.is_dir():
        print(f"ERROR: VIDEO_DIR does not exist or is not a directory: {videos_root}", file=sys.stderr)
        return 2

    overwrite = not args.append
    exts = args.ext

    # Pre-scan for validation and camera discovery
    items, total_video_files, cameras = discover_videos(videos_root, exts)

    if total_video_files == 0:
        print(
            f"ERROR: No video files with extensions {exts} found under VIDEO_DIR: {videos_root}\n"
            f"       Make sure your videos are in that directory.",
            file=sys.stderr,
        )
        return 3

    # If a camera filter is requested, verify it exists among matched stems
    if args.camera:
        if args.camera not in cameras:
            # If there are video files but none match the expected pattern, mention that
            pattern_hint = ""
            if len(items) == 0 and total_video_files > 0:
                pattern_hint = (
                    "\nNote: Found video files, but none match the expected naming pattern:\n"
                    "      <camera>_<YYYYMMDD>-<HHMMSS>-<HHMMSS>.<ext>\n"
                )

            cam_list = ", ".join(sorted(cameras)) if cameras else "(none discovered)"
            print(
                f"ERROR: Camera '{args.camera}' was not found in VIDEO_DIR.\n"
                f"       Available cameras from matched filenames: {cam_list}{pattern_hint}",
                file=sys.stderr,
            )
            return 4

        # Filter items to only that camera
        items = [t for t in items if t[1] == args.camera]

        if not items:
            print(
                f"ERROR: No videos matched for camera '{args.camera}' after filtering.",
                file=sys.stderr,
            )
            return 5

    # If there are matched items but zero (i.e., all videos are off-pattern), surface that
    if not items:
        print(
            "ERROR: No files matched the expected naming pattern:\n"
            "       <camera>_<YYYYMMDD>-<HHMMSS>-<HHMMSS>.<ext>\n"
            f"       under VIDEO_DIR: {videos_root}",
            file=sys.stderr,
        )
        return 6

    # Do the processing and write CSVs
    camera_csvs, err_csvs = process_items_and_write_csvs(
        videos_root=videos_root,
        items=items,
        overwrite=overwrite,
    )

    print("Following metadata files were created/updated:")
    print(format_dict_as_lines(camera_csvs))
    print("\nFollowing error log files were created/updated:")
    print(format_dict_as_lines(err_csvs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))