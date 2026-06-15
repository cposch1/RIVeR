#!/usr/bin/env python3
"""
extract_frames.py — Extract frames from videos under VIDEO_DIR into FRAMES_DIR.

Environment (from your setup.sh):
  - VIDEO_DIR: root folder that contains the videos
  - FRAMES_DIR: destination root for extracted frames

Filename convention (required to parse camera/date/time):
  <camera>_<YYYYMMDD>-<HHMMSS>-<HHMMSS>.<ext>
Example:
  ilh-cam1-pt_20250426-120000-121500.mp4

Outputs:
  FRAMES_DIR/<camera>/<YYYYMMDD>/<HHMMSS>/frame_000000.jpg  (created by your RIVeR functions)
  FRAMES_DIR/_frame_paths.parquet and FRAMES_DIR/_frame_paths.csv (index of all frames)

Arguments (defaults as requested):
  --every 20                 Extract every 20th frame (default: 20)
  --start-frame-number 0     Start at frame index 0 (default: 0)
  --end-frame-number None    Process until end of video (default: None)
  --overwrite                Delete the target frames folder before extraction (default: OFF)
  --no-overwrite             Keep existing frames folder (default behavior)
  --camera CAM               Only process this camera (prefix before first '_')
  --start-date YYYYMMDD      Filter start date (inclusive; default: none)
  --end-date YYYYMMDD        Filter end date (inclusive; default: none)
  --start-time HHMMSS        Filter start time (inclusive; default: none)
  --end-time HHMMSS          Filter end time (inclusive; default: none)
  --ext .avi .mp4 ...        Video extensions to include (default: .avi .mp4)
  --verbose                  Show INFO logs from river.config (default: off)

Notes:
  - Uses metadata files (_<camera>_meta.csv) discovered under VIDEO_DIR to whitelist
    which (camera, date, start) entries are allowed. If none are found, it will WARN.
  - The frame extraction itself is performed by `video_to_frames` imported from `river.config`
    with the SAME signature you used (we do not change the function’s logic).
"""

from __future__ import annotations

# ---- Quiet-by-default logging setup (must be before importing river.config) ----
import os as _os
import sys as _sys
import logging as _logging

# Enable INFO logs only when --verbose is present on the command line
# or EXTRACT_FRAMES_VERBOSE=1 is set in the environment.
_verbose = ("--verbose" in _sys.argv) or (_os.environ.get("EXTRACT_FRAMES_VERBOSE") == "1")
_logging.basicConfig(level=_logging.INFO if _verbose else _logging.WARNING)
_logging.getLogger("river.config").setLevel(_logging.INFO if _verbose else _logging.WARNING)
# -------------------------------------------------------------------------------

# ✅ Import your processing logic exactly as in the notebook
from river.config import *  # noqa: F401,F403  (video_to_frames, etc.)

import argparse
import csv
import re
import sys
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # fallback if tqdm not installed
        return x


# ---------- Naming / discovery ----------

STEM_RE = re.compile(
    r'^(?P<camera>.+)_(?P<date>\d{8})-(?P<start>\d{6})-(?P<end>\d{6})\.(?P<ext>avi|mp4)$',
    re.IGNORECASE
)

DEFAULT_INPUT_EXTS = [".avi", ".mp4"]


def parse_video_name(video_path: Path) -> Tuple[str, str, str, str]:
    """Returns (camera, date, clock_start, clock_end) if filename matches the convention."""
    m = STEM_RE.match(video_path.name)
    if not m:
        raise ValueError(f"Video filename does not match expected pattern: {video_path.name}")
    return (
        m.group("camera"),
        m.group("date"),
        m.group("start"),
        m.group("end"),
    )


def iter_videos(root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    """Yield video files under root with allowed suffixes."""
    root = Path(root)
    suffixes_lower = {s.lower() for s in suffixes}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(("._", ":_")):  # macOS/edge junk
            continue
        if p.suffix.lower() in suffixes_lower:
            yield p


def _normalize_clock(val: Optional[str]) -> str:
    """Normalize to HHMMSS format; return '' if invalid/missing."""
    if val is None:
        return ""
    v = val.strip().replace(":", "").replace("-", "").replace(" ", "")
    return v if len(v) == 6 and v.isdigit() else ""


def _norm(val: Optional[str]) -> Optional[str]:
    """Normalize user filter input by removing separators and whitespace; None if blank."""
    if val is None:
        return None
    v = val.strip().replace(":", "").replace("-", "").replace(" ", "")
    return v if v else None


def target_frames_dir_for(video_path: Path, frames_root: Path) -> Path:
    """FRAMES_DIR/<camera>/<date>/<clock_start>/"""
    camera, date, clock_start, _ = parse_video_name(video_path)
    return frames_root / camera / date / clock_start


def _fix_windows_path(p: str) -> str:
    """Convert Git Bash paths (/c/...) to Windows paths (C:/...)."""
    if p and len(p) > 2 and p[0] == "/" and p[2] == "/":
        drive = p[1].upper()
        return f"{drive}:/{p[3:]}"
    return p


# ---------- Metadata whitelist ----------

def load_allowed_from_meta(video_root: Path) -> Set[Tuple[str, str, str]]:
    """
    Reads meta CSV files (_{camera}_meta.csv) under video_root (recursively).
    Returns set of (camera, date_yyyymmdd, clock_start_hhmmss).
    """
    allowed: Set[Tuple[str, str, str]] = set()
    video_root = Path(video_root)

    for meta_csv in video_root.rglob("_*_meta.csv"):
        name = meta_csv.name
        if not (name.startswith("_") and name.endswith("_meta.csv")):
            continue

        camera = name[1:-9]  # strip leading "_" and trailing "_meta.csv"

        with meta_csv.open("r", newline="", encoding="utf-8-sig") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)
            if reader.fieldnames is None:
                continue

            key_map = { (h or "").strip().lower(): h for h in reader.fieldnames }

            def get(row, *keys):
                for k in keys:
                    src = key_map.get(k)
                    if src and (val := row.get(src)) is not None:
                        return val
                return None

            for row in reader:
                date = (get(row, "date", "date_yyyymmdd") or "").strip()
                clock_raw = (get(row, "time_hhmmss", "clock_start") or "").strip()
                clock = _normalize_clock(clock_raw)
                if not date or not clock:
                    continue
                allowed.add((camera, date, clock))

    return allowed


# ---------- Filters ----------

def validate_filters(
    allowed: Set[Tuple[str, str, str]],
    camera_filter: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
) -> None:
    """
    Validate filters against the metadata (if present). If no metadata,
    validation will be relaxed and camera existence is checked later
    against filenames.
    """
    cams  = {c for (c, _, _) in allowed}
    dates = {d for (_, d, _) in allowed}

    if allowed:
        if camera_filter and camera_filter not in cams:
            raise ValueError(f"Camera '{camera_filter}' not found in metadata. Existing cameras: {sorted(cams)}")

        if start_date and start_date not in dates:
            raise ValueError(f"start_date '{start_date}' not found in metadata")
        if end_date and end_date not in dates:
            raise ValueError(f"end_date '{end_date}' not found in metadata")

    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date > end_date")

    if start_time and len(start_time) != 6:
        raise ValueError("start_time must be HHMMSS")
    if end_time and len(end_time) != 6:
        raise ValueError("end_time must be HHMMSS")
    if start_time and end_time and start_time > end_time:
        raise ValueError("start_time > end_time")


def passes_filters(
    camera: str, date: str, clock_start: str,
    camera_filter: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
) -> bool:
    if camera_filter and camera != camera_filter:
        return False
    if start_date and date < start_date:
        return False
    if end_date and date > end_date:
        return False
    if start_time and clock_start < start_time:
        return False
    if end_time and clock_start > end_time:
        return False
    return True


# ---------- CLI ----------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="extract_frames",
        description="Extract frames from videos under VIDEO_DIR into FRAMES_DIR, "
                    "optionally filtering by camera/date/time.",
    )
    # Filters (default: none)
    p.add_argument("--camera", help="Only process this camera (prefix before first '_').")
    p.add_argument("--start-date", default=None, help="YYYYMMDD (default: none)")
    p.add_argument("--end-date", default=None, help="YYYYMMDD (default: none)")
    p.add_argument("--start-time", default=None, help="HHMMSS (default: none)")
    p.add_argument("--end-time", default=None, help="HHMMSS (default: none)")

    # Frame extraction defaults
    p.add_argument("--every", type=int, default=20, help="Take every Nth frame (default: 20).")
    p.add_argument("--start-frame-number", type=int, default=0,
                   help="Begin at this frame index (default: 0).")
    p.add_argument("--end-frame-number", type=int, default=None,
                   help="Stop at this frame index (inclusive, default: none).")

    # Overwrite behavior: default is NO overwrite
    ow = p.add_mutually_exclusive_group()
    ow.add_argument(
        "--overwrite",
        action="store_true",
        dest="overwrite",
        help="Delete frame folder before extraction (default: False)."
    )
    ow.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing frames (default behavior)."
    )
    p.set_defaults(overwrite=False)

    # Extensions
    p.add_argument("--ext", nargs="+", default=DEFAULT_INPUT_EXTS,
                   help="Video file extensions to include (default: .avi .mp4).")

    # Verbose (already parsed pre-import to affect import-time logs)
    p.add_argument("--verbose", action="store_true",
                   help="Show INFO logs from river.config for this run.")

    return p.parse_args(argv)


# ---------- Main ----------

def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Resolve env (setup.sh should have set these)
    video_dir_env = os.environ.get("VIDEO_DIR")
    frames_dir_env = os.environ.get("FRAMES_DIR")
    
    video_dir_env = _fix_windows_path(video_dir_env)
    frames_dir_env = _fix_windows_path(frames_dir_env)


    if not video_dir_env:
        print("ERROR: VIDEO_DIR is not set. Run 'source setup.sh' in this shell.", file=sys.stderr)
        return 2
    if not frames_dir_env:
        print("ERROR: FRAMES_DIR is not set. Run 'source setup.sh' in this shell.", file=sys.stderr)
        return 2

    
    video_dir = Path(video_dir_env)
    frames_dir = Path(frames_dir_env)


    if not video_dir.exists() or not video_dir.is_dir():
        print(f"ERROR: VIDEO_DIR does not exist or is not a directory: {video_dir}", file=sys.stderr)
        return 2
    if not frames_dir.exists():
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos (with allowed extensions)
    videos = list(iter_videos(video_dir, args.ext))
    if not videos:
        print(f"ERROR: No video files with extensions {args.ext} found under VIDEO_DIR: {video_dir}", file=sys.stderr)
        return 3

    # Discover cameras from filenames (helps validate --camera even without metadata)
    cameras_from_files: Set[str] = set()
    for vp in videos:
        m = STEM_RE.match(vp.name)
        if m:
            cameras_from_files.add(m.group("camera"))

    # Load metadata whitelist
    allowed_triples = load_allowed_from_meta(video_dir)
    if not allowed_triples:
        print("WARNING: No allowed entries found in metadata (_*_meta.csv). "
              "You may need to run your metadata extraction step first.", file=sys.stderr)

    # Normalize filters
    camera_filter = args.camera
    start_date = _norm(args.start_date)
    end_date   = _norm(args.end_date)
    start_time = _norm(args.start_time)
    end_time   = _norm(args.end_time)

    # Validate filters (use metadata if available to improve error messages)
    try:
        validate_filters(allowed_triples, camera_filter, start_date, end_date, start_time, end_time)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    # Additionally validate camera against filenames when metadata is absent/insufficient
    if camera_filter and camera_filter not in {c for (c, _, _) in allowed_triples} and camera_filter not in cameras_from_files:
        cam_list = ", ".join(sorted(cameras_from_files)) if cameras_from_files else "(none discovered)"
        print(
            f"ERROR: Camera '{camera_filter}' was not found among discovered filenames.\n"
            f"       Available cameras: {cam_list}",
            file=sys.stderr,
        )
        return 5

    # Process
    processed = 0
    skipped = 0

    for vp in tqdm(videos, desc="Extracting frames"):
        try:
            camera, date, clock_start_raw, clock_end = parse_video_name(vp)
        except ValueError:
            skipped += 1
            continue

        clock_start = _normalize_clock(clock_start_raw)

        # Must be listed in metadata (if present) and pass user filters
        if allowed_triples and (camera, date, clock_start) not in allowed_triples:
            skipped += 1
            continue
        if not passes_filters(camera, date, clock_start, camera_filter, start_date, end_date, start_time, end_time):
            skipped += 1
            continue

        dest = target_frames_dir_for(vp, frames_dir)

        # Overwrite behaviour:
        # - If --overwrite: delete the folder then recreate it.
        # - If --no-overwrite (default): keep folder and skip already-written frames (we pass overwrite=False).
        if args.overwrite:
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            per_file_overwrite = False  # original notebook used overwrite=False in video_to_frames
        else:
            dest.mkdir(parents=True, exist_ok=True)
            per_file_overwrite = False  # keep existing files; do not rewrite

        # Call your RIVeR function with the SAME signature you used in the notebook
        extract_config = {
            "video_path": vp,
            "frames_dir": dest,
            "start_frame_number": args.start_frame_number,
            "end_frame_number": args.end_frame_number,
            "every": args.every,
            "overwrite": per_file_overwrite,
        }

        try:
            _first = video_to_frames(**extract_config)  # from river.config
            processed += 1
        except Exception as e:
            # Write a simple error file in the dest to aid debugging
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / "_extract_error.txt").open("a", encoding="utf-8") as ef:
                ef.write(f"{vp}\n{e}\n\n")
            skipped += 1

    print(f"\nDONE.\nProcessed: {processed} video(s).\nSkipped: {skipped} video(s).")

    if processed == 0 and not allowed_triples:
        print("NOTE: No metadata entries were found, and no videos were processed. "
              "Run your metadata extraction to create _<camera>_meta.csv files.", file=sys.stderr)

    # ---------- Build / update frames index (always, even with --no-overwrite) ----------

    def collect_frame_paths(frame_dir: Path) -> pd.DataFrame:
        """
        Walk a frames directory structured like:
            frame_dir / camera / date / time_hhmmss / *.jpg

        Returns a DataFrame with columns:
            camera, date_yyyymmdd, time_hhmmss, frame_path
        """
        frame_dir = frame_dir.resolve()
        rows = []

        # Loop structure: camera → date → time_hhmmss → frames
        for camera_dir in frame_dir.iterdir():
            if not camera_dir.is_dir():
                continue
            camera = camera_dir.name

            for date_dir in camera_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                date = date_dir.name  # e.g. "20250426"

                for time_dir in date_dir.iterdir():
                    if not time_dir.is_dir():
                        continue

                    # folder name is time_hhmmss (already HHMMSS)
                    time_hhmmss = time_dir.name

                    # Collect all JPG frames
                    for jpg in time_dir.glob("*.jpg"):
                        rows.append({
                            "camera": camera,
                            "date_yyyymmdd": date,
                            "time_hhmmss": time_hhmmss,
                            "frame_path": str(jpg.resolve())
                        })

        df = pd.DataFrame(
            rows,
            columns=["camera", "date_yyyymmdd", "time_hhmmss", "frame_path"]
        )
        return df

    try:
        df_frames = collect_frame_paths(frames_dir)
        frames_file_parquet = frames_dir / "_frame_paths.parquet"
        frames_file_csv = frames_dir / "_frame_paths.csv"

        # Try parquet first; if engine missing, fall back to CSV only
        parquet_ok = True
        try:
            df_frames.to_parquet(frames_file_parquet, index=False)
        except Exception:
            parquet_ok = False

        # Always (re)write CSV
        df_frames.to_csv(frames_file_csv, index=False)

        # Minimal console message
        star = str(frames_file_parquet).rstrip(".parquet") + ".*"
        if parquet_ok:
            print(f"Frames DF saved to {star}")
        else:
            print(f"Frames DF saved to {frames_file_csv} (parquet engine not available)")
    except Exception as e:
        print(f"WARNING: Failed to build frames index: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))