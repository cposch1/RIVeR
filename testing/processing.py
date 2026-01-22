# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from river.config import *


# %% [markdown]
# ## Step 1: Frame Extraction
#
# Extract video infos

# %%
# Function that extracts video properties
def check_video_info(video_path: Path) -> dict:
    """Check video properties and estimate storage requirements."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read one frame to estimate size
        ret, frame = cap.read()
        if not ret:
            raise VideoHasNoFrames("Could not read frames from video")
            
        frame_size_mb = frame.nbytes / (1024 * 1024)  # Size in MB
        total_size_gb = (frame_size_mb * total_frames) / 1024  # Total size in GB
        
        return {
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "estimated_size_gb": total_size_gb
        }
    finally:
        cap.release()


# Function that scans all videos

from pathlib import Path
import re
import csv

def scan_videos_and_write_csvs(
    videos_root: Path,
    out_root: Path,
    suffix: str = ".avi",
    overwrite: bool = True,
) -> dict:
    """
    Scan videos under `videos_root` expecting names like:
      {camera}_{date}_{clockstart}_{clockend}.avi
    or with hyphens:
      {camera}_{date}-{clockstart}-{clockend}.avi

    Creates one CSV per camera:  {camera}_meta.csv  in  out_root / "meta"

    CSV columns: date, clock, total_frames, fps, resolution, estimated_size_gb
    Returns a dict: camera_name -> path_to_csv
    """
    videos_root = videos_root.resolve()

    # Regex: capture camera, date (YYYYMMDD), start/end (hhmmss), allowing '_' or '-' as separators
    pat = re.compile(
        r'^(?P<camera>.+?)_(?P<date>\d{8})[-_](?P<start>\d{6})[-_](?P<end>\d{6})\.avi$',
        re.IGNORECASE
    )

    # Gather rows per camera
    rows_per_camera = {}
    # NEW: gather error rows per camera (date, clock, error_message)
    errors_per_camera = {}

    # Walk directories: expect nested per-day/per-window folders, but be flexible
    for path in videos_root.rglob(f"*{suffix}"):
        
        fname = path.name

        # >>> Ignore AppleDouble/metadata artifacts <<<
        if fname.startswith(('._')):
            continue
        m = pat.match(fname)
        if not m:
            continue

        camera = m.group("camera")
        date = m.group("date")
        clock_start = m.group("start")
        clock_end = m.group("end")
        clock = f"{clock_start}_{clock_end}"

        # Extract video info; on error, log to per-camera error CSV accumulator
        try:
            info = check_video_info(path)
        except Exception as e:
            errors_per_camera.setdefault(camera, []).append({
                "date": date,
                "clock": clock,
                "error_message": str(e),
            })
            continue

        row = {
            "date": date,
            "clock": clock,  # combined as requested; if you want separate cols, add them too
            "total_frames": info["total_frames"],
            "fps": info["fps"],
            "resolution": info["resolution"],
            "estimated_size_gb": info["estimated_size_gb"],
            # "relative_path": str(path.relative_to(videos_root)),
        }
        rows_per_camera.setdefault(camera, []).append(row)

    # Write meta CSVs (unchanged except removed console prints)
    camera_to_csv = {}
    for camera, rows in rows_per_camera.items():
        # Sort by date then start time inside 'clock'
        rows.sort(key=lambda r: (r["date"], r["clock"]))

        csv_path = video_dir / f"_{camera}_meta.csv"
        camera_to_csv[camera] = csv_path

        if overwrite or not csv_path.exists():
            # Write with header
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["date", "clock", "total_frames", "fps", "resolution", "estimated_size_gb"],
                )
                writer.writeheader()
                writer.writerows(rows)
        else:
            # Append mode; do not write header
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["date", "clock", "total_frames", "fps", "resolution", "estimated_size_gb"],
                )
                writer.writerows(rows)

    # NEW: Write per-camera ERROR CSVs alongside the meta CSVs
    for camera, rows in errors_per_camera.items():
        # Sort by date then clock for consistency
        rows.sort(key=lambda r: (r["date"], r["clock"]))

        err_csv_path = video_dir / f"_{camera}_error_log.csv"
        write_header = overwrite or (not err_csv_path.exists())

        mode = "w" if write_header else "a"
        with err_csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["date", "clock", "error_message"],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    if not rows_per_camera and not errors_per_camera:
        # No videos matched or everything skipped silently; keep quiet per your requirement
        pass

    return camera_to_csv

# Run the functions
camera_csvs = scan_videos_and_write_csvs(video_dir, results_dir, suffix=".avi", overwrite=True)
camera_csvs

# %% [markdown]
# Extract frames
# - `every`: Extract every nth frame (e.g., every=2 takes every second frame)
# - `start_frame_number`: Begin extraction from this frame
# - `end_frame_number`: Stop extraction at this frame
# - `chunk_size`: Number of frames per processing chunk (affects memory usage)

# %%
start_frame_number = 0
end_frame_number = None    # process all frames
every = 25 

# ------------------------------------------------------------------------------
# 1) Filename parsing
#    Accepts:
#      ilh-cam1-pt_20250426-180000-185900.avi
#      camA_20250101-000000-010000.avi
#      camA_20250101_000000_010000.avi
# ------------------------------------------------------------------------------
VID_NAME_RE = re.compile(
    r'^(?P<camera>.+?)_(?P<date>\d{8})[-_](?P<start>\d{6})[-_](?P<end>\d{6})\.(?P<ext>avi|mp4)$',
    re.IGNORECASE
)

def parse_video_name(video_path: Path):
    """
    Parse the video filename and return (camera, date, clock)
    where clock is 'hhmmss_hhmmss'.
    """
    m = VID_NAME_RE.match(video_path.name)
    if not m:
        raise ValueError(f"Video filename does not match expected pattern: {video_path.name}")
    camera = m.group("camera")
    date = m.group("date")
    start = m.group("start")
    end = m.group("end")
    clock = f"{start}_{end}"
    return camera, date, clock


def target_frames_dir_for(video_path: Path, base_frames_dir: Path) -> Path:
    """
    frames/<camera>/<date>/<clock>/
    """
    camera, date, clock = parse_video_name(video_path)
    return base_frames_dir / camera / date / clock


# ------------------------------------------------------------------------------
# 2) Iterate videos under video_dir (skipping AppleDouble artifacts)
# ------------------------------------------------------------------------------
def iter_videos(root: Path, suffixes=(".avi", ".mp4")):
    root = Path(root)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        # Ignore AppleDouble / metadata artifacts
        if name.startswith(("._", ":_")):
            continue
        if p.suffix.lower() in suffixes:
            yield p


# ------------------------------------------------------------------------------
# 3) Load allowed (camera, date, clock) from per-camera meta CSVs in video_dir
#    Expected CSV columns: date, clock, total_frames, fps, resolution, estimated_size_gb
#    Meta filenames: _{camera}_meta.csv (e.g., _ilh-cam1-pt_meta.csv)
# ------------------------------------------------------------------------------
def load_allowed_from_meta(video_root: Path) -> set[tuple[str, str, str]]:
    allowed = set()
    video_root = Path(video_root)
    for meta_csv in video_root.glob("_*_meta.csv"):
        name = meta_csv.name
        # Ensure pattern _{camera}_meta.csv
        if not (name.startswith("_") and name.endswith("_meta.csv")):
            continue
        # camera name = strip leading "_" and trailing "_meta.csv"
        camera = name[1:-9]  # remove 1 char "_" and 9 chars "_meta.csv"

        with meta_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get("date")
                clock = row.get("clock")
                if not date or not clock:
                    continue
                allowed.add((camera, date, clock))
    return allowed


# ------------------------------------------------------------------------------
# 4) Build allowlist from metadata and extract frames ONLY for allowed videos
#    No try/except around extraction: errors will surface (as requested).
# ------------------------------------------------------------------------------
allowed_triples = load_allowed_from_meta(video_dir)

processed = 0
skipped = 0

for vp in tqdm(list(iter_videos(video_dir)), desc="Extracting frames (allowed in meta)"):
    # Only process videos that are present in per-camera meta CSVs
    try:
        camera, date, clock = parse_video_name(vp)
    except ValueError:
        # Filename not matching the convention won't be in metadata anyway
        skipped += 1
        continue

    if (camera, date, clock) not in allowed_triples:
        skipped += 1
        continue

    dest = target_frames_dir_for(vp, frames_dir)
    dest.mkdir(parents=True, exist_ok=True)

    extract_config = {
        "video_path": vp,
        "frames_dir": dest,          # hierarchical destination (camera/date/clock)
        "start_frame_number": start_frame_number,
        "end_frame_number": end_frame_number,    # process all frames
        "every": every,                 # keep your sampling
        "overwrite": False
    }

    # Do NOT catch errors here—let them raise if something is wrong
    _first_frame = video_to_frames(**extract_config)
    processed += 1

print(f"\nDone. Processed: {processed} video(s). Skipped (not in meta or bad name): {skipped}.")

# %%
