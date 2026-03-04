# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RIVeR: Rectification of Image Velocimetry Results
#
# RIVeR (Rectification of Image Velocimetry Results) is a Python package designed for processing river flow videos to obtain velocity fields and discharge estimates. It supports three main filming scenarios:
#
# **RIVeR-ICE: Rectification of Image Velocimetry Results - Integrated Channel Evolution**
#
# RIVeR-ICE (Rectification of Image Velocimetry Results - Integrated Channel Evolution) is an extension of RIVeR that....
#
# **Prerequisites**
#
# Before starting, ensure you have:
#
# - Python 3.11 or later
# - RIVeR package installed
# - Required dependencies (numpy, opencv-python, scipy)
#
# **Required folder hierarchy**
#
# ...
#
# **Required file terminology**
#
# ...

# %% [markdown]
# # Step 0: Imports
#
# Imports functions and dependencies from the package

# %%
from river.config import *


# %% [markdown]
# # Step 1: Metadata Extraction
#
# Automated video metadata extraction from data in the video folder

# %%
def _fmt_duration_hhmmss(total_seconds: Optional[float]) -> str:
    """Format seconds as hh:mm:ss (always include hours, zero-padded). Empty string if None."""
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

    if duration_s is None or duration_s == 0:
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


# ---------- Main scan + CSV writing ----------

def scan_videos_and_write_csvs(
    videos_root: Path,
    suffixes: Iterable[str] = (".mp4", ".avi", ".mkv", ".mov"),
    overwrite: bool = True,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:

    videos_root = videos_root.resolve()
    suffixes_set = {s.lower() for s in suffixes}

    pat = re.compile(
        r'^(?P<camera>.+?)_(?P<date>\d{8})-(?P<start>\d{6})-(?P<end>\d{6})$',
        re.IGNORECASE
    )

    rows_per_camera = {}
    errors_per_camera = {}

    for path in videos_root.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() not in suffixes_set:
            continue

        stem = path.stem
        if stem.startswith("._"):
            continue

        m = pat.match(stem)
        if not m:
            continue

        camera = m.group("camera")
        date = m.group("date")
        time = m.group("start")
        # clock_end parsed but not used anymore

        try:
            info = check_video_info(path)

            row = {
                "date_yyyymmdd": date,
                "time_hhmmss": time,
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
                "time_hhmmss": time,
                "duration_hhmmss": info["duration_hhmmss"],
                "total_frames": info["total_frames"],
                "fps": info["fps"],
                "resolution": info["resolution"],
                "bitrate_mbps": info["bitrate_mbps"],
                "size_gb": info["size_gb"],
                "error_message": str(e)
            })

    # Write meta CSVs
    camera_to_csv = {}
    for camera, rows in rows_per_camera.items():
        rows.sort(key=lambda r: (r["date_yyyymmdd"], r["time_hhmmss"]))

        csv_path = videos_root / camera / f"_{camera}_meta.csv"
        camera_to_csv[camera] = csv_path
        write_header = overwrite or not csv_path.exists()

        with csv_path.open("w" if write_header else "a", newline="", encoding="utf-8") as f:
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
    err_to_csv = {}
    for camera, rows in errors_per_camera.items():
        rows.sort(key=lambda r: (r["date_yyyymmdd"], r["time_hhmmss"]))

        err_csv_path = videos_root / camera / f"_{camera}_error_log.csv"
        err_to_csv[camera] = err_csv_path
        write_header = overwrite or not err_csv_path.exists()

        with err_csv_path.open("w" if write_header else "a", newline="", encoding="utf-8") as f:
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


# List created files
camera_csvs, err_csvs = scan_videos_and_write_csvs(video_dir, overwrite=True)

def format_dict_as_lines(d: dict) -> str:
    if not d:
        return "  (none)"
    return "\n".join(f"  {cam} -> {path}" for cam, path in sorted(d.items()))

print("Following metadata files were created:")
print(format_dict_as_lines(camera_csvs))
print("\nFollowing error log files were created:")
print(format_dict_as_lines(err_csvs))

# %% [markdown]
# # Step 2: Frame Extraction
#
# **Why Extract Frames?**
# - PIV analysis requires sequential image pairs
# - Easier memory management than processing full videos
# - Allows for quality control and frame selection
# - Enables parallel processing in later steps
#
# **Prerequisites**
# - RIVeR package installed
# - Video file(s) of river flow
# - Sufficient storage space for frames (tip: estimate ~0.5-2MB per frame)
#
# **Parameters**
# - `every`: Extract every nth frame (e.g., every=2 takes every second frame)
# - `start_frame_number`: Begin extraction from this frame
# - `end_frame_number`: Stop extraction at this frame
# - `overwrite_frames`: Option for overwriting existing data in frames folder
# - 
# **Filters**
# - `camera_filter`: Extract only for this camera
# - `start_date`: Begin extraction from this date
# - `end_date`: Stop extraction at this date
# - `start_time`: Begin extraction from this time
# - `end_time`: Stop extraction at this time

# %%
#########################
### DEFINE PARAMETERS ###
#########################

every = 1  # take every Nth frame
start_frame_number = 0
end_frame_number = None    # process all frames            
overwrite_frames = True    # or True if you want to force re-extraction/deletion of older frames


#########################
### USER FILTERS ###
#########################

camera_filter = None       # e.g. "ilh-cam1-pt"
start_date = None          # e.g. "20250426"
end_date = None
start_time = None          # e.g. "120000"
end_time = None


###############################################
# Filename parsing: <camera>_<YYYYMMDD>-<HHMMSS>-<HHMMSS>.<ext>
###############################################

VID_NAME_RE = re.compile(
    r'^(?P<camera>.+)_(?P<date>\d{8})-(?P<start>\d{6})-(?P<end>\d{6})\.(?P<ext>avi|mp4)$',
    re.IGNORECASE
)

def parse_video_name(video_path: Path) -> Tuple[str, str, str, str]:
    """Returns (camera, date, clock_start, clock_end)."""
    m = VID_NAME_RE.match(video_path.name)
    if not m:
        raise ValueError(f"Video filename does not match expected pattern: {video_path.name}")
    return (
        m.group("camera"),
        m.group("date"),
        m.group("start"),
        m.group("end"),
    )


def target_frames_dir_for(video_path: Path, base_frames_dir: Path) -> Path:
    """frames/<camera>/<date>/<clock_start>/"""
    camera, date, clock_start, _clock_end = parse_video_name(video_path)
    return base_frames_dir / camera / date / clock_start


def iter_videos(root: Path, suffixes: Iterable[str] = (".avi", ".mp4")):
    """Yield video files under root with allowed suffixes."""
    root = Path(root)
    suffixes_lower = {s.lower() for s in suffixes}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(("._", ":_")):
            continue
        if p.suffix.lower() in suffixes_lower:
            yield p


###############################################
### Helpers
###############################################

def _normalize_clock_start(val: str) -> str:
    """Normalize to HHMMSS format."""
    if val is None:
        return ""
    v = val.strip().replace(":", "").replace("-", "").replace(" ", "")
    return v if len(v) == 6 and v.isdigit() else ""


def _norm(val):
    """Normalize user filter input."""
    if val is None:
        return None
    val = val.strip().replace(":", "").replace("-", "").replace(" ", "")
    return val if val else None

start_date = _norm(start_date)
end_date   = _norm(end_date)
start_time = _norm(start_time)
end_time   = _norm(end_time)

def prepare_frames_dir(dest: Path, overwrite: bool):
    """
    Handles replacing or keeping existing frames folder before extraction.
    """
    if dest.exists():
        if overwrite:
            shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
        else:
            # If folder not empty → skip extraction entirely
            if any(dest.iterdir()):
                return False
    else:
        dest.mkdir(parents=True, exist_ok=True)
    return True


###############################################
### LOAD ALLOWED METADATA
###############################################

def load_allowed_from_meta(video_root: Path) -> Set[Tuple[str, str, str]]:
    """
    Reads meta CSV files (_{camera}_meta.csv)
    Returns set of (camera, date, clock_startHHMMSS).
    """
    allowed: Set[Tuple[str, str, str]] = set()
    video_root = Path(video_root)

    for meta_csv in video_root.rglob("_*_meta.csv"):
        name = meta_csv.name
        if not name.startswith("_") or not name.endswith("_meta.csv"):
            continue

        camera = name[1:-9]  # strip "_" and "_meta.csv"

        with meta_csv.open("r", newline="", encoding="utf-8-sig") as f:
            sample = f.read(8192); f.seek(0)
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

                clock = _normalize_clock_start(clock_raw)
                if not date or not clock:
                    continue

                allowed.add((camera, date, clock))

    return allowed


allowed_triples = load_allowed_from_meta(video_dir)
print(f"Loaded {len(allowed_triples)} allowed entries.")

if not allowed_triples:
    print("WARNING: No allowed entries found in metadata.")


###############################################
### VALIDATION OF USER FILTERS
###############################################

def validate_filters(allowed: Set[Tuple[str, str, str]]):
    cams  = {c for (c,_,_) in allowed}
    dates = {d for (_,d,_) in allowed}

    if camera_filter and camera_filter not in cams:
        raise ValueError(f"Camera '{camera_filter}' not found. Existing cameras: {sorted(cams)}")

    if start_date and start_date not in dates:
        raise ValueError(f"start_date '{start_date}' not found in metadata")
    if end_date and end_date not in dates:
        raise ValueError(f"end_date '{end_date}' not found in metadata")
    if start_date and end_date and start_date > end_date:
        raise ValueError(f"start_date > end_date")

    if start_time and len(start_time) != 6:
        raise ValueError("start_time must be HHMMSS")
    if end_time and len(end_time) != 6:
        raise ValueError("end_time must be HHMMSS")
    if start_time and end_time and start_time > end_time:
        raise ValueError(f"start_time > end_time")

validate_filters(allowed_triples)


###############################################
### FILTER CHECK
###############################################

def passes_filters(camera: str, date: str, clock_start: str) -> bool:
    """Return True if this video satisfies the user-defined filters."""
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


###############################################
### MAIN EXTRACTION LOOP
###############################################

processed = 0
skipped = 0

for vp in tqdm(list(iter_videos(video_dir)), desc="Extracting frames"):
    try:
        camera, date, clock_start_raw, clock_end = parse_video_name(vp)
    except ValueError:
        skipped += 1
        continue

    clock_start = _normalize_clock_start(clock_start_raw)

    key = (camera, date, clock_start)
    if key not in allowed_triples or not passes_filters(camera, date, clock_start):
        skipped += 1
        continue

    
    
    dest = target_frames_dir_for(vp, frames_dir)
    
    # Option B behavior:
    # overwrite_frames = True  -> delete + recreate
    # overwrite_frames = False -> keep folder and ALWAYS extract
    if overwrite_frames:
        import shutil
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)  # create if missing, keep existing as-is



    extract_config = {
        "video_path": vp,
        "frames_dir": dest,
        "start_frame_number": start_frame_number,
        "end_frame_number": end_frame_number,
        "every": every,
        "overwrite": False
    }

    _first_frame = video_to_frames(**extract_config)  # noqa: F821
    processed += 1

print(f"\nDONE.\nProcessed: {processed} video(s).\nSkipped: {skipped} video(s).")

# %% [markdown]
# # Step 3: Orthrectification
#
# Performs coordinate transformation for oblique (side-view) river videos using RIVeR that accounts for perspective distortion.
#
# **Prerequisites**
#
# - Completed frame extraction
# - An oblique view frame to work with
# - 4 GCPs (ground control points) with known real-world coordinates 
#
# **Analysis requirements**
#
# - GCP well distributed across the frame
# - Include points at different depths in the scene
# - GCP selection at the surface, not at the top of the GCP marker
# - Real-world coordinates saved in the gcps folder
# - Distances between points are automatically calculated
#
# Point ordering is critical for correct transformation:
# - Point 1 must be the most upstream and leftmost point in your view
# - Remaining points (2, 3, and 4) must be defined in counterclockwise order
# - Example ordering:
#   * Point 1: Upstream-left
#   * Point 2: Upstream-right
#   * Point 3: Downstream-right
#   * Point 4: Downstream-left

# %% [markdown]
# ***Repeat lines until "End of Step 3" for each station***

# %%
# Adapt so you loop through it and do it once a day for each station

# %%
#########################
### DEFINE PARAMETERS ###
#########################

gcp_cam = "chamb_02"
gcp_date = "20260223"         # in format YYYYMMDD
gcp_time = "120000"           # in format HHMMSS

#########################

# %%
# Function that creates frame_path df
def collect_frame_paths(frame_dir: Path) -> pd.DataFrame:
    """
    Walk a frames directory structured like:

        frame_dir / camera / date / time_hhmmss / *.jpg

    Returns a DataFrame with:
        camera, date, time_hhmmss, frame_path
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

                # folder name *is* time_hhmmss (already HHMMSS)
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


# Create frame path df
df_frames = collect_frame_paths(frames_dir)
frames_file = frames_dir/"_frame_paths.parquet"
df_frames.to_parquet(frames_file, index=False)
df_frames.to_csv(frames_dir/"_frame_paths.csv", index=False)
print(f'Frames DF saved to {str(frames_file).strip(".parquet")+".*"}')

# %%
# Load frame
df_frames = pd.read_parquet(frames_dir/"_frame_paths.parquet")
frame, frame_rgb, frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time) # loads by default 0th frame

# %%
# Select GCPs in image
# %matplotlib widget
plt.close()

img = mpimg.imread(str(frame_path))
points = []

fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(img)
ax.set_title(f"Select GCPs:\n1) left upstream\n2) right upstream\n3) right downstreamm\n4) left downstream\n\n{frame_path}")
plt.axis("off")


# Function to get image coordinates
def onclick(event):
    # Ensure click is inside image
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: x={x}, y={y}")
        points.append((x, y))
        
        n = len(points)
        if n == 1:
            ax.plot(x, y, 'o', color='#ED6B57', markersize=3)  # first point red
            ax.text(x, y, "1", color='#ED6B57', fontsize=8, ha='left', va='bottom')
        elif n in (2, 3):
            ax.plot(x, y, 'o', color='#6CD4FF', markersize=3)  # 2nd & 3rd blue
            ax.text(x, y, str(n), color='#6CD4FF', fontsize=8, ha='left', va='bottom')
        elif n == 4:
            ax.plot(x, y, 'o', color='#6CD4FF', markersize=3)  # 4th blue
            ax.text(x, y, str(n), color='#6CD4FF', fontsize=8, ha='left', va='bottom')
            
            fig.canvas.draw()
            fig.canvas.mpl_disconnect(cid)
            print("4 points collected:", points)
            return

# Get image coordinates
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Select GCPs in the displayed picture above

# %%
raise SystemExit

# %%
# Save GCP image coordinates
if points == []:
    print(f"No GCPs selected and saved")
elif len(points)!=4:
    print("Select exactly 4 GCPs!")
else:
    print(f"Selected image coordinates (X/Y):\n{points}")

    
    gcps_img_file = gcps_dir / gcp_cam / (f"{gcp_cam}_gcps_img_{gcp_date}_{gcp_time}.csv")
    gcps_img_file.parent.mkdir(parents=True, exist_ok=True)
    with open(gcps_img_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(points)
    print(f"\nImage coordinates saved to:\n{gcps_img_file}")
    
    gcps_img = gcps_dir / gcp_cam / (f"{gcp_cam}_gcps_img_{gcp_date}_{gcp_time}.png")
    gcps_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(gcps_img))
    plt.close()
    print(f"\nImage coordinates picture saved to:\n{gcps_img}")


# %%
# Function for calculating euclidian distances
def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function for calculating all distances
def calc_dist(gcp_cam):
    point_coords_world = load_gcps_real(gcp_cam)
    distances = {
        'd12': dist(point_coords_world['point1'], point_coords_world['point2']),
        'd23': dist(point_coords_world['point2'], point_coords_world['point3']),
        'd34': dist(point_coords_world['point3'], point_coords_world['point4']),
        'd41': dist(point_coords_world['point4'], point_coords_world['point1']),
        'd13': dist(point_coords_world['point1'], point_coords_world['point3']),  # diagonal
        'd24': dist(point_coords_world['point2'], point_coords_world['point4'])   # diagonal
    }
    return distances

# Load distances
distances = calc_dist(gcp_cam)

# Print real world distances
distances_print = {name: round(value, 2) for name, value in distances.items()}
print("GCPs real world distances:")
print(distances_print)

# Save real world distances
gcps_dist_file = gcps_dir / gcp_cam / (f"{gcp_cam}_gcps_dist.csv")
with open(gcps_dist_file, "w", newline="") as f:
    writer = csv.writer(f)
    for name, value in distances_print.items():
        writer.writerow([value])

print(f"\nGCPs real world distances saved to:\n{gcps_dist_file}")

# %%
# Do transformation
transformation = transform(df_frames,gcp_cam,gcp_date,gcp_time)
#transformation = transform(df_frames,"le5-cam1-pt","20250426","120000")

# %%
# Function for visualizing transformation
def vis_transf(df_frames,gcp_cam,gcp_date,gcp_time,transformation):
    # %matplotlib inline
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ### First subplot with original image and control points ###
    ax1.imshow(img)
    
    # Load image points
    points = load_gcps_img(gcp_cam,gcp_date,gcp_time)
    x1_pix, y1_pix = points['point1']
    x2_pix, y2_pix = points['point2']
    x3_pix, y3_pix = points['point3']
    x4_pix, y4_pix = points['point4']
    
    # Draw lines with specific colors
    ax1.plot([x1_pix, x2_pix], [y1_pix, y2_pix], color='#6CD4FF', linewidth=2) # Line 1-2
    ax1.plot([x2_pix, x3_pix], [y2_pix, y3_pix], color='#62C655', linewidth=2) # Line 2-3
    ax1.plot([x3_pix, x4_pix], [y3_pix, y4_pix], color='#ED6B57', linewidth=2) # Line 3-4
    ax1.plot([x4_pix, x1_pix], [y4_pix, y1_pix], color='#F5BF61', linewidth=2) # Line 4-1
    ax1.plot([x1_pix, x3_pix], [y1_pix, y3_pix], color='#CC4BC2', linewidth=2) # Diagonal 1-3
    ax1.plot([x2_pix, x4_pix], [y2_pix, y4_pix], color='#7765E3', linewidth=2) # Diagonal 2-4
    
    ## Plot points
    # Point 1 in red
    ax1.plot(x1_pix, y1_pix, 'o', color='#ED6B57', markersize=3)
    ax1.text(x1_pix, y1_pix, "1", color='#ED6B57', fontsize=8, ha='left', va='bottom')
    # Points 2-4 in blue
    ax1.plot([x2_pix, x3_pix, x4_pix], [y2_pix, y3_pix, y4_pix], 'o', color='#6CD4FF', markersize=3)
    pts = [(x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix)]
    [ax1.text(x, y, str(i), color='#6CD4FF', fontsize=8, ha='left', va='bottom') for i, (x, y) in enumerate(pts, start=2)]
    
    ax1.axis('off')
    ax1.set_title('Original Image')  # Fixed from ax1.title to ax1.set_title
    
    
    ### Second subplot with orthorectified image ###
    if 'transformed_img' in transformation and 'extent' in transformation:
        extent = transformation['extent']
        ax2.imshow(transformation['transformed_img'], extent=extent)
        
        ## Add scale bar
        # Calculate appropriate scale length
        map_width = extent[1] - extent[0]
        magnitude = 10 ** np.floor(np.log10(map_width * 0.2))
        scale_length = np.round(map_width * 0.2 / magnitude) * magnitude
        scale_length_rounded = int(scale_length) if scale_length < 10 else scale_length
        
        # Define scale bar position (in data coordinates)
        margin = (extent[1] - extent[0]) * 0.05  # 5% margin from edges
        bar_height = (extent[3] - extent[2]) * 0.015  # Height of bar
        x_pos = extent[1] - margin - scale_length_rounded
        y_pos = extent[2] + margin
        
        # Add scale bar
        rect = Rectangle((x_pos, y_pos), scale_length_rounded, bar_height,
                         fc='white', ec='black')
        ax2.add_patch(rect)
        
        # Add text label for the scale bar
        ax2.text(x_pos + scale_length_rounded/2, y_pos + 2*bar_height,
                f'{int(scale_length_rounded)} m',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, pad=2))
        
        ## Convert GCP pixel coordinates to real-world coordinates to display in the second image
        real_world_points = []
        for x, y in [(x1_pix, y1_pix), (x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix)]:
            rw_point = transform_pixel_to_real_world(x, y, transformation['transformation_matrix'])
            real_world_points.append(rw_point)
        
        real_world_points = np.array(real_world_points)
        
        # Get individual point coordinates
        x1_rw, y1_rw = real_world_points[0]
        x2_rw, y2_rw = real_world_points[1]
        x3_rw, y3_rw = real_world_points[2]
        x4_rw, y4_rw = real_world_points[3]
        print(real_world_points)
        
        # Draw lines with the same colors as in the first plot
        ax2.plot([x1_rw, x2_rw], [y1_rw, y2_rw], color='#6CD4FF', linewidth=2) # Line 1-2
        ax2.plot([x2_rw, x3_rw], [y2_rw, y3_rw], color='#62C655', linewidth=2) # Line 2-3
        ax2.plot([x3_rw, x4_rw], [y3_rw, y4_rw], color='#ED6B57', linewidth=2) # Line 3-4
        ax2.plot([x4_rw, x1_rw], [y4_rw, y1_rw], color='#F5BF61', linewidth=2) # Line 4-1
        ax2.plot([x1_rw, x3_rw], [y1_rw, y3_rw], color='#CC4BC2', linewidth=2) # Diagonal 1-3
        ax2.plot([x2_rw, x4_rw], [y2_rw, y4_rw], color='#7765E3', linewidth=2) # Diagonal 2-4
        
        ## Plot points
        # Point 1 in red
        ax2.plot(x1_rw, y1_rw, 'o', color='#ED6B57', markersize=3)
        ax2.text(x1_rw, y1_rw, "1", color='#ED6B57', fontsize=8, ha='left', va='bottom')
        # Points 2-4 in blue
        ax2.plot([x2_rw, x3_rw, x4_rw], [y2_rw, y3_rw, y4_rw], 'o', color='#6CD4FF', markersize=3)
        pts = [(x2_rw, y2_rw), (x3_rw, y3_rw), (x4_rw, y4_rw)]
        [ax2.text(x, y, str(i), color='#6CD4FF', fontsize=8, ha='left', va='bottom') for i, (x, y) in enumerate(pts, start=2)]
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Orthorectified Image')
    
    plt.tight_layout()
    
    ortho_img = rect_dir / gcp_cam / (f"{gcp_cam}_orthorectification_{gcp_date}_{gcp_time}.png")
    ortho_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(ortho_img))
    plt.show()
    #plt.close(fig)

    print(f"\nOrthorectification image saved to:\n{ortho_img}")

# %%
# Visualize transformation
vis_transf(df_frames,gcp_cam,gcp_date,gcp_time,transformation)

# %%
# Save transformation
transformation_matrix = transformation['transformation_matrix']
transf_file = rect_dir / gcp_cam / (f"{gcp_cam}_transform_{gcp_date}_{gcp_time}.json")
transf_file.parent.mkdir(parents=True, exist_ok=True)
with open(transf_file, 'w') as f:
    json.dump(transformation_matrix, f, indent=1)
print(f"Transformation matrix saved to\n{transf_file}")

# %% [markdown]
# ***End of Step 3 (repeat for each station)***

# %% [markdown]
# # Step 4: Cross-Section Selection and Bathymetry
#
# This step defines and analyzes river cross-sections using RIVeR.
# - Define cross-section lines from orthorectified image
# - Define water depth and calculate idealized bathymetry data 
# - Calculate section properties like area and width
# - Prepare cross-sections for velocity analysis
#
# **Prerequisites**
#
# - Completed orthrectification
# - Transformation matrix from previous steps
# - Water depth
#
# **Parameters**
#
# - `num_stations`: Defines number of steps (resolution) in the cross-section for bahymetry calculation and PIV vectors
# - `alpha_vel`: Ratio between surface and depth-averaged velocity (typically 0.85-1.0)

# %%
# Define number of "stations", i.e. analysis/bathymetry points across section
num_stations = 15

# Define vertical velocity correction coefficient
alpha_vel = 1

# %%
#########################
### DEFINE PARAMETERS ###
#########################

gcp_cam = "chamb_02"
gcp_date = "20260223"         # in format YYYYMMDD
gcp_time = "120000"           # in format HHMMSS

#########################

pt_name = "pt_02"

# %%
# Load image
df_frames = pd.read_parquet(frames_dir/"_frame_paths.parquet")
_,frame_rgb,frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time)
img = mpimg.imread(str(frame_path))

# Do transformation
transformation = transform(df_frames,gcp_cam,gcp_date,gcp_time)

# Load transformation matrix
transf_file = rect_dir / gcp_cam / (f"{gcp_cam}_transform_{gcp_date}_{gcp_time}.json")
with open(transf_file, 'r') as f:
    transformation_matrix = np.array(json.load(f))

# Load first FCP coordinates
gcps_real_file = gcps_dir / gcp_cam / f"{gcp_cam}_gcps_real.csv"
with open(gcps_real_file, newline="") as f:
    reader = csv.reader(f)
    first_row = next(reader)   # read only the first line
off_x = float(first_row[0])
off_y = float(first_row[1])


# Load PT depth data
df_pt = load_pt(pt_name)

# Load PT real world coord data
pt_points_coords = load_pt_real(pt_name)

# Load video metadata
meta_file = video_dir / gcp_cam / f"_{gcp_cam}_meta.csv"
df_meta = pd.read_csv(meta_file)

# %%
# Subset pt data based on video date, time and duration

# Filter the row matching date + time
row = df_meta[
    (df_meta["date_yyyymmdd"] == int(gcp_date)) &
    (df_meta["time_hhmmss"] == int(gcp_time))
].iloc[0]

# Compute start & end timestamp
start_ts = pd.to_datetime(f"{gcp_date} {gcp_time}", format="%Y%m%d %H%M%S")
duration_str = str(row["duration_hhmmss"]).zfill(6)
duration_s = f"{duration_str[0:2]}:{duration_str[2:4]}:{duration_str[4:6]}"
duration_td = pd.to_timedelta(duration_s)
end_ts = start_ts + duration_td

print("Video start:", start_ts)
print("Video end  :", end_ts)

# Convert df_pt timestamp and filter by video window
df_pt["timestamp"] = pd.to_datetime(
    df_pt["date_yyyymmdd"].astype(str) + " " + df_pt["time_hhmmss"].astype(str),
    format="%Y%m%d %H%M%S"
)

mask = (df_pt["timestamp"] >= start_ts) & (df_pt["timestamp"] <= end_ts)
df_window = df_pt.loc[mask]

# Compute mean depth for that video clip
depth_avg = df_window["depth_m"].mean()
print("Average depth over video period:", round(depth_avg,2), "m")

# Save depth results
dep_res_csv = dep_dir / pt_name / f"{pt_name}_dep_{gcp_date}_{gcp_time}.csv"
dep_res_csv.parent.mkdir(parents=True, exist_ok=True)
with open(dep_res_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([depth_avg])     # value
print("Saved average depth to: ", dep_res_csv)

# %%
# Select cross section in image

# %matplotlib widget

plt.close()
plt.figure(figsize=(14, 10))
ax = plt.gca()
ax.minorticks_on()
ax.grid(True, which='both', color='black', linewidth=0.5, alpha=0.2)

extent = transformation['extent']
ax.imshow(transformation['transformed_img'], extent=extent)

#########################################################################################
## Plot PT loc

# Transform PT loc to orthrectified local space
x1_pt_co, y1_pt_co = pt_points_coords[0]
x1_pt_rw = x1_pt_co - off_x
y1_pt_rw = y1_pt_co - off_y


## Plot PT points
ax.plot(x1_pt_rw, y1_pt_rw, 'o', color='blue', markersize=5)
ax.text(x1_pt_rw, y1_pt_rw, "PT", color='blue', fontsize=8, ha='left', va='bottom')
###########################################################################################

# Add scale bar
# Calculate appropriate scale length
map_width = extent[1] - extent[0]
magnitude = 10 ** np.floor(np.log10(map_width * 0.2))
scale_length = np.round(map_width * 0.2 / magnitude) * magnitude
scale_length_rounded = int(scale_length) if scale_length < 10 else scale_length

# Define scale bar position (in data coordinates)
margin = (extent[1] - extent[0]) * 0.05  # 5% margin from edges
bar_height = (extent[3] - extent[2]) * 0.015  # Height of bar
x_pos = extent[1] - margin - scale_length_rounded
y_pos = extent[2] + margin

# Add scale bar
rect = Rectangle((x_pos, y_pos), scale_length_rounded, bar_height,
                 fc='white', ec='black')
ax.add_patch(rect)

# Add text label for the scale bar
ax.text(x_pos + scale_length_rounded/2, y_pos + 2*bar_height,
        f'{int(scale_length_rounded)} m',
        ha='center', va='bottom', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, pad=2))


ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Cross section selection in orthorectified image')

#plt.tight_layout()

#mage_output_file = output_dir / "05_cross_sec_orthorect.png"
#plt.savefig(str(image_output_file))

# --- Interactive picking of two points in real-world coordinates ---
points_rw = []
point_coords_xs = {}  # {'point1': (x1, y1), 'point2': (x2, y2)}

# We'll store the variables you used in your other code:
x1_rw = y1_rw = x2_rw = y2_rw = None

def onclick(event):
    nonlocal_vars = ('x1_rw', 'y1_rw', 'x2_rw', 'y2_rw', 'point_coords_xs')  # for clarity in this cell
    # In a notebook cell, use 'global' to assign to top-level names:
    global x1_rw, y1_rw, x2_rw, y2_rw, point_coords_xs

    # Only react to clicks inside the axes with valid data coordinates
    if (event.inaxes is not ax) or (event.xdata is None) or (event.ydata is None):
        return

    x, y = float(event.xdata), float(event.ydata)
    print(f"Clicked at: East={x:.3f} m, North={y:.3f} m")
    points_rw.append((x, y))

    # First point (left bank) — plot red marker
    if len(points_rw) == 1:
        x1_rw, y1_rw = points_rw[0]
        ax.plot(x1_rw, y1_rw, 'o', color='#ED6B57', markersize=3, zorder=4)
        fig.canvas.draw_idle()

    # Second point (right bank) — plot green marker, draw the connecting line, disconnect
    elif len(points_rw) == 2:
        x2_rw, y2_rw = points_rw[1]
        ax.plot(x2_rw, y2_rw, 'o', color='#62C655', markersize=3, zorder=4)
        # Draw line connecting the two banks
        ax.plot([x1_rw, x2_rw], [y1_rw, y2_rw], color='#F5BF61', linewidth=2, zorder=4)

        # Save into a dict for clean access
        point_coords_xs = [
            (x1_rw, y1_rw),
            (x2_rw, y2_rw),
        ]

        print("2 points collected (real-world coords):")
        print(point_coords_xs)

        # Disconnect the event handler after two points
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.draw_idle()

        # --- Optional: save figure automatically once selected ---
        # image_output_file = output_dir / "05_cross_sec_orthorect_picked.png"
        # fig.savefig(str(image_output_file), dpi=150, bbox_inches='tight')
        # print(f"Saved: {image_output_file}")


# Connect the callback
cid = ax.figure.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Select cross section in the displayed picture above

# %%
raise SystemExit

# %%
print(point_coords_xs)

# %%
if not point_coords_xs:
    print("No GCPs selected and saved")
else:
    print(f"Selected cross-section coordinates (X/Y):\n{point_coords_xs}")

    xs_file = bathy_dir / gcp_cam / f"{gcp_cam}_xs_coord_{gcp_date}_{gcp_time}.csv"
    xs_file.parent.mkdir(parents=True, exist_ok=True)

    # Ensure list (not set)
    point_coords_xs = list(point_coords_xs)

    with open(xs_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write floats in Excel-friendly format
        for x, y in point_coords_xs:
            writer.writerow([f"{x:.15f}", f"{y:.15f}"])

    print(f"\nCross-section coordinates saved to:\n{xs_file}")

    xs_img = bathy_dir/gcp_cam/f'{gcp_cam}_xs_{gcp_date}_{gcp_time}.png'
    plt.savefig(xs_img)
    print(f"\nCross-section image saved to:\n{xs_img}")

# %%
# Load cross section point coordinates
points_cross = load_xs_img(gcp_cam,gcp_date,gcp_time)

x_le = points_cross[0][0]
y_le = points_cross[0][1]
x_ri = points_cross[1][0]
y_ri = points_cross[1][1]

print(f"Selected cross-section coordinates (X/Y):\n{points_cross}")

# %%
### HERE: Fix bathymetry estimation (not anymore mid_point based)

# %%
## Midpoint bathymetry version
# Define bathymetry
bath_file = bathy_dir / gcp_cam / (f"{gcp_cam}_bath_{gcp_date}_{gcp_time}.csv")

lvl = depth_avg

le_bath = x_le - x_le
ri_bath = x_ri - x_le
mid_bath = (ri_bath - le_bath) / 2

xL = le_bath
xM = mid_bath
xR = ri_bath
lvl = lvl

def p(x):
    return lvl * (x - xM)**2 / ((xL - xM)*(xR - xM)) * (-1)

xs = np.linspace(xL, xR, num_stations)
points = [(float(x), float(p(x))) for x in xs]



# Write CSV
with open(bath_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["d", "h"])   # header
    writer.writerows(points)      # the data rows

# %%
## Sensor-dyanmic greatest depth bathymetry version
# Define bathymetry
bath_file = bathy_dir / gcp_cam / (f"{gcp_cam}_bath_{gcp_date}_{gcp_time}.csv")
lvl = depth_avg
#x1_pt_rw = 2.5    # for testing

# Horizontal coordinates (left=0)
xL = x_le - x_le            # 0
xR = x_ri - x_le
xS = x1_pt_rw               # sensor horizontal position (same coordinate system)

# Elevation levels:
# 'lvl' is the elevation at the banks (top of bed profile in your convention).
# The deepest point is 0 by requirement.
# Sensor elevation:
hS = lvl - depth_avg        # must be in [0, lvl]

# Guard against tiny numeric issues
eps = 1e-9
hS = max(0.0, min(float(hS), float(lvl)))

# Ratio r = sqrt(hS/lvl). r in [0,1]. r=0 -> sensor at the deepest point; r=1 -> sensor at bank level.
r = 0.0 if lvl <= eps else float(np.sqrt(hS / max(lvl, eps)))

def choose_x0(xL, xR, xS, r):
    """
    Compute candidate vertex locations x0 that ensure:
      - h(x0) = 0 (minimum)
      - h(xL) = h(xR) = lvl
      - h(xS) = hS
    Try both 'sensor on left branch' and 'sensor on right branch' formulas.
    Prefer a valid x0 within (xL, xR). Fall back to midpoint if needed.
    """
    # Handle edge cases explicitly
    if r < 1e-9:
        # Sensor is at the deepest level -> vertex at the sensor
        return float(xS)

    if 1.0 - r < 1e-9:
        # Sensor elevation equals bank elevation (depth_avg ≈ 0): degenerate.
        # Use midpoint vertex as a reasonable default.
        return float(0.5 * (xL + xR))

    # Candidate assuming sensor is on LEFT branch (xS <= x0)
    x0_left = (xS - r * xL) / (1.0 - r)
    # Candidate assuming sensor is on RIGHT branch (xS >= x0)
    x0_right = (xS - r * xR) / (1.0 - r)

    valid_left = (xL < x0_left < xR) and (xS <= x0_left + 1e-12)
    valid_right = (xL < x0_right < xR) and (xS >= x0_right - 1e-12)

    if valid_left and not valid_right:
        return float(x0_left)
    if valid_right and not valid_left:
        return float(x0_right)
    if valid_left and valid_right:
        # If both are valid, choose the one that positions x0 closer to xS (milder asymmetry)
        return float(x0_left) if abs(x0_left - xS) <= abs(x0_right - xS) else float(x0_right)

    # If neither candidate is valid (very rare due to rounding), fall back to midpoint
    return float(0.5 * (xL + xR))

# Compute the vertex (deepest point)
x0 = choose_x0(xL, xR, xS, r)

# Define piecewise parabola guaranteeing min=0 and banks=lvl
def p(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    # Coefficients from bank constraints
    aL = lvl / ((xL - x0) ** 2)
    aR = lvl / ((xR - x0) ** 2)

    # Left branch (x <= x0): h = aL (x - x0)^2
    left_mask = (x <= x0)
    out[left_mask] = aL * (x[left_mask] - x0) ** 2

    # Right branch (x >= x0): h = aR (x - x0)^2
    right_mask = ~left_mask
    out[right_mask] = aR * (x[right_mask] - x0) ** 2

    # Numerical clipping to guarantee never below 0, never above lvl by > tiny epsilon
    return np.clip(out, 0.0, lvl)

# Sample and write CSV
xs = np.linspace(xL, xR, num_stations)
points = [(float(x), float(p(x))) for x in xs]

with open(bath_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["d", "h"])   # header
    writer.writerows(points)

# %%
# Define initial cross-sections dictionary
xsections = {
    "section1": {
        "east_l": x_le,      # Left bank easting
        "north_l": y_le,      # Left bank northing
        "east_r": x_ri,      # Right bank easting
        "north_r": y_ri,      # Right bank northing
        "level": lvl,       # Water level
        "num_stations": num_stations,   # Number of analysis points
        "alpha": alpha_vel,           # Velocity correction coefficient
        "bath": str(bath_file),  # Path to bathymetry file
        "left_station": 0   # Offset for first station from left bank
    }
}

# Calculate XS pixel coordinates from real-world coordinates
left_pixel = transform_real_world_to_pixel(xsections["section1"]["east_l"], 
                                         xsections["section1"]["north_l"], 
                                         transformation_matrix)
print("Image coordinates of cross section points")
print(f"Left river bank: {left_pixel}")
right_pixel = transform_real_world_to_pixel(xsections["section1"]["east_r"], 
                                          xsections["section1"]["north_r"], 
                                          transformation_matrix)
print(f"Left river bank: {right_pixel}")
# Calculate real-world length of the section
rw_length = np.sqrt((xsections["section1"]["east_r"] - xsections["section1"]["east_l"])**2 + 
                    (xsections["section1"]["north_r"] - xsections["section1"]["north_l"])**2)
print("\nReal world cross section length")
print(f"%0.2f" %rw_length + " m")

# Update the dictionary with additional values
xsections["section1"].update({
    "xl": left_pixel[0],     # Left bank x-pixel coordinate
    "yl": left_pixel[1],     # Left bank y-pixel coordinate
    "xr": right_pixel[0],    # Right bank x-pixel coordinate
    "yr": right_pixel[1],    # Right bank y-pixel coordinate
    "rw_length": rw_length
})

# Save cross-sections to JSON
sect_file = bathy_dir / gcp_cam / (f"{gcp_cam}_xs_{gcp_date}_{gcp_time}.json")
with open(sect_file, 'w') as f:
    json.dump(xsections, f, indent=2)
print(f"\nCross-sections data saved to {sect_file}")

# %%
# Calculate PT pixel coordinates from real-world coordinates
pt_pix = transform_real_world_to_pixel(x1_pt_rw, y1_pt_rw, transformation_matrix)

x1_pt_pix = pt_pix[0]
y1_pt_pix = pt_pix[1]

print(x1_pt_rw,y1_pt_rw)
print(pt_pix)

# Save PT img coord to csv
pt_img_coord = pts_dir / (f"{pt_name}_img.csv")
with open(pt_img_coord, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([x1_pt_pix,y1_pt_pix])
print(f"\nPT image coordinates saved to:\n{pt_img_coord}")

# Save PT loc coord to csv
pt_loc_coord = pts_dir / (f"{pt_name}_loc.csv")
with open(pt_loc_coord, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([x1_pt_rw,y1_pt_rw])
print(f"\nPT local coordinates saved to:\n{pt_loc_coord}")

# %%
# %matplotlib inline

# Load and plot bathymetry data
data = Dataset()
with open(bath_file, 'r') as f:
    data.load(f, format='csv', headers=False)

# Extract stations and stages
stations = [float(i) if i is not None else 0.0 for i in data.get_col(0)[1:]]
stages = [float(i) if i is not None else 0.0 for i in data.get_col(1)[1:]]
stations = np.array(stations)
stages = np.array(stages)

# Create figure with two subplots
plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# First subplot: Frame with cross-section points
ax1.imshow(frame_rgb)
ax1.plot([xsections["section1"]["xl"], xsections["section1"]["xr"]], 
         [xsections["section1"]["yl"], xsections["section1"]["yr"]], 
         color='#F5BF61', linewidth=2)  # Line connecting points
ax1.plot(xsections["section1"]["xl"], xsections["section1"]["yl"], 'o', color='#ED6B57', markersize=3)  # XS Left point
ax1.plot(xsections["section1"]["xr"], xsections["section1"]["yr"], 'o', color='#62C655', markersize=3)  # XS Right point
ax1.plot(x1_pt_pix, y1_pt_pix, 'o', color='blue', markersize=3)  # PT
ax1.text(x1_pt_pix, y1_pt_pix, "PT", color='blue', fontsize=8, ha='left', va='bottom')

ax1.set_title('Cross-Section and PT Location')
ax1.axis('off')

# Second subplot: Bathymetry profile
ax2.plot(stations, stages, 'k-', linewidth=2, label='River bed')
ax2.axhline(y=xsections["section1"]["level"], color='#6CD4FF', linestyle='--', label='Water level')
ax2.fill_between(stations, stages, xsections["section1"]["level"], 
                where=(stages <= xsections["section1"]["level"]), 
                alpha=0.3, color='#6CD4FF', label='Wet area')


# ---- ADD PRESSURE TRANSDUCER POINT ----
pt_x = x1_pt_rw                         # horizontal coordinate
pt_y = xsections["section1"]["level"] - depth_avg   # elevation from depth

ax2.plot(pt_x, pt_y, 'o', color='blue', markersize=6, label='PT')
ax2.text(pt_x, pt_y, "PT", color='blue', fontsize=9, ha='left', va='bottom')
# ----------------------------------------

ax2.grid(True)
ax2.set_xlabel('Distance from left bank (m)')
ax2.set_ylabel('Elevation (m)')
ax2.set_title('Bathymetry Profile')
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# Save bathymetry image
bath_img = bathy_dir / gcp_cam / (f"{gcp_cam}_bath_{gcp_date}_{gcp_time}.png")
plt.savefig(bath_img)

# %% [markdown]
# # Step 5: PIV Analysis
#
# This steps configures and performs the Particle Image Velocimetry (PIV) analysis using RIVeR.
# - Sets interrogation window parameters
# - Calculates optimal ROI height
# - Creates masks for analysis regions from selected cross-section
# - Runs PIV analysis
#
# **Prerequisites**
#
# - Completed cross-section definition
# - Extracted video frames ready for analysis
# - Transformation matrix and cross-section data saved
#
# **Parameters**
# - `interrogation_area_1`: First pass window size should be larger to capture larger displacements
# - `interrogation_area_2`: Second pass window size should be smaller for better spatial resolution
# - `overlap`: Window overlap determines the density of velocity vectors
# - `window_size`: Base window size for height calculation

# %%
interrogation_area_1 = 128  # Size of first interrogation window
interrogation_area_2 = 64  # Size of second interrogation window
overlap = 64             # Size overlap
window_size = 32           # Base window size for height calculation

# %%
gcp_cam = "chamb_02"
gcp_date = "20260223"         # in format YYYYMMDD
gcp_time = "120000"           # in format HHMMSS

# %%
# Load transformation matrix
transf_file = rect_dir / gcp_cam / (f"{gcp_cam}_transform_{gcp_date}_{gcp_time}.json")
with open(transf_file, 'r') as f:
    transformation_matrix = np.array(json.load(f))

# Load cross-sections data
sect_file = bathy_dir / gcp_cam / (f"{gcp_cam}_xs_{gcp_date}_{gcp_time}.json")
with open(sect_file, 'r') as f:
    xsections = json.load(f)

# Load image
frame,frame_rgb,frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time)
img = mpimg.imread(str(frame_path))

# Load and display frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(frame_rgb)
plt.axis('off')
plt.title('Analysis Frame')
plt.show()

# %%
# Define PIV analysis parameters
piv_params = {
    "interrogation_area_1": interrogation_area_1,  # Size of first interrogation window
    "interrogation_area_2": interrogation_area_2,   # Size of second interrogation window
    "overlap": overlap,              # Size overlap
    "num_stations": xsections["section1"]["num_stations"],
    "window_size": window_size           # Base window size for height calculation
}

print("PIV Analysis Parameters:")
print(f"First pass window size: {piv_params['interrogation_area_1']} pixels")
print(f"Second pass window size: {piv_params['interrogation_area_2']} pixels")
print(f"Window overlap: {piv_params['overlap']/piv_params['interrogation_area_1']*100}%")
print(f"Number of stations: {piv_params['num_stations']}")

# %% [markdown]
# ## Step 5.1: Analysis Mask and Bounding Box
#
# Create a mask and bounding box to optimize PIV analysis performance and focus on relevant areas:
#
# **ROI (Region of Interest) Bounding Box**: 
#   - Defines a rectangular region that encompasses all cross-sections
#   - Significantly reduces computation time by limiting PIV analysis to only this region
#   - All areas outside this box are excluded from processing entirely
#
# **Analysis Mask**:
#   - Further refines the analysis area within the ROI
#   - White areas (mask value = 1) indicate regions where PIV calculations will be retained
#   - Black areas (mask value = 0) indicate regions where PIV results will be filtered out
#   - Helps eliminate spurious velocities from areas not relevant to the flow analysis
#   - Particularly useful for removing:
#     - Bank areas
#     - Vegetation
#     - Static objects
#     - Areas outside the water surface
#
# The combination of ROI and mask ensures that:
# 1. Processing time is minimized by focusing only on relevant areas
# 2. Memory usage is optimized by excluding unnecessary regions
# 3. Final results contain only meaningful velocity measurements from the areas of interest

# %%
# Calculate recommended ROI height
height_roi = recommend_height_roi(
    xsections,
    piv_params["interrogation_area_1"],
    transformation_matrix
)

print(f"\nRecommended ROI height: {height_roi:.2f} meters")

# %%
# Plot analysis mask and bounding box
# %matplotlib inline

# Create mask and get bounding box
mask, bbox = create_mask_and_bbox(
    frame,
    xsections,
    transformation_matrix,
    height_roi
)

plt.close()

# Visualize mask overlaid on the frame
plt.figure(figsize=(12, 8))

# Display the original frame first
plt.imshow(frame_rgb)

# Create black overlay for the entire image
overlay = np.zeros(frame_rgb.shape[:2])  # Single channel black overlay
plt.imshow(overlay, alpha=0.1, cmap='gray')  # Apply with 0.1 opacity

# Remove the overlay where mask is 1
overlay_mask = np.ones(frame_rgb.shape[:2]) * 0.5  # Start with 0.5 opacity everywhere
overlay_mask[mask == 1] = 0  # Make completely transparent where mask is 1
plt.imshow(overlay, alpha=overlay_mask, cmap='gray')


# Add the bounding box with dashed light blue lines
rect = plt.Rectangle(
    (bbox[0], bbox[1]), bbox[2], bbox[3],
    linewidth=2, 
    edgecolor='#6CD4FF',  # Light blue color
    facecolor='none',
    linestyle='--'        # Dashed line
)
plt.gca().add_patch(rect)

# Add the Cross Section
plt.plot([xsections["section1"]["xl"], xsections["section1"]["xr"]], 
         [xsections["section1"]["yl"], xsections["section1"]["yr"]], 
         color='#F5BF61', linewidth=2)  # Line connecting points
plt.plot(xsections["section1"]["xl"], xsections["section1"]["yl"], 'o', color='#ED6B57', markersize=10)  # Left point
plt.plot(xsections["section1"]["xr"], xsections["section1"]["yr"], 'o', color='#62C655', markersize=10)  # Right point

plt.title('Frame with Analysis Region and Mask')
plt.axis('off')
plt.tight_layout()

# Save analysis region and mask image
piv_reg_img = piv_dir / gcp_cam / f"{gcp_cam}_piv_mask_{gcp_date}_{gcp_time}.png"
piv_reg_img.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(piv_reg_img)

plt.show()

print("\nBounding Box Parameters:")
print(f"x: {bbox[0]:.1f}")
print(f"y: {bbox[1]:.1f}")
print(f"width: {bbox[2]:.1f}")
print(f"height: {bbox[3]:.1f}")

# %% [markdown]
# ## Step 5.2: Test PIV Analysis
#
# Performs a test PIV analysis on a pair of consecutive frames to verify our configuration and visualize the results. This test will help us validate our ROI and mask settings before running the full analysis.
#
# Important: The PIV analysis at this stage produces a displacement field measured in pixels - this represents how far features have moved between the two frames. These displacements are not yet true velocities (which would require:
# - Conversion from pixel space to real-world coordinates using our transformation matrix
# - Division by the time interval between frames to get velocity units (e.g., m/s)
#
# For this test, we'll use default values for the optional parameters. Here are the available options:
#
# - `mask_auto` (default=True): Automatically applies a Gaussian filter to limit peak search area
# - `multipass` (default=True): Performs multiple passes to improve accuracy, using the first pass result to guide the second
# - `standard_filter` (default=True): Removes outliers based on standard deviation of velocities
# - `standard_threshold` (default=4): Number of standard deviations for outlier detection
# - `median_test_filter` (default=True): Additional outlier removal using local median test
# - `epsilon` (default=0.02): Tolerance parameter for median test filtering
# - `threshold` (default=2): Threshold for normalized fluctuations in median test
# - `filter_grayscale` (default=True): Converts images to grayscale before processing
# - `filter_clahe` (default=True): Applies Contrast Limited Adaptive Histogram Equalization
# - `clip_limit_clahe` (default=5): Upper limit for contrast enhancement in CLAHE
#
# This test will help us validate that our PIV settings are appropriate before proceeding with the full analysis and conversion to real-world velocities.

# %%
# Test PIV

# %matplotlib inline

frame_dir = frames_dir / (f"{gcp_cam}/{gcp_date}/120000")
print(frame_dir)

# Get two consecutive frames
frames = sorted(frame_dir.glob("*.jpg"))
frame1_path = frames[0]
frame2_path = frames[1]

print(frame1_path)
print(frame2_path)

# Run PIV test
piv_results = run_test(
    image_1=frame1_path,
    image_2=frame2_path,
    mask=mask,
    bbox=bbox,
    interrogation_area_1=piv_params["interrogation_area_1"],
    interrogation_area_2=piv_params["interrogation_area_2"]
)

plt.close()

# Plot results
plt.figure(figsize=(12, 8))

# Display the first frame
frame1 = cv2.imread(str(frame1_path))
frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
plt.imshow(frame1_rgb)

# Reshape results for quiver plot
x = np.array(piv_results['x']).reshape(piv_results['shape'])
y = np.array(piv_results['y']).reshape(piv_results['shape'])
u = np.array(piv_results['u']).reshape(piv_results['shape'])
v = np.array(piv_results['v']).reshape(piv_results['shape'])

# Display the mask
plt.imshow(overlay, alpha=overlay_mask, cmap='gray')

# Create quiver plot with all displacement vectors
# Create quiver plot with all displacement vectors
# Note: We use -v because the image coordinate system has y-axis inverted
# In images, y increases downward, while in plotting y increases upward
# This inversion ensures the vectors point in the correct physical direction
plt.quiver(x, y, u, -v,color='blue')


plt.title('PIV Test Results')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5.3: Full PIV Analysis and Results Saving
#
# Now that we've validated our PIV configuration through testing, we'll perform the complete analysis on all frames in our dataset. This step:
# 1. Processes all image pairs in the sequence
# 2. Computes median displacement fields
# 3. Visualizes the results
# 4. Saves the analysis output for later use
#
# **Key Elements of Full Analysis**
#
# - **Multiple Frame Processing**: Unlike our test which used just two frames, this analyzes all sequential frame pairs
# - **Median Statistics**: Computes statistical measures across all frames to provide:
#   - Median displacements (more robust than mean)
#   - Temporal variations in the flow field
#   - Gradient information for seeding quality assessment
#
# **Output Data Structure**
#
# The `piv_results` dictionary contains:
# - `shape`: Dimensions of the velocity field grid
# - `x`, `y`: Coordinate arrays for vector positions
# - `u_median`, `v_median`: Median displacement components
# - `u`, `v`: Full displacement time series
# - `gradient`: Seeding quality metrics
#
# **Important Notes**
#
# 1. **Processing Time**: Full analysis may take several minutes depending on:
#    - Number of frames
#    - Size of ROI
#    - Computer processing power
#
# 2. **Memory Usage**: Large datasets may require significant RAM
#    - Monitor system resources during processing
#    - Consider reducing ROI size if memory issues occur
#
# 3. **Vector Interpretation**:
#    - Displacements are still in pixel units
#    - Negative v-values are plotted inverted due to image coordinate system
#    - Vectors show median pattern across all frames
#
# 4. **Data Storage**:
#    - Results are saved in JSON format
#    - Large datasets may create substantial files
#    - Consider compression for long-term storage
#
# The saved results will be used in subsequent notebooks for:
# - Conversion to real-world velocities
# - Discharge calculations

# %%
# Run full PIV Analisis
piv_results = run_analyze_all(
    frame_dir,
    mask=mask,
    bbox=bbox,
    interrogation_area_1=piv_params["interrogation_area_1"],
    interrogation_area_2=piv_params["interrogation_area_2"]
)

# %%
# Plot results
plt.figure(figsize=(10, 8))

# Display the first frame
plt.imshow(frame1_rgb)
x,y,u,v = np.array(piv_results['x']),np.array(piv_results['y']),np.array(piv_results['u_median']),np.array(piv_results['v_median'])

# Display the mask
plt.imshow(overlay, alpha=overlay_mask, cmap='gray')

# Create quiver plot with all displacement vectors
# Create quiver plot with all displacement vectors
# Note: We use -v because the image coordinate system has y-axis inverted
# In images, y increases downward, while in plotting y increases upward
# This inversion ensures the vectors point in the correct physical direction
plt.quiver(x, y, u, -v,color='blue')

plt.title('PIV Median Results')
plt.axis('off')
plt.tight_layout()

# Save piv results image
piv_res_img = piv_dir / gcp_cam / f"{gcp_cam}_piv_result_{gcp_date}_{gcp_time}.png"
piv_res_img.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(piv_res_img)
plt.show()
print(f"\nPIV results image saved to {piv_res_img}")

# Save piv results
piv_res_file = piv_dir / gcp_cam / f"{gcp_cam}_piv_result_{gcp_date}_{gcp_time}.json"
with open(piv_res_file, 'w') as f:
    json.dump(piv_results, f, indent=2)
print(f"\nPIV results data saved to {piv_res_file}")

# %% [markdown]
# # Step 6: Discharge Calculation
#
# This step calculates river discharge using PIV results and cross-section data. We'll convert pixel displacements to real-world velocities and combine them with bathymetry data to compute volumetric flow rates.
#
# **Theory**
#
# 1. **Discharge Calculation**:
#    - Q = V × A (Velocity × Area)
#    - Integration across cross-section
#    - Depth-averaged velocity estimation
#
# 2. **Velocity Components**:
#    - Conversion from pixel to real-world coordinates
#    - Consideration of camera frame rate
#    - Alpha coefficient for surface-to-depth velocity ratio
#
# 3. **Cross-section Elements**:
#    - Bathymetry profile
#    - Station spacing
#    - Water level
#
#
# **Essential Parameters:**
# - `fps`: Video capture frequency from video exif(e.g., 30 fps)
# - `step`: Number of frames between PIV pairs from frames extraction (affects time between velocity measurements)
# - **Alpha Coefficient**: PREVIOUSLY DEFINED
# - **Number of Stations**: PREVIOUSLY DEFINED
# - **Interpolation**: Whether to fill data gaps using interpolation
#
# **Optional Parameters:**
# - **Artificial Seeding**: Whether the tracer used was artificialy seeded
# - **Multipass**: Use multiple PIV passes for improved accuracy
# - **Standard Filter**: Apply standard deviation filtering to velocity measurements
# - **Median Test Filter**: Remove outliers using median comparison
#
# **Function: update_current_x_section**
#
# This function performs three main tasks:
# 1. Updates velocity profiles for the cross-section
# 2. Calculates discharge using the configured parameters
# 3. Returns statistical summaries including:
#    - Total discharge (Q)
#    - Average velocity
#    - Cross-sectional area
#    - Depth
#
# -----------------------------------------------------------
# **Visualization**
# -----------------------------------------------------------
#
# **Left Panel: Spatial Visualization**
# - Frame from the video showing the cross-section location
# - Color-coded endpoints (red: left bank, green: right bank)
# - Velocity vectors scaled and colored by magnitude
#
# **Right Panels: Quantitative Analysis**
#
# 1. **Discharge Distribution** (Top)
#    - Bar plot showing proportion of total discharge along the cross-section
#    - Color-coded bars indicating contribution levels:
#      - Red: High contribution (>10% of total)
#      - Yellow: Medium contribution (5-10%)
#      - Green: Low contribution (<5%)
#
# 2. **Velocity Profile** (Middle)
#    - Green shaded area: ±1 standard deviation
#    - Red shaded area: 5th to 95th percentile range
#
# 3. **Depth Profile** (Bottom)
#    - Shows channel morphology along cross-section
#
# The visualization integrates spatial context with quantitative measurements, allowing for comprehensive interpretation of the flow characteristics at the cross-section.

# %%
# Define parameters (check from video metadata and frame extraction)
fps = 30
step = 1

# %%
gcp_cam = "chamb_02"
gcp_date = "20260223"         # in format YYYYMMDD
gcp_time = "120000"           # in format HHMMSS

########################

pt_name = "pt_02"

# %%
## Define paths

transformation_file = rect_dir / gcp_cam / f"{gcp_cam}_transform_{gcp_date}_{gcp_time}.json"
xsections_file = bathy_dir / gcp_cam / f"{gcp_cam}_xs_{gcp_date}_{gcp_time}.json"
piv_results_file = piv_dir / gcp_cam / f"{gcp_cam}_piv_result_{gcp_date}_{gcp_time}.json"

# %%
## Load data

# Load transformation matrix
with open(transformation_file, 'r') as f:
    transformation_matrix = np.array(json.load(f))

# Load cross-sections data
with open(xsections_file, 'r') as f:
    xsections = json.load(f)

# Load piv results data
with open(piv_results_file, 'r') as f:
    piv_results = json.load(f)

# Load the image
df_frames = pd.read_parquet(frames_dir/"_frame_paths.parquet")
frame, frame_rgb, frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time) # loads by default 0th frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Load PT img coord
pt_pix = load_pt_img(pt_name)
(x1_pt_pix, y1_pt_pix) = pt_pix[0]

# Load PT loc coord
pt_loc = load_pt_loc(pt_name)
(x1_pt_rw, y1_pt_rw) = pt_loc[0]

# Load PT depth
lvl = load_pt_dep(pt_name,gcp_date,gcp_time)

# %%
alpha = xsections['section1']['alpha']
num_stations = xsections['section1']['num_stations']
summary = update_current_x_section(xsections,
                                   piv_results,
                                   transformation_matrix,
                                   step,
                                   fps,
                                   0,
                                   alpha = alpha,
                                   num_stations= num_stations,
                                   interpolate=True)

# %%
# Visualize mask overlaid on the frame
plt.figure(figsize=(12, 8))

# Display the original frame first
plt.imshow(frame_rgb)
# Add the Cross Section
plt.plot([xsections["section1"]["xl"], xsections["section1"]["xr"]], 
         [xsections["section1"]["yl"], xsections["section1"]["yr"]], 
         color='#F5BF61', linewidth=2)  # Line connecting points
plt.plot(xsections["section1"]["xl"], xsections["section1"]["yl"], 'o', color='#ED6B57', markersize=10)  # Left point
plt.plot(xsections["section1"]["xr"], xsections["section1"]["yr"], 'o', color='#62C655', markersize=10)  # Right point

width_arrow = 0.8* np.mean(np.diff(summary['section1']['distance']))
arrows, magnitude_range = calculate_multiple_arrows(
    summary['section1']['east'],
    summary['section1']['north'],
    summary['section1']['filled_streamwise_velocity_magnitude'],
    transformation_matrix,
    frame_rgb.shape[0],
    width=width_arrow
)

# Plot each arrow
for arrow in arrows:
    plt.fill(arrow['points'][:, 0], arrow['points'][:, 1], color=arrow['color'],alpha=0.7)

plt.title('Frame with Velocity Profile')
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# Create figure with custom grid layout
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 1], height_ratios=[1, 1, 1])

# Left column: Frame with velocity arrows
ax0 = plt.subplot(gs[:, 0])  # Spans all rows in first column
ax0.imshow(frame_rgb)

# Plot cross-section line
ax0.plot([xsections["section1"]["xl"], xsections["section1"]["xr"]], 
         [xsections["section1"]["yl"], xsections["section1"]["yr"]], 
         color='#F5BF61', linewidth=2)

# Plot end points
ax0.plot(xsections["section1"]["xl"], xsections["section1"]["yl"], 
         'o', color='#ED6B57', markersize=10, label='Left bank')
ax0.plot(xsections["section1"]["xr"], xsections["section1"]["yr"], 
         'o', color='#62C655', markersize=10, label='Right bank')

# Plot PT point
ax0.plot(x1_pt_pix, y1_pt_pix, 'o', color='blue', markersize=5, label='PT')  # PT

# Calculate and plot velocity arrows
width_arrow = 0.8 * np.mean(np.diff(summary['section1']['distance']))
arrows, magnitude_range = calculate_multiple_arrows(
    summary['section1']['east'],
    summary['section1']['north'],
    summary['section1']['filled_streamwise_velocity_magnitude'],
    transformation_matrix,
    frame_rgb.shape[0],
    width=width_arrow
)

# Plot arrows
for arrow in arrows:
    ax0.fill(arrow['points'][:, 0], arrow['points'][:, 1], 
             color=arrow['color'], alpha=0.7)
ax0.set_title('Cross-section with Velocity Vectors', pad=20)
ax0.axis('off')
ax0.legend()

# Stats text
textstr_stat = (
    f"Total Discharge: {summary['section1']['total_Q']:.2f} ± {summary['section1']['total_q_std']:.2f} m³/s\n"
    f"Mean Velocity: {summary['section1']['mean_V']:.2f} m/s\n"
    f"Total Width: {summary['section1']['total_W']:.2f} m\n"
    f"Maximum Depth: {summary['section1']['max_depth']:.2f} m"
)

ax0.text(
    0.01, -0.05, textstr_stat,
    transform=ax0.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# Title text
textstr_tit = (
    f"Camera: {gcp_cam}\n"
    f"Date: {gcp_date}\n"
    f"Time: {gcp_time}"
)

ax0.text(
    0.01, 1.05, textstr_tit,
    transform=ax0.transAxes,
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# Right column plots
# 1. Discharge Distribution (as bars with colors based on values)
ax1 = plt.subplot(gs[0, 1])
bar_width = 0.7 * np.mean(np.diff(summary['section1']['distance']))

# Create color array based on values
colors = []
for q in summary['section1']['Q_portion']:
    if q > 0.1:
        colors.append('#ED6B57')
    elif q > 0.05:
        colors.append('#F5BF61')
    else:
        colors.append('#62C655')

ax1.bar(summary['section1']['distance'], 
        summary['section1']['Q_portion'],
        width=bar_width,
        color=colors,
        alpha=0.9,
        align='center')
ax1.set_title('Discharge Distribution')
ax1.set_ylabel('Proportion of Total Discharge')
ax1.grid(True, alpha=0.3)

# Plot vertical PT line
pt_x_1 = x1_pt_rw
pt_y_1 = 0
ax1.axvline(x=pt_x_1, color='blue', linestyle='--', linewidth=1, label='PT')

# Add legend for discharge colors
legend_elements = [
    Patch(facecolor='#ED6B57', alpha=0.9, label='> 0.1'),
    Patch(facecolor='#F5BF61', alpha=0.9, label='0.05 - 0.1'),
    Patch(facecolor='#62C655', alpha=0.9, label='< 0.05')
]
ax1.legend(handles=legend_elements, title='Discharge Proportion')

# 2. Velocity Profile - black line with points
ax2 = plt.subplot(gs[1, 1])
# Add 5th and 95th percentile area
ax2.fill_between(summary['section1']['distance'],
                 summary['section1']['5th_percentile'],
                 summary['section1']['95th_percentile'],
                 color='#ED6B57', alpha=0.3, label='5th-95th percentile')
# Add standard deviation area
ax2.fill_between(summary['section1']['distance'],
                 summary['section1']['minus_std'],
                 summary['section1']['plus_std'],
                 color='#62C655', alpha=0.3, label='±1 std')
ax2.plot(summary['section1']['distance'], 
         summary['section1']['filled_streamwise_velocity_magnitude'],
         'k-', linewidth=2)  # black line
ax2.plot(summary['section1']['distance'], 
         summary['section1']['filled_streamwise_velocity_magnitude'],
         'ko', markersize=6)  # black points

# Plot vertical PT line
pt_x_2 = x1_pt_rw
pt_y_2 = 0
ax2.axvline(x=pt_x_2, color='blue', linestyle='--', linewidth=1, label='PT')


ax2.set_title('Velocity Profile')
ax2.set_ylabel('Velocity (m/s)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Depth Profile - black line with light blue fill
ax3 = plt.subplot(gs[2, 1])
ax3.plot(summary['section1']['distance'], 
         summary['section1']['depth'],
         'k-', linewidth=2)  # black line
ax3.fill_between(summary['section1']['distance'],
                 summary['section1']['depth'],
                 color='#6CD4FF', alpha=0.5)  # light blue fill

# Plot vertical PT line
pt_x_3 = x1_pt_rw
pt_y_3 = lvl
ax3.axvline(x=pt_x_3, color='blue', linestyle='--', linewidth=1, label='PT')


ax3.set_title('Depth Profile')
ax3.set_xlabel('Distance from Left Bank (m)')
ax3.set_ylabel('Depth (m)')
ax3.grid(True, alpha=0.3)
ax3.invert_yaxis()  # Invert y-axis to show depth properly

ax1.get_shared_x_axes().join(ax1, ax2, ax3)

# Add overall title and adjust layout
plt.suptitle('Cross-section Analysis Summary', y=1.02, fontsize=16)
plt.tight_layout()

# Save discharge results image
dis_res_img = disch_dir / gcp_cam / f"{gcp_cam}_dis_img_{gcp_date}_{gcp_time}.png"
dis_res_img.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(dis_res_img)

plt.show()

# Print numerical summary
print(f"Total Discharge: {summary['section1']['total_Q']:.2f} ± {summary['section1']['total_q_std']:.2f} m³/s")
print(f"Mean Velocity: {summary['section1']['mean_V']:.2f} m/s")
print(f"Total Width: {summary['section1']['total_W']:.2f} m")
print(f"Maximum Depth: {summary['section1']['max_depth']:.2f} m")

# Print saving info
print("Discharge results image saved to:")
print(dis_res_img)

# %%
# Save results metadata and data
sec1 = summary["section1"]

# Identify which keys are arrays (per-station) vs scalars (metadata)
array_like_keys = []
scalar_keys = []

for k, v in sec1.items():
    if isinstance(v, (list, tuple, np.ndarray)):
        array_like_keys.append(k)
    else:
        scalar_keys.append(k)

# A) Metadata: one row with scalar fields
meta = {k: sec1[k] for k in scalar_keys}
# Keep a pointer to the original paths, or replace with relative path if desired
df_meta = pd.DataFrame([meta])

# B) Stations table: stack arrays into rows
# Ensure consistent lengths
lengths = {k: len(sec1[k]) for k in array_like_keys}
if len(set(lengths.values())) != 1:
    raise ValueError(f"Inconsistent array lengths in per-station arrays: {lengths}")

df_stations = pd.DataFrame({k: sec1[k] for k in array_like_keys})

# Optionally, add contextual fields for joins/filtering
for extra in ["east_l", "north_l", "east_r", "north_r", "level", "alpha"]:
    if extra in sec1:
        df_meta[extra] = sec1[extra]  # stored once in meta
# If you want to “carry” identifiers in the stations table
df_stations.insert(0, "section", "section1")

# Save both
dis_res_data = (disch_dir / gcp_cam)

# Parquet engines: 'pyarrow' (recommended) or 'fastparquet'
engine = "pyarrow"

dis_res_meta_file = dis_res_data / f"{gcp_cam}_dis_meta_{gcp_date}_{gcp_time}."
df_meta.to_parquet(f"{dis_res_meta_file}parquet", index=False, engine=engine)
df_meta.to_csv(f"{dis_res_meta_file}csv", index=False)
print(f"Saved discharge metadata to {dis_res_meta_file}*")

dis_res_data_file = dis_res_data / f"{gcp_cam}_dis_data_{gcp_date}_{gcp_time}."
df_stations.to_parquet(f"{dis_res_data_file}parquet", index=False, engine=engine)
df_stations.to_csv(f"{dis_res_data_file}csv", index=False)
print(f"Saved discharge data to {dis_res_data_file}*")

# %%
# Save results summary
agg = summary["summary"]

df_agg_wide = []
for stat_name, metrics in agg.items():
    row = {"stat": stat_name, **metrics}
    df_agg_wide.append(row)
df_agg_wide = pd.DataFrame(df_agg_wide)
df_agg_wide.insert(0, "section", "section1")

dis_res_sum = (disch_dir / gcp_cam)

dis_res_sum_file = dis_res_sum/ f"{gcp_cam}_dis_sum_{gcp_date}_{gcp_time}."
df_agg_wide.to_parquet(f"{dis_res_sum_file}parquet", index=False, engine=engine)
df_agg_wide.to_csv(f"{dis_res_sum_file}csv", index=False)
print(f"Saved discharge summary to {dis_res_sum_file}*")

# %%
# Load results data
df_meta = pd.read_parquet(dis_res_data / f"{gcp_cam}_dis_meta_{gcp_date}_{gcp_time}.parquet")
df_data = pd.read_parquet(dis_res_data / f"{gcp_cam}_dis_data_{gcp_date}_{gcp_time}.parquet")
df_sum = pd.read_parquet(dis_res_data / f"{gcp_cam}_dis_sum_{gcp_date}_{gcp_time}.parquet")

# %%
df_sum

# %%
# to do:
# 2) Make Step 3 (orthrectification) dynamic
# 3) Make Step 4 (cross section selection) dynamic
# 4) Make Steo 5 (water depth and bathymetry) dynamic

# %%

# %%

# %%

# %%
## Overlay comparison

# %matplotlib widget
plt.ioff()  # 🔴 prevent Matplotlib auto-displaying the figure

###################################################
gcp_cam = "ilh-cam1-pt"
gcp_date = "20250426"         # in format YYYYMMDD
gcp_time = "120000"           # in format HHMMSS

_,_,frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time)

img1 = mpimg.imread(str(frame_path))
###################################################
gcp_cam = "le5-cam1-pt"
gcp_date = "20250426"         # in format YYYYMMDD
gcp_time = "120000"           # in format HHMMSS

_,_,frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time)

img2 = mpimg.imread(str(frame_path))
###################################################

# --- Plot with alpha slider ---

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_title("Overlay comparison (foreground opacity)")
ax.axis("off")

bg = ax.imshow(img1)
fg = ax.imshow(img2, alpha=0.5)  # start half transparent

alpha_slider = FloatSlider(
    value=0.5, min=0.0, max=1.0, step=0.1,
    description='Date 1', continuous_update=True, readout=False
)

def on_alpha_change(change):
    fg.set_alpha(change['new'])
    fig.canvas.draw_idle()

alpha_slider.observe(on_alpha_change, names='value')
alpha_label = Label("Date 2")
slider_row = HBox([alpha_slider, alpha_label])


ui = VBox([slider_row, fig.canvas])
display(ui)  # ✅ show exactly once

# %%
# Export auxiliary imagery of later dates to check stable camera position

gcp_check_dates = ["20250426","20250427"]         # in format YYYYMMDD
gcp_check_time = "130000"                         # in format HHMMSS

# Function that loads auxiliary imagery paths
def get_gcp_frame_paths(
    df_frames,
    camera: str,
    dates: list[str],
    start_time: str,  # "HHMMSS"
    target_basename: str = "0000000000.jpg"
) -> list[str | None]:
    """
    For each date in `dates`, returns the frame_path matching:
      camera == camera
      date == normalized YYYYMMDD
      clock == HHMMSS-(HHMMSS+59min)
      basename == target_basename
    If a given date has no match, returns None at that position.
    """
    norm_dates = [str(d).strip().zfill(8) for d in dates]
    check_clock_exact = gcp_check_time+"-"+str(int(gcp_check_time)+5900)

    # Pre-filter by camera/date/clock for speed
    df_check = df_frames[
        (df_frames["camera"] == camera) &
        (df_frames["date"].astype(str).isin(norm_dates)) &
        (df_frames["clock_start"] == gcp_check_time)
    ].copy()

    # Compute lowercase basenames
    df_check["basename"] = (
        df_check["frame_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .apply(lambda p: Path(p).name.lower())
    )

    # Filter to target basename only
    hits = df_check[df_check["basename"].eq(target_basename.lower())].copy()

    # Build a lookup (date -> list of paths); usually 1 per date
    by_date = {}
    for _, row in hits.iterrows():
        d = str(row["date"]).strip().zfill(8)
        by_date.setdefault(d, []).append(str(Path(row["frame_path"]).as_posix()))

    # Return results aligned with input dates, using None when missing
    check_results = []
    for d in norm_dates:
        paths_for_d = by_date.get(d, [])
        check_results.append({
            "date": d,
            "clock_start": clock_start,
            "frame_path": paths_for_d[0] if paths_for_d else None
        })
    
    return pd.DataFrame(check_results)


# Load auxiliary imagery paths
check_results = get_gcp_frame_paths(df_frames, gcp_cam, gcp_check_dates, gcp_check_time)
print("Following auxiliary imagery has been exported:")

# Save auxiliary imagery
n_aux = 1
for ch_dat, ch_clo, ch_pat in zip(check_results.date,check_results.clock_start,check_results.frame_path):
    img = mpimg.imread(str(ch_pat))
    plt.close()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.set_title(f"Select GCPs:\n1) left upstream\n2) right upstream\n3) right downstreamm\n4) left downstream\n\n{ch_pat}")
    ax.axis("off")
    plt.tight_layout()
    
    check_img = gcps_dir / (f"{gcp_cam}_gcps_img_{ch_dat}_{ch_clo}_{n_aux:02d}_aux.png")
    n_aux = n_aux + 1
    print(check_img)
    fig.savefig(str(check_img))
    plt.close()
