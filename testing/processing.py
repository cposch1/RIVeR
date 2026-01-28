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
# # Step 1: Metadata Extraction
#
# Extract video metadata

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
def scan_videos_and_write_csvs(
    videos_root: Path,
    suffix: str = ".avi",
    overwrite: bool = True,
) -> dict:
    """
    Scan videos under `videos_root` expecting naming convention:
      {camera}_{date}-{clockstart}-{clockend}.avi

    Creates one CSV per camera:  {camera}_meta.csv  in  out_root / "meta"

    CSV columns: date, clock, total_frames, fps, resolution, estimated_size_gb
    Returns a dict: camera_name -> path_to_csv
    """
    videos_root = videos_root.resolve()

    # Regex: capture camera, date (YYYYMMDD), start/end (hhmmss), allowing '_' or '-' as separators
    pat = re.compile(r'^(?P<camera>.+?)_(?P<date>\d{8})-(?P<start>\d{6})-(?P<end>\d{6})\.avi$',re.IGNORECASE)

    # Arrays that gather rows and error rows per camera
    rows_per_camera = {}
    errors_per_camera = {}

    # Loop directories to extract info
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
        clock = f"{clock_start}-{clock_end}"

        # Extract video info; on error, log to per-camera error CSV accumulator
        try:
            info = check_video_info(path)
        
        except Exception as e:
            errors_per_camera.setdefault(camera, []).append({
                "date": date,
                "clock": clock,
                "error_message": str(e),
                "estimated_size_gb": round(os.path.getsize(path) / (1024**3),2),
                "path": str(path.relative_to(videos_root)),
            })
            continue

        row = {
            "date": date,
            "clock": clock,  # combined as requested; if you want separate cols, add them too
            "total_frames": info["total_frames"],
            "fps": info["fps"],
            "resolution": info["resolution"],
            "estimated_size_gb": round(info["estimated_size_gb"],2),
            "path": str(path.relative_to(videos_root)),
        }
        rows_per_camera.setdefault(camera, []).append(row)

    # Write meta CSVs
    camera_to_csv = {}
    for camera, rows in rows_per_camera.items():
        # Sort by date then start time inside 'clock'
        rows.sort(key=lambda r: (r["date"], r["clock"]))

        csv_path = video_dir / f"_{camera}_meta.csv"
        camera_to_csv[camera] = csv_path
        write_header = overwrite or not csv_path.exists()
        mode = "w" if write_header else "a"
        
        with csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["date", "clock", "total_frames", "fps", "resolution", "estimated_size_gb", "path"],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        
    # Write error CSVs
    err_to_csv = {}
    for camera, rows in errors_per_camera.items():
        # Sort by date then clock for consistency
        rows.sort(key=lambda r: (r["date"], r["clock"]))

        err_csv_path = video_dir / f"_{camera}_error_log.csv"
        err_to_csv[camera] = err_csv_path
        err_write_header = overwrite or not err_csv_path.exists()
        mode = "w" if err_write_header else "a"
        
        with err_csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["date", "clock", "error_message", "estimated_size_gb", "path"],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    if not rows_per_camera and not errors_per_camera:
        # No videos matched or everything skipped silently; keep quiet per your requirement
        pass

    return camera_to_csv, err_to_csv

# Run the functions
camera_csvs, err_csvs = scan_videos_and_write_csvs(video_dir, suffix=".avi", overwrite=True)

print(f"DONE.\n\nFollowing metadata files have been created:\n{camera_csvs}\n\nFollowing error log files have been created:\n{err_csvs}")

# %% [markdown]
# # Step 2: Frame Extraction
#
# Extract video frames
# - `every`: Extract every nth frame (e.g., every=2 takes every second frame)
# - `start_frame_number`: Begin extraction from this frame
# - `end_frame_number`: Stop extraction at this frame
# - `chunk_size`: Number of frames per processing chunk (affects memory usage)

# %%
#########################
### DEFINE PARAMETERS ###
#########################

start_frame_number = 0
end_frame_number = None    # process all frames
every = 25 

#########################

# File parsing
VID_NAME_RE = re.compile(
    r'^(?P<camera>.+?)_(?P<date>\d{8})-(?P<start>\d{6})-(?P<end>\d{6})\.(?P<ext>avi|mp4)$',
    re.IGNORECASE
)

# Function that gets video names
def parse_video_name(video_path: Path):
    """
    Parse the video filename and return (camera, date, clock)
    where clock is 'hhmmss-hhmmss'.
    """
    m = VID_NAME_RE.match(video_path.name)
    if not m:
        raise ValueError(f"Video filename does not match expected pattern: {video_path.name}")
    camera = m.group("camera")
    date = m.group("date")
    start = m.group("start")
    end = m.group("end")
    clock = f"{start}-{end}"
    return camera, date, clock


# Function that creates directories
def target_frames_dir_for(video_path: Path, base_frames_dir: Path) -> Path:
    """
    frames/<camera>/<date>/<clock>/
    """
    camera, date, clock = parse_video_name(video_path)
    return base_frames_dir / camera / date / clock


# Function that loops over videos (skipping AppleDouble artifacts)
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


# Function that extracts only valid videos
def load_allowed_from_meta(video_root: Path) -> set[tuple[str, str, str]]:
    allowed = set()
    video_root = Path(video_root)
    for meta_csv in video_root.glob("_*_meta.csv"):
        name = meta_csv.name
        # Ensure pattern _{camera}_meta.csv
        if not (name.startswith("_") and name.endswith("_meta.csv")):
            continue
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


# Extracting frames from valid videos that appear in the metadata list
allowed_triples = load_allowed_from_meta(video_dir)

processed = 0
skipped = 0

for vp in tqdm(list(iter_videos(video_dir)), desc="Extracting frames"):
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

print(f"\nDONE.\nProcessed: {processed} video(s).\nSkipped: {skipped} video(s).")


# %%
# Function that constructs paths to frames
def collect_frame_paths(frame_dir: Path) -> pd.DataFrame:
    """
    Walks a frames directory structured like:
    
        frame_dir / camera / date / clock / *.jpg
        
    and returns a DataFrame with columns:
        camera, date, clock, frame_path
    """
    all_rows = []

    frame_dir = frame_dir.resolve()

    # Loop structure: camera → date → clock → frames
    for camera_dir in frame_dir.iterdir():
        if not camera_dir.is_dir():
            continue
        camera = camera_dir.name

        for date_dir in camera_dir.iterdir():
            if not date_dir.is_dir():
                continue
            date = date_dir.name  # expect YYYYMMDD

            for clock_dir in date_dir.iterdir():
                if not clock_dir.is_dir():
                    continue
                clock = clock_dir.name  # expect hhmmss-hhmmss

                # Now collect all JPG files in this folder
                for jpg in clock_dir.glob("*.jpg"):
                    all_rows.append({
                        "camera": camera,
                        "date": date,
                        "clock": clock,
                        "frame_path": jpg.relative_to(frame_dir.parent.parent)
                    })

    return pd.DataFrame(all_rows)

df_frames = collect_frame_paths(frames_dir)

# %% [markdown]
# # Step 3: Orthrectification

# %% [markdown]
# ### Repeat lines until "End of Step 3" for each station

# %%
#########################
### DEFINE PARAMETERS ###
#########################

gcp_cam = "ilh-cam1-pt"
gcp_date = "20250426"         # in format YYYYMMDD
gcp_time = "120000"           # in format HHMMSS

#########################

# Function that loads the frame image
def load_frame(df_frames,gcp_cam,gcp_date,gcp_time):
    clock_exact = gcp_time+"-"+str(int(gcp_time)+5900)
    df_sub = df_frames[
        (df_frames["camera"] == gcp_cam) &
        (df_frames["date"] == gcp_date) &
        (df_frames["clock"] == clock_exact)
    ].copy()
    
    if df_sub.empty:
        raise FileNotFoundError("Camera/date/time not valid")
    
    df_sub["basename"] = (
        df_sub["frame_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .apply(lambda p: Path(p).name.lower())
    )
    hits = df_sub[df_sub["basename"].eq("0000000000.jpg")]
    if hits.empty:
        raise FileNotFoundError("No '0000000000.jpg' in that segment")
    
    frame_path = str(Path(hits.iloc[0]["frame_path"]).as_posix())
    frame = cv2.imread(str(frame_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, frame_rgb, frame_path

frame, frame_rgb, frame_path = load_frame(df_frames,gcp_cam,gcp_date,gcp_time)

# %%
# Display image and select GCPs

# %matplotlib widget
plt.close()

img = mpimg.imread(str(frame_path))

points = []

fig, ax = plt.subplots(figsize=(14,10))
ax.imshow(img)
ax.set_title(f"Select GCPs:\n1) left upstream\n2) right upstream\n3) right downstreamm\n4) left downstream\n\n{frame_path}")
plt.axis("off")

def onclick(event):
    # Ensure click is inside image
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: x={x}, y={y}")
        points.append((x, y))
        
        n = len(points)
        if n == 1:
            ax.plot(x, y, 'o', color='#ED6B57', markersize=3)  # first point red
        elif n in (2, 3):
            ax.plot(x, y, 'o', color='#6CD4FF', markersize=3)  # 2nd & 3rd blue
        elif n == 4:
            ax.plot(x, y, 'o', color='#6CD4FF', markersize=3)  # 4th blue
            fig.canvas.draw()
            fig.canvas.mpl_disconnect(cid)
            print("4 points collected:", points)
            return


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
else:
    print(f"Selected image coordinates (X/Y):\n{points}")

    gcps_file = gcps_dir / (f"{gcp_cam}_gcps_img_{gcp_date}_{clock_exact}.csv")
    with open(gcps_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(points)
    print(f"\nImage coordinates saved to:\n{gcps_file}")
    
    gcps_img = gcps_dir / (f"{gcp_cam}_gcps_img_{gcp_date}_{clock_exact}_main.png")
    plt.savefig(str(gcps_img))
    plt.close()
    print(f"\nImage coordinates selection saved to:\n{gcps_img}")

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
        (df_frames["clock"] == check_clock_exact)
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
            "clock": clock_exact,
            "frame_path": paths_for_d[0] if paths_for_d else None
        })
    
    return pd.DataFrame(check_results)


# Load auxiliary imagery paths
check_results = get_gcp_frame_paths(df_frames, gcp_cam, gcp_check_dates, gcp_check_time)
print("Following auxiliary imagery has been exported:")

# Save auxiliary imagery
for ch_dat, ch_clo, ch_pat in zip(check_results.date,check_results.clock,check_results.frame_path):
    img = mpimg.imread(str(ch_pat))
    plt.close()
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(img)
    ax.set_title(f"Select GCPs:\n1) left upstream\n2) right upstream\n3) right downstreamm\n4) left downstream\n\n{ch_pat}")
    ax.axis("off")
    plt.tight_layout()
    
    check_img = gcps_dir / (f"{gcp_cam}_gcps_img_{ch_dat}_{ch_clo}_aux.png")
    print(check_img)
    fig.savefig(str(check_img))
    plt.close()


# %%
# Load GCP image coordinates
points_img = []
with open(gcps_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        x = int(float(row[0]))
        y = int(float(row[1]))
        points_img.append((x, y))

point_img_keys = [f"point{i}" for i in range(1, len(points_img) + 1)]
point_coords_pixel = dict(zip(point_img_keys, points_img))
print(f"GCPs image coordinates:\n{point_coords_pixel}")

# %%
# Load GCP real world coordinates
points_real = []
gcps_real_file = gcps_dir / f"{gcp_cam}_gcps_real.csv"
with open(gcps_real_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        x = int(float(row[0]))
        y = int(float(row[1]))
        points_real.append((x, y))

point_real_keys = [f"point{i}" for i in range(1, len(points_real) + 1)]
point_coords_world = dict(zip(point_real_keys, points_real))

# function for calculating euclidian distances
def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

distances = {
    'd12': dist(point_coords_world['point1'], point_coords_world['point2']),
    'd23': dist(point_coords_world['point2'], point_coords_world['point3']),
    'd34': dist(point_coords_world['point3'], point_coords_world['point4']),
    'd41': dist(point_coords_world['point4'], point_coords_world['point1']),
    'd13': dist(point_coords_world['point1'], point_coords_world['point3']),  # diagonal
    'd24': dist(point_coords_world['point2'], point_coords_world['point4'])   # diagonal
}
print("GCPs real world coordinates and distances:")
print(point_coords_world)
print(distances)

# %%
# Do test orthorectification and show/save test image

# Extract coordinates for transformation
x1_pix, y1_pix = point_coords_pixel['point1']
x2_pix, y2_pix = point_coords_pixel['point2']
x3_pix, y3_pix = point_coords_pixel['point3']
x4_pix, y4_pix = point_coords_pixel['point4']

# Calculate transformation matrix
transformation = oblique_view_transformation_matrix(
    x1_pix, y1_pix,
    x2_pix, y2_pix,
    x3_pix, y3_pix,
    x4_pix, y4_pix,
    distances['d12'],
    distances['d23'],
    distances['d34'],
    distances['d41'],
    distances['d13'],
    distances['d24'],
    image_path=frame_path,
)

transformation_matrix = transformation['transformation_matrix']

# Visualize the points on the frame
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# First subplot with original image and control points
ax1.imshow(img)
# Draw lines with specific colors
# Line 1-2
ax1.plot([x1_pix, x2_pix], [y1_pix, y2_pix], color='#6CD4FF', linewidth=2)
# Line 2-3
ax1.plot([x2_pix, x3_pix], [y2_pix, y3_pix], color='#62C655', linewidth=2)
# Line 3-4
ax1.plot([x3_pix, x4_pix], [y3_pix, y4_pix], color='#ED6B57', linewidth=2)
# Line 4-1
ax1.plot([x4_pix, x1_pix], [y4_pix, y1_pix], color='#F5BF61', linewidth=2)
# Diagonal 1-3
ax1.plot([x1_pix, x3_pix], [y1_pix, y3_pix], color='#CC4BC2', linewidth=2)
# Diagonal 2-4
ax1.plot([x2_pix, x4_pix], [y2_pix, y4_pix], color='#7765E3', linewidth=2)
# Plot points
# Point 1 in red
ax1.plot(x1_pix, y1_pix, 'o', color='#ED6B57', markersize=3)
# Points 2-4 in blue
ax1.plot([x2_pix, x3_pix, x4_pix], [y2_pix, y3_pix, y4_pix], 'o', color='#6CD4FF', markersize=3)
ax1.axis('off')
ax1.set_title('Original Image')  # Fixed from ax1.title to ax1.set_title

# Second subplot with orthorectified image
if 'transformed_img' in transformation and 'extent' in transformation:
    extent = transformation['extent']
    ax2.imshow(transformation['transformed_img'], extent=extent)
    
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
    ax2.add_patch(rect)
    
    # Add text label for the scale bar
    ax2.text(x_pos + scale_length_rounded/2, y_pos + 2*bar_height,
            f'{int(scale_length_rounded)} m',
            ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, pad=2))
    
    # Convert GCP pixel coordinates to real-world coordinates to display in the second image
    real_world_points = []
    for x, y in [(x1_pix, y1_pix), (x2_pix, y2_pix), (x3_pix, y3_pix), (x4_pix, y4_pix)]:
        rw_point = transform_pixel_to_real_world(x, y, transformation_matrix)
        real_world_points.append(rw_point)
    
    real_world_points = np.array(real_world_points)
    
    # Get individual point coordinates
    x1_rw, y1_rw = real_world_points[0]
    x2_rw, y2_rw = real_world_points[1]
    x3_rw, y3_rw = real_world_points[2]
    x4_rw, y4_rw = real_world_points[3]
    
    # Draw lines with the same colors as in the first plot
    # Line 1-2
    ax2.plot([x1_rw, x2_rw], [y1_rw, y2_rw], color='#6CD4FF', linewidth=2)
    # Line 2-3
    ax2.plot([x2_rw, x3_rw], [y2_rw, y3_rw], color='#62C655', linewidth=2)
    # Line 3-4
    ax2.plot([x3_rw, x4_rw], [y3_rw, y4_rw], color='#ED6B57', linewidth=2)
    # Line 4-1
    ax2.plot([x4_rw, x1_rw], [y4_rw, y1_rw], color='#F5BF61', linewidth=2)
    # Diagonal 1-3
    ax2.plot([x1_rw, x3_rw], [y1_rw, y3_rw], color='#CC4BC2', linewidth=2)
    # Diagonal 2-4
    ax2.plot([x2_rw, x4_rw], [y2_rw, y4_rw], color='#7765E3', linewidth=2)
    
    # Plot points
    # Point 1 in red
    ax2.plot(x1_rw, y1_rw, 'o', color='#ED6B57', markersize=3)
    # Points 2-4 in blue
    ax2.plot([x2_rw, x3_rw, x4_rw], [y2_rw, y3_rw, y4_rw], 'o', color='#6CD4FF', markersize=3)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Orthorectified Image')

plt.tight_layout()

ortho_check_file = gcps_dir / (f"{gcp_cam}_ortho_check_{ch_dat}_{ch_clo}.png")
plt.savefig(str(ortho_check_file))
plt.show()
#plt.close(fig)

# %%
# Save transformation
transf_file = gcps_dir / (f"{gcp_cam}_transform_{ch_dat}_{ch_clo}.json")
with open(transf_file, 'w') as f:
    json.dump(transformation_matrix, f, indent=1)
print(f"Transformation matrix saved to\n{transf_file}")

# Save raw orthorectified image
#ortho_file = gcps_dir / (f"{gcp_cam}_orthorect_{ch_dat}_{ch_clo}.png")
#plt.imsave(str(ortho_file), transformation['transformed_img'])

# %% [markdown]
# ### End of Step 3 (repeat for each station)

# %% [markdown]
# # Step 4: Cross Section Selection

# %%
# %matplotlib widget
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from ipywidgets import FloatSlider, VBox, Label, HBox

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
    value=0.5, min=0.0, max=1.0, step=0.01,
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
# Load transformation matrix
with open(transf_file, 'r') as f:
    transformation_matrix = np.array(json.load(f))

# %%
transformation

# %%
transformation_matrix

# %%
# %matplotlib widget

plt.close()
plt.figure(figsize=(14, 10))
ax = plt.gca()


extent = transformation['extent']
ax.imshow(transformation['transformed_img'], extent=extent)

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


ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title('Cross-section selection in orthorectified image')

#plt.tight_layout()

#mage_output_file = output_dir / "05_cross_sec_orthorect.png"
#plt.savefig(str(image_output_file))

# --- Interactive picking of two points in real-world coordinates ---
points_rw = []
point_coords_rw = {}  # {'point1': (x1, y1), 'point2': (x2, y2)}

# We'll store the variables you used in your other code:
x1_rw = y1_rw = x2_rw = y2_rw = None

def onclick(event):
    nonlocal_vars = ('x1_rw', 'y1_rw', 'x2_rw', 'y2_rw', 'point_coords_rw')  # for clarity in this cell
    # In a notebook cell, use 'global' to assign to top-level names:
    global x1_rw, y1_rw, x2_rw, y2_rw, point_coords_rw

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
        point_coords_rw = {
            (x1_rw, y1_rw),
            (x2_rw, y2_rw),
        }

        print("2 points collected (real-world coords):")
        print(point_coords_rw)

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

# %%
point_coords_rw

# %%
