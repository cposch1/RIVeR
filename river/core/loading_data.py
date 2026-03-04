from pathlib import Path
import pandas as pd
import csv
import cv2
from river.config import gcps_dir
from river.config import bathy_dir
from river.config import pts_dir
from river.config import dep_dir

# Function that loads the frame image
def load_frame(df_frames,gcp_cam,gcp_date,gcp_time):
    df_sub = df_frames[
        (df_frames["camera"] == gcp_cam) &
        (df_frames["date_yyyymmdd"] == gcp_date) &
        (df_frames["time_hhmmss"] == gcp_time)
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


# Function that loads GCP image coordinates
def load_gcps_img(gcp_cam,gcp_date,gcp_time):
    points_img = []
    gcps_img_file = gcps_dir / gcp_cam / (f"{gcp_cam}_gcps_img_{gcp_date}_{gcp_time}.csv")
    with open(gcps_img_file, newline="") as f:
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

    return point_coords_pixel


# Function that loads GCP real world coordinates
def load_gcps_real(gcp_cam):
    points_real = []
    gcps_real_file = gcps_dir / gcp_cam / f"{gcp_cam}_gcps_real.csv"
    with open(gcps_real_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x = float(row[0])
            y = float(row[1])
            points_real.append((x, y))
    
    point_real_keys = [f"point{i}" for i in range(1, len(points_real) + 1)]
    point_coords_world = dict(zip(point_real_keys, points_real))
    
    # Print real world coordinates
    print("GCPs real world coordinates:")
    print(point_coords_world)

    return point_coords_world


# Function that loads GCP real world distances
def load_dist(gcp_cam):
    gcps_dist_file = gcps_dir / gcp_cam / f"{gcp_cam}_gcps_dist.csv"
    distances=[]
    with open(gcps_dist_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            dist = float(row[0])
            distances.append(dist)
    
    # Print real world coordinates
    print("GCPs real world distances:")
    print(distances)
    
    return distances


# Function that loads XS img data
def load_xs_img(gcp_cam,gcp_date,gcp_time):
    csv_cross_path = bathy_dir / gcp_cam / (f"{gcp_cam}_xs_coord_{gcp_date}_{gcp_time}.csv")

    points_cross = []
    with open(csv_cross_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x = float(row[0])
            y = float(row[1])
            points_cross.append((x, y))

    return points_cross


# Function that loads PT depth data
def load_pt(pt_name):

    pt_data_file = pts_dir / (f"{pt_name}_depth.csv")
    
    # Load CSV from string
    df = pd.read_csv(pt_data_file)
    
    # Convert timestamp column
    df["Time"] = pd.to_datetime(df["Time"])
    
    # Create required columns
    df["date_yyyymmdd"] = df["Time"].dt.strftime("%Y%m%d")
    df["time_hhmmss"] = df["Time"].dt.strftime("%H%M%S")
    df["depth_m"] = df["device_frmpayload_data_Water_deep_cm.mean"]/100
    
    # Select only required columns
    df = df[["date_yyyymmdd", "time_hhmmss", "depth_m"]]
    
    return df


# Function that loads PT img coord data
def load_pt_img(pt_name):
    pt_img_path = pts_dir / (f"{pt_name}_img.csv")

    points_pt = []
    with open(pt_img_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x = float(row[0])
            y = float(row[1])
            points_pt.append((x, y))

    return points_pt


# Function that loads PT real world coord data
def load_pt_real(pt_name):
    pt_real_path = pts_dir / (f"{pt_name}_real.csv")

    points_pt = []
    with open(pt_real_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x = float(row[0])
            y = float(row[1])
            points_pt.append((x, y))

    return points_pt


# Function that loads PT loc data
def load_pt_loc(pt_name):
    pt_loc_path = pts_dir / (f"{pt_name}_loc.csv")

    points_pt = []
    with open(pt_loc_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x = float(row[0])
            y = float(row[1])
            points_pt.append((x, y))

    return points_pt


# Function that loads PT depth data
def load_pt_dep(pt_name,gcp_date,gcp_time):

    pt_dep_file = dep_dir / pt_name / f"{pt_name}_dep_{gcp_date}_{gcp_time}.csv"
        
    with open(pt_dep_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    return float(rows[0][0])
