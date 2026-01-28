# config.py
from __future__ import annotations
import os
from pathlib import Path
import warnings
import logging
import sys
import numpy as np
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import json
import csv
import math
import pandas as pd
import re
from IPython.display import display
from tablib import Dataset
from PIL import Image
from ipywidgets import FloatSlider, VBox, Label, HBox

# ---- Helpers ---------------------------------------------------------------

def _env_path(var_name: str, default: Path) -> Path:
    """Read a path from environment with a Path fallback; expand ~ and resolve."""
    val = os.environ.get(var_name)
    if val:
        return Path(val).expanduser().resolve()
    return default.expanduser().resolve()


# ---- Project roots & paths -------------------------------------------------

# Default project root: directory that contains this file
_default_root = Path(__file__).resolve().parent

root_dir   = _env_path("PROJECT_DIR", _default_root)
data_dir   = _env_path("DATA_DIR",   root_dir / "data")
results_dir= _env_path("RESULTS_DIR",root_dir / "results")
video_dir  = _env_path("VIDEO_DIR",  data_dir / "videos")
frames_dir = _env_path("FRAMES_DIR", data_dir / "frames")
gcps_dir =  _env_path("GCPS_DIR", data_dir / "gcps")
bathy_dir =  _env_path("BATHY_DIR", data_dir / "bathymetry")


# Ensure base folders exist
for p in (data_dir,results_dir,video_dir, frames_dir,gcps_dir,bathy_dir):
    p.mkdir(parents=True, exist_ok=True)


# ---- Logging & warnings ----------------------------------------------------

warnings.filterwarnings("always")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("river.config")
logger.info("Configured paths:")
for name, p in {
    "root_dir": root_dir,
    "data_dir": data_dir,
    "results_dir": results_dir,
    "video_dir": video_dir,
    "frames_dir": frames_dir,
    "gcps_dir": gcps_dir,
    "bathy_dir": bathy_dir
}.items():
    logger.info("  %-12s -> %s", name, p)


# ---- Import convenience (optional) -----------------------------------------
# Provide convenient re-exports for notebooks.
try:
    from river.core.video_to_frames import video_to_frames
    from river.core.exceptions import VideoHasNoFrames
    from river.core.coordinate_transform import (oblique_view_transformation_matrix,transform_pixel_to_real_world,transform_real_world_to_pixel)
    from river.core.coordinate_transform import get_camera_solution    
    from river.core.compute_section import (calculate_station_coordinates,divide_segment_to_dict,add_pixel_coordinates,calculate_river_section_properties)
    from river.core.define_roi_masks import (recommend_height_roi,create_mask_and_bbox)
    from river.core.piv_pipeline import (run_test,run_analyze_all)
    from river.core.compute_section import update_current_x_section

    from river.utils.arrow_utils import calculate_multiple_arrows
    from river.utils.visualization import plot_camera_solution

except Exception as e:
    # Avoid hard failure if import path not ready; user can still import explicitly.
    logger.warning("Could not import RIVeR modules yet: %s", e)
    video_to_frames = None
    VideoHasNoFrames = None    
    oblique_view_transformation_matrix = None
    transform_pixel_to_real_world = None
    transform_real_world_to_pixel = None
    get_camera_solution = None
    plot_camera_solution = None
    calculate_station_coordinates = None
    divide_segment_to_dict = None
    add_pixel_coordinates = None
    calculate_river_section_properties = None
    recommend_height_roi = None
    create_mask_and_bbox = None
    run_test = None
    run_analyze_all = None
    update_current_x_section = None
    calculate_multiple_arrows = None
