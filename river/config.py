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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import json
import csv
import math
import pandas as pd
import re
from IPython.display import display
from tablib import Dataset
from PIL import Image
from ipywidgets import FloatSlider, VBox, Label, HBox
from typing import Iterable, Tuple, Dict, Optional, Set
import shutil

# ---- Helpers ---------------------------------------------------------------

def _env_path(var_name: str, fallback: Path) -> Path:
    val = os.environ.get(var_name)

    if val:
        # ✅ Convert Git Bash → Windows
        if val.startswith("/c/"):
            val = "C:/" + val[3:]

        # ✅ DO NOT call resolve() here
        return Path(val).expanduser()

    return fallback.expanduser()


# ---- Project root (single source of truth) ---------------------------------

# Prefer PROJECT_DIR from setup.sh, fallback: repo structure
root_dir = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).resolve()


# ---- Data + result directories ---------------------------------------------

data_dir = _env_path("DATA_DIR", root_dir / "testing" / "data")
results_dir = _env_path("RESULTS_DIR", root_dir / "testing" / "results")

video_dir  = _env_path("VIDEO_DIR", data_dir / "videos")
frames_dir = _env_path("FRAMES_DIR", data_dir / "frames")
gcps_dir   = _env_path("GCPS_DIR", data_dir / "gcps")
bathy_dir  = _env_path("BATH_DIR", data_dir / "bathymetry")
rect_dir   = _env_path("RECT_DIR", data_dir / "orthorectification")
pts_dir    = _env_path("PTS_DIR", data_dir / "pts")

piv_dir   = _env_path("PIV_DIR", results_dir / "piv")
disch_dir = _env_path("DISCH_DIR", results_dir / "discharge")
dep_dir   = _env_path("DEP_DIR", results_dir / "depth")


# ---- Ensure directories exist ---------------------------------------------

for p in [
    data_dir, video_dir, frames_dir, gcps_dir, bathy_dir,
    rect_dir, pts_dir, results_dir, piv_dir, disch_dir, dep_dir
]:
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
    "video_dir": video_dir,
    "frames_dir": frames_dir,
    "gcps_dir": gcps_dir,
    "bathy_dir": bathy_dir,
    "rect_dir": rect_dir,
    "pts_dir": pts_dir,
    "results_dir": results_dir,
    "piv_dir": piv_dir,
    "disch_dir": disch_dir,
    "dep_dir": dep_dir,
}.items():
    logger.info("  %-12s -> %s", name, p)


# ---- Import convenience (optional) -----------------------------------------

try:
    from river.core.video_to_frames import video_to_frames
    from river.core.exceptions import VideoHasNoFrames
    from river.core.coordinate_transform import (
        oblique_view_transformation_matrix,
        transform_pixel_to_real_world,
        transform_real_world_to_pixel,
        get_camera_solution
    )
    from river.core.compute_section import (
        calculate_station_coordinates,
        divide_segment_to_dict,
        add_pixel_coordinates,
        calculate_river_section_properties,
        update_current_x_section
    )
    from river.core.define_roi_masks import (
        recommend_height_roi,
        create_mask_and_bbox
    )
    from river.core.piv_pipeline import (
        run_test,
        run_analyze_all
    )

    from river.utils.arrow_utils import calculate_multiple_arrows
    from river.utils.visualization import plot_camera_solution

    from river.core.loading_data import (
        load_frame, load_gcps_img, load_gcps_real,
        load_dist, load_xs_img,
        load_pt, load_pt_img, load_pt_real, load_pt_loc, load_pt_dep
    )

    from river.core.image_rectification import transform

except Exception as e:
    logger.warning("Could not import RIVeR modules yet: %s", e)

    # Optional safe fallbacks
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