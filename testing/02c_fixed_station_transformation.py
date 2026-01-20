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

# %% [markdown]
# # Fixed Station Transformation
#
# ## Introduction
#
# This notebook demonstrates how to perform coordinate transformation for fixed-station river videos using RIVeR. Fixed-station videos are recorded from a stationary position, typically on a riverbank or bridge, and require camera calibration to translate from pixel coordinates to real-world coordinates.
#
# ### Prerequisites
#
# - Completed frame extraction (01_video_to_frames.ipynb)
# - Ground Reference Points (GRPs) with known 3D coordinates (X, Y, Z)
#   * Minimum of 6 points is required for a basic solution
#   * At least 8-10 points are recommended for optimal accuracy and redundancy
#   * Points should be well-distributed across the region of interest
#   * Include points at different distances from the camera when possible
# - Either:
#   * A single clear frame where all GRPs are visible, or
#   * Multiple frames showing different subsets of GRPs that can be combined
#   * For multi-frame GRP collection, camera position must remain absolutely fixed
#
#
# ### Theory
#
# Fixed-station transformation uses camera calibration techniques to establish a relationship between image coordinates and real-world coordinates. Unlike simpler transformations (scaling or homography), this approach:
#
# - Accounts for 3D perspective effects
# - Requires at least 6 Ground Reference Points (GRPs) with known X, Y, Z coordinates
# - Allows for computation of camera position and orientation

# %%
## Required Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import json
import pandas as pd
from IPython.display import display
import sys
import csv

# Import RIVeR modules using relative imports
from river.core.coordinate_transform import (
    get_camera_solution,
    transform_pixel_to_real_world,
    transform_real_world_to_pixel
)

# Add the notebook directory to Python path
notebook_dir = Path.cwd()
if str(notebook_dir) not in sys.path:
    sys.path.append(str(notebook_dir))

# Import the function to plot the camera solution
from utils.visualization import plot_camera_solution

# Set up paths 
frame_path = Path("data/frames/ilh_20250426-200000-205900/0000000000.jpg")
grps_path = Path("data/grps/ilh.json")
output_dir = Path("results/ilh")
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1: Load and Display the Frame
#
# First, we'll load our sample frame and prepare it for transformation. For oblique views, it's important to select a frame where all GRPs are clearly visible.

# %%
# Load the image
frame = cv2.imread(str(frame_path))
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the frame
plt.figure(figsize=(12, 8))
plt.imshow(frame_rgb)
plt.axis('off')
plt.title('Original Frame')
plt.show()

# %% [markdown]
# ## Step 2: Load and Visualize Ground Reference Points (GRPs)
#
# For fixed-station transformation, we need Ground Reference Points with known 3D coordinates (X, Y, Z). These points are critical for establishing the relationship between image space and real-world space. Effective GRPs should:
#
# - Be well distributed across the frame, especially in the area of interest
# - Include points at different distances from the camera to capture depth
# - Have accurately measured real-world coordinates with minimal error
# - Be visible and identifiable in the video frames
# - Include features that are stable over time (e.g., permanent structures, installed markers)
#
# In practice, collecting good GRPs often requires field surveys using total stations, RTK-GPS, or other surveying equipment. For this example, we'll load pre-measured GRPs from a JSON file.
#

# %%
# Load the GRPs from the JSON file
# The file contains dictionaries with 'X', 'Y', 'Z' (real-world coordinates)
# and 'x', 'y' (pixel coordinates) keys
with open(grps_path, 'r') as f:
    grp_dict = json.load(f)

# Create a DataFrame for better visualization and inspection of the GRP data
# This helps us verify the quality and distribution of our reference points
import pandas as pd

# Create point IDs for easier reference
# Get the number of points from any coordinate array in the dictionary
num_points = len(grp_dict['X'])
point_ids = [f"Point {i+1}" for i in range(num_points)]

# Create a dataframe with all GRP information
# Using .get() with default values ensures the code doesn't break if a key is missing
df = pd.DataFrame({
    'Point ID': point_ids,
    'X (real world)': grp_dict.get('X', [None] * num_points),  # Easting or longitudinal coordinate
    'Y (real world)': grp_dict.get('Y', [None] * num_points),  # Northing or lateral coordinate
    'Z (real world)': grp_dict.get('Z', [None] * num_points),  # Elevation
    'x (pixel)': grp_dict.get('x', [None] * num_points),       # Horizontal pixel coordinate
    'y (pixel)': grp_dict.get('y', [None] * num_points)        # Vertical pixel coordinate
})

# Set Point ID as index for cleaner display
df.set_index('Point ID', inplace=True)

# Display the table with formatted values
print("Ground Reference Points (GRPs)")
display(df.style.format({
    'X (real world)': '{:.3f}',  # 3 decimal places for real-world coordinates (meter precision)
    'Y (real world)': '{:.3f}',
    'Z (real world)': '{:.3f}',
    'x (pixel)': '{:.2f}',       # 2 decimal places for pixel coordinates
    'y (pixel)': '{:.2f}'
}))

# %% [raw]
#
# import csv
# import numpy as np
# import pandas as pd
#
# # 1) Load the CSV into a list-of-dicts
# with open(grps_path, newline='') as f:
#     reader = csv.DictReader(f)
#     rows = list(reader)
#
# # Optional: quick sanity check
# if not rows:
#     raise ValueError("CSV appears empty or has no data rows.")
#
# # 2) Make a DataFrame
# df = pd.DataFrame(rows)
#
# # 3) Normalize/rename columns if needed (case/spacing issues)
# #    Adjust these mappings if your headers differ
# rename_map = {
#     'Point ID': 'Point ID',
#     'X': 'X',
#     'Y': 'Y',
#     'Z': 'Z',
#     'x': 'x',
#     'y': 'y',
# }
# df = df.rename(columns=rename_map)
#
# # 4) Ensure required columns exist
# required = ['X', 'Y', 'Z', 'x', 'y']
# missing = [c for c in required if c not in df.columns]
# if missing:
#     raise ValueError(f"Missing required columns in CSV: {missing}. "
#                      f"Found columns: {list(df.columns)}")
#
# # 5) Convert to numeric
# for c in required + ['Point ID']:
#     if c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors='coerce')
#
# # 6) Drop rows with NaNs in required columns (or handle differently)
# df = df.dropna(subset=required)
#
# # 7) Build the dict-of-arrays expected by get_camera_solution
# grp_dict = {c: df[c].to_numpy(dtype=float) for c in required}
#
# # Optional: you can keep Point IDs separately if you need them later
# point_ids = df['Point ID'].to_numpy(dtype=int) if 'Point ID' in df.columns else None

# %% [markdown]
# ## Step 3: Calculate Camera Solution
#
# Now we'll use the `get_camera_solution` function to compute the camera matrix that transforms between pixel and real-world coordinates.
#

# %%
cam_solution = get_camera_solution(grp_dict, image_path=frame_path)

# Print camera position and error
print(f"Camera position (X, Y, Z): {cam_solution['camera_position']}")
print(f"Mean reprojection error: {cam_solution['error']:.4f} pixels")

# Optimized camera solution
print("\nCalculating optimized camera solution...")
try:
    cam_solution_optimized = get_camera_solution(
        grp_dict, 
        image_path=frame_path, 
        optimize_solution=True
    )
    print(f"Used {cam_solution_optimized['num_points']} points for optimization")
    print(f"Points used: {cam_solution_optimized['point_indices']}")
    print(f"Camera position (X, Y, Z): {cam_solution_optimized['camera_position']}")
    print(f"Mean reprojection error: {cam_solution_optimized['error']:.4f} pixels")
    
    # Use the optimized solution if available
    cam_solution = cam_solution_optimized
except Exception as e:
    print(f"Optimization failed: {e}")
    print("Using basic camera solution instead.")


# %% [markdown]
# ## Step 4: Visualize the Transformation
#
# Let's visualize how the transformation works by showing the original image with GRPs and the orthorectified view.
#

# %%
fig = plot_camera_solution(frame_rgb, grp_dict, cam_solution)
plt.show()

# %% [markdown]
# ## Step 5: Save the Camera Matrix for Future Use
#
# Finally, we'll save the camera solution to a JSON file for use in subsequent processing steps.
#

# %%
# Convert camera matrix (numpy array) to list for JSON serialization
output_file = output_dir / "transformation.json"

# Create a serializable version of the camera solution
cam_solution = {
    'camera_matrix': cam_solution['camera_matrix'],
    'camera_position': cam_solution['camera_position'],
    'error': float(cam_solution['error']),
}

# Save to JSON file
with open(output_file, 'w') as f:
    json.dump(cam_solution, f, indent=2)

print(f"Camera solution saved to {output_file}")

# %% [markdown]
# ## Common Issues and Troubleshooting
#
# 1. **High Reprojection Error:**
#    - Ensure GRP measurements are accurate
#    - Check if GRPs are correctly marked in the image
#    - Add more GRPs or redistribute them across the scene
#    - Use the optimized solution which automatically selects the best subset of points
#
# 2. **Distorted Orthorectified View:**
#    - GRPs may be coplanar (all points at same elevation) - ensure points are at different elevations
#    - Ensure GRPs cover a wide area of the image
#    - Verify that the Z-coordinate measurements are accurate
#
# Remember that fixed-station transformation requires precise GRP measurements.

# %%
