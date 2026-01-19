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
# # Oblique Video Transformation
#
# This notebook demonstrates how to perform coordinate transformation for oblique (side-view) river videos using RIVeR. Unlike nadir views, oblique videos require a more complex homography transformation to account for perspective distortion.
#
# ## Prerequisites
#
# - Completed frame extraction (00_introduction.ipynb) 
# - GRP measurements from field survey including known distances between points
# - An oblique view frame to work with
# - 4 points with known real-world distances 
#
# ## Theory
#
# Oblique transformation requires:
# - More GRPs than nadir view (4)
# - Known distances between points
# - Careful point selection across the frame to capture perspective

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import json

# Import RIVeR modules using relative imports
from river.core.coordinate_transform import (
    oblique_view_transformation_matrix,
    transform_pixel_to_real_world,
    transform_real_world_to_pixel
)

# Set up paths 
frame_path = Path("data/frames/ilh_20250426-200000-205900/0000000000.jpg")
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
# ## Step 2: Define Ground Control Points (GCPs) and Visualize Transformation
#
# For oblique transformation, we need 4 points with known distances between them. The points should be:
# - Well distributed across the frame
# - Include points at different depths in the scene
# - Have accurately measured distances between them
#
# Point ordering is critical for correct transformation:
# - Point 1 must be the most upstream and leftmost point in your view
# - Remaining points (2, 3, and 4) must be defined in counterclockwise order
# - Example ordering:
#   * Point 1: Upstream-left
#   * Point 2: Upstream-right
#   * Point 3: Downstream-right
#   * Point 4: Downstream-left
#
# This specific ordering ensures proper calculation of the homography transformation and accurate real-world coordinate mapping.
#
# The visualization consists of two parts:
# 1. The original image with the GCPs marked and connected
# 2. The orthorectified image showing the same points in real-world coordinates
#
# In the orthorectified image:
# - The scale bar indicates distances in meters
# - The colored lines correspond to the same connections between points as in the original image
# - The transformation preserves the relative distances between points according to the measured values

# %%
# Example GRP coordinates (replace with your actual measurements)
# Format: pixel coordinates (x, y) for 4 points forming a quadrilateral
point_coords_pixel = {
    'point1': (1498, 313),   # left-upstream
    'point2': (1313, 146),  # right-upstream
    'point3': (1084, 141), # left-downstream
    'point4': (727, 395)   # left-downstream
}

# Known distances between points in meters
distances = {
    'd12': 28.00,  # Distance between points 1 and 2
    'd23': 12.21,   # Distance between points 2 and 3
    'd34': 28.00,  # Distance between points 3 and 4
    'd41': 12.21,   # Distance between points 4 and 1
    'd13': 23.26,  # Diagonal distance between points 1 and 3
    'd24': 36.40   # Diagonal distance between points 2 and 4
}

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# First subplot with original image and control points
ax1.imshow(frame_rgb)
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
ax1.plot(x1_pix, y1_pix, 'o', color='#ED6B57', markersize=10)
# Points 2-4 in blue
ax1.plot([x2_pix, x3_pix, x4_pix], [y2_pix, y3_pix, y4_pix], 'o', color='#6CD4FF', markersize=10)
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
    ax2.plot(x1_rw, y1_rw, 'o', color='#ED6B57', markersize=10)
    # Points 2-4 in blue
    ax2.plot([x2_rw, x3_rw, x4_rw], [y2_rw, y3_rw, y4_rw], 'o', color='#6CD4FF', markersize=10)
    
    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax2.set_title('Orthorectified Image')

plt.tight_layout()

image_output_file = output_dir / "02_orthorect_process.png"
plt.savefig(str(image_output_file))
plt.show()  # Fixed from ax1.show() to plt.show()

# %% [markdown]
# ## Step 3: Save Transformation Matrix
#
# We'll save the transformation matrix to a JSON file for use in subsequent processing steps.

# %%
# Save transformation matrix to JSON
output_file = output_dir / "transformation.json"
with open(output_file, 'w') as f:
    json.dump(transformation_matrix, f, indent=1)
print(f"Transformation matrix saved to {output_file}")

# Save the transformed image
image_output_file = output_dir / "03_orthorect_output.png"
plt.imsave(str(image_output_file), transformation['transformed_img'])
print(f"Transformed image saved to {image_output_file}")

# Verify transformation by converting some test points
test_pixel = (800, 900)  # Example pixel coordinates
real_world = transform_pixel_to_real_world(test_pixel[0], test_pixel[1], transformation_matrix)
print("\nTransformation Test:")
print(f"Pixel coordinates: {test_pixel}")
print(f"Real-world coordinates (meters): ({real_world[0]:.2f}, {real_world[1]:.2f})")

# %% [markdown]
# ## Common Issues and Troubleshooting
#
# 1. Poor Transformation Results:
#    - Ensure GRP measurements are accurate
#    - Distribute points evenly across the frame
#    - Avoid colinear points
#
# 2. Transformation Matrix Errors:
#    - Check for typos in coordinates
#    - Verify distance measurements
#    - Ensure points form a proper quadrilateral
#
# 3. Perspective Issues:
#    - Include points at different depths in the scene
#    - Use points that bound the area of interest
#    - Consider lens distortion effects
#
# ## Next Steps
#
# After successful transformation:
# - Proceed to PIV analysis (03_cross_sections.ipynb)
# - Use transformation matrix for velocity calculations
# - Convert results back to real-world coordinates

# %% [markdown]
# ## Step 4: Select river cross-section
#
# By clicking in the picture frame - first left river bank, then right

# %%
# %matplotlib widget
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

first_frame = r'C:\Users\cposch1\RIVeR\testing\data\frames\ilh_20250426-200000-205900\0000000000.jpg'

img = mpimg.imread(str(first_frame))
#img = mpimg.imread(image_output_file)

points = []
point_coords_pixel = {}   # make it a dict from the start

fig, ax = plt.subplots(figsize=(18,9))
ax.imshow(img)
ax.set_title("Select 1) left and 2) right river bank")
plt.axis("off")

def onclick(event):
    global point_coords_pixel  # <<< IMPORTANT

    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: x={x}, y={y}")
        points.append((x, y))

        if len(points) == 1:
            ax.plot(x, y, 'o', color='#ED6B57', markersize=10)
            fig.canvas.draw()

        # When 2 points are collected, name them
        elif len(points) == 2:
            ax.plot(x, y, 'o', color='#62C655', markersize=10)
            fig.canvas.draw()
            
            fig.canvas.mpl_disconnect(cid)

            point_coords_pixel = {
                'point1': points[0],
                'point2': points[1]
            }

            print("2 points collected:")
            print(point_coords_pixel)

            ax.plot([point_coords_pixel['point1'][0], point_coords_pixel['point2'][0]],
                    [point_coords_pixel['point1'][1], point_coords_pixel['point2'][1]], color='#F5BF61', linewidth=2)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()

# %%
image_output_file = output_dir / "04_cross_sec_select.png"
plt.savefig(str(image_output_file))
plt.close()

# %%
point_coords_pixel

# %%
x1_pix, y1_pix = point_coords_pixel['point1']
x2_pix, y2_pix = point_coords_pixel['point2']

 # %%
 # Convert GCP pixel coordinates to real-world coordinates to display in the second image
real_world_points = []
for x, y in [(x1_pix, y1_pix), (x2_pix, y2_pix)]:
    rw_point = transform_pixel_to_real_world(x, y, transformation_matrix)
    real_world_points.append(rw_point)
    
real_world_points = np.array(real_world_points)
 
# Get individual point coordinates
x1_rw, y1_rw = real_world_points[0]
x2_rw, y2_rw = real_world_points[1]

# %%
real_world_points

# %%
# %matplotlib inline
plt.plot(figsize=(18, 9))
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



# Draw lines with the same colors as in the first plot
# Line 1-2
#ax.plot([x1_rw, x2_rw], [y1_rw, y2_rw], color='#6CD4FF', linewidth=2)
ax.plot([x1_rw, x2_rw], [y1_rw, y2_rw], color='#F5BF61', linewidth=2)  # Line connecting points

# Plot points
# Point 1 in red
ax.plot(x1_rw, y1_rw, 'o', color='#ED6B57', markersize=10)
# Points 2-4 in blue
ax.plot(x2_rw, y2_rw, 'o', color='#62C655', markersize=10)

ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title('Cross-section in orthorectified image')

plt.tight_layout()

image_output_file = output_dir / "05_cross_sec_orthorect.png"
plt.savefig(str(image_output_file))

plt.show()  # Fixed from ax1.show() to plt.show()
plt.close()

# %%
