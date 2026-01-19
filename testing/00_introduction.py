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
# # Introduction to RIVeR: Rectification of  Image Velocimetry Results
#
# Welcome to the RIVeR tutorial series! This notebook provides an introduction to the RIVeR package, which enables  processing of river flow videos for velocity and discharge estimation using Large Scale Particle Image Velocimetry (LSPIV).
#
# ## What is RIVeR?
#
# RIVeR (Rectification of  Image Velocimetry Resultse) is a Python package designed for processing river flow videos to obtain velocity fields and discharge estimates. It supports three main filming scenarios:
#
# 1. **Nadir View**: Camera positioned directly above the river
# 2. **Oblique View**: Camera at an angle to the river surface
# 3. **Fixed station**: Fixed camera with 3D perspective considerations
#
# ## Tutorial Series Overview
#
# This tutorial series consists of the following notebooks:
#
# 1. Introduction (current notebook)
# 2. Video Processing and Frame Extraction
# 3. Coordinate Transformation
# 4. PIV Analysis
# 5. Cross-section Definition and Discharge Calculation
#
# ## Prerequisites
#
# Before starting, ensure you have:
#
# - Python 3.11 or later
# - RIVeR package installed
# - Required dependencies (numpy, opencv-python, scipy)
# - Sample data downloaded (will be covered in this notebook)

# %%
# Required imports for environment setup
from pathlib import Path
import sys
import numpy as np
import cv2

# Add the root directory to Python path for relative imports
root_dir = Path().absolute().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Import RIVeR modules
from river.core.exceptions import VideoHasNoFrames
from river.core.video_to_frames import video_to_frames

# %% [markdown]
# ## Package Structure
#
# RIVeR is organized into several core modules:
#
# ```bash
# river/
# ├── core/
# │   ├── video_to_frames.py      # Video processing
# │   ├── coordinate_transform.py  # Coordinate transformations
# │   ├── piv_pipeline.py         # PIV analysis pipeline
# │   ├── compute_section.py      # Cross-section computations
# │   └── exceptions.py           # Custom exceptions
# └── examples/
#     └── data/                   # Sample datasets
# ```
#
