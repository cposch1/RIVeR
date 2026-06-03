#!/usr/bin/env bash

# --------------------------------------------------
# Find absolute paths (robust, no guessing)
# --------------------------------------------------

# Directory of this script (testing/)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd -W )"

# Project root (one level above testing/)
PROJECT_DIR="$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd -W )"

export PROJECT_DIR

# --------------------------------------------------
# Data + results (relative to testing/)
# --------------------------------------------------

DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results"

export DATA_DIR
export RESULTS_DIR

export VIDEO_DIR="$DATA_DIR/videos"
export FRAMES_DIR="$DATA_DIR/frames"
export GCPS_DIR="$DATA_DIR/gcps"
export BATH_DIR="$DATA_DIR/bathymetry"
export RECT_DIR="$DATA_DIR/orthorectification"
export PTS_DIR="$DATA_DIR/pts"

export PIV_DIR="$RESULTS_DIR/piv"
export DISCH_DIR="$RESULTS_DIR/discharge"
export DEP_DIR="$RESULTS_DIR/depth"

# --------------------------------------------------
# Python import (project root contains "river/")
# --------------------------------------------------

export PYTHONPATH="$PROJECT_DIR"

# --------------------------------------------------
# Ensure directories exist
# --------------------------------------------------

mkdir -p "$DATA_DIR" "$VIDEO_DIR" "$FRAMES_DIR" \
         "$GCPS_DIR" "$BATH_DIR" "$RECT_DIR" "$PTS_DIR" \
         "$RESULTS_DIR" "$PIV_DIR" "$DISCH_DIR" "$DEP_DIR"

# --------------------------------------------------
# Debug output
# --------------------------------------------------

echo "PROJECT_DIR = $PROJECT_DIR"
echo "SCRIPT_DIR  = $SCRIPT_DIR"
echo "DATA_DIR    = $DATA_DIR"
echo "VIDEO_DIR   = $VIDEO_DIR"
echo "RESULTS_DIR = $RESULTS_DIR"
echo "PYTHONPATH  = $PYTHONPATH"
echo "### Environment is ready ###"