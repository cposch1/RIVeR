#!/usr/bin/env bash

micromamba activate velo

# Use current directory
PROJECT_DIR="$HOME/RIVeR"
SCRIPT_DIR="$PROJECT_DIR/testing"

# Use user-writable directories
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

VIDEO_DIR="$DATA_DIR/videos"
FRAMES_DIR="$DATA_DIR/frames"
GCPS_DIR="$DATA_DIR/gcps"
BATH_DIR="$DATA_DIR/bathymetry"
RECT_DIR="$DATA_DIR/orthorectification"
PTS_DIR="$DATA_DIR/pts"

PIV_DIR="$RESULTS_DIR/piv"
DISCH_DIR="$RESULTS_DIR/discharge"
DEP_DIR="$RESULTS_DIR/depth"

mkdir -p "$DATA_DIR" "$VIDEO_DIR" "$FRAMES_DIR" \
         "$GCPS_DIR" "$BATH_DIR" "$RECT_DIR" "$PTS_DIR" \
         "$RESULTS_DIR" "$PIV_DIR" "$DISCH_DIR" "$DEP_DIR"

echo "PROJECT_DIR = $PROJECT_DIR"
echo "SCRIPT_DIR  = $SCRIPT_DIR"
echo "DATA_DIR    = $DATA_DIR"
echo "RESULTS_DIR = $RESULTS_DIR"

echo "### Environment is ready ###"
