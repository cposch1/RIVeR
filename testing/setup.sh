#!/usr/bin/env bash

# Absolute path to the directory containing this script (RIVeR/testing)
PROJECT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PROJECT_DIR

export DATA_DIR="$PROJECT_DIR/data"

export VIDEO_DIR="$DATA_DIR/videos"
export FRAMES_DIR="$DATA_DIR/frames"
export GCPS_DIR="$DATA_DIR/gcps"
export BATH_DIR="$DATA_DIR/bathymetry"
export RECT_DIR="$DATA_DIR/orthorectification"
export PTS_DIR="$DATA_DIR/pts"

export RESULTS_DIR="$PROJECT_DIR/results"
export PIV_DIR="$RESULTS_DIR/piv"
export DISCH_DIR="$RESULTS_DIR/discharge"

case ":$PYTHONPATH:" in
  *":$PROJECT_DIR:"*) ;;
  *) export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

mkdir -p "$DATA_DIR" "$VIDEO_DIR" "$FRAMES_DIR" "$GCPS_DIR" "$BATH_DIR" "$RECT_DIR" "$PTS_DIR" "$RESULTS_DIR" "$PIV_DIR" "$DISCH_DIR"

echo "PROJECT_DIR = $PROJECT_DIR"
echo "DATA_DIR = $DATA_DIR"
echo "RESULTS_DIR = $RESULTS_DIR"
echo "### Environment is ready. Start Jupyter Lab. ###"
