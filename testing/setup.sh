#!/usr/bin/env bash

# Absolute path to the directory containing this script (RIVeR/testing)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
export PROJECT_ROOT

export DATA_DIR="$PROJECT_ROOT/data"
export VIDEO_DIR="$DATA_DIR/videos"
export FRAMES_DIR="$DATA_DIR/frames"
export RESULTS_DIR="$PROJECT_ROOT/results"

case ":$PYTHONPATH:" in
  *":$PROJECT_ROOT:"*) ;;
  *) export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

mkdir -p "$DATA_DIR" "$VIDEO_DIR" "$FRAMES_DIR" "$RESULTS_DIR"

echo "[setup.sh] PROJECT_ROOT=$PROJECT_ROOT"
echo "[setup.sh] DATA_DIR=$DATA_DIR"
echo "[setup.sh] PYTHONPATH=$PYTHONPATH"
echo "[setup.sh] Environment is ready. Start Jupyter."
