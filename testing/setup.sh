#!/usr/bin/env bash

# Absolute path to the directory containing this script (RIVeR/testing)
PROJECT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PROJECT_DIR
export DATA_DIR="$PROJECT_DIR/data"
export VIDEO_DIR="$DATA_DIR/videos"
export FRAMES_DIR="$DATA_DIR/frames"
export RESULTS_DIR="$PROJECT_DIR/results"

case ":$PYTHONPATH:" in
  *":$PROJECT_DIR:"*) ;;
  *) export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

mkdir -p "$DATA_DIR" "$VIDEO_DIR" "$FRAMES_DIR" "$RESULTS_DIR"

echo "PROJECT_DIR = $PROJECT_DIR"
echo "DATA_DIR = $DATA_DIR"
echo "RESULTS_DIR = $RESULTS_DIR"
echo "### Environment is ready. Start Jupyter Lab. ###"
