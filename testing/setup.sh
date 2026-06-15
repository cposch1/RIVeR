#!/usr/bin/env bash

# --------------------------------------------------
# Activate environment (micromamba or conda)
# --------------------------------------------------
if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate velo

elif command -v conda &> /dev/null; then
    # Ensure conda works in non-interactive shells (HPC safe)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate velo

else
    echo "Error: neither micromamba nor conda found"
    return 1
fi

# --------------------------------------------------
# Resolve paths RELATIVE to this script only
# --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export SCRIPT_DIR
export PROJECT_DIR

# --------------------------------------------------
# Data + results (always relative)
# --------------------------------------------------
export DATA_DIR="$SCRIPT_DIR/data"
export RESULTS_DIR="$SCRIPT_DIR/results"

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
# Python path (project root)
# --------------------------------------------------
# Convert only if cygpath exists (i.e. on Git Bash / Windows)
if command -v cygpath &> /dev/null; then
    export PYTHONPATH="$(cygpath -w "$PROJECT_DIR")"
else
    export PYTHONPATH="$PROJECT_DIR"
fi

# --------------------------------------------------
# Create directories safely
# --------------------------------------------------
mkdir -p \
  "$VIDEO_DIR" "$FRAMES_DIR" "$GCPS_DIR" \
  "$BATH_DIR" "$RECT_DIR" "$PTS_DIR" \
  "$PIV_DIR" "$DISCH_DIR" "$DEP_DIR"

# --------------------------------------------------
# Debug output
# --------------------------------------------------
echo "PROJECT_DIR = $PROJECT_DIR"
echo "SCRIPT_DIR  = $SCRIPT_DIR"
echo "DATA_DIR    = $DATA_DIR"
echo "RESULTS_DIR = $RESULTS_DIR"
echo "PYTHONPATH  = $PYTHONPATH"
echo "Environment is ready"