#!/usr/bin/env bash
#
# Download the first frame (0000000000.jpg) of every date/time folder from the
# server, keeping the folder structure, plus the _frame_paths.* index files.
#
# Works from any level: a single camera folder or a folder holding several
# cameras (e.g. the whole frames directory).
#
# Usage:
#   bash c01_first_frame_download.sh [REMOTE_DIR] [LOCAL_DIR]
#
#   REMOTE_DIR  folder on the server   (default: $REMOTE_DEFAULT below)
#   LOCAL_DIR   local target folder    (default: ~/Desktop/<name of REMOTE_DIR>)
#               Git Bash (/c/...) and Windows (C:/...) paths both work.
#
# Examples:
#   bash c01_first_frame_download.sh                                   # all cameras
#   bash c01_first_frame_download.sh /home/cposch/RIVeR/testing/data/frames/lev5-cam2-lake

set -o pipefail

SSH="cposch@octopus.unil.ch"
REMOTE_DEFAULT="/home/cposch/RIVeR/testing/data/frames"

REMOTE_BASE="${1:-$REMOTE_DEFAULT}"
REMOTE_BASE="${REMOTE_BASE%/}"
LOCAL_BASE="${2:-$HOME/Desktop/$(basename "$REMOTE_BASE")}"
LOCAL_BASE=$(cygpath -u "$LOCAL_BASE" 2>/dev/null || echo "$LOCAL_BASE")

echo "[INFO] $SSH:$REMOTE_BASE -> $LOCAL_BASE"
mkdir -p "$LOCAL_BASE" || exit 1

# One connection: pack the first frames (+ index files at the top level) on
# the server and unpack them locally with their relative paths
ssh "$SSH" "cd '$REMOTE_BASE' && { \
        find . -maxdepth 1 -type f -name '_frame_paths.*' -print0; \
        find . -type f -name 0000000000.jpg -print0; \
    } | tar --null -cf - -T -" \
    | tar -xf - -C "$LOCAL_BASE"

if [ $? -ne 0 ]; then
    echo "[ERROR] Download failed (check the remote folder and your SSH connection)."
    exit 1
fi

# Report date/time folders on the server that have no first frame
MISSING=$(ssh "$SSH" "cd '$REMOTE_BASE' && find . -type d -regextype posix-extended \
    -regex '(.*/)?[0-9]{8}/[0-9]{6}' ! -exec test -e '{}/0000000000.jpg' \; -print")

N=$(find "$LOCAL_BASE" -type f -name 0000000000.jpg | wc -l)
echo "[OK] $N first frames in $LOCAL_BASE"
ls "$LOCAL_BASE"/_frame_paths.* >/dev/null 2>&1 && echo "[OK] _frame_paths.* index files included"

if [ -n "$MISSING" ]; then
    echo "[MISSING] no 0000000000.jpg in:"
    echo "$MISSING" | sed 's|^\./|  |'
fi
