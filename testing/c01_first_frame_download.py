#!/usr/bin/env bash

REMOTE_BASE="/home/cposch/RIVeR/testing/data/frames/ilh-cam1-pt"
LOCAL_BASE="/c/Users/cposch1/Desktop/ilh-cam1-pt"
SSH="cposch@octopus.unil.ch"

# Find all time directories (date/time level)
ssh $SSH "find $REMOTE_BASE -mindepth 2 -maxdepth 2 -type d" | while read REMOTE_DIR; do

    # Extract relative path: date/time
    REL=${REMOTE_DIR#"$REMOTE_BASE"/}

    LOCAL_DIR="$LOCAL_BASE/$REL"

    echo "[INFO] $REL"

    mkdir -p "$LOCAL_DIR"

    scp "$SSH:$REMOTE_DIR/0000000000.jpg" "$LOCAL_DIR/" 2>/dev/null && \
        echo "[OK] $REL" || \
        echo "[MISSING] $REL"
done