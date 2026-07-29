#!/bin/bash
# Waits for the currently-running TNG50-1 high-z batch fetch (PID passed as
# $1) to finish, then selects ~1500 of the MOST MASSIVE remaining TNG50-1
# candidates (logM >= 11.0 preferred, lens-galaxy priority) across the full
# z=0-6.5 catalog and starts fetching them. Intended to be launched in the
# background once and left alone.
set -e
source ~/.zshrc >/dev/null 2>&1

WAIT_PID="$1"
CATALOG=/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet
CANDIDATES=/Volumes/extHD/tng_local_catalog/tng50_batch3_candidates.csv
PY=/Users/gozalig1/anaconda3/envs/astro-clean/bin/python

cd /Users/gozalig1/Projects/jwst-mock-lens-simulator

echo "[auto_tng50_batch3] waiting for PID $WAIT_PID to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "[auto_tng50_batch3] previous fetch done, selecting new (mass-prioritized) candidates..."

"$PY" scripts/select_tng50_massive_lens_candidates.py \
    --catalog "$CATALOG" \
    --output "$CANDIDATES" \
    --n-total 1500 \
    --logm-min 11.0

echo "[auto_tng50_batch3] starting batch fetch..."
"$PY" scripts/batch_fetch_galaxygenius_stamps.py \
    --candidates "$CANDIDATES" \
    --output-dir /Volumes/extHD/galaxygenius_build/workspace/data \
    --sleep 0.5

echo "[auto_tng50_batch3] done."
