#!/bin/bash
# Waits for build_tng_local_catalog.py --sim TNG50-1 to produce
# tng50-1_local_catalog.parquet, then automatically runs candidate
# selection + the particle-cutout batch fetch for TNG50-1. Intended to be
# launched in the background once and left alone.
set -e

CATALOG=/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet
CANDIDATES=/Volumes/extHD/tng_local_catalog/tng50_candidates.csv
PY=/Users/gozalig1/anaconda3/envs/astro-clean/bin/python

cd /Users/gozalig1/Projects/jwst-mock-lens-simulator

echo "[auto_tng50] waiting for $CATALOG ..."
while [ ! -f "$CATALOG" ]; do
    sleep 30
done
echo "[auto_tng50] catalog found, selecting candidates..."

"$PY" scripts/select_tng50_lowmass_candidates.py \
    --catalog "$CATALOG" \
    --output "$CANDIDATES" \
    --n-total 1500

echo "[auto_tng50] starting batch fetch..."
"$PY" scripts/batch_fetch_galaxygenius_stamps.py \
    --candidates "$CANDIDATES" \
    --output-dir /Volumes/extHD/galaxygenius_build/workspace/data \
    --sleep 0.5

echo "[auto_tng50] done."
