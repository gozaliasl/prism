#!/bin/bash
# Extends the TNG50-1 local catalog to the full TNG100-like snapshot list
# (z=0..6.5), covering the high-z source/lens range (z~1.4-6.5) not covered
# by the initial low-z (z<=1.3) TNG50-1 catalog build. Already-built
# per_snapshot/TNG50-1/snap*.parquet files are reused, so this only fetches
# the new high-z snapshots (12-42).
set -e
source ~/.zshrc >/dev/null 2>&1

cd /Users/gozalig1/Projects/jwst-mock-lens-simulator

python3 scripts/build_tng_local_catalog.py \
    --sim TNG50-1 \
    --logM-min 8.0 --logM-max 12.5 \
    --logM-stratify-threshold 10.0 \
    --max-per-snapshot 600 \
    --sleep 0.3 \
    --output /Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet
