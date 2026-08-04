#!/bin/bash
set -euo pipefail
cd /Volumes/exthd-prism/prism-lensing
export PYTHONUNBUFFERED=1
OUT=/Volumes/exthd-prism/prism-data/outputs/hydris_prism_50_1arcmin
LOG=/Volumes/exthd-prism/prism-data/outputs/logs/hydris_prism_50_1arcmin.log
mkdir -p "$OUT" "$(dirname "$LOG")"
exec python -u -m prism.core.simulator \
  --config configs/hydris_jwst_50_1arcmin.yaml \
  --cosmos_catalog /Volumes/exthd-prism/prism-data/catalogs/cosmos/cosmos_web_lens_structural_properties.csv \
  --output_dir "$OUT" \
  --n_lenses 50 --n_non_lenses 0 --numpix 1936 --seed 43 --no_date_suffix \
  > "$LOG" 2>&1
