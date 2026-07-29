#!/usr/bin/env bash
# Launch Euclid Q1 selection-function production run:
#   20 000 lenses + 20 000 non-lenses, 96×96 @ 0.10″/pix, VIS/Y/J/H
#
# Prerequisites (must be mounted):
#   /Volumes/extHD          — TNG catalogs + particle cutouts
#   /Volumes/exthd-prism    — output destination (~0.5–1 TB free recommended)
#
# Usage:
#   ./scripts/local/run_euclid_q1_sf_prod_20k.sh           # full 20k+20k
#   ./scripts/local/run_euclid_q1_sf_prod_20k.sh --smoke    # 20+20 validation
#   ./scripts/local/run_euclid_q1_sf_prod_20k.sh --fg       # foreground (no nohup)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/euclid_q1_sf_prod_20k_96px.yaml"
OUT_DIR="/Volumes/exthd-prism/euclid_q1_sf_prod_20k_96px"
N_LENSES=20000
N_NON_LENSES=20000
# 434 COSMOS bases × 47 ≈ 20 398, truncated by --n_lenses to 20 000
VARIATIONS_PER_BASE=47
SEED=42
FOREGROUND=0
SMOKE=0

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --fg|--foreground) FOREGROUND=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  N_LENSES=20
  N_NON_LENSES=20
  VARIATIONS_PER_BASE=1
  OUT_DIR="/Volumes/exthd-prism/euclid_q1_sf_prod_20k_96px_smoke"
  echo "[SMOKE] Running 20 lenses + 20 non-lenses → ${OUT_DIR}"
fi

# --- Preflight ---------------------------------------------------------------
fail=0
[[ -d /Volumes/extHD ]] || { echo "ERROR: /Volumes/extHD not mounted"; fail=1; }
[[ -d /Volumes/exthd-prism ]] || { echo "ERROR: /Volumes/exthd-prism not mounted"; fail=1; }
[[ -f /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet ]] || {
  echo "ERROR: TNG100 local catalog missing on extHD"; fail=1;
}
[[ -d data/euclid_q1_psf/tiles ]] || {
  echo "ERROR: Euclid Q1 PSF library missing (data/euclid_q1_psf/tiles)"; fail=1;
}
[[ -f data/cosmos_web_lens_structural_properties.csv ]] || {
  echo "ERROR: cosmos catalog missing"; fail=1;
}
n_psf=$(find data/euclid_q1_psf/tiles -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "[PREFLIGHT] Q1 PSF tiles: ${n_psf} (expect 335)"
[[ "$n_psf" -ge 100 ]] || { echo "ERROR: too few PSF tiles"; fail=1; }
[[ "$fail" -eq 0 ]] || exit 1

mkdir -p "$OUT_DIR"
cp -f "$CONFIG" "$OUT_DIR/run_config.yaml"
cp -f configs/euclid_q1_sf_prod_20k_96px.DATASET.md "$OUT_DIR/DATASET_README.md" 2>/dev/null || true

LOG="$OUT_DIR/run.log"
echo "[RUN] config=$CONFIG"
echo "[RUN] out=$OUT_DIR"
echo "[RUN] n_lenses=$N_LENSES  n_non_lenses=$N_NON_LENSES  variations_per_base=$VARIATIONS_PER_BASE  seed=$SEED"
echo "[RUN] log=$LOG"
echo "[RUN] estimated wall time (full): ~20–30 h on this machine"

CMD=(
  python -m prism.core.simulator
  --config "$CONFIG"
  --cosmos_catalog data/cosmos_web_lens_structural_properties.csv
  --lens_analysis_catalog data/lens_analysis_catalog.csv
  --merged_field_catalog data/merged_lens_field_catalog.csv
  --output_dir "$OUT_DIR"
  --n_lenses "$N_LENSES"
  --n_non_lenses "$N_NON_LENSES"
  --n_field_max 20
  --variations_per_base "$VARIATIONS_PER_BASE"
  --seed "$SEED"
  --add_artifacts
  --save_intermediate
  --no_date_suffix
)

if [[ "$FOREGROUND" -eq 1 ]]; then
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
else
  nohup "${CMD[@]}" >>"$LOG" 2>&1 &
  echo $! >"$OUT_DIR/run.pid"
  echo "[RUN] started pid=$(cat "$OUT_DIR/run.pid")"
  echo "[RUN] monitor:  tail -f $LOG"
fi
