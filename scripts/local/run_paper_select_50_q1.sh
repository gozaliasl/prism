#!/usr/bin/env bash
# Full Q1 + κ/γ/μ/flexion rerun of 50 Euclid 1′ paper candidates (Trilogy RGB).
#
# Usage:
#   ./scripts/local/run_paper_select_50_q1.sh
#   ./scripts/local/run_paper_select_50_q1.sh --fg

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/euclid_paper_select_50_q1.yaml"
OUT_DIR="outputs/euclid_paper_select_50_q1"
N_LENSES=50
N_FIELD_MAX=80
SEED=271828
FOREGROUND=0

for arg in "$@"; do
  case "$arg" in
    --fg|--foreground) FOREGROUND=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown: $arg" >&2; exit 1 ;;
  esac
done

fail=0
[[ -d /Volumes/extHD ]] || { echo "ERROR: /Volumes/extHD not mounted"; fail=1; }
[[ -d data/euclid_q1_psf/tiles ]] || { echo "ERROR: Euclid Q1 PSF missing"; fail=1; }
[[ -f data/euclid_q1_psf/modeling_lens_mass.csv ]] || { echo "ERROR: Q1 mass catalog missing"; fail=1; }
[[ -f data/galaxy_catalog.fits ]] || { echo "ERROR: galaxy_catalog.fits missing"; fail=1; }
[[ "$fail" -eq 0 ]] || exit 1

mkdir -p "$OUT_DIR"
cp -f "$CONFIG" "$OUT_DIR/run_config.yaml"
LOG="$OUT_DIR/run.log"

CMD=(
  python -u -m prism.core.simulator
  --config "$CONFIG"
  --cosmos_catalog data/cosmos_web_lens_structural_properties.csv
  --lens_analysis_catalog data/lens_analysis_catalog.csv
  --merged_field_catalog data/merged_lens_field_catalog.csv
  --output_dir "$OUT_DIR"
  --n_lenses "$N_LENSES"
  --n_non_lenses 0
  --n_field_max "$N_FIELD_MAX"
  --variations_per_base 1
  --seed "$SEED"
  --add_artifacts
  --save_intermediate
  --no_date_suffix
)

echo "[RUN] Q1+kappa out=$OUT_DIR n=$N_LENSES seed=$SEED"
if [[ "$FOREGROUND" -eq 1 ]]; then
  "${CMD[@]}" 2>&1 | tee "$LOG"
else
  nohup "${CMD[@]}" >"$LOG" 2>&1 &
  echo $! >"$OUT_DIR/run.pid"
  echo "[BG] pid=$(cat "$OUT_DIR/run.pid")  tail -f $LOG"
fi
