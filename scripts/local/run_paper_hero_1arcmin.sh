#!/usr/bin/env bash
# Generate 1′×1′ Euclid Q1 hybrid BGG strong-lens candidates for paper figures.
#
# Output: outputs/euclid_paper_hero_1arcmin/
# After the run completes:
#   python scripts/local/package_paper_hero_figure.py outputs/euclid_paper_hero_1arcmin
#
# Usage:
#   ./scripts/local/run_paper_hero_1arcmin.sh           # 8 candidates (default)
#   ./scripts/local/run_paper_hero_1arcmin.sh --fg      # foreground
#   ./scripts/local/run_paper_hero_1arcmin.sh --smoke   # 1 lens quick test

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/euclid_paper_hero_1arcmin.yaml"
OUT_DIR="outputs/euclid_paper_hero_1arcmin"
N_LENSES=8
N_FIELD_MAX=45
SEED=314
FOREGROUND=0
SMOKE=0
RUN_BG=0

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --fg|--foreground) FOREGROUND=1 ;;
    --bg) RUN_BG=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  N_LENSES=1
  OUT_DIR="outputs/euclid_paper_hero_1arcmin_smoke"
  echo "[SMOKE] 1 lens -> ${OUT_DIR}"
fi

fail=0
[[ -d /Volumes/extHD ]] || { echo "ERROR: /Volumes/extHD not mounted"; fail=1; }
[[ -d data/euclid_q1_psf/tiles ]] || { echo "ERROR: Euclid Q1 PSF library missing"; fail=1; }
[[ -f data/cosmos_web_lens_structural_properties.csv ]] || { echo "ERROR: cosmos catalog missing"; fail=1; }
[[ "$fail" -eq 0 ]] || exit 1

mkdir -p "$OUT_DIR"
cp -f "$CONFIG" "$OUT_DIR/run_config.yaml"

LOG="$OUT_DIR/run.log"
echo "[RUN] config=$CONFIG"
echo "[RUN] out=$OUT_DIR  n_lenses=$N_LENSES  seed=$SEED"
echo "[RUN] FOV=1.0 arcmin @ 0.10 arcsec/pix (600×600)"
echo "[RUN] log=$LOG"
echo "[RUN] est. wall time: ~15–45 min for full candidate set"

CMD=(
  python -u src/jwst_lens_simulator.py
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

if [[ "$FOREGROUND" -eq 1 ]]; then
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
  python scripts/local/package_paper_hero_figure.py "$OUT_DIR"
elif [[ "$RUN_BG" -eq 1 ]]; then
  "${CMD[@]}" >>"$LOG" 2>&1
  PYTHON_BIN="$(command -v python || command -v python3)"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/local/package_paper_hero_figure.py" "$OUT_DIR" >>"$LOG" 2>&1
else
  nohup bash "$REPO_ROOT/scripts/local/run_paper_hero_1arcmin.sh" --bg >>"$LOG" 2>&1 &
  echo $! >"$OUT_DIR/run.pid"
  echo "[RUN] started pid=$(cat "$OUT_DIR/run.pid")"
  echo "[RUN] monitor: tail -f $LOG"
fi
