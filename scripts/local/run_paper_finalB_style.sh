#!/usr/bin/env bash
# Generate final_B-style 1′ Euclid paper candidates (large θ_E, sparse bright field).
#
# Target aesthetic: outputs/euclid_final/final_B_fullmaps.jpg
#   • θ_E ~ 4–9″ (near-complete blue ring)
#   • ~12–18 bright group members, smooth elliptical BGG
#
# Usage:
#   ./scripts/local/run_paper_finalB_style.sh           # 12 candidates
#   ./scripts/local/run_paper_finalB_style.sh --smoke    # 2 lenses quick test
#   ./scripts/local/run_paper_finalB_style.sh --fg

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/euclid_paper_finalB_style.yaml"
OUT_DIR="outputs/euclid_paper_finalB_style"
N_LENSES=12
N_FIELD_MAX=18
SEED=76
FOREGROUND=0
SMOKE=0

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --fg|--foreground) FOREGROUND=1 ;;
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
  N_LENSES=2
  OUT_DIR="outputs/euclid_paper_finalB_style_smoke"
  echo "[SMOKE] ${N_LENSES} lenses -> ${OUT_DIR}"
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
echo "[RUN] FOV=1.0 arcmin @ 0.10 arcsec/pix (600×600), θ_E hard max=10″"
echo "[RUN] log=$LOG"

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

if [[ "$FOREGROUND" -eq 1 ]]; then
  "${CMD[@]}" 2>&1 | tee "$LOG"
else
  nohup "${CMD[@]}" >"$LOG" 2>&1 &
  echo $! >"$OUT_DIR/run.pid"
  echo "[BG] pid=$(cat "$OUT_DIR/run.pid")  tail -f $LOG"
fi
