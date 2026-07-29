#!/usr/bin/env bash
# Run 1′ Euclid sims with COSMOS-Web density + shear + FP physics.
#
# Usage:
#   ./scripts/local/run_paper_physics_1arcmin.sh
#   ./scripts/local/run_paper_physics_1arcmin.sh --smoke
#   ./scripts/local/run_paper_physics_1arcmin.sh --fg

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/euclid_paper_physics_1arcmin.yaml"
OUT_DIR="outputs/euclid_paper_physics_1arcmin"
N_LENSES=10
N_FIELD_MAX=80
SEED=211
FOREGROUND=0
SMOKE=0

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --fg|--foreground) FOREGROUND=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown: $arg" >&2; exit 1 ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  N_LENSES=3
  OUT_DIR="outputs/euclid_paper_physics_1arcmin_smoke"
fi

fail=0
[[ -d /Volumes/extHD ]] || { echo "ERROR: /Volumes/extHD not mounted"; fail=1; }
[[ -d data/euclid_q1_psf/tiles ]] || { echo "ERROR: Euclid Q1 PSF missing"; fail=1; }
[[ -f data/galaxy_catalog.fits ]] || { echo "ERROR: galaxy_catalog.fits missing"; fail=1; }
[[ "$fail" -eq 0 ]] || exit 1

# Print COSMOS density table used for this run
python - <<'PY'
from prism.core.simulator import cosmos_field_density_per_arcmin2, field_galaxy_count_target
print("[COSMOS-Web] surface density (data/galaxy_catalog.fits):")
for m in (21.5, 23.5, 24.5, 26.0):
    d = cosmos_field_density_per_arcmin2(m)
    print(f"  mag < {m:.1f}:  {d:6.1f} /arcmin²   →  N≈{d:.0f} in 1′ (isolated)")
cfg = {"field": {"density_mag_limit": 23.5}, "catalogs": {"galaxy_catalog_fits": "data/galaxy_catalog.fits"}}
for name, mean in [("isolated", 2.5), ("pair", 3.0), ("group", 4.5)]:
    m, s = field_galaxy_count_target(600, 0.10, {"galaxy_count_mean": mean}, cfg)
    print(f"  1′ target [{name:8s}]: N = {m:.1f} ± {s:.1f}")
PY

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

echo "[RUN] out=$OUT_DIR n=$N_LENSES seed=$SEED"
if [[ "$FOREGROUND" -eq 1 ]]; then
  "${CMD[@]}" 2>&1 | tee "$LOG"
else
  nohup "${CMD[@]}" >"$LOG" 2>&1 &
  echo $! >"$OUT_DIR/run.pid"
  echo "[BG] pid=$(cat "$OUT_DIR/run.pid")  tail -f $LOG"
fi
