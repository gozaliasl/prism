#!/usr/bin/env bash
# run_jades_10k.sh
# Generate a 10,000-lens JADES-matched JWST/NIRCam strong-lens dataset
# (F115W/F150W/F277W/F444W, JADES Deep-tier depth/pixel scale) for use as
# training data for strong-lens detection in the JADES survey.
#
# Output goes to /Volumes/extHD (per project convention), runs in the
# background so it can be left running overnight.
#
# Usage:
#   ./scripts/local/run_jades_10k.sh [N_LENSES] [SEED]
#
# Defaults: N_LENSES=10000, SEED=42
# At ~8 s/lens (measured w/ 8 bands), 10,000 lenses takes ~22 hours.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    PYTHON="${CONDA_BASE}/envs/astro-clean/bin/python"
else
    PYTHON="python"
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "       Activate the astro-clean conda environment and re-run."
    exit 1
fi

N_LENSES="${1:-10000}"
SEED="${2:-42}"

# variations_per_base: 434 base COSMOS-Web configurations x 23 ~= 9982 ~= 10k
VARIATIONS_PER_BASE=23

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/Volumes/extHD/jwst-lens-similator-output/jades_${N_LENSES}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "  JELSIM — JADES 10k JWST Strong-Lens Generation"
echo "  Started:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "  N lenses:   ${N_LENSES}"
echo "  Seed:       ${SEED}"
echo "  Output:     ${OUTPUT_DIR}"
echo "========================================================"

nohup "$PYTHON" -m prism.core.simulator \
    --config "$PROJECT_ROOT/configs/jades_config.yaml" \
    --cosmos_catalog "$PROJECT_ROOT/data/cosmos_web_lens_structural_properties.csv" \
    --lens_analysis_catalog "$PROJECT_ROOT/data/lens_analysis_catalog.csv" \
    --merged_field_catalog "$PROJECT_ROOT/data/merged_lens_field_catalog.csv" \
    --output_dir "$OUTPUT_DIR" \
    --n_lenses "$N_LENSES" --n_non_lenses 0 \
    --variations_per_base "$VARIATIONS_PER_BASE" \
    --n_field_max 5 \
    --seed "$SEED" --no_date_suffix --add_artifacts \
    > "$OUTPUT_DIR/run_log.txt" 2>&1 &

PID=$!
echo "$PID" > "$OUTPUT_DIR/run.pid"
echo ""
echo "Launched in background with PID $PID"
echo "Monitor with:"
echo "  tail -f $OUTPUT_DIR/run_log.txt"
echo "  ls $OUTPUT_DIR/unified_npz | wc -l"
