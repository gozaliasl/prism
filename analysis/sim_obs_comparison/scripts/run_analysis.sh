#!/bin/bash
# Master orchestration script for sim vs obs comparison pipeline
# 
# Usage:
#   bash run_analysis.sh [--step 1|2|3|all] [--max-pairs 10]
#
# Steps:
#  1. Extract real lens properties from FITS
#  2. Match simulated to observed lenses
#  3. Generate visualizations
#  4. Compute metrics & generate report

set -e

STEP=${1:-"all"}
MAX_PAIRS=${2:-10}
WORKSPACE_ROOT="/Users/gozalig1/Projects/jwst-mock-lens-simulator"
ANALYSIS_DIR="${WORKSPACE_ROOT}/analysis/sim_obs_comparison"
SCRIPTS_DIR="${ANALYSIS_DIR}/scripts"

echo "======================================================"
echo "Sim vs Obs Comparison Pipeline"
echo "======================================================"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Analysis:  ${ANALYSIS_DIR}"
echo ""

# Step 1: Extract real lens properties
if [[ "$STEP" == "1" ]] || [[ "$STEP" == "all" ]]; then
    echo "[Step 1] Extracting real lens properties..."
    python3 "${SCRIPTS_DIR}/extract_real_lens_properties.py"
    echo ""
fi

# Step 2: Match simulated to observed
if [[ "$STEP" == "2" ]] || [[ "$STEP" == "all" ]]; then
    echo "[Step 2] Matching simulated to observed lenses..."
    python3 "${SCRIPTS_DIR}/match_sim_to_obs.py"
    echo ""
fi

# Step 3: Generate visualizations
if [[ "$STEP" == "3" ]] || [[ "$STEP" == "all" ]]; then
    echo "[Step 3] Generating comparison visualizations..."
    python3 "${SCRIPTS_DIR}/visualize_comparisons.py" --max-pairs "${MAX_PAIRS}" --stretch log
    echo ""
fi

# Step 4: Generate report (coming)
if [[ "$STEP" == "4" ]] || [[ "$STEP" == "all" ]]; then
    echo "[Step 4] Computing metrics and generating report..."
    echo "  (Not yet implemented)"
    echo ""
fi

echo "======================================================"
echo "Pipeline complete!"
echo "Check ${ANALYSIS_DIR} for outputs"
echo "======================================================"
