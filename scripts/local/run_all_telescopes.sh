#!/usr/bin/env bash
# run_all_telescopes.sh
# Run JELSIM for all supported telescopes: JWST, Roman, Euclid, Subaru, LSST
#
# Usage:
#   ./scripts/local/run_all_telescopes.sh [OPTIONS]
#
# Options:
#   --n-lenses N        Number of lenses per telescope  (default: 50)
#   --n-non-lenses N    Number of non-lens systems      (default: 0)
#   --seed N            Random seed                     (default: 42)
#   --output-dir DIR    Base output directory           (default: outputs/all_telescopes_TIMESTAMP)
#   --telescopes LIST   Comma-separated subset          (default: jwst,roman,euclid,subaru,lsst)
#   --no-intermediate   Do not save intermediate images
#   --no-artifacts      Do not add observational artifacts
#   --help              Show this help

set -euo pipefail

# ── Locate project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# ── Resolve Python from astro-clean conda env ────────────────────────────────
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

# ── Defaults ─────────────────────────────────────────────────────────────────
N_LENSES=50
N_NON_LENSES=0
SEED=42
OUTPUT_DIR=""
TELESCOPES="jwst,roman,euclid,subaru,lsst"
SAVE_INTERMEDIATE=true
ADD_ARTIFACTS=true

# ── Argument parsing ─────────────────────────────────────────────────────────
show_usage() {
    sed -n '2,20p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-lenses)        N_LENSES="$2";        shift 2 ;;
        --n-non-lenses)    N_NON_LENSES="$2";    shift 2 ;;
        --seed)            SEED="$2";             shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";       shift 2 ;;
        --telescopes)      TELESCOPES="$2";       shift 2 ;;
        --no-intermediate) SAVE_INTERMEDIATE=false; shift ;;
        --no-artifacts)    ADD_ARTIFACTS=false;   shift ;;
        --help|-h)         show_usage ;;
        *) echo "Unknown option: $1"; show_usage ;;
    esac
done

# ── Output directory ─────────────────────────────────────────────────────────
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$PROJECT_ROOT/outputs/all_telescopes_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

# ── Build per-telescope YAML configs ─────────────────────────────────────────
CONFIG_DIR="$OUTPUT_DIR/configs"
mkdir -p "$CONFIG_DIR"

# Known telescopes (used for validation — no bash associative arrays for zsh compat)
KNOWN_TELESCOPES="jwst roman euclid subaru lsst"

$PYTHON - <<PYEOF
import yaml, copy, os, sys

with open("configs/default_config.yaml") as f:
    base = yaml.safe_load(f)

tel_bands = {
    "jwst":   ["F115W", "F150W", "F277W", "F444W"],
    "roman":  ["ROMAN_F106", "ROMAN_F129", "ROMAN_F158", "ROMAN_F184"],
    "euclid": ["EUCLID_VIS", "EUCLID_Y", "EUCLID_J", "EUCLID_H"],
    "subaru": ["SUBARU_G", "SUBARU_R", "SUBARU_I", "SUBARU_Z"],
    "lsst":   ["LSST_G", "LSST_R", "LSST_I", "LSST_Z"],
}

config_dir = "${CONFIG_DIR}"
os.makedirs(config_dir, exist_ok=True)

for tel, bands in tel_bands.items():
    cfg = copy.deepcopy(base)
    cfg["telescope"] = tel
    cfg["bands"]     = bands
    cfg["detector_chain"]["enabled"] = True
    cfg["add_artifacts"] = True
    cfg["save_intermediate_images"] = True
    out = os.path.join(config_dir, f"config_{tel}.yaml")
    with open(out, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

print(f"Configs written to {config_dir}")
PYEOF

# ── Common simulator arguments ────────────────────────────────────────────────
CATALOGS="--cosmos_catalog $PROJECT_ROOT/data/cosmos_web_lens_structural_properties.csv \
          --lens_analysis_catalog $PROJECT_ROOT/data/lens_analysis_catalog.csv \
          --merged_field_catalog $PROJECT_ROOT/data/merged_lens_field_catalog.csv"

COMMON="--n_lenses $N_LENSES --n_non_lenses $N_NON_LENSES --seed $SEED --no_date_suffix"
[[ "$ADD_ARTIFACTS"      == "true" ]] && COMMON="$COMMON --add_artifacts"
[[ "$SAVE_INTERMEDIATE"  == "true" ]] && COMMON="$COMMON --save_intermediate"

# ── Per-telescope runner ──────────────────────────────────────────────────────
run_telescope() {
    local TEL=$1
    local OUT="$OUTPUT_DIR/$TEL"
    mkdir -p "$OUT"

    echo ""
    echo "========================================"
    echo "  Telescope: $(echo "$TEL" | tr '[:lower:]' '[:upper:]')   ($(date '+%H:%M:%S'))"
    echo "  Output:    $OUT"
    echo "========================================"

    "$PYTHON" -m prism.core.simulator \
        --config "$CONFIG_DIR/config_${TEL}.yaml" \
        $CATALOGS \
        --output_dir "$OUT" \
        $COMMON \
        2>&1 | tee "$OUT/run_log.txt"

    echo "  $(echo "$TEL" | tr '[:lower:]' '[:upper:]') DONE  ($(date '+%H:%M:%S'))"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
IFS=',' read -ra TEL_LIST <<< "$TELESCOPES"

echo ""
echo "========================================================"
echo "  JELSIM — Multi-Telescope Simulation"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Telescopes: ${TELESCOPES}"
echo "  Lenses/tel: ${N_LENSES}  Non-lenses/tel: ${N_NON_LENSES}"
echo "  Seed: ${SEED}"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================================"

for TEL in "${TEL_LIST[@]}"; do
    TEL="$(echo "$TEL" | tr -d ' ')"
    if ! echo "$KNOWN_TELESCOPES" | grep -qw "$TEL"; then
        echo "WARNING: Unknown telescope '$TEL' — skipping"
        continue
    fi
    run_telescope "$TEL"
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  ALL TELESCOPES COMPLETE  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "========================================================"
printf "  %-10s %8s %12s %10s\n" "Telescope" "Saved" "Shape" "FOV"
printf "  %-10s %8s %12s %10s\n" "---------" "-----" "-----" "---"

$PYTHON - <<PYEOF
import numpy as np, os, json

tel_ps = {"jwst":0.031,"roman":0.11,"euclid":0.10,"subaru":0.168,"lsst":0.200}
out_base = "${OUTPUT_DIR}"

for tel in "${TELESCOPES}".split(","):
    tel = tel.strip()
    npz_dir = os.path.join(out_base, tel, "unified_npz")
    if not os.path.exists(npz_dir):
        print(f"  {tel:<10s} {'N/A':>8s}")
        continue
    files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    n = len(files)
    if not files:
        print(f"  {tel:<10s} {n:>8d}")
        continue
    try:
        d = np.load(os.path.join(npz_dir, sorted(files)[0]), allow_pickle=True)
        sh = d["image_final"].shape
        ps = tel_ps.get(tel, 0)
        fov = f"{sh[-1]*ps:.1f}\""
        print(f"  {tel:<10s} {n:>8d} {str(sh):>12s} {fov:>10s}")
    except Exception as e:
        print(f"  {tel:<10s} {n:>8d}  (error: {e})")
PYEOF

echo ""
echo "Output directory: $OUTPUT_DIR"
