#!/bin/bash
#
# Time Delay Simulation Script
# Simulates lens systems with time delays for variable sources (quasar, supernova, AGN)
#

set -e  # Exit on error

# Default values
N_LENSES=500
N_NON_LENSES=0
VARIATIONS=1
SEED=42
OUTPUT_DIR=""
SKIP_ML_TRAINING=true
FRACTION_VARIABLE=1.0  # Fraction of lenses with variable sources (1.0 = all lenses)
QUASAR_FRACTION=0.60   # Fraction of variable sources that are quasars
SUPERNOVA_FRACTION=0.25 # Fraction of variable sources that are supernovae
AGN_FRACTION=0.15      # Fraction of variable sources that are AGN

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --n-lenses N           Number of lens systems (default: 500)"
    echo "  --seed N               Random seed (default: 42)"
    echo "  --output-dir DIR       Output directory (default: auto-generated)"
    echo "  --variations N         Variations per base lens (default: 1)"
    echo "  --fraction-variable F  Fraction of lenses with variable sources (default: 1.0 = all)"
    echo "  --quasar-fraction F    Fraction of variable sources that are quasars (default: 0.60)"
    echo "  --supernova-fraction F Fraction of variable sources that are supernovae (default: 0.25)"
    echo "  --agn-fraction F       Fraction of variable sources that are AGN (default: 0.15)"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --n-lenses 500 --seed 42"
    echo "  $0 --n-lenses 1000 --fraction-variable 1.0  # All lenses have variable sources"
    echo "  $0 --n-lenses 500 --quasar-fraction 0.5 --supernova-fraction 0.3 --agn-fraction 0.2"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-lenses)
            N_LENSES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --variations)
            VARIATIONS="$2"
            shift 2
            ;;
        --fraction-variable)
            FRACTION_VARIABLE="$2"
            shift 2
            ;;
        --quasar-fraction)
            QUASAR_FRACTION="$2"
            shift 2
            ;;
        --supernova-fraction)
            SUPERNOVA_FRACTION="$2"
            shift 2
            ;;
        --agn-fraction)
            AGN_FRACTION="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Normalize source type fractions to sum to 1.0
TOTAL_FRACTION=$(echo "$QUASAR_FRACTION + $SUPERNOVA_FRACTION + $AGN_FRACTION" | bc -l)
if (( $(echo "$TOTAL_FRACTION != 1.0" | bc -l) )); then
    echo "⚠️  Warning: Source type fractions sum to $TOTAL_FRACTION, normalizing to 1.0"
    QUASAR_FRACTION=$(echo "$QUASAR_FRACTION / $TOTAL_FRACTION" | bc -l)
    SUPERNOVA_FRACTION=$(echo "$SUPERNOVA_FRACTION / $TOTAL_FRACTION" | bc -l)
    AGN_FRACTION=$(echo "$AGN_FRACTION / $TOTAL_FRACTION" | bc -l)
fi

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    source ~/.zshrc 2>/dev/null || true
    conda activate astro-clean 2>/dev/null || echo "Note: conda environment not activated"
fi

# Configuration
CONFIG="configs/default_config.yaml"
COSMOS_CATALOG="data/cosmos_web_lens_structural_properties.csv"
LENS_ANALYSIS="data/lens_analysis_catalog.csv"
FIELD_CATALOG="data/merged_lens_field_catalog.csv"

# Output directory setup
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/time_delays_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"

# Create temporary config file with custom time delay settings
TEMP_CONFIG="/tmp/jwst_time_delays_${TIMESTAMP}.yaml"
cp "$CONFIG" "$TEMP_CONFIG"

# Update time delay settings in temp config using Python
python3 << PYEOF
import yaml
import sys

# Load config
with open("$CONFIG", 'r') as f:
    config = yaml.safe_load(f)

# Update time delay settings
config['time_delays']['fraction_variable_sources'] = float("$FRACTION_VARIABLE")
config['time_delays']['source_types']['quasar']['fraction'] = float("$QUASAR_FRACTION")
config['time_delays']['source_types']['supernova']['fraction'] = float("$SUPERNOVA_FRACTION")
config['time_delays']['source_types']['agn']['fraction'] = float("$AGN_FRACTION")

# Save updated config
with open("$TEMP_CONFIG", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Updated config saved to: $TEMP_CONFIG")
PYEOF

CONFIG="$TEMP_CONFIG"

echo "🚀 TIME DELAY SIMULATION"
echo "========================"
echo ""
echo "Configuration:"
echo "  - Lenses: $N_LENSES"
echo "  - Seed: $SEED"
echo "  - Variations per base: $VARIATIONS"
echo "  - Fraction with variable sources: $FRACTION_VARIABLE ($(echo "$FRACTION_VARIABLE * 100" | bc -l | xargs printf "%.0f")%)"
echo "  - Output: $OUTPUT_DIR"
echo ""
echo "Source type distribution:"
echo "  - Quasars: $(echo "$QUASAR_FRACTION * 100" | bc -l | xargs printf "%.1f")%"
echo "  - Supernovae: $(echo "$SUPERNOVA_FRACTION * 100" | bc -l | xargs printf "%.1f")%"
echo "  - AGN: $(echo "$AGN_FRACTION * 100" | bc -l | xargs printf "%.1f")%"
echo ""
echo "Expected variable sources:"
N_VARIABLE=$(echo "$N_LENSES * $FRACTION_VARIABLE" | bc -l | xargs printf "%.0f")
echo "  - Total: ~$N_VARIABLE systems"
echo "  - Quasars: ~$(echo "$N_VARIABLE * $QUASAR_FRACTION" | bc -l | xargs printf "%.0f")"
echo "  - Supernovae: ~$(echo "$N_VARIABLE * $SUPERNOVA_FRACTION" | bc -l | xargs printf "%.0f")"
echo "  - AGN: ~$(echo "$N_VARIABLE * $AGN_FRACTION" | bc -l | xargs printf "%.0f")"
echo ""

# Run simulation
echo "🎬 Generating time delay simulations..."
echo ""

python3 src/jwst_lens_simulator.py \
    --config "$CONFIG" \
    --cosmos_catalog "$COSMOS_CATALOG" \
    --lens_analysis_catalog "$LENS_ANALYSIS" \
    --merged_field_catalog "$FIELD_CATALOG" \
    --output_dir "$OUTPUT_DIR" \
    --n_lenses $N_LENSES \
    --n_non_lenses $N_NON_LENSES \
    --variations_per_base $VARIATIONS \
    --seed $SEED \
    --add_artifacts \
    --numpix 300

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Time delay simulation completed!"
    echo "   Output: $OUTPUT_DIR"
    echo ""
    
    # Clean up temp config
    rm -f "$TEMP_CONFIG"
    
    # Check time delay catalog
    if [ -f "$OUTPUT_DIR/time_delay_catalog.csv" ]; then
        echo "📊 Time delay systems generated:"
        python3 << PYEOF
import pandas as pd
import sys
try:
    td_catalog = pd.read_csv("$OUTPUT_DIR/time_delay_catalog.csv")
    print(f"   Total: {len(td_catalog)} systems")
    print(f"   Source types:")
    for stype, count in td_catalog['source_type'].value_counts().items():
        pct = count / len(td_catalog) * 100
        print(f"     - {stype}: {count} ({pct:.1f}%)")
except Exception as e:
    print(f"   Error reading catalog: {e}")
PYEOF
    else
        echo "⚠️  No time_delay_catalog.csv found"
    fi
    
    echo ""
    echo "💡 Next steps:"
    echo "   - Check time_delay_catalog.csv for available systems"
    echo "   - Create demo figures using:"
    echo "     python3 scripts/time_delays/create_time_delay_demo_figure.py \\"
    echo "       --output-dir $OUTPUT_DIR \\"
    echo "       --lens-id <ID> \\"
    echo "       --source-type <quasar|supernova|agn> \\"
    echo "       --n-epochs 4 \\"
    echo "       --output-fig docs/jwst_slsim_paper/figures/time_delay_demo_<type>_<ID>.png"
else
    echo "❌ Simulation failed with exit code $EXIT_CODE"
    exit 1
fi

