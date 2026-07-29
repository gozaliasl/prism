#!/bin/bash
#
# Complete Segmentation Training Pipeline
#
# This script automates the full pipeline:
# 1. Run lens simulations (with optional time delays)
# 2. Prepare segmentation training data
# 3. Train U-Net segmentation model
# 4. (Optional) Run detection/annotation on simulated images
#
# Usage:
#   ./segmentation_training_pipeline.sh [OPTIONS]
#

set -e  # Exit on error

# Ensure numba can cache in writable location (helps in containers/HPC)
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$NUMBA_CACHE_DIR"
export NUMBA_CACHE_DIR
export NUMBA_DISABLE_CACHE="${NUMBA_DISABLE_CACHE:-1}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

# Default values
N_LENSES=10000
N_NON_LENSES=0
VARIATIONS=""  # Will be auto-calculated if not provided
SEED=42
OUTPUT_DIR=""
TIME_DELAY_FRACTION=0.0  # Default: no time delays (simpler, no epochs)
TRAINING_EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001
DEVICE="cpu"
RUN_DETECTION=false
LENS_IDS_FOR_DETECTION=""

# Number of available true lenses in COWLS catalog
COWLS_N_LENSES=434

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Complete pipeline for segmentation-based lensed source detection:
  1. Run lens simulations
  2. Prepare training data (segmentation masks)
  3. Train U-Net model
  4. (Optional) Run detection/annotation

OPTIONS:
  --n-lenses N              Number of lens systems (default: 10000)
  --n-non-lenses N          Number of non-lens systems (default: 0)
  --variations N            Variations per base lens (default: 1)
  --seed N                  Random seed (default: 42)
  --output-dir DIR          Output directory (default: auto-generated)
  --time-delay-fraction F   Fraction with time delays, 0.0-1.0 (default: 0.0 = no time delays)
  --training-epochs N       Training epochs (default: 50)
  --batch-size N            Batch size (default: 16)
  --learning-rate F         Learning rate (default: 0.001)
  --device DEVICE           Device: cpu or cuda (default: cpu)
  --run-detection           Run detection/annotation after training
  --detect-lens-ids IDS     Comma-separated lens IDs for detection (e.g., "0,1,2")
  --help                    Show this help message

EXAMPLES:
  # Full pipeline with 10k lenses, no time delays, train model
  $0 --n-lenses 10000 --time-delay-fraction 0.0

  # With time delays (15% of lenses)
  $0 --n-lenses 10000 --time-delay-fraction 0.15

  # Train and then run detection on specific lenses
  $0 --n-lenses 5000 --run-detection --detect-lens-ids "0,1,2,3,4"

  # Use GPU for training
  $0 --n-lenses 10000 --device cuda --training-epochs 100

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-lenses)
            N_LENSES="$2"
            shift 2
            ;;
        --n-non-lenses)
            N_NON_LENSES="$2"
            shift 2
            ;;
        --variations)
            VARIATIONS="$2"
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
        --time-delay-fraction)
            TIME_DELAY_FRACTION="$2"
            shift 2
            ;;
        --training-epochs)
            TRAINING_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --run-detection)
            RUN_DETECTION=true
            shift
            ;;
        --detect-lens-ids)
            LENS_IDS_FOR_DETECTION="$2"
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
    OUTPUT_DIR="outputs/segmentation_training_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"

# Create temporary config file with custom time delay settings
TEMP_CONFIG="/tmp/jwst_segmentation_${TIMESTAMP}.yaml"
cp "$CONFIG" "$TEMP_CONFIG"

# Update time delay settings in temp config
python3 << PYEOF
import yaml
import sys

# Load config
with open("$CONFIG", 'r') as f:
    config = yaml.safe_load(f)

# Update time delay settings
if 'time_delays' not in config:
    config['time_delays'] = {}

config['time_delays']['enabled'] = float("$TIME_DELAY_FRACTION") > 0.0
config['time_delays']['fraction_variable_sources'] = float("$TIME_DELAY_FRACTION")

# Save updated config
with open("$TEMP_CONFIG", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Updated config saved to: $TEMP_CONFIG")
PYEOF

CONFIG="$TEMP_CONFIG"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 SEGMENTATION TRAINING PIPELINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  - Lenses: $N_LENSES"
echo "  - Non-lenses: $N_NON_LENSES"
echo "  - Variations: $VARIATIONS"
echo "  - Seed: $SEED"
# Calculate percentage without bc (bash-native)
TIME_DELAY_PERCENT=$(awk "BEGIN {printf \"%.0f\", $TIME_DELAY_FRACTION * 100}")
echo "  - Time delay fraction: $TIME_DELAY_FRACTION (${TIME_DELAY_PERCENT}%)"
echo "  - Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# STEP 1: Run Simulations
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📸 STEP 1: Running Lens Simulations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 -m prism.core.simulator \
    --config "$CONFIG" \
    --cosmos_catalog "$COSMOS_CATALOG" \
    --lens_analysis_catalog "$LENS_ANALYSIS" \
    --merged_field_catalog "$FIELD_CATALOG" \
    --output_dir "$OUTPUT_DIR" \
    --n_lenses $N_LENSES \
    --n_non_lenses $N_NON_LENSES \
    --variations $VARIATIONS \
    --seed $SEED \
    --add_artifacts \
    --numpix 300

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Simulation failed with exit code $EXIT_CODE"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

echo ""
echo "✅ Simulations completed!"
echo ""

# ============================================================================
# STEP 2: Prepare Training Data
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STEP 2: Preparing Segmentation Training Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TRAINING_DIR="$OUTPUT_DIR/segmentation_training"

python3 scripts/local/prepare_segmentation_training_data.py \
    --output-dir "$OUTPUT_DIR" \
    --training-dir "$TRAINING_DIR" \
    --patch-size 128 \
    --stride 64

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Training data preparation failed with exit code $EXIT_CODE"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

# Check if training data was created
if [ ! -f "$TRAINING_DIR/training_metadata.csv" ]; then
    echo "❌ ERROR: Training metadata file not found!"
    echo "   Expected: $TRAINING_DIR/training_metadata.csv"
    echo "   This means no training patches were extracted."
    echo ""
    echo "   Possible causes:"
    echo "   - Missing image_positions_arcsec in catalog"
    echo "   - lenstronomy calculation failed"
    echo "   - Image files not found"
    echo "   - Mask creation failed (0 pixels in masks)"
    echo ""
    echo "   Check the output above for warnings/errors."
    rm -f "$TEMP_CONFIG"
    exit 1
fi

# Check if we have any training patches
N_PATCHES=$(wc -l < "$TRAINING_DIR/training_metadata.csv" 2>/dev/null || echo "0")
N_PATCHES=$((N_PATCHES - 1))  # Subtract header line

if [ "$N_PATCHES" -le 0 ]; then
    echo "❌ ERROR: No training patches extracted!"
    echo "   Found $N_PATCHES patches (need at least 1)"
    echo "   Check the output above for warnings/errors."
    rm -f "$TEMP_CONFIG"
    exit 1
fi

echo "   ✅ Found $N_PATCHES training patches"

echo ""
echo "✅ Training data preparation completed!"
echo ""

# ============================================================================
# STEP 3: Train U-Net Model
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 STEP 3: Training U-Net Segmentation Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MODEL_PATH="$OUTPUT_DIR/unet_segmentation_model.pth"

python3 scripts/local/train_segmentation_model.py \
    --training-dir "$TRAINING_DIR" \
    --model-output "$MODEL_PATH" \
    --epochs $TRAINING_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --device "$DEVICE"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Model training failed with exit code $EXIT_CODE"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

echo ""
echo "✅ Model training completed!"
echo "   Model saved to: $MODEL_PATH"
echo ""

# ============================================================================
# STEP 4: (Optional) Run Detection/Annotation
# ============================================================================
if [ "$RUN_DETECTION" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 STEP 4: Running Detection/Annotation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [[ -z "$LENS_IDS_FOR_DETECTION" ]]; then
        # Detect on first 10 lenses by default
        LENS_IDS_FOR_DETECTION="0,1,2,3,4,5,6,7,8,9"
        echo "No lens IDs specified, using first 10: $LENS_IDS_FOR_DETECTION"
    fi
    
    # Convert comma-separated IDs to array
    IFS=',' read -ra LENS_ID_ARRAY <<< "$LENS_IDS_FOR_DETECTION"
    
    for LENS_ID in "${LENS_ID_ARRAY[@]}"; do
        LENS_ID=$(echo "$LENS_ID" | xargs)  # Trim whitespace
        
        echo "  Processing lens $LENS_ID..."
        
        python3 scripts/local/identify_lensed_images.py \
            --output-dir "$OUTPUT_DIR" \
            --lens-id "$LENS_ID" \
            --epoch 0 \
            --method segmentation \
            --model-path "$MODEL_PATH" || echo "  ⚠️  Failed to process lens $LENS_ID"
    done
    
    echo ""
    echo "✅ Detection/annotation completed!"
    echo "   Annotated images saved to: $OUTPUT_DIR/annotated"
    echo ""
fi

# Cleanup
rm -f "$TEMP_CONFIG"

# ============================================================================
# Summary
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Output Directory: $OUTPUT_DIR"
echo "📊 Training Data: $TRAINING_DIR"
echo "🤖 Trained Model: $MODEL_PATH"
echo ""
echo "💡 Next Steps:"
echo "   - Use the trained model for detection:"
echo "     python3 scripts/local/identify_lensed_images.py \\"
echo "       --output-dir <output_dir> \\"
echo "       --lens-id <id> \\"
echo "       --method segmentation \\"
echo "       --model-path $MODEL_PATH"
echo ""
echo "   - Or run detection on multiple lenses:"
echo "     for i in {0..9}; do"
echo "       python3 scripts/local/identify_lensed_images.py \\"
echo "         --output-dir $OUTPUT_DIR \\"
echo "         --lens-id \$i \\"
echo "         --method segmentation \\"
echo "         --model-path $MODEL_PATH"
echo "     done"
echo ""

# Final step markers
echo "STEP_1: Simulation output -> $OUTPUT_DIR" | tee "$OUTPUT_DIR/STEP_1.txt"
echo "STEP_2: Training data -> $TRAINING_DIR" | tee "$OUTPUT_DIR/STEP_2.txt"
echo "STEP_3: Trained model -> $MODEL_PATH" | tee "$OUTPUT_DIR/STEP_3.txt"
echo ""
