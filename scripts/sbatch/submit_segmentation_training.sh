#!/bin/bash
#
# Submit Segmentation Training Pipeline to Mahti
#
# This script submits the segmentation training pipeline job to Mahti.
# It sets up the job parameters and submits using sbatch.
#
# Usage:
#   ./submit_segmentation_training.sh [OPTIONS]
#

set -e

# Default values
N_LENSES=10000
N_NON_LENSES=0
VARIATIONS=1
SEED=42
TIME_DELAY_FRACTION=0.0
TRAINING_EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
DEVICE=cuda
RUN_DETECTION=false
LENS_IDS_FOR_DETECTION=""
ACCOUNT="${ACCOUNT:-ituomine}"  # Default account
VENV_PATH=""

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Submit segmentation training pipeline job to Mahti.

REQUIRED:
  --account ACCOUNT        Project account (e.g., project_XXXXXXX)

OPTIONS:
  --n-lenses N            Number of lens systems (default: 10000)
  --n-non-lenses N        Number of non-lens systems (default: 0)
  --variations N          Variations per base lens (default: 1)
  --seed N                Random seed (default: 42)
  --time-delay-fraction F Fraction with time delays, 0.0-1.0 (default: 0.0)
  --training-epochs N      Training epochs (default: 50)
  --batch-size N          Batch size (default: 32)
  --learning-rate F       Learning rate (default: 0.001)
  --run-detection         Run detection/annotation after training
  --detect-lens-ids IDS   Comma-separated lens IDs for detection
  --help                  Show this help message
  --venv-path PATH        Path to virtualenv to use inside job (optional)

EXAMPLES:
  # Basic submission (10k lenses, no time delays)
  $0 --account project_XXXXXXX

  # With time delays
  $0 --account project_XXXXXXX --time-delay-fraction 0.15

  # Smaller test run
  $0 --account project_XXXXXXX --n-lenses 1000 --training-epochs 20

  # With detection
  $0 --account project_XXXXXXX --run-detection --detect-lens-ids "0,1,2,3,4"

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
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
        --run-detection)
            RUN_DETECTION=true
            shift
            ;;
        --detect-lens-ids)
            LENS_IDS_FOR_DETECTION="$2"
            shift 2
            ;;
        --venv-path)
            VENV_PATH="$2"
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

# Account defaults to ituomine if not specified
if [[ -z "$ACCOUNT" ]]; then
    ACCOUNT="ituomine"
    echo "ℹ️  Using default account: $ACCOUNT"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_segmentation_training.sh"

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "❌ Error: sbatch script not found: $SBATCH_SCRIPT"
    exit 1
fi

# Build export string for environment variables
EXPORT_VARS="N_LENSES=$N_LENSES"
EXPORT_VARS="$EXPORT_VARS,N_NON_LENSES=$N_NON_LENSES"
EXPORT_VARS="$EXPORT_VARS,VARIATIONS=$VARIATIONS"
EXPORT_VARS="$EXPORT_VARS,SEED=$SEED"
EXPORT_VARS="$EXPORT_VARS,TIME_DELAY_FRACTION=$TIME_DELAY_FRACTION"
EXPORT_VARS="$EXPORT_VARS,TRAINING_EPOCHS=$TRAINING_EPOCHS"
EXPORT_VARS="$EXPORT_VARS,BATCH_SIZE=$BATCH_SIZE"
EXPORT_VARS="$EXPORT_VARS,LEARNING_RATE=$LEARNING_RATE"
EXPORT_VARS="$EXPORT_VARS,DEVICE=$DEVICE"
EXPORT_VARS="$EXPORT_VARS,RUN_DETECTION=$RUN_DETECTION"
if [[ -n "$VENV_PATH" ]]; then
    EXPORT_VARS="$EXPORT_VARS,VENV_PATH=$VENV_PATH"
fi

if [[ -n "$LENS_IDS_FOR_DETECTION" ]]; then
    EXPORT_VARS="$EXPORT_VARS,LENS_IDS_FOR_DETECTION=$LENS_IDS_FOR_DETECTION"
fi

# Update account in sbatch script (temporary)
TEMP_SBATCH="/tmp/sbatch_segmentation_${USER}_$$.sh"
sed "s/project_XXXXXXX/$ACCOUNT/" "$SBATCH_SCRIPT" > "$TEMP_SBATCH"

# Submit job
echo "🚀 Submitting segmentation training pipeline to Mahti..."
echo ""
echo "Configuration:"
echo "  - Account: $ACCOUNT"
echo "  - Lenses: $N_LENSES"
echo "  - Time delay fraction: $TIME_DELAY_FRACTION"
echo "  - Training epochs: $TRAINING_EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Device: $DEVICE"
echo ""

JOB_ID=$(sbatch --export="$EXPORT_VARS" "$TEMP_SBATCH" | grep -oP '\d+')

if [ $? -eq 0 ] && [ -n "$JOB_ID" ]; then
    echo "✅ Job submitted successfully!"
    echo "   Job ID: $JOB_ID"
    echo ""
    echo "📋 Monitor job:"
    echo "   squeue -j $JOB_ID"
    echo ""
    echo "📄 View output:"
    echo "   tail -f logs/segmentation_training_${JOB_ID}.out"
    echo ""
    echo "📄 View errors:"
    echo "   tail -f logs/segmentation_training_${JOB_ID}.err"
    echo ""
    echo "❌ Cancel job:"
    echo "   scancel $JOB_ID"
else
    echo "❌ Job submission failed!"
    exit 1
fi

# Cleanup
rm -f "$TEMP_SBATCH"

