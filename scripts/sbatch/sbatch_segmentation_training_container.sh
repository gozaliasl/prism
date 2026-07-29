#!/bin/bash
#SBATCH --job-name=seg_train
#SBATCH --time=24:00:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=ituomine
# Note: Output/error paths will be set after detecting project directory

#============================================================================
# Segmentation Training Pipeline - Mahti (Container Version)
#============================================================================
# This script uses a Singularity/Apptainer container to avoid environment
# conflicts. The container includes all necessary packages.
#============================================================================

set -e

# Note: Output redirection happens after detecting project directory
# Initial echo statements go to default output until redirection

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect scratch directory
if [ -n "$SCRATCH" ]; then
    SCRATCH_BASE="$SCRATCH"
elif [ -n "$SLURM_ACCOUNT" ] && [ -n "$USER" ]; then
    SCRATCH_BASE="/scratch/$SLURM_ACCOUNT/$USER"
elif [ -n "$ACCOUNT" ] && [ -n "$USER" ]; then
    SCRATCH_BASE="/scratch/$ACCOUNT/$USER"
else
    SCRATCH_BASE="/scratch/$USER"
fi

# Use the actual project directory path
if [ -d "/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator" ]; then
    WORK_DIR="/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator"
    OUTPUT_BASE="/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator/output"
else
    WORK_DIR="$SCRATCH_BASE/jwst-mock-lens-simulator"
    OUTPUT_BASE="$SCRATCH_BASE/jwst-mock-lens-simulator/output"
fi

# Container path - automatically detect in common locations
# Priority: 1) Environment variable, 2) Project directory, 3) Scratch root
if [ -n "$CONTAINER_PATH" ] && [ -f "$CONTAINER_PATH" ]; then
    # Use explicitly set path
    :
elif [ -f "/scratch/ituomine/gozaliasl/prism.sif" ]; then
    # Use known project location
    CONTAINER_PATH="/scratch/ituomine/gozaliasl/prism.sif"
elif [ -f "$SCRATCH_BASE/prism.sif" ]; then
    # Use scratch root
    CONTAINER_PATH="$SCRATCH_BASE/prism.sif"
elif [ -f "$WORK_DIR/prism.sif" ]; then
    # Use project directory
    CONTAINER_PATH="$WORK_DIR/prism.sif"
else
    # Default fallback
    CONTAINER_PATH="$SCRATCH_BASE/prism.sif"
fi

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "ERROR: Container not found: $CONTAINER_PATH"
    echo ""
    echo "Please:"
    echo "  1. Build container: cd container && ./build_container.sh"
    echo "  2. Transfer to Mahti: scp prism.sif <user>@mahti.csc.fi:$SCRATCH_BASE/"
    echo "  3. Set CONTAINER_PATH environment variable or update this script"
    exit 1
fi

echo "Working directory: $WORK_DIR"
echo "Output directory: $OUTPUT_BASE"
echo "Container: $CONTAINER_PATH"
echo ""

# Parse command-line arguments (passed via sbatch --export or script)
N_LENSES="${N_LENSES:-10000}"
N_NONLENSES="${N_NONLENSES:-0}"
# Variations will be auto-calculated in segmentation_training_pipeline.sh if not set
N_VARIATIONS="${N_VARIATIONS:-auto}"
SEED="${SEED:-42}"

# Number of available true lenses in COWLS catalog
COWLS_N_LENSES=434
TIME_DELAY_FRACTION="${TIME_DELAY_FRACTION:-0.0}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"

# Auto-calculate variations if needed
if [ "$N_VARIATIONS" = "auto" ] || [ -z "$N_VARIATIONS" ]; then
    if [ "$N_LENSES" -gt 0 ]; then
        N_VARIATIONS=$(( (N_LENSES + COWLS_N_LENSES - 1) / COWLS_N_LENSES ))
    else
        N_VARIATIONS=1
    fi
fi

echo "Configuration:"
echo "  - Lenses: $N_LENSES"
echo "  - Non-lenses: $N_NONLENSES"
echo "  - Variations per base: $N_VARIATIONS (will generate ~$((N_VARIATIONS * COWLS_N_LENSES)) total lenses)"
echo "  - Seed: $SEED"
echo "  - Time delay fraction: $TIME_DELAY_FRACTION"
echo "  - Training epochs: $TRAINING_EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Device: $DEVICE"
echo ""

# Create directories
mkdir -p "$WORK_DIR"
mkdir -p "$OUTPUT_BASE"

# Create logs directory in project (not system directory)
LOG_DIR="$WORK_DIR/logs/mahti"
mkdir -p "$LOG_DIR"

# Log paths are set by submit script via #SBATCH directives
# Just echo where they should be for reference
LOG_OUTPUT="$LOG_DIR/segmentation_training_${SLURM_JOB_ID}.out"
LOG_ERROR="$LOG_DIR/segmentation_training_${SLURM_JOB_ID}.err"

echo "============================================================================"
echo "Segmentation Training Pipeline - Mahti (Container)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Log output: $LOG_OUTPUT"
echo "Log error: $LOG_ERROR"
echo ""

# Copy project to scratch (if not already there)
if [ ! -d "$WORK_DIR/src" ]; then
    echo "Copying project to scratch..."
    rsync -av \
        --exclude='.git' \
        --exclude='venv*' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='outputs' \
        --exclude='logs' \
        --exclude='job*' \
        --exclude='*_*.*' \
        --exclude='hwloc_topo_*.xml' \
        --exclude='/var/spool/slurmd' \
        --exclude='slurm-*.out' \
        --exclude='slurm-*.err' \
        "$PROJECT_ROOT/" "$WORK_DIR/" 2>&1 | grep -v "Permission denied" || true
fi

# Generate output directory name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$OUTPUT_BASE/segmentation_training_${TIMESTAMP}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run pipeline inside container
echo "============================================================================"
echo "Running segmentation training pipeline in container..."
echo "============================================================================"
echo ""

# Initialize module system for PyTorch (required since container doesn't include it)
initialize_module_system() {
    if command -v module &>/dev/null; then
        return 0
    fi
    local candidates=(
        /appl/profile/zz-csc-env.sh
        /appl/profile/zz-mahti-modules.sh
        /appl/Modules/init/bash
        /usr/share/lmod/lmod/init/bash
        /usr/share/modules/init/bash
        /etc/profile.d/modules.sh
    )
    local init
    for init in "${candidates[@]}"; do
        if [ -f "$init" ]; then
            source "$init"
            if command -v module &>/dev/null; then
                return 0
            fi
        fi
    done
    return 1
}

if initialize_module_system; then
    echo "Loading PyTorch module (required for container)..."
    module load pytorch/2.0 || {
        echo "WARNING: Could not load pytorch/2.0 module"
        echo "PyTorch may not be available in container"
    }
else
    echo "WARNING: Module system not available"
    echo "PyTorch may not be available in container"
fi

# Use Singularity/Apptainer to run the pipeline script
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD=apptainer
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD=singularity
else
    echo "ERROR: Neither 'apptainer' nor 'singularity' found"
    exit 1
fi

NUMBA_CACHE_DIR_HOST="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$NUMBA_CACHE_DIR_HOST"
export NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR_HOST"
export NUMBA_DISABLE_CACHE=1

MPLCONFIGDIR_HOST="${MPLCONFIGDIR:-/tmp/mplconfig_${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR_HOST"
export MPLCONFIGDIR="$MPLCONFIGDIR_HOST"

# Run container with module-provided PyTorch accessible
# The container will use PyTorch from the host's module system
# Bind the actual project directory to /workspace in container
$CONTAINER_CMD exec \
    --nv \
    --bind "$WORK_DIR:/workspace" \
    --bind "$OUTPUT_BASE:/output" \
    --pwd /workspace \
    --env NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}" \
    --env NUMBA_DISABLE_CACHE=1 \
    --env MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID:-$$}" \
    "$CONTAINER_PATH" \
    bash -c "
        mkdir -p /tmp/numba_cache_${SLURM_JOB_ID:-$$} /tmp/mplconfig_${SLURM_JOB_ID:-$$} && \
        cd /workspace && \
        bash scripts/segmentation_training_pipeline.sh \
            --output-dir /output/segmentation_training_${TIMESTAMP} \
            --n-lenses $N_LENSES \
            --n-non-lenses $N_NONLENSES \
            --variations $N_VARIATIONS \
            --seed $SEED \
            --time-delay-fraction $TIME_DELAY_FRACTION \
            --training-epochs $TRAINING_EPOCHS \
            --batch-size $BATCH_SIZE \
            --device $DEVICE
    "

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ Pipeline completed successfully!"
    echo "============================================================================"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "To check results:"
    echo "  ls -lh $OUTPUT_DIR"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "❌ Pipeline failed with exit code: $EXIT_CODE"
    echo "============================================================================"
    echo ""
    echo "Check logs:"
    echo "  tail -f $LOG_ERROR"
    echo ""
    exit $EXIT_CODE
fi
