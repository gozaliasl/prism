#!/bin/bash
#SBATCH --job-name=seg_sim
#SBATCH --time=12:00:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=ituomine

set -e

# Arguments (with defaults)
N_LENSES="${N_LENSES:-10000}"
N_NON_LENSES="${N_NON_LENSES:-0}"
N_VARIATIONS="${N_VARIATIONS:-auto}"
SEED="${SEED:-42}"
TIME_DELAY_FRACTION="${TIME_DELAY_FRACTION:-0.0}"
OUTPUT_BASE_DEFAULT="/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator/output"
OUTPUT_BASE="${OUTPUT_BASE:-$OUTPUT_BASE_DEFAULT}"
WORK_DIR_DEFAULT="/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator"
WORK_DIR="${WORK_DIR:-$WORK_DIR_DEFAULT}"
CONTAINER_PATH="${CONTAINER_PATH:-/scratch/ituomine/gozaliasl/prism.sif}"

# Derive variations if auto
COWLS_N_LENSES=434
if [ "$N_VARIATIONS" = "auto" ] || [ -z "$N_VARIATIONS" ]; then
    if [ "$N_LENSES" -gt 0 ]; then
        N_VARIATIONS=$(( (N_LENSES + COWLS_N_LENSES - 1) / COWLS_N_LENSES ))
    else
        N_VARIATIONS=1
    fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$OUTPUT_BASE/segmentation_training_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WORK_DIR/logs/mahti"

echo "============================================================================"
echo "Simulation Step (Container)"
echo "============================================================================"
echo "Work dir: $WORK_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Config:"
echo "  - Lenses: $N_LENSES"
echo "  - Non-lenses: $N_NON_LENSES"
echo "  - Variations: $N_VARIATIONS"
echo "  - Time delay fraction: $TIME_DELAY_FRACTION"
echo "  - Seed: $SEED"
echo "============================================================================"

# Initialize module system for GPU access
if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load pytorch/2.0 || true

# Use apptainer/singularity
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD=apptainer
else
    CONTAINER_CMD=singularity
fi

# Avoid numba caching issues
export NUMBA_CACHE_DIR=/tmp/numba_cache
export NUMBA_DISABLE_CACHING=1

$CONTAINER_CMD exec \
  --nv \
  --bind "$WORK_DIR:/workspace" \
  --bind "$OUTPUT_BASE:/output" \
  --env NUMBA_CACHE_DIR=/tmp/numba_cache \
  --env NUMBA_DISABLE_CACHING=1 \
  --pwd /workspace \
  "$CONTAINER_PATH" \
  bash -c "
    set -e
    mkdir -p /output && \
    python3 -m prism.core.simulator \
      --mode training \
      --output-dir /output/segmentation_training_${TIMESTAMP} \
      --n-lenses $N_LENSES \
      --n-non-lenses $N_NON_LENSES \
      --variations $N_VARIATIONS \
      --seed $SEED \
      --time-delay-fraction $TIME_DELAY_FRACTION \
      --numpix 300
  "

echo "$OUTPUT_DIR" > "$WORK_DIR/logs/mahti/last_seg_output_dir.txt"
echo "Simulation finished. Output: $OUTPUT_DIR"


