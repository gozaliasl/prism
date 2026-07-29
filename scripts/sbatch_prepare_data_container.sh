#!/bin/bash
#SBATCH --job-name=seg_prep
#SBATCH --time=04:00:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=ituomine

set -e

WORK_DIR_DEFAULT="/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator"
WORK_DIR="${WORK_DIR:-$WORK_DIR_DEFAULT}"
CONTAINER_PATH="${CONTAINER_PATH:-/scratch/ituomine/gozaliasl/jwst_lens_simulator.sif}"

# Use last output dir if not provided
if [ -z "$SIM_OUTPUT_DIR" ] && [ -f "$WORK_DIR/logs/mahti/last_seg_output_dir.txt" ]; then
  SIM_OUTPUT_DIR="$(cat "$WORK_DIR/logs/mahti/last_seg_output_dir.txt")"
fi

if [ -z "$SIM_OUTPUT_DIR" ]; then
  echo "ERROR: SIM_OUTPUT_DIR not set and last output dir not found."
  exit 1
fi

echo "============================================================================"
echo "Prepare Training Data (Container)"
echo "============================================================================"
echo "Work dir: $WORK_DIR"
echo "Sim output: $SIM_OUTPUT_DIR"
echo "============================================================================"

if [ -f /usr/share/modules/init/bash ]; then
  source /usr/share/modules/init/bash
elif [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
module load pytorch/2.0 || true

if command -v apptainer &> /dev/null; then
  CONTAINER_CMD=apptainer
else
  CONTAINER_CMD=singularity
fi

export NUMBA_CACHE_DIR=/tmp/numba_cache
export NUMBA_DISABLE_CACHING=1

$CONTAINER_CMD exec \
  --nv \
  --bind "$WORK_DIR:/workspace" \
  --env NUMBA_CACHE_DIR=/tmp/numba_cache \
  --env NUMBA_DISABLE_CACHING=1 \
  --pwd /workspace \
  "$CONTAINER_PATH" \
  bash -c "
    set -e
    python3 scripts/prepare_segmentation_training_data.py \
      --output-dir \"$SIM_OUTPUT_DIR\"
  "

echo "Data preparation finished."


