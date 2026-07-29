#!/bin/bash
#SBATCH --job-name=seg_train_only
#SBATCH --time=12:00:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=ituomine

set -e

WORK_DIR_DEFAULT="/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator"
WORK_DIR="${WORK_DIR:-$WORK_DIR_DEFAULT}"
CONTAINER_PATH="${CONTAINER_PATH:-/scratch/ituomine/gozaliasl/prism.sif}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"

# Locate training dir from last simulation if not provided
if [ -z "$SIM_OUTPUT_DIR" ] && [ -f "$WORK_DIR/logs/mahti/last_seg_output_dir.txt" ]; then
  SIM_OUTPUT_DIR="$(cat "$WORK_DIR/logs/mahti/last_seg_output_dir.txt")"
fi

TRAIN_DIR_DEFAULT="$SIM_OUTPUT_DIR/segmentation_training"
TRAINING_DIR="${TRAINING_DIR:-$TRAIN_DIR_DEFAULT}"

if [ -z "$TRAINING_DIR" ] || [ ! -d "$TRAINING_DIR" ]; then
  echo "ERROR: TRAINING_DIR not found: $TRAINING_DIR"
  exit 1
fi

echo "============================================================================"
echo "Train Segmentation Model (Container)"
echo "============================================================================"
echo "Work dir: $WORK_DIR"
echo "Training dir: $TRAINING_DIR"
echo "Epochs: $TRAINING_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
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
  --pwd /workspace \
  --env NUMBA_CACHE_DIR=/tmp/numba_cache \
  --env NUMBA_DISABLE_CACHING=1 \
  "$CONTAINER_PATH" \
  bash -c "
    set -e
    python3 scripts/local/train_segmentation_model.py \
      --training-dir \"$TRAINING_DIR\" \
      --epochs $TRAINING_EPOCHS \
      --batch-size $BATCH_SIZE \
      --device $DEVICE
  "

echo "Training finished."


