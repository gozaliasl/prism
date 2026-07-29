#!/bin/bash
# Submit simulate -> prepare -> train as a chained workflow using afterok dependencies
# Usage:
#   ./scripts/submit_segmentation_chain_container.sh --account ituomine --n-lenses 10000

set -e

ACCOUNT=""
N_LENSES="10000"
TIME_DELAY_FRACTION="0.0"
N_VARIATIONS="auto"
TRAINING_EPOCHS="50"
BATCH_SIZE="32"
DEVICE="cuda"
CONTAINER_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) ACCOUNT="$2"; shift 2 ;;
    --n-lenses) N_LENSES="$2"; shift 2 ;;
    --variations) N_VARIATIONS="$2"; shift 2 ;;
    --time-delay-fraction) TIME_DELAY_FRACTION="$2"; shift 2 ;;
    --training-epochs) TRAINING_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --container-path) CONTAINER_PATH="$2"; shift 2 ;;
    --help)
      cat << EOF
Usage: $0 --account ACCOUNT [options]
Options:
  --n-lenses N              Number of lenses (default: 10000)
  --variations N            Variations per base (default: auto)
  --time-delay-fraction F   Fraction with time delays (default: 0.0)
  --training-epochs N       Training epochs (default: 50)
  --batch-size N            Batch size (default: 32)
  --device DEVICE           cuda or cpu (default: cuda)
  --container-path PATH     Override container path
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$ACCOUNT" ]; then
  echo "ERROR: --account is required"; exit 1
fi

export N_LENSES N_VARIATIONS TIME_DELAY_FRACTION TRAINING_EPOCHS BATCH_SIZE DEVICE
if [ -n "$CONTAINER_PATH" ]; then
  export CONTAINER_PATH
fi

echo "Submitting chained jobs (simulate -> prepare -> train)"

SIM_JOB=$(sbatch --account "$ACCOUNT" scripts/sbatch/sbatch_step_1_simulate_container.sh | awk '{print $4}')
echo "Submitted simulate job: $SIM_JOB"

PREP_JOB=$(sbatch --account "$ACCOUNT" --dependency=afterok:$SIM_JOB scripts/sbatch/sbatch_step_2_prepare_container.sh | awk '{print $4}')
echo "Submitted prepare job:  $PREP_JOB (afterok:$SIM_JOB)"

TRAIN_JOB=$(sbatch --account "$ACCOUNT" --dependency=afterok:$PREP_JOB scripts/sbatch/sbatch_step_3_train_container.sh | awk '{print $4}')
echo "Submitted train job:    $TRAIN_JOB (afterok:$PREP_JOB)"

echo ""
echo "Monitor:"
echo "  squeue -j $SIM_JOB,$PREP_JOB,$TRAIN_JOB"


