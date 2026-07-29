# Scripts Directory - Organized Structure

This directory contains scripts organized by usage type: **local execution**, **HPC/SLURM**, and **time delay specific**.

## 📁 Directory Structure

```
scripts/
├── local/              # Local execution scripts (bash & Python)
├── sbatch/             # HPC/SLURM sbatch scripts and submit wrappers
├── time_delays/        # Time delay specific scripts
└── segmentation_training_pipeline.sh  # Shared pipeline (used by both local and sbatch)
```

## 🚀 Quick Start

### Local Execution

**Complete workflow (with ML training):**
```bash
./scripts/local/complete_workflow.sh --mode ml-standard
```

**Time delay simulations:**
```bash
./scripts/time_delays/simulate_time_delays.sh --n-lenses 500
```

### HPC/SLURM (Mahti)

**Chained workflow (simulate → prepare → train):**
```bash
./scripts/sbatch/submit_segmentation_chain_container.sh \
    --account ituomine \
    --n-lenses 10000
```

**Individual steps (container-based):**
```bash
# Step 1: Simulate
sbatch --account ituomine scripts/sbatch/sbatch_step_1_simulate_container.sh

# Step 2: Prepare training data (after simulation completes)
sbatch --account ituomine scripts/sbatch/sbatch_step_2_prepare_container.sh

# Step 3: Train model (after data preparation completes)
sbatch --account ituomine scripts/sbatch/sbatch_step_3_train_container.sh
```

## 📂 Script Categories

### `local/` - Local Execution Scripts

**Bash Scripts:**
- `complete_workflow.sh` - Main workflow for local simulations with ML training
- `push_to_github.sh` - Git push utility (excludes docs/outputs/data)

**Python Scripts:**
- `analyze_simulations.py` - Analyze simulation results (completeness, purity, etc.)
- `identify_lensed_images.py` - Detect and annotate lensed images
- `prepare_segmentation_training_data.py` - Prepare U-Net training data
- `prepare_training_data.py` - Prepare ML training data
- `train_environment_models.py` - Train environment prediction models
- `train_segmentation_model.py` - Train U-Net segmentation model
- `test_csv_training.py` - Test CSV data loading

### `sbatch/` - HPC/SLURM Scripts

**Container-based (kept):**
- `sbatch_step_1_simulate_container.sh` - Run simulations in container
- `sbatch_step_2_prepare_container.sh` - Prepare training data in container
- `sbatch_step_3_train_container.sh` - Train model in container
- `submit_segmentation_chain_container.sh` - Chain simulate→prepare→train (recommended)

**Non-container (kept):**
- `sbatch_segmentation_training.sh` - Full pipeline using modules
- `submit_segmentation_training.sh` - Submit wrapper for non-container pipeline

### `time_delays/` - Time Delay Scripts

- `simulate_time_delays.sh` - Generate time delay lens systems
- `create_time_delay_demo_figure.py` - Create demo figures for paper

## 🔗 Shared Scripts

- `segmentation_training_pipeline.sh` - Shared pipeline script (used by both local and sbatch)

## 📋 Usage Examples

### Local: Complete Workflow

```bash
# Production mode (5k lenses + 5k non-lenses)
./scripts/local/complete_workflow.sh

# ML training mode (50k lenses + 50k non-lenses)
./scripts/local/complete_workflow.sh --mode ml-standard

# Custom
./scripts/local/complete_workflow.sh --mode custom --n-lenses 20000
```

### Local: Time Delay Simulations

```bash
# Basic time delay simulation
./scripts/time_delays/simulate_time_delays.sh --n-lenses 500

# With specific source fractions
./scripts/time_delays/simulate_time_delays.sh \
    --n-lenses 500 \
    --quasar-fraction 0.4 \
    --supernova-fraction 0.3 \
    --agn-fraction 0.3

# Create demo figure
python3 scripts/time_delays/create_time_delay_demo_figure.py \
    --output-dir outputs/time_delays_XXX \
    --lens-id 25 \
    --source-type quasar
```

### HPC: Chained Workflow (Recommended)

```bash
# Full pipeline (3 jobs: simulate → prepare → train)
./scripts/sbatch/submit_segmentation_chain_container.sh \
    --account ituomine \
    --n-lenses 10000 \
    --training-epochs 50 \
    --batch-size 32
```

### HPC: Non-container Pipeline

```bash
# Submit non-container full pipeline
./scripts/sbatch/submit_segmentation_training.sh --account ituomine
```

### HPC: Individual Steps (Container)

```bash
# Step 1: Simulate (export variables first)
export N_LENSES=10000
export N_VARIATIONS=auto
export TIME_DELAY_FRACTION=0.0
sbatch --account ituomine scripts/sbatch/sbatch_step_1_simulate_container.sh

# Step 2: Prepare (reads output from Step 1 automatically)
sbatch --account ituomine scripts/sbatch/sbatch_step_2_prepare_container.sh

# Step 3: Train (reads training data from Step 2 automatically)
export TRAINING_EPOCHS=50
export BATCH_SIZE=32
export DEVICE=cuda
sbatch --account ituomine scripts/sbatch/sbatch_step_3_train_container.sh
```

## 🔧 Container Requirements

All container-based scripts require:
- Container built: `bash container/build_on_mahti.sh`
- Container path: `/scratch/ituomine/gozaliasl/jwst_lens_simulator.sif` (or set `CONTAINER_PATH`)

## 📝 Notes

- The sbatch directory now contains only two series: container-based (3 steps + chained submit) and non-container (single full pipeline + submit wrapper)
- Local directory keeps the original complete workflow script; Python utilities remain for analysis/training/detection
