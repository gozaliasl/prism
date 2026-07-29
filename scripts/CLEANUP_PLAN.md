# Script Cleanup Plan

## Scripts to KEEP (6 essential scripts)

1. **complete_workflow.sh** - Main local simulation workflow
   - Handles production, ML training, and custom modes
   - Supports all simulation configurations
   - **Includes ML model training** (calls train_environment_models.py)

2. **sbatch_production.sh** - HPC batch submission script
   - Configured for Mahti (CSC Finland)
   - Can be adapted for other HPC systems
   - Uses complete_workflow.sh internally

3. **simulate_time_delays.sh** - Time delay simulation script
   - Specialized for variable source simulations
   - Supports quasar, supernova, and AGN sources

4. **analyze_simulations.py** - Simulation analysis script
   - Generates completeness/purity analysis
   - Creates confusion matrices
   - Produces sensitivity analysis plots
   - **Required for paper figures** (Figure 9 in results section)

5. **train_environment_models.py** - ML model training
   - Called by complete_workflow.sh for environment prediction
   - Trains models from COSMOS data
   - **Required for ML functionality**

6. **test_csv_training.py** - Data validation script
   - Called by complete_workflow.sh to validate data loading
   - **Required for ML training workflow**

## Scripts to REMOVE

### Analysis Scripts (redundant - keep only analyze_simulations.py)
- analyze_real_jwst_observations.py
- analyze_simulation.sh
- analyze_threshold_stats.py
- compare_equal_size_thresholds.py
- compare_thresholds.py

### Cleanup Scripts (self-removing after use)
- cleanup_analysis_directory.sh
- cleanup_lensing_figures.sh
- cleanup_redundant_scripts.sh

### Demo/Test Scripts
- demo_time_delays.py
- test_environment_training.py (redundant - test_csv_training.py is used)

### Training Scripts (redundant)
- train_models.py (functionality in train_environment_models.py)

### Utility Scripts (redundant - keep push_to_github.sh if needed)
- create_massive_galaxies_catalog.py
- run_custom20k_analysis.sh
- run_full_custom_pipeline.sh
- run_threshold_sweep.sh
- select_best_showcase_images.py
- select_best_showcase_images_v2.py
- select_nonlens_by_category.py
- verify_setup.sh

### Redundant Workflow Scripts
- nersc_complete_workflow.sh (functionality in sbatch_production.sh)
- sbatch_complete_workflow_nersc.sh (redundant)
- sbatch_custom_pipeline.sh (redundant)
- sbatch_nersc.sh (redundant)

### Specialized Scripts (optional - for figure generation)
- create_time_delay_demo_figure.py (can be kept if needed for paper figures)

## Cleanup Execution

To remove all unnecessary files, run:
```bash
cd scripts
rm -f analyze_*.py analyze_*.sh
rm -f cleanup_*.sh
rm -f compare_*.py
rm -f create_massive_galaxies_catalog.py
rm -f demo_time_delays.py
rm -f nersc_complete_workflow.sh
# Keep push_to_github.sh (useful utility)
# rm -f push_to_github.sh
rm -f run_*.sh
rm -f select_*.py
rm -f test_environment_training.py
rm -f train_models.py
rm -f verify_setup.sh
rm -f sbatch_complete_workflow_nersc.sh
rm -f sbatch_custom_pipeline.sh
rm -f sbatch_nersc.sh
```

Or use the cleanup script (if it exists):
```bash
./cleanup_redundant_scripts.sh
```

## Final Script List (6 scripts)

After cleanup, only these will remain:
1. complete_workflow.sh
2. sbatch_production.sh
3. simulate_time_delays.sh
4. analyze_simulations.py
5. train_environment_models.py
6. test_csv_training.py
