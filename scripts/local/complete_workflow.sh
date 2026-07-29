#!/bin/bash
#
# Complete JWST Mock Lens Workflow
# 1. Train ML models from COSMOS data
# 2. Generate simulation batches (production, ML training, or custom)

# Ensures proper conda environment (astro-clean) and error handling
#

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode MODE           Simulation mode (default: production)"
    echo "                        - production: 5k lenses + 5k non-lenses (paper validation)"
    echo "                        - ml-standard: 50k lenses + 50k non-lenses (ML training)"
    echo "                        - ml-large: 100k lenses + 100k non-lenses (large-scale ML)"
    echo "                        - ml-quick: 10k lenses + 10k non-lenses (quick ML test)"
    echo "                        - custom: Use custom parameters"
    echo "  --n-lenses N          Number of lenses (required for custom mode)"
    echo "  --n-non-lenses N      Number of non-lenses (required for custom mode)"
    echo "  --variations N        Variations per base lens (optional, default: 1)"
    echo "  --output-dir DIR      Output directory (default: external drive)"
    echo "  --lens-types TYPES    Lens type distribution (default: single=45%,binary=35%,group=20%)"
    echo "                        Format: single=45,binary=35,group=20"
    echo "  --binary-types TYPES  Binary lens subtypes (default: nfw=33%,sie=33%,shear=34%)"
    echo "                        Format: nfw=33,sie=33,shear=34 (applies to binary lenses only)"
    echo "  --skip-ml-training    Skip ML model training"
    echo "  --legacy-storage      Disable unified .npz storage (use separate .npy files)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Production mode (default)"
    echo "  $0 --mode ml-standard                # 50k lenses for ML training"
    echo "  $0 --mode ml-large                   # 100k lenses for large-scale ML"
    echo "  $0 --mode custom --n-lenses 200 --n-non-lenses 200"
    echo "  $0 --mode custom --n-lenses 200 --n-non-lenses 200 --variations 5"
    echo "  $0 --mode custom --n-lenses 200 --n-non-lenses 100 --lens-types single=45,binary=35,group=20"
    echo "  $0 --mode custom --n-lenses 1000 --n-non-lenses 500 --binary-types nfw=40,sie=40,shear=20"
}

# Default values
MODE="production"
SKIP_ML_TRAINING=false
UNIFIED_STORAGE=true
OUTPUT_DIR=""
CUSTOM_N_LENSES=""
CUSTOM_N_NON_LENSES=""
CUSTOM_VARIATIONS=""
LENS_TYPES_SINGLE=45
LENS_TYPES_BINARY=35
LENS_TYPES_GROUP=20
BINARY_TYPE_NFW=33
BINARY_TYPE_SIE=33
BINARY_TYPE_SHEAR=34

# Cleanup temporary files on exit
cleanup() {
    if [[ -n "${TEMP_CONFIG:-}" && -f "$TEMP_CONFIG" ]]; then
        rm -f "$TEMP_CONFIG"
    fi
}
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --n-lenses)
            CUSTOM_N_LENSES="$2"
            shift 2
            ;;
        --n-non-lenses)
            CUSTOM_N_NON_LENSES="$2"
            shift 2
            ;;
        --variations)
            CUSTOM_VARIATIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-ml-training)
            SKIP_ML_TRAINING=true
            shift
            ;;
        --legacy-storage)
            UNIFIED_STORAGE=false
            shift
            ;;
        --lens-types)
            # Parse lens type distribution: single=45,binary=35,group=20
            LENS_TYPES_STR="$2"
            # Extract values using grep and awk
            LENS_TYPES_SINGLE=$(echo "$LENS_TYPES_STR" | grep -o "single=[0-9]*" | cut -d= -f2)
            LENS_TYPES_BINARY=$(echo "$LENS_TYPES_STR" | grep -o "binary=[0-9]*" | cut -d= -f2)
            LENS_TYPES_GROUP=$(echo "$LENS_TYPES_STR" | grep -o "group=[0-9]*" | cut -d= -f2)
            shift 2
            ;;
        --binary-types)
            # Parse binary type distribution: nfw=33,sie=33,shear=34
            BINARY_TYPES_STR="$2"
            # Extract values using grep and awk
            BINARY_TYPE_NFW=$(echo "$BINARY_TYPES_STR" | grep -o "nfw=[0-9]*" | cut -d= -f2)
            BINARY_TYPE_SIE=$(echo "$BINARY_TYPES_STR" | grep -o "sie=[0-9]*" | cut -d= -f2)
            BINARY_TYPE_SHEAR=$(echo "$BINARY_TYPES_STR" | grep -o "shear=[0-9]*" | cut -d= -f2)
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

# Function to get simulation parameters from config
get_simulation_params() {
    local mode="$1"
    
    case "$mode" in
        "production")
            echo "5000 5000 1"
            ;;
        "ml-standard")
            echo "50000 50000 115"
            ;;
        "ml-large")
            echo "100000 100000 230"
            ;;
        "ml-quick")
            echo "10000 10000 23"
            ;;
        "custom")
            if [[ -z "$CUSTOM_N_LENSES" || -z "$CUSTOM_N_NON_LENSES" ]]; then
                echo "❌ Custom mode requires --n-lenses and --n-non-lenses" >&2
                echo "❌ Optional: --variations (default: 1)" >&2
                show_usage
                exit 1
            fi
            # Use default variations of 1 if not specified
            local variations="${CUSTOM_VARIATIONS:-1}"
            echo "$CUSTOM_N_LENSES $CUSTOM_N_NON_LENSES $variations"
            ;;
        *)
            echo "❌ Unknown mode '$mode'" >&2
            show_usage
            exit 1
            ;;
    esac
}

# Get simulation parameters
read N_LENSES N_NON_LENSES VARIATIONS <<< $(get_simulation_params "$MODE")

echo "🚀 COMPLETE JWST MOCK LENS WORKFLOW"
echo "===================================="
echo ""
echo "Mode: $MODE"
echo "Configuration:"
echo "  - Lenses: $N_LENSES"
echo "  - Non-lenses: $N_NON_LENSES"
echo "  - Variations per base: $VARIATIONS"
echo "  - Lens type distribution:"
echo "    • Single field: ${LENS_TYPES_SINGLE}%"
echo "    • Binary lenses: ${LENS_TYPES_BINARY}%"
echo "      - NFW+NFW: ${BINARY_TYPE_NFW}% of binaries"
echo "      - SIE+SIE: ${BINARY_TYPE_SIE}% of binaries"
echo "      - SHEAR-only: ${BINARY_TYPE_SHEAR}% of binaries"
echo "    • Group lenses: ${LENS_TYPES_GROUP}%"
echo "  - Skip ML training: $SKIP_ML_TRAINING"
echo "  - Unified storage: $UNIFIED_STORAGE"
echo ""

# Verify conda environment
echo "🔧 Verifying conda environment..."
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Activate conda environment
echo "🔧 Activating astro-clean conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.zshrc
conda activate astro-clean

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $PYTHON_VERSION"
ACTIVE_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
echo "✅ Conda environment: $ACTIVE_ENV"
if [[ "$ACTIVE_ENV" != "astro-clean" ]]; then
    echo "❌ Expected astro-clean environment, got $ACTIVE_ENV"
    exit 1
fi
echo ""

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in scripts/local/, so go up 2 levels to project root
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# ML Training Step
if [[ "$SKIP_ML_TRAINING" == "false" ]]; then
    echo "📊 Step 1: Training ML Models from COSMOS Data"
    echo "=============================================="
    echo ""

    # Check if models already exist
    if [ -d "models" ] && [ -f "models/env_classifier.pkl" ]; then
        echo "✅ ML models already exist, skipping training"
        echo "   (Delete models/ directory to retrain)"
    else
        echo "🧠 Training environment models from COSMOS data..."
        echo ""
        
        # Test data loading first
        echo "🔍 Testing data loading..."
        python scripts/local/test_csv_training.py
        
        if [ $? -eq 0 ]; then
            echo "✅ Data loading test passed"
            echo ""
            
            # Train models
            echo "🎯 Training ML models with lens-centric approach..."
            python scripts/local/train_environment_models.py
            
            if [ $? -eq 0 ]; then
                echo "✅ ML models trained successfully"
            else
                echo "❌ Model training failed"
                exit 1
            fi
        else
            echo "❌ Data loading test failed"
            exit 1
        fi
    fi
else
    echo "⏭️  Skipping ML model training (--skip-ml-training)"
fi

echo ""
echo "🎬 Step 2: Generating Simulation Batch"
echo "======================================"
echo ""

# Configuration
CONFIG="configs/default_config.yaml"
COSMOS_CATALOG="data/cosmos_web_lens_structural_properties.csv"
LENS_ANALYSIS="data/lens_analysis_catalog.csv"
FIELD_CATALOG="data/merged_lens_field_catalog.csv"

# Validate required input files
for req in "$CONFIG" "$COSMOS_CATALOG" "$LENS_ANALYSIS" "$FIELD_CATALOG"; do
    if [[ ! -f "$req" ]]; then
        echo "❌ Required file not found: $req"
        exit 1
    fi
done

# Output directory setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Override unified storage setting if requested
TEMP_CONFIG=""
if [[ "$UNIFIED_STORAGE" == "false" ]]; then
    TEMP_CONFIG="/tmp/jwst_sim_unified_${TIMESTAMP}.yaml"
    echo "🧩 Disabling unified .npz storage (temp config: $TEMP_CONFIG)"
    TIMESTAMP="$TIMESTAMP" python - << 'PY'
import yaml
from pathlib import Path
import os

cfg_path = Path("configs/default_config.yaml")
out_path = Path("/tmp") / Path(f"jwst_sim_unified_{os.environ['TIMESTAMP']}.yaml")

with cfg_path.open("r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("output", {})
cfg["output"]["unified_storage"] = False

with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
    CONFIG="$TEMP_CONFIG"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    # Default to project outputs directory
    OUT_ROOT="$PROJECT_ROOT/outputs"
    mkdir -p "$OUT_ROOT"
    OUTPUT_DIR="$OUT_ROOT/${MODE}_${TIMESTAMP}"
else
    OUT_ROOT="$(dirname "$OUTPUT_DIR")"
    mkdir -p "$OUT_ROOT"
fi

mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Copy output data guide into the run directory
if [[ -f "$PROJECT_ROOT/docs/OUTPUT_DATA_GUIDE.md" ]]; then
    cp "$PROJECT_ROOT/docs/OUTPUT_DATA_GUIDE.md" "$OUTPUT_DIR/README_OUTPUT_DATA.md"
fi

# Determine noise configuration based on mode
if [[ "$MODE" == "production" ]]; then
    # Production mode: Generate both empirical and ablation batches
    echo "🎬 Generating Production Batches (Empirical + Ablation)"
    echo "======================================================"
    
    # Batch A: Empirical noise
    OUT_A="$OUTPUT_DIR/batchA_empirical"
    mkdir -p "$OUT_A"
    
    echo "📊 Batch A: Empirical noise/PSF"
    echo "Output: $OUT_A"
    python src/jwst_lens_simulator.py \
        --config "$CONFIG" \
        --cosmos_catalog "$COSMOS_CATALOG" \
        --lens_analysis_catalog "$LENS_ANALYSIS" \
        --merged_field_catalog "$FIELD_CATALOG" \
        --output_dir "$OUT_A" \
        --n_lenses $N_LENSES \
        --n_non_lenses $N_NON_LENSES \
        --variations_per_base $VARIATIONS \
        --seed 42 \
        --add_artifacts \
        --save_intermediate \
        --no_date_suffix
    
    EXIT_CODE_A=$?
    
    # Batch B: Ablation (median noise)
    OUT_B="$OUTPUT_DIR/batchB_ablation_median"
    mkdir -p "$OUT_B"
    
    # Create ablation config
    ABLATION_CFG="/tmp/jwst_sim_ablation_${TIMESTAMP}.yaml"
    cp "$CONFIG" "$ABLATION_CFG"
    cat >> "$ABLATION_CFG" << 'EOF'

noise:
  sampling_method: 'median'
EOF
    
    echo ""
    echo "📊 Batch B: Ablation study (median noise/PSF)"
    echo "Output: $OUT_B"
    python src/jwst_lens_simulator.py \
        --config "$ABLATION_CFG" \
        --cosmos_catalog "$COSMOS_CATALOG" \
        --lens_analysis_catalog "$LENS_ANALYSIS" \
        --merged_field_catalog "$FIELD_CATALOG" \
        --output_dir "$OUT_B" \
        --n_lenses $N_LENSES \
        --n_non_lenses $N_NON_LENSES \
        --variations_per_base $VARIATIONS \
        --seed 43 \
        --add_artifacts \
        --save_intermediate \
        --no_date_suffix
    
    EXIT_CODE_B=$?
    
    # Check results
    if [ $EXIT_CODE_A -eq 0 ] && [ $EXIT_CODE_B -eq 0 ]; then
        echo ""
        echo "✅ Production batches completed successfully!"
        echo "   - Batch A (empirical): $OUT_A"
        echo "   - Batch B (ablation):  $OUT_B"
    else
        echo "❌ Production batch generation failed"
        exit 1
    fi
    
else
    # ML training modes: Generate batches with lens type distribution
    echo "🎬 Step 2: Generating Simulation Batch with Lens Types"
    echo "======================================================="
    echo "Mode: $MODE"
    echo "Total Lenses: $N_LENSES"
    echo "Total Non-lenses: $N_NON_LENSES"
    echo "Variations per base: $VARIATIONS"
    echo ""
    
    # Calculate number of lenses for each type
    N_LENSES_SINGLE=$((N_LENSES * LENS_TYPES_SINGLE / 100))
    N_LENSES_BINARY=$((N_LENSES * LENS_TYPES_BINARY / 100))
    N_LENSES_GROUP=$((N_LENSES * LENS_TYPES_GROUP / 100))
    
    # Account for rounding
    REMAINDER=$((N_LENSES - N_LENSES_SINGLE - N_LENSES_BINARY - N_LENSES_GROUP))
    N_LENSES_SINGLE=$((N_LENSES_SINGLE + REMAINDER))
    
    echo "📊 Lens Distribution:"
    echo "  - Single field lenses: $N_LENSES_SINGLE (${LENS_TYPES_SINGLE}%)"
    echo "  - Binary lenses: $N_LENSES_BINARY (${LENS_TYPES_BINARY}%)"
    echo "  - Group lenses: $N_LENSES_GROUP (${LENS_TYPES_GROUP}%)"
    echo ""
    
    # Create subdirectories for each lens type
    OUT_SINGLE="$OUTPUT_DIR/single_field"
    OUT_BINARY="$OUTPUT_DIR/binary_lenses"
    OUT_GROUP="$OUTPUT_DIR/group_lenses"
    
    mkdir -p "$OUT_SINGLE" "$OUT_BINARY" "$OUT_GROUP"
    
    # Calculate non-lenses distribution (proportional)
    N_NONLENSES_SINGLE=$((N_NON_LENSES * LENS_TYPES_SINGLE / 100))
    N_NONLENSES_BINARY=$((N_NON_LENSES * LENS_TYPES_BINARY / 100))
    N_NONLENSES_GROUP=$((N_NON_LENSES * LENS_TYPES_GROUP / 100))
    
    # Account for rounding
    REMAINDER_NL=$((N_NON_LENSES - N_NONLENSES_SINGLE - N_NONLENSES_BINARY - N_NONLENSES_GROUP))
    N_NONLENSES_SINGLE=$((N_NONLENSES_SINGLE + REMAINDER_NL))
    
    SEED_OFFSET=0
    
    # Generate Single Field Lenses
    if [ $N_LENSES_SINGLE -gt 0 ]; then
        echo "📊 Generating Single Field Lenses ($N_LENSES_SINGLE lenses + $N_NONLENSES_SINGLE non-lenses)"
        echo "Output: $OUT_SINGLE"
        
        # Create single field config to force 100% single field lenses
        SF_CFG="/tmp/jwst_sim_sf_${TIMESTAMP}.yaml"
        cp "$CONFIG" "$SF_CFG"
        cat >> "$SF_CFG" << 'EOF'

lens_class_distribution:
  enabled: true
  single_field:
    fraction: 1.0
  binary_sie_sie:
    fraction: 0.0
  binary_nfw_nfw:
    fraction: 0.0
  binary_shear_only:
    fraction: 0.0
  group:
    fraction: 0.0
EOF
        
        python src/jwst_lens_simulator.py \
            --config "$SF_CFG" \
            --cosmos_catalog "$COSMOS_CATALOG" \
            --lens_analysis_catalog "$LENS_ANALYSIS" \
            --merged_field_catalog "$FIELD_CATALOG" \
            --output_dir "$OUT_SINGLE" \
            --n_lenses $N_LENSES_SINGLE \
            --n_non_lenses $N_NONLENSES_SINGLE \
            --variations_per_base $VARIATIONS \
            --seed $((42 + SEED_OFFSET)) \
            --add_artifacts \
            --save_intermediate \
            --numpix 300 \
            --no_date_suffix
        
        if [ $? -ne 0 ]; then
            echo "❌ Single field lens generation failed"
            exit 1
        fi
        SEED_OFFSET=$((SEED_OFFSET + 1))
        echo "✅ Single field lenses completed"
        echo ""
    fi
    
    # Generate Binary Lenses
    if [ $N_LENSES_BINARY -gt 0 ]; then
        echo "📊 Generating Binary Lenses ($N_LENSES_BINARY lenses + $N_NONLENSES_BINARY non-lenses)"
        echo "Output: $OUT_BINARY"
        
        # Create binary config with custom distribution
        BINARY_CFG="/tmp/jwst_sim_binary_${TIMESTAMP}.yaml"
        cp "$CONFIG" "$BINARY_CFG"
        
        # Calculate normalized fractions for binary types (they sum to 1.0 within binary category)
        NFW_FRACTION=$(echo "scale=4; $BINARY_TYPE_NFW / 100" | bc)
        SIE_FRACTION=$(echo "scale=4; $BINARY_TYPE_SIE / 100" | bc)
        SHEAR_FRACTION=$(echo "scale=4; $BINARY_TYPE_SHEAR / 100" | bc)
        
        # Set lens_class_distribution to force 100% binary lenses
        cat >> "$BINARY_CFG" << EOF

lens_class_distribution:
  enabled: true
  single_field:
    fraction: 0.0
  group:
    fraction: 0.0
  binary_sie_sie:
    fraction: $SIE_FRACTION
  binary_nfw_nfw:
    fraction: $NFW_FRACTION
  binary_shear_only:
    fraction: $SHEAR_FRACTION

binary_lenses:
  enabled: true
  mass_profile_types:
    sie_sie: $SIE_FRACTION
    nfw_nfw: $NFW_FRACTION
    shear_only: $SHEAR_FRACTION
EOF
        
        python src/jwst_lens_simulator.py \
            --config "$BINARY_CFG" \
            --cosmos_catalog "$COSMOS_CATALOG" \
            --lens_analysis_catalog "$LENS_ANALYSIS" \
            --merged_field_catalog "$FIELD_CATALOG" \
            --output_dir "$OUT_BINARY" \
            --n_lenses $N_LENSES_BINARY \
            --n_non_lenses $N_NONLENSES_BINARY \
            --variations_per_base $VARIATIONS \
            --seed $((42 + SEED_OFFSET)) \
            --add_artifacts \
            --save_intermediate \
            --numpix 300 \
            --no_date_suffix
        
        if [ $? -ne 0 ]; then
            echo "❌ Binary lens generation failed"
            exit 1
        fi
        SEED_OFFSET=$((SEED_OFFSET + 1))
        echo "✅ Binary lenses completed"
        echo ""
    fi
    
    # Generate Group Lenses
    if [ $N_LENSES_GROUP -gt 0 ]; then
        echo "📊 Generating Group Lenses ($N_LENSES_GROUP lenses + $N_NONLENSES_GROUP non-lenses)"
        echo "Output: $OUT_GROUP"
        
        # Create group config to force 100% group lenses
        GROUP_CFG="/tmp/jwst_sim_group_${TIMESTAMP}.yaml"
        cp "$CONFIG" "$GROUP_CFG"
        cat >> "$GROUP_CFG" << 'EOF'

lens_class_distribution:
  enabled: true
  single_field:
    fraction: 0.0
  binary_sie_sie:
    fraction: 0.0
  binary_nfw_nfw:
    fraction: 0.0
  binary_shear_only:
    fraction: 0.0
  group:
    fraction: 1.0

group_lenses:
  enabled: true
  n_members: 3
EOF
        
        python src/jwst_lens_simulator.py \
            --config "$GROUP_CFG" \
            --cosmos_catalog "$COSMOS_CATALOG" \
            --lens_analysis_catalog "$LENS_ANALYSIS" \
            --merged_field_catalog "$FIELD_CATALOG" \
            --output_dir "$OUT_GROUP" \
            --n_lenses $N_LENSES_GROUP \
            --n_non_lenses $N_NONLENSES_GROUP \
            --variations_per_base $VARIATIONS \
            --seed $((42 + SEED_OFFSET)) \
            --add_artifacts \
            --save_intermediate \
            --numpix 300 \
            --no_date_suffix
        
        if [ $? -ne 0 ]; then
            echo "❌ Group lens generation failed"
            exit 1
        fi
        SEED_OFFSET=$((SEED_OFFSET + 1))
        echo "✅ Group lenses completed"
        echo ""
    fi
    
    # Combine results
    echo "📊 Combining all lens types into single catalog..."
    if [ -d "$OUT_SINGLE" ]; then
        cp "$OUT_SINGLE/cosmos_lens_training_catalog.csv" "$OUTPUT_DIR/cosmos_lens_training_catalog.csv" 2>/dev/null || true
        cp "$OUT_SINGLE/cosmos_nonlens_training_catalog.csv" "$OUTPUT_DIR/cosmos_nonlens_training_catalog.csv" 2>/dev/null || true
    fi
    
    echo ""
    echo "✅ All lens type batches completed successfully!"
    echo "   - Single field: $OUT_SINGLE"
    echo "   - Binary:       $OUT_BINARY"
    echo "   - Groups:       $OUT_GROUP"
fi

echo ""
echo "🎉 WORKFLOW COMPLETE!"
echo "===================="
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo ""
echo "📊 Lens Type Distribution:"
echo "  - Single field (SF): ${LENS_TYPES_SINGLE}%"
echo "  - Binary (BR): ${LENS_TYPES_BINARY}%"
echo "    • NFW+NFW: ${BINARY_TYPE_NFW}% of binaries"
echo "    • SIE+SIE: ${BINARY_TYPE_SIE}% of binaries"
echo "    • SHEAR-only: ${BINARY_TYPE_SHEAR}% of binaries"
echo "  - Groups (GR): ${LENS_TYPES_GROUP}%"
echo ""
echo "📂 Generated Output:"
if [[ "$MODE" == "production" ]]; then
    echo "   - Batch A (empirical): $OUTPUT_DIR/batchA_empirical"
    echo "   - Batch B (ablation):  $OUTPUT_DIR/batchB_ablation_median"
else
    echo "   - Single field: $OUTPUT_DIR/single_field"
    echo "   - Binary:       $OUTPUT_DIR/binary_lenses"
    echo "   - Groups:       $OUTPUT_DIR/group_lenses"
fi
echo ""
echo "📊 Files in each batch:"
echo "   - cosmos_lens_training_catalog.csv"
echo "   - cosmos_nonlens_training_catalog.csv"
echo "   - cosmos_training_catalog_lens_and_nonlens.csv"
echo "   - jpg_rgb/ (image files)"
if [[ "$UNIFIED_STORAGE" == "true" ]]; then
    echo "   - unified_npz/ (single-file compressed samples)"
else
    echo "   - npy/ (numpy arrays)"
fi
echo ""

if [[ "$MODE" == "ml-standard" || "$MODE" == "ml-large" || "$MODE" == "ml-quick" ]]; then
    echo "🎯 Ready for ML model training!"
    echo "   Use the generated images and catalogs to train your lens detection models."
    echo ""
    echo "🔍 Lens type analysis:"
    echo "   - Single field (*_SF_*): Common isolated lenses"
    echo "   - Binary (BR): Paired lenses, good for training robustness"
    echo "   - Groups (GR): Complex multi-lens systems"
elif [[ "$MODE" == "production" ]]; then
    echo "📊 Ready for analysis and validation!"
    echo "   Use the generated batches for completeness/purity analysis and paper figures."
fi

echo ""
echo "💡 Next steps:"
echo "   - Run analysis scripts on the generated data"
echo "   - Train ML models using the training datasets"
echo "   - Validate on real JWST observations"
echo ""
echo "📝 Lens Type Naming Convention:"
echo "   - *_SF_*: Single Field lenses"
echo "   - *_BR_*: Binary (pair) lenses"
echo "   - *_GR_*: Group lenses"
