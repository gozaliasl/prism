#!/bin/bash
#
# Run realistic lens simulations with three different pair galaxy treatments
# 1. SIE+SIE: Binary SIE (fast, simple)
# 2. NFW+NFW: Binary NFW (realistic dark matter)
# 3. Shear-only: Pair contributes only to shear (simplified)
#
# Produces output catalog tracking which treatment each lens belongs to
#

set -e
set -u
set -o pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode MODE           Simulation mode (default: quick)"
    echo "                        - quick: 200 lenses + 200 non-lenses per treatment (fast test)"
    echo "                        - standard: 1000 lenses + 1000 non-lenses per treatment"
    echo "                        - production: 5000 lenses + 5000 non-lenses per treatment"
    echo "  --n-lenses N          Number of lenses per treatment (overrides mode)"
    echo "  --n-non-lenses N      Number of non-lenses per treatment (overrides mode)"
    echo "  --output-dir DIR      Output directory (default: outputs/pair_treatments)"
    echo "  --skip-ml-training    Skip ML model training"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Quick mode (default)"
    echo "  $0 --mode standard                           # Standard mode"
    echo "  $0 --n-lenses 500 --n-non-lenses 500        # Custom sizes"
}

# Default values
MODE="quick"
SKIP_ML_TRAINING=false
OUTPUT_DIR=""
CUSTOM_N_LENSES=""
CUSTOM_N_NON_LENSES=""

# Parse arguments
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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-ml-training)
            SKIP_ML_TRAINING=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set sample sizes based on mode
if [[ -z "$CUSTOM_N_LENSES" ]]; then
    case $MODE in
        quick)
            N_LENSES=10
            N_NON_LENSES=0
            ;;
        standard)
            N_LENSES=1000
            N_NON_LENSES=1000
            ;;
        production)
            N_LENSES=5000
            N_NON_LENSES=5000
            ;;
        *)
            echo "❌ Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac
else
    N_LENSES="$CUSTOM_N_LENSES"
    N_NON_LENSES="$CUSTOM_N_NON_LENSES"
fi

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Setup output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/outputs/pair_treatments_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PAIR GALAXY TREATMENT COMPARISON - REALISTIC SIMULATIONS${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Mode: $MODE"
echo "  Lenses per treatment: $N_LENSES"
echo "  Non-lenses per treatment: $N_NON_LENSES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Activate conda environment
echo -e "${YELLOW}🔧 Verifying conda environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source ~/.zshrc
conda activate astro-clean

# Create config files for each treatment
echo -e "${YELLOW}📝 Creating configuration files...${NC}"

CONFIG_DIR="$OUTPUT_DIR/configs"
mkdir -p "$CONFIG_DIR"

# Base config (use default)
BASE_CONFIG="$PROJECT_ROOT/configs/default_config.yaml"

# 1. SIE+SIE Configuration (both pair galaxies as SIE lens)
SIE_SIE_CONFIG="$CONFIG_DIR/pair_sie_sie.yaml"
cat > "$SIE_SIE_CONFIG" << 'EOF'
# SIE+SIE Binary Lens Configuration
# Both pair galaxies contribute to lens mass as SIE

binary_lenses:
  enabled: true
  fraction: 1.0       # 100% binary pairs for this treatment
  mass_profile_types:
    sie_sie: 1.0        # 100% SIE+SIE (no NFW in this config)
    nfw_nfw: 0.0        # 0% NFW
  mass_ratio:
    min: 0.3
    max: 1.0
    distribution: 'uniform'
  separation:
    min_in_einstein_radii: 0.5
    max_in_einstein_radii: 2.0
    distribution: 'uniform'
  redshift_config:
    same_redshift_fraction: 0.8
    max_delta_z: 0.3
  orientation:
    random_position_angle: true
    correlated_ellipticities: true

environment:
  types:
    isolated_field:
      fraction: 0.45
      galaxy_count_mean: 2.5
      galaxy_count_std: 1.2
      min_galaxies: 0
      max_galaxies: 3
      separation_mean: 2.5
      avoid_factor: 1.25
      max_radius_arcsec: 3.0
      shear_min: 0.01
      shear_max: 0.05
    galaxy_pair:
      fraction: 0.35
      galaxy_count_mean: 3.0
      galaxy_count_std: 1.3
      min_galaxies: 1
      max_galaxies: 4
      separation_mean: 1.8
      avoid_factor: 1.1
      max_radius_arcsec: 3.5
      shear_min: 0.03
      shear_max: 0.08
    group:
      fraction: 0.20
      galaxy_count_mean: 4.5
      galaxy_count_std: 1.6
      min_galaxies: 2
      max_galaxies: 5
      separation_mean: 1.2
      avoid_factor: 1.0
      max_radius_arcsec: 4.0
      shear_min: 0.05
      shear_max: 0.15
EOF

# 2. NFW+NFW Configuration (realistic dark matter)
NFW_NFW_CONFIG="$CONFIG_DIR/pair_nfw_nfw.yaml"
cat > "$NFW_NFW_CONFIG" << 'EOF'
# NFW+NFW Binary Lens Configuration
# Both pair galaxies contribute to lens mass as NFW profiles (realistic dark matter)

binary_lenses:
  enabled: true
  fraction: 1.0       # 100% binary pairs for this treatment
  mass_profile_types:
    sie_sie: 0.0        # 0% SIE
    nfw_nfw: 1.0        # 100% NFW+NFW (realistic dark matter halos)
  mass_ratio:
    min: 0.3
    max: 1.0
    distribution: 'uniform'
  separation:
    min_in_einstein_radii: 0.5
    max_in_einstein_radii: 2.0
    distribution: 'uniform'
  redshift_config:
    same_redshift_fraction: 0.8
    max_delta_z: 0.3
  orientation:
    random_position_angle: true
    correlated_ellipticities: true

environment:
  types:
    isolated_field:
      fraction: 0.45
      galaxy_count_mean: 2.5
      galaxy_count_std: 1.2
      min_galaxies: 0
      max_galaxies: 3
      separation_mean: 2.5
      avoid_factor: 1.25
      max_radius_arcsec: 3.0
      shear_min: 0.01
      shear_max: 0.05
    galaxy_pair:
      fraction: 0.35
      galaxy_count_mean: 3.0
      galaxy_count_std: 1.3
      min_galaxies: 1
      max_galaxies: 4
      separation_mean: 1.8
      avoid_factor: 1.1
      max_radius_arcsec: 3.5
      shear_min: 0.03
      shear_max: 0.08
    group:
      fraction: 0.20
      galaxy_count_mean: 4.5
      galaxy_count_std: 1.6
      min_galaxies: 2
      max_galaxies: 5
      separation_mean: 1.2
      avoid_factor: 1.0
      max_radius_arcsec: 4.0
      shear_min: 0.05
      shear_max: 0.15
EOF

# 3. Shear-Only Configuration (pair contributes only to shear)
SHEAR_ONLY_CONFIG="$CONFIG_DIR/pair_shear_only.yaml"
cat > "$SHEAR_ONLY_CONFIG" << 'EOF'
# Shear-Only Configuration
# Pair galaxies contribute only to external environmental shear (no binary lensing)

binary_lenses:
  enabled: false        # Disable binary lensing
  fraction: 0.0

environment:
  types:
    isolated_field:
      fraction: 0.45
      galaxy_count_mean: 2.5
      galaxy_count_std: 1.2
      min_galaxies: 0
      max_galaxies: 3
      separation_mean: 2.5
      avoid_factor: 1.25
      max_radius_arcsec: 3.0
      shear_min: 0.01
      shear_max: 0.05
    galaxy_pair:
      fraction: 0.35
      galaxy_count_mean: 3.0
      galaxy_count_std: 1.3
      min_galaxies: 1
      max_galaxies: 4
      separation_mean: 1.8
      avoid_factor: 1.1
      max_radius_arcsec: 3.5
      shear_min: 0.03
      shear_max: 0.08
    group:
      fraction: 0.20
      galaxy_count_mean: 4.5
      galaxy_count_std: 1.6
      min_galaxies: 2
      max_galaxies: 5
      separation_mean: 1.2
      avoid_factor: 1.0
      max_radius_arcsec: 4.0
      shear_min: 0.05
      shear_max: 0.15
EOF

echo -e "${GREEN}✓ Configuration files created${NC}"
echo "  - SIE+SIE: $SIE_SIE_CONFIG"
echo "  - NFW+NFW: $NFW_NFW_CONFIG"
echo "  - Shear-only: $SHEAR_ONLY_CONFIG"
echo ""

# ML Training (only once)
if [[ "$SKIP_ML_TRAINING" == "false" ]]; then
    if [ -d "models" ] && [ -f "models/env_classifier.pkl" ]; then
        echo -e "${GREEN}✅ ML models already exist, skipping training${NC}"
    else
        echo -e "${YELLOW}📊 Step 0: Training ML Models${NC}"
        echo "============================================"
        python scripts/local/train_environment_models.py
        echo -e "${GREEN}✅ ML models trained${NC}"
    fi
fi
echo ""

# Catalog files for tracking
TRACKING_CATALOG="$OUTPUT_DIR/pair_treatments_catalog.csv"
DETAILED_CATALOG="$OUTPUT_DIR/pair_treatments_detailed.csv"

# Initialize catalog headers
echo "lens_id,treatment,source_file,n_images,magnification_sum,time_delay_days" > "$TRACKING_CATALOG"
echo "lens_id,treatment,source_file,n_images,magnification_sum,time_delay_days,theta_E_arcsec,z_lens,z_source,environment,comments" > "$DETAILED_CATALOG"

# Function to run simulation and track results
run_treatment() {
    local TREATMENT=$1
    local CONFIG=$2
    local SEED=$3
    
    local TREATMENT_DIR="$OUTPUT_DIR/$TREATMENT"
    mkdir -p "$TREATMENT_DIR"
    
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ Running $TREATMENT treatment${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Configuration: $CONFIG"
    echo "Output directory: $TREATMENT_DIR"
    echo "Seed: $SEED"
    echo ""
    
    # Run simulation (lens-only, diagnostic images optional)
    python src/jwst_lens_simulator.py \
        --config "$CONFIG" \
        --cosmos_catalog "data/cosmos_web_lens_structural_properties.csv" \
        --lens_analysis_catalog "data/lens_analysis_catalog.csv" \
        --merged_field_catalog "data/merged_lens_field_catalog.csv" \
        --output_dir "$TREATMENT_DIR" \
        --n_lenses "$N_LENSES" \
        --n_non_lenses 0 \
        --variations_per_base 1 \
        --seed "$SEED" \
        --add_artifacts \
        --no_date_suffix \
        --numpix 300
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ $TREATMENT simulation completed${NC}"
        
        # Extract and update tracking catalog
        if [ -f "$TREATMENT_DIR/cosmos_lens_training_catalog.csv" ]; then
            # Add treatment info to each lens row
            tail -n +2 "$TREATMENT_DIR/cosmos_lens_training_catalog.csv" | \
            awk -v treatment="$TREATMENT" '{print $0 "," treatment}' >> "$TRACKING_CATALOG.tmp"
        fi
        
        return 0
    else
        echo -e "${RED}❌ $TREATMENT simulation failed${NC}"
        return 1
    fi
}

# Run all three treatments
echo -e "${YELLOW}🎬 Running simulations for all three treatments...${NC}"
echo ""

# Capture exit codes
EXIT_CODES=()

# 1. SIE+SIE (seed 42)
run_treatment "sie_sie" "$SIE_SIE_CONFIG" 42
EXIT_CODES+=(${PIPESTATUS[0]})

# 2. NFW+NFW (seed 43)
run_treatment "nfw_nfw" "$NFW_NFW_CONFIG" 43
EXIT_CODES+=(${PIPESTATUS[0]})

# 3. Shear-only (seed 44)
run_treatment "shear_only" "$SHEAR_ONLY_CONFIG" 44
EXIT_CODES+=(${PIPESTATUS[0]})

echo ""

# Check if all succeeded
ALL_SUCCESS=true
for code in "${EXIT_CODES[@]}"; do
    if [ $code -ne 0 ]; then
        ALL_SUCCESS=false
        break
    fi
done

if [[ "$ALL_SUCCESS" == "true" ]]; then
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║ ✅ ALL TREATMENTS COMPLETED SUCCESSFULLY   ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}📊 Output Summary:${NC}"
    echo "  SIE+SIE output:  $OUTPUT_DIR/sie_sie"
    echo "  NFW+NFW output:  $OUTPUT_DIR/nfw_nfw"
    echo "  Shear-only output: $OUTPUT_DIR/shear_only"
    echo ""
    echo -e "${YELLOW}📁 Tracking catalogs:${NC}"
    echo "  Quick reference: $TRACKING_CATALOG"
    echo "  Detailed info:   $DETAILED_CATALOG"
    echo ""
    echo -e "${YELLOW}💡 Next steps:${NC}"
    echo "  1. Review the output images in each treatment directory"
    echo "  2. Compare lensing signatures (arcs, rings, etc.)"
    echo "  3. Check the tracking catalogs to find specific lenses"
    echo "  4. Use for paper comparison figures"
    echo ""
    
    # Create summary script for easy browsing
    cat > "$OUTPUT_DIR/browse_results.sh" << 'BROWSESH'
#!/bin/bash
echo "Treatment results:"
echo "  1. SIE+SIE    (fast, simple point mass)"
echo "  2. NFW+NFW    (realistic dark matter)"
echo "  3. Shear-only (environmental effect only)"
echo ""
echo "Example viewing:"
echo "  open sie_sie/jpg_rgb/*.jpg"
echo "  open nfw_nfw/jpg_rgb/*.jpg"
echo "  open shear_only/jpg_rgb/*.jpg"
echo ""
ls -lh sie_sie/jpg_rgb | head -5
ls -lh nfw_nfw/jpg_rgb | head -5
ls -lh shear_only/jpg_rgb | head -5
BROWSESH
    chmod +x "$OUTPUT_DIR/browse_results.sh"
    
else
    echo -e "${RED}❌ Some treatments failed!${NC}"
    exit 1
fi
