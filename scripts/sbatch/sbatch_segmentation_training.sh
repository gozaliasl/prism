#!/bin/bash
#SBATCH --job-name=seg_train
#SBATCH --account=ituomine
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/segmentation_training_%j.out
#SBATCH --error=logs/segmentation_training_%j.err

# Initialize module system ASAP (before any 'module' commands)
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
            # shellcheck disable=SC1090
            source "$init"
            if command -v module &>/dev/null; then
                return 0
            fi
        fi
    done
    return 1
}

# Clean PATH to avoid singularity conflicts (if present)
strip_ai_paths() {
    local var_name="$1"
    local current_value="${!var_name}"
    if [ -z "$current_value" ]; then
        return
    fi
    local oldIFS="$IFS"
    IFS=':' read -r -a components <<< "$current_value"
    IFS="$oldIFS"
    local rebuilt=()
    local entry
    for entry in "${components[@]}"; do
        if [ -z "$entry" ]; then
            continue
        fi
        case "$entry" in
            *"/appl/soft/ai"*)
                continue
                ;;
        esac
        rebuilt+=("$entry")
    done
    if [ ${#rebuilt[@]} -gt 0 ]; then
        local new_value
        IFS=':'; new_value="${rebuilt[*]}"; IFS="$oldIFS"
        eval "export $var_name=\"$new_value\""
    else
        eval "unset $var_name"
    fi
}

sanitize_env_paths() {
    strip_ai_paths PATH
    strip_ai_paths LD_LIBRARY_PATH
    strip_ai_paths PYTHONPATH
    hash -r 2>/dev/null || true
}

sanitize_env_paths

# Ensure module command is available (Mahti requirement)
if ! initialize_module_system; then
    echo "❌ Error: Unable to initialize Mahti module system (module command not found)."
    echo "   Tried known CSC/Modules init scripts (zz-csc-env.sh, lmod, etc.)."
    echo "   Please run from a Mahti login environment or source the module init script before sbatch."
    exit 1
fi

# ============================================================================
# Simple Mahti Segmentation Training Pipeline
# ============================================================================
# This script runs the complete pipeline:
#   1. Simulate lens systems
#   2. Prepare segmentation training data
#   3. Train U-Net model
#   4. (Optional) Run detection/annotation
#
# To modify resources, edit the #SBATCH directives above.
# To modify pipeline parameters, edit the variables in the Configuration section.
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration - EDIT THESE AS NEEDED
# ============================================================================

# Job parameters (can be overridden with sbatch --export)
N_LENSES=${N_LENSES:-10000}
N_NON_LENSES=${N_NON_LENSES:-0}
VARIATIONS=${VARIATIONS:-1}
SEED=${SEED:-42}
TIME_DELAY_FRACTION=${TIME_DELAY_FRACTION:-0.0}
TRAINING_EPOCHS=${TRAINING_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
DEVICE=${DEVICE:-cuda}
RUN_DETECTION=${RUN_DETECTION:-false}
LENS_IDS_FOR_DETECTION=${LENS_IDS_FOR_DETECTION:-""}
VENV_PATH=${VENV_PATH:-""}
USE_SYSTEM_SITE_PACKAGES=${USE_SYSTEM_SITE_PACKAGES:-false}
FORCE_RECREATE_VENV=${FORCE_RECREATE_VENV:-false}

# Pip/index configuration (override via sbatch --export)
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}
PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-}
PIP_DISABLE_CACHE=${PIP_DISABLE_CACHE:-true}

# Output directory
OUTPUT_BASE="/scratch/ituomine/gozaliasl/jwst-mock-lens-simulator/output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/segmentation_training_${TIMESTAMP}"

# Project root - tries multiple locations
if [ -d "${USERAPPL}/jwst-mock-lens-simulator" ]; then
    PROJECT_ROOT="${USERAPPL}/jwst-mock-lens-simulator"
elif [ -d "/users/${USER}/jwst-mock-lens-simulator" ]; then
    PROJECT_ROOT="/users/${USER}/jwst-mock-lens-simulator"
elif [ -d "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(pwd)"
fi

cd "$PROJECT_ROOT"
mkdir -p logs

# ============================================================================
# Environment Setup
# ============================================================================

echo "============================================================================"
echo "Segmentation Training Pipeline - Mahti"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Clean PATH again (module system might have added singularity paths)
sanitize_env_paths

# Load PyTorch module FIRST (before detecting Python, so we get module's Python)
USE_PIP_TORCH=false
if command -v module &> /dev/null; then
    echo "Loading modules..."
    module --force purge 2>/dev/null || module purge 2>/dev/null || true
    
    # Clean PATH after purge (just to be safe)
    sanitize_env_paths
    
    # Load PyTorch module (includes Python 3.9+, scikit-learn, matplotlib, seaborn)
    if module load pytorch/2.0 2>/dev/null; then
        echo "Loaded pytorch/2.0 (includes Python 3.9+, scikit-learn, matplotlib, seaborn)"
        USE_PIP_TORCH=false
    elif module load pytorch/2.1 2>/dev/null; then
        echo "Loaded pytorch/2.1"
        USE_PIP_TORCH=false
    elif module load pytorch 2>/dev/null; then
        echo "Loaded pytorch (default)"
        USE_PIP_TORCH=false
    else
        echo "Warning: Could not load pytorch module, will use pip"
        USE_PIP_TORCH=true
    fi
    
    # Clean PATH after module load
    sanitize_env_paths
    
    # Verify PyTorch is available
    if [ "$USE_PIP_TORCH" = "false" ]; then
        if python -c "import torch" 2>/dev/null; then
            echo "PyTorch module loaded successfully"
            python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1 || true
        else
            echo "Warning: PyTorch import failed, will use pip"
            USE_PIP_TORCH=true
        fi
    fi
else
    echo "Warning: Module system not available, will use pip"
    USE_PIP_TORCH=true
fi

# Determine Python command AFTER module loading (so we get module's Python if available)
detect_python() {
    local candidates=()
    
    # First, try to find Python from module (should be in PATH after module load)
    # Check common module Python paths - ALLOW pytorch module paths under /appl/soft/ai/
    if command -v python3 &> /dev/null; then
        local py3_path=$(command -v python3)
        # Allow /appl/soft/ai/pytorch/*/bin/python3 (module Python)
        # But exclude /appl/soft/ai/bin/python3 (wrapper scripts)
        if printf "%s" "$py3_path" | grep -q "/appl/soft/ai/pytorch/"; then
            candidates+=("$py3_path")
        elif printf "%s" "$py3_path" | grep -qv "^/usr/bin/" && printf "%s" "$py3_path" | grep -qv "/appl/soft/ai/bin"; then
            candidates+=("$py3_path")
        fi
    fi
    if command -v python &> /dev/null; then
        local py_path=$(command -v python)
        if printf "%s" "$py_path" | grep -q "/appl/soft/ai/pytorch/"; then
            candidates+=("$py_path")
        elif printf "%s" "$py_path" | grep -qv "^/usr/bin/" && printf "%s" "$py_path" | grep -qv "/appl/soft/ai/bin"; then
            candidates+=("$py_path")
        fi
    fi
    
    # Check common module installation paths directly
    local pytorch_paths=(
        "/appl/soft/ai/pytorch/2.0/bin/python3"
        "/appl/soft/ai/pytorch/2.1/bin/python3"
        "/appl/soft/ai/pytorch/bin/python3"
    )
    for path in "${pytorch_paths[@]}"; do
        if [ -x "$path" ]; then
            candidates+=("$path")
        fi
    done
    
    # Fallback to system Python (but check version)
    if command -v python3 &> /dev/null; then
        local py3_path=$(command -v python3)
        if printf "%s" "$py3_path" | grep -q "^/usr/bin/"; then
            candidates+=("$py3_path")
        fi
    fi
    if command -v python &> /dev/null; then
        local py_path=$(command -v python)
        if printf "%s" "$py_path" | grep -q "^/usr/bin/"; then
            candidates+=("$py_path")
        fi
    fi
    
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ] && [ -n "$candidate" ]; then
            # Exclude wrapper scripts in /appl/soft/ai/bin (but allow module paths)
            if printf "%s" "$candidate" | grep -q "/appl/soft/ai/bin$"; then
                continue
            fi
            # Verify it's actually Python 3.8+
            local version=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            local major=$(echo "$version" | cut -d. -f1)
            local minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

if PYTHON_CMD=$(detect_python); then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    echo "Using Python: $PYTHON_CMD (version $PYTHON_VERSION)"
    
    # Check if Python version is compatible (need 3.8+ for numpy>=1.21.0)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo "Error: Python $PYTHON_VERSION is too old (need 3.8+ for numpy>=1.21.0)"
        echo "The pytorch module should provide Python 3.9+. Check if module loaded correctly."
        exit 1
    else
        echo "Python version $PYTHON_VERSION is compatible"
    fi
else
    echo "Error: Could not locate a Python interpreter outside /appl/soft/ai/bin"
    exit 1
fi
echo ""

resolve_realpath() {
    local target="$1"
    if command -v realpath &> /dev/null; then
        realpath "$target"
    else
        /usr/bin/python3 - "$target" <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
    fi
}

# Virtual environment
if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
    VENV_DIR="$VENV_PATH"
elif [ -d "$PROJECT_ROOT/venv_mahti" ]; then
    VENV_DIR="$PROJECT_ROOT/venv_mahti"
else
    VENV_DIR="$PROJECT_ROOT/venv_mahti"
fi

if [ "$FORCE_RECREATE_VENV" = "true" ] && [ -d "$VENV_DIR" ]; then
    echo "FORCE_RECREATE_VENV=true -> removing existing virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR (system-site-packages: $USE_SYSTEM_SITE_PACKAGES)"
    if [ "$USE_SYSTEM_SITE_PACKAGES" = "true" ]; then
        "$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR"
    else
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
fi

echo "Activating virtual environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"
hash -r 2>/dev/null || true
PYTHON_VENV_BIN="$(command -v python)"
PYTHON_VENV_REAL=$(resolve_realpath "$PYTHON_VENV_BIN")
if printf "%s" "$PYTHON_VENV_BIN $PYTHON_VENV_REAL" | grep -q "/appl/soft/ai"; then
    echo "Warning: Virtualenv python resolved to $PYTHON_VENV_REAL (contains /appl/soft/ai); rebuilding with base interpreter $PYTHON_CMD"
    deactivate 2>/dev/null || true
    rm -rf "$VENV_DIR"
    if [ "$USE_SYSTEM_SITE_PACKAGES" = "true" ]; then
        "$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR"
    else
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    hash -r 2>/dev/null || true
    PYTHON_VENV_BIN="$(command -v python)"
    PYTHON_VENV_REAL=$(resolve_realpath "$PYTHON_VENV_BIN")
fi

echo "Venv python: $PYTHON_VENV_BIN"

# Clean PATH one more time after venv activation (venv might add paths)
sanitize_env_paths

run_pip() {
    # Build pip command with proper argument order
    # Format: python -m pip [global-options] <command> [command-options] [packages]
    local cmd=( "$PYTHON_VENV_BIN" -m pip )
    
    # Global options (before command)
    if [ "$PIP_DISABLE_CACHE" = "true" ]; then
        cmd+=( --no-cache-dir )
    fi
    
    # Get the command (first argument)
    local pip_command="$1"
    shift
    
    # Add command
    cmd+=( "$pip_command" )
    
    # Command-specific options (for install command)
    if [ "$pip_command" = "install" ]; then
        # Add index URLs if specified
        if [ -n "$PIP_INDEX_URL" ]; then
            cmd+=( --index-url "$PIP_INDEX_URL" )
        fi
        if [ -n "$PIP_EXTRA_INDEX_URL" ]; then
            cmd+=( --extra-index-url "$PIP_EXTRA_INDEX_URL" )
        fi
    fi
    
    # Add all remaining arguments (options and packages)
    cmd+=( "$@" )
    
    # Execute
    "${cmd[@]}"
}

# Install packages
echo "Installing/updating packages..."
run_pip install --upgrade pip --quiet

if [ "$USE_PIP_TORCH" = "true" ]; then
    echo "Installing PyTorch via pip..."
    if [ "$DEVICE" = "cuda" ]; then
        PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118" run_pip install torch torchvision --quiet || true
    else
        PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" run_pip install torch torchvision --quiet || true
    fi
else
    echo "Using PyTorch from module system (includes scikit-learn, matplotlib, seaborn)"
fi

# Check which packages are already available from module
check_package() {
    "$PYTHON_VENV_BIN" -c "import $1" 2>/dev/null && echo "yes" || echo "no"
}

echo "Checking packages available from module..."
HAS_NUMPY=$(check_package numpy)
HAS_SCIPY=$(check_package scipy)
HAS_PANDAS=$(check_package pandas)
HAS_MATPLOTLIB=$(check_package matplotlib)
HAS_SEABORN=$(check_package seaborn)
HAS_ASTROPY=$(check_package astropy)
HAS_YAML=$(check_package yaml)

# Install only missing packages
PACKAGES_TO_INSTALL=()
if [ "$HAS_NUMPY" = "no" ]; then
    PACKAGES_TO_INSTALL+=(numpy)
fi
if [ "$HAS_SCIPY" = "no" ]; then
    PACKAGES_TO_INSTALL+=(scipy)
fi
if [ "$HAS_PANDAS" = "no" ]; then
    PACKAGES_TO_INSTALL+=(pandas)
fi
if [ "$HAS_MATPLOTLIB" = "no" ]; then
    PACKAGES_TO_INSTALL+=(matplotlib)
fi
if [ "$HAS_SEABORN" = "no" ]; then
    PACKAGES_TO_INSTALL+=(seaborn)
fi
if [ "$HAS_ASTROPY" = "no" ]; then
    PACKAGES_TO_INSTALL+=(astropy)
fi
if [ "$HAS_YAML" = "no" ]; then
    PACKAGES_TO_INSTALL+=(pyyaml)
fi

if [ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]; then
    echo "Installing missing packages: ${PACKAGES_TO_INSTALL[*]}"
    run_pip install "${PACKAGES_TO_INSTALL[@]}" --quiet || true
else
    echo "All key packages already available from module (numpy, scipy, pandas, matplotlib, seaborn, astropy, yaml)"
fi

# Install other requirements
run_pip install -r requirements.txt --quiet

# Verify key packages
echo "Verifying key packages..."
\"$PYTHON_VENV_BIN\" -c "
import sys
packages = ['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'astropy', 'yaml', 'torch']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'OK: {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'Missing: {pkg}')
if missing:
    print(f'Warning: Missing packages: {missing}')
    sys.exit(1)
" 2>&1 || echo "Warning: Some packages may be missing"

# Verify PyTorch
echo "Verifying PyTorch..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1 || echo "Warning: PyTorch verification failed"
echo ""

# ============================================================================
# Paths
# ============================================================================

CONFIG="$PROJECT_ROOT/configs/default_config.yaml"
COSMOS_CATALOG="$PROJECT_ROOT/data/cosmos_web_lens_structural_properties.csv"
LENS_ANALYSIS="$PROJECT_ROOT/data/lens_analysis_catalog.csv"
FIELD_CATALOG="$PROJECT_ROOT/data/merged_lens_field_catalog.csv"

# Update config for time delays
TEMP_CONFIG="/tmp/jwst_segmentation_${SLURM_JOB_ID}.yaml"
cp "$CONFIG" "$TEMP_CONFIG"

$PYTHON_CMD << PYEOF
import yaml
with open("$CONFIG", 'r') as f:
    config = yaml.safe_load(f)
if 'time_delays' not in config:
    config['time_delays'] = {}
config['time_delays']['enabled'] = float("$TIME_DELAY_FRACTION") > 0.0
config['time_delays']['fraction_variable_sources'] = float("$TIME_DELAY_FRACTION")
with open("$TEMP_CONFIG", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
PYEOF

CONFIG="$TEMP_CONFIG"

# ============================================================================
# Step 1: Run Simulations
# ============================================================================

echo "============================================================================"
echo "Step 1: Running Lens Simulations"
echo "============================================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

$PYTHON_CMD "$PROJECT_ROOT/src/jwst_lens_simulator.py" \
    --config "$CONFIG" \
    --cosmos_catalog "$COSMOS_CATALOG" \
    --lens_analysis_catalog "$LENS_ANALYSIS" \
    --merged_field_catalog "$FIELD_CATALOG" \
    --output_dir "$OUTPUT_DIR" \
    --n_lenses $N_LENSES \
    --n_non_lenses $N_NON_LENSES \
    --variations $VARIATIONS \
    --seed $SEED \
    --add_artifacts \
    --numpix 300

if [ $? -ne 0 ]; then
    echo "Error: Simulation failed!"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

echo "Simulations completed!"
echo ""

# ============================================================================
# Step 2: Prepare Training Data
# ============================================================================

echo "============================================================================"
echo "Step 2: Preparing Segmentation Training Data"
echo "============================================================================"
echo ""

TRAINING_DIR="$OUTPUT_DIR/segmentation_training"

$PYTHON_CMD "$PROJECT_ROOT/scripts/local/prepare_segmentation_training_data.py" \
    --output-dir "$OUTPUT_DIR" \
    --training-dir "$TRAINING_DIR" \
    --patch-size 128 \
    --stride 64

if [ $? -ne 0 ]; then
    echo "Error: Training data preparation failed!"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

if [ ! -f "$TRAINING_DIR/training_metadata.csv" ]; then
    echo "Error: Training metadata file not found!"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

N_PATCHES=$(wc -l < "$TRAINING_DIR/training_metadata.csv" 2>/dev/null || echo "0")
N_PATCHES=$((N_PATCHES - 1))

if [ "$N_PATCHES" -le 0 ]; then
    echo "Error: No training patches extracted!"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

echo "Found $N_PATCHES training patches"
echo "Training data preparation completed!"
echo ""

# ============================================================================
# Step 3: Train U-Net Model
# ============================================================================

echo "============================================================================"
echo "Step 3: Training U-Net Segmentation Model"
echo "============================================================================"
echo ""

MODEL_PATH="$OUTPUT_DIR/unet_segmentation_model.pth"

$PYTHON_CMD "$PROJECT_ROOT/scripts/local/train_segmentation_model.py" \
    --training-dir "$TRAINING_DIR" \
    --model-output "$MODEL_PATH" \
    --epochs $TRAINING_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "Error: Model training failed!"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

echo "Model training completed!"
echo "Model saved to: $MODEL_PATH"
echo ""

# ============================================================================
# Step 4: (Optional) Run Detection/Annotation
# ============================================================================

if [ "$RUN_DETECTION" = "true" ]; then
    echo "============================================================================"
    echo "Step 4: Running Detection/Annotation"
    echo "============================================================================"
    echo ""
    
    LENS_IDS_FOR_DETECTION=${LENS_IDS_FOR_DETECTION:-"0,1,2,3,4,5,6,7,8,9"}
    IFS=',' read -ra LENS_ID_ARRAY <<< "$LENS_IDS_FOR_DETECTION"
    
    for LENS_ID in "${LENS_ID_ARRAY[@]}"; do
        LENS_ID=$(echo "$LENS_ID" | xargs)
        echo "Processing lens $LENS_ID..."
        
        $PYTHON_CMD "$PROJECT_ROOT/scripts/local/identify_lensed_images.py" \
            --output-dir "$OUTPUT_DIR" \
            --lens-id "$LENS_ID" \
            --epoch 0 \
            --method segmentation \
            --model-path "$MODEL_PATH" || echo "Warning: Failed to process lens $LENS_ID"
    done
    
    echo "Detection/annotation completed!"
    echo ""
fi

# Cleanup
rm -f "$TEMP_CONFIG"

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "Pipeline Completed Successfully!"
echo "============================================================================"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "Training Data: $TRAINING_DIR"
echo "Trained Model: $MODEL_PATH"
echo "Training Patches: $N_PATCHES"
echo ""
echo "To download results:"
echo "  scp -r <username>@mahti.csc.fi:$OUTPUT_DIR ./"
echo ""
