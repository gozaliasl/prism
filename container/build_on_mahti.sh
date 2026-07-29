#!/bin/bash
# Build Singularity container on Mahti
# Usage: Run this script on Mahti login node

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="jwst_lens_simulator.sif"
DEF_FILE="$SCRIPT_DIR/Singularity.def"

echo "============================================================================"
echo "Building Singularity Container on Mahti"
echo "============================================================================"
echo ""
echo "Container name: $CONTAINER_NAME"
echo "Definition file: $DEF_FILE"
echo ""

# Check for Singularity/Apptainer (try multiple locations)
echo "Checking for Singularity/Apptainer..."
SINGULARITY_CMD=""

# Check common system paths
for cmd in apptainer singularity; do
    for path in \
        "/usr/bin/$cmd" \
        "/usr/local/bin/$cmd" \
        "/opt/singularity/bin/$cmd" \
        "/appl/soft/ai/singularity/bin/$cmd" \
        "$(command -v $cmd 2>/dev/null)"; do
        if [ -n "$path" ] && [ -x "$path" ]; then
            SINGULARITY_CMD="$path"
            echo "Found: $path"
            break 2
        fi
    done
done

# If not found, try loading as module (suppress errors)
if [ -z "$SINGULARITY_CMD" ]; then
    echo "Not found in standard paths, trying modules..."
    if module load singularity 2>/dev/null && command -v singularity &>/dev/null; then
        SINGULARITY_CMD=singularity
        echo "Loaded: singularity module"
    elif module load apptainer 2>/dev/null && command -v apptainer &>/dev/null; then
        SINGULARITY_CMD=apptainer
        echo "Loaded: apptainer module"
    fi
fi

# Final check
if [ -z "$SINGULARITY_CMD" ] || ! command -v "$SINGULARITY_CMD" &>/dev/null; then
    echo ""
    echo "============================================================================"
    echo "ERROR: Singularity/Apptainer not found"
    echo "============================================================================"
    echo ""
    echo "Tried:"
    echo "  - Standard system paths"
    echo "  - Module system (singularity, apptainer)"
    echo ""
    echo "Please check:"
    echo "  1. Is Singularity installed on Mahti?"
    echo "     Contact CSC support or check: https://docs.csc.fi/computing/containers/"
    echo ""
    echo "  2. Alternative: Build container on a different system and transfer it"
    echo ""
    echo "  3. Alternative: Use the non-container sbatch script instead:"
    echo "     ./scripts/submit_segmentation_training.sh --account ituomine"
    echo ""
    exit 1
fi

echo "Using: $SINGULARITY_CMD"

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    exit 1
fi

# Determine output location (scratch directory)
# Try to detect the correct scratch path
if [ -n "$CONTAINER_OUTPUT_DIR" ]; then
    # Allow explicit override via environment variable
    OUTPUT_DIR="$CONTAINER_OUTPUT_DIR"
elif [ -d "/scratch/ituomine/gozaliasl" ]; then
    # Use the project directory with big space
    OUTPUT_DIR="/scratch/ituomine/gozaliasl"
elif [ -n "$SCRATCH" ]; then
    OUTPUT_DIR="$SCRATCH"
elif [ -n "$SLURM_ACCOUNT" ] && [ -n "$USER" ]; then
    OUTPUT_DIR="/scratch/$SLURM_ACCOUNT/$USER"
else
    OUTPUT_DIR="/scratch/$USER"
fi

OUTPUT_PATH="$OUTPUT_DIR/$CONTAINER_NAME"

# Use scratch directory for temporary files (avoid /tmp which may be full)
TMPDIR="$OUTPUT_DIR/tmp_singularity_build"
mkdir -p "$TMPDIR"

# Set all possible environment variables for Apptainer/Singularity
export TMPDIR="$TMPDIR"
export APPTAINER_TMPDIR="$TMPDIR"
export SINGULARITY_TMPDIR="$TMPDIR"
export APPTAINER_CACHEDIR="$OUTPUT_DIR/.apptainer_cache"
export SINGULARITY_CACHEDIR="$OUTPUT_DIR/.singularity_cache"

# Clean up any existing build artifacts to free space
echo "Cleaning up old build artifacts..."
rm -rf "$TMPDIR"/* 2>/dev/null || true
rm -rf "$APPTAINER_CACHEDIR"/* 2>/dev/null || true
rm -rf "$SINGULARITY_CACHEDIR"/* 2>/dev/null || true

# Create cache directories
mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p "$SINGULARITY_CACHEDIR"

# Check available space
echo "Checking available space..."
df -h "$OUTPUT_DIR" | tail -1
AVAILABLE_SPACE=$(df -BG "$OUTPUT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space: ${AVAILABLE_SPACE}GB"
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo "WARNING: Less than 10GB available. Container build may fail."
    echo "Consider cleaning up old files in: $OUTPUT_DIR"
fi

echo "Output location: $OUTPUT_PATH"
echo "Temporary directory: $TMPDIR"
echo "Cache directory: $APPTAINER_CACHEDIR"
echo ""
echo "Building container (this may take 10-20 minutes)..."
echo ""

# Build container with custom temp directory
# Note: --tmpdir flag might not work, so we rely on environment variables
$SINGULARITY_CMD build \
    --fakeroot \
    --tmpdir "$TMPDIR" \
    "$OUTPUT_PATH" \
    "$DEF_FILE"

BUILD_EXIT_CODE=$?

# Clean up temporary directory after build
if [ -d "$TMPDIR" ]; then
    echo "Cleaning up temporary files..."
    rm -rf "$TMPDIR"
fi

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ Container built successfully!"
    echo "============================================================================"
    echo ""
    echo "Container location: $OUTPUT_PATH"
    echo ""
    echo "Size: $(du -h "$OUTPUT_PATH" | cut -f1)"
    echo ""
    echo "To use in sbatch scripts, set:"
    echo "  export CONTAINER_PATH=$OUTPUT_PATH"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "❌ Container build failed"
    echo "============================================================================"
    exit 1
fi

