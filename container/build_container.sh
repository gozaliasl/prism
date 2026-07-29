#!/bin/bash
# Build Singularity container for Mahti
# Usage: ./build_container.sh [container_name.sif]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_NAME="${1:-prism.sif}"
CONTAINER_DIR="$SCRIPT_DIR"
DEF_FILE="$CONTAINER_DIR/Singularity.def"

echo "============================================================================"
echo "Building Singularity Container for JWST Lens Simulator"
echo "============================================================================"
echo ""
echo "Container name: $CONTAINER_NAME"
echo "Definition file: $DEF_FILE"
echo ""

# Check if Singularity/Apptainer is available
if command -v apptainer &> /dev/null; then
    SINGULARITY_CMD=apptainer
    echo "Using: apptainer"
elif command -v singularity &> /dev/null; then
    SINGULARITY_CMD=singularity
    echo "Using: singularity"
else
    echo "ERROR: Neither 'apptainer' nor 'singularity' found in PATH"
    echo "Please install Singularity/Apptainer or load the module on Mahti"
    exit 1
fi

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    exit 1
fi

# Build container
echo ""
echo "Building container (this may take 10-20 minutes)..."
echo ""

$SINGULARITY_CMD build \
    --fakeroot \
    "$CONTAINER_DIR/$CONTAINER_NAME" \
    "$DEF_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ Container built successfully!"
    echo "============================================================================"
    echo ""
    echo "Container location: $CONTAINER_DIR/$CONTAINER_NAME"
    echo ""
    echo "To use on Mahti:"
    echo "  1. Transfer container to Mahti:"
    echo "     scp $CONTAINER_DIR/$CONTAINER_NAME <username>@mahti.csc.fi:/scratch/<account>/<user>/"
    echo ""
    echo "  2. Update sbatch script to use container path"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "❌ Container build failed"
    echo "============================================================================"
    exit 1
fi

