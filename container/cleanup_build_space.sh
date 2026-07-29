#!/bin/bash
# Cleanup script to free space for container build on Mahti

set -e

OUTPUT_DIR="${1:-/scratch/ituomine/gozaliasl}"

echo "============================================================================"
echo "Cleaning Up Build Space"
echo "============================================================================"
echo ""
echo "Target directory: $OUTPUT_DIR"
echo ""

# Check current space
echo "Current space usage:"
df -h "$OUTPUT_DIR" | tail -1
echo ""

# Clean up build artifacts
echo "Cleaning up build artifacts..."
rm -rf "$OUTPUT_DIR/tmp_singularity_build"/* 2>/dev/null || true
rm -rf "$OUTPUT_DIR/.apptainer_cache"/* 2>/dev/null || true
rm -rf "$OUTPUT_DIR/.singularity_cache"/* 2>/dev/null || true

# Clean up any failed builds
echo "Cleaning up failed builds..."
find "$OUTPUT_DIR" -name "*.sif.*" -type f -delete 2>/dev/null || true
find "$OUTPUT_DIR" -name "build-temp-*" -type d -exec rm -rf {} + 2>/dev/null || true

# Show space after cleanup
echo ""
echo "Space after cleanup:"
df -h "$OUTPUT_DIR" | tail -1
echo ""

echo "✅ Cleanup complete!"

