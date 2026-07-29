#!/bin/bash
#
# Diagnostic script to check for common simulation failures
#

echo "🔍 JWST Lens Simulator - Diagnostic Report"
echo "=========================================="
echo ""

# Check Python environment
echo "1️⃣  Checking Python Environment..."
which python
python --version
echo ""

# Check conda environment
echo "2️⃣  Checking Conda Environment..."
conda info | head -5
echo ""

# Check required data files
echo "3️⃣  Checking Data Files..."
echo "   - Galaxy catalog: $([ -f 'data/galaxy_catalog.csv' ] && echo '✅ Found' || echo '❌ Missing')"
echo "   - COSMOS structural: $([ -f 'data/cosmos_web_lens_structural_properties.csv' ] && echo '✅ Found' || echo '❌ Missing')"
echo "   - Lens analysis: $([ -f 'data/lens_analysis_catalog.csv' ] && echo '✅ Found' || echo '❌ Missing')"
echo "   - Field catalog: $([ -f 'data/merged_lens_field_catalog.csv' ] && echo '✅ Found' || echo '❌ Missing')"
echo ""

# Check PSF data
echo "4️⃣  Checking PSF Data..."
if [ -d "data/psf_v5_30mas" ]; then
    PSF_COUNT=$(find data/psf_v5_30mas -name "*.npy" -o -name "*.pkl" | wc -l)
    echo "   PSF files: ✅ Found ($PSF_COUNT files)"
else
    echo "   PSF directory: ❌ Missing"
fi
echo ""

# Check output directory
echo "5️⃣  Checking Output Directory..."
OUTPUT_BASE="/Volumes/extHD/jwst-lens-similator-output"
if [ -d "$OUTPUT_BASE" ]; then
    echo "   Base directory: ✅ Found"
    RECENT=$(ls -td "$OUTPUT_BASE"/*/ 2>/dev/null | head -1)
    if [ -n "$RECENT" ]; then
        echo "   Most recent: $(basename "$RECENT")"
        echo "   Size: $(du -sh "$RECENT" 2>/dev/null | awk '{print $1}')"
    fi
else
    echo "   Base directory: ❌ Not mounted"
fi
echo ""

# Check disk space
echo "6️⃣  Checking Disk Space..."
df -h /Volumes/extHD/ 2>/dev/null | awk 'NR>1 {printf "   Used: %s / %s (%.1f%%)\n", $3, $2, $5}'
echo ""

# Test a small simulation
echo "7️⃣  Running Small Test Simulation (10 lenses, 10 non-lenses)..."
echo "   This will help identify any runtime errors..."
echo ""

python -m prism.core.simulator \
    --config configs/default_config.yaml \
    --cosmos_catalog data/cosmos_web_lens_structural_properties.csv \
    --lens_analysis_catalog data/lens_analysis_catalog.csv \
    --merged_field_catalog data/merged_lens_field_catalog.csv \
    --output_dir /tmp/jwst_diagnostic_test \
    --n_lenses 10 \
    --n_non_lenses 10 \
    --variations_per_base 1 \
    --seed 42 \
    --add_artifacts \
    --numpix 300 \
    --no_date_suffix 2>&1 | tail -50

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Small test simulation completed successfully!"
    echo "   Check /tmp/jwst_diagnostic_test for output"
else
    echo ""
    echo "❌ Small test simulation failed - check error messages above"
fi

echo ""
echo "📊 Diagnostic complete!"
