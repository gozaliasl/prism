#!/bin/bash
# Verify that batch separation is working correctly
# Usage: ./verify_batch_separation.sh /path/to/output/dir

OUTPUT_DIR="${1:-.}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Output directory not found: $OUTPUT_DIR"
    exit 1
fi

echo "🔍 Verifying Batch Separation"
echo "==============================="
echo "Output: $OUTPUT_DIR"
echo ""

# Check single_field directory
echo "📂 SINGLE_FIELD Directory:"
if [ -d "$OUTPUT_DIR/single_field/unified_npz/" ]; then
    SF_SF=$(ls "$OUTPUT_DIR/single_field/unified_npz/" | grep "^PRISM_lens_SF" | wc -l)
    SF_BR=$(ls "$OUTPUT_DIR/single_field/unified_npz/" | grep "^PRISM_lens_BR" | wc -l)
    SF_GR=$(ls "$OUTPUT_DIR/single_field/unified_npz/" | grep "^PRISM_lens_GR" | wc -l)
    SF_TOTAL=$(ls "$OUTPUT_DIR/single_field/unified_npz/" | grep "^PRISM_lens_" | wc -l)
    
    echo "   SF files: $SF_SF"
    echo "   BR files: $SF_BR"
    echo "   GR files: $SF_GR"
    echo "   Total:    $SF_TOTAL"
    
    if [ $SF_BR -gt 0 ] || [ $SF_GR -gt 0 ]; then
        echo "   ❌ ERROR: Found non-SF lenses in single_field directory!"
    else
        echo "   ✅ OK: Only SF lenses found"
    fi
else
    echo "   ⚠️  Directory doesn't exist yet"
fi
echo ""

# Check binary_lenses directory
echo "📂 BINARY_LENSES Directory:"
if [ -d "$OUTPUT_DIR/binary_lenses/unified_npz/" ]; then
    BR_SF=$(ls "$OUTPUT_DIR/binary_lenses/unified_npz/" | grep "^PRISM_lens_SF" | wc -l)
    BR_BR=$(ls "$OUTPUT_DIR/binary_lenses/unified_npz/" | grep "^PRISM_lens_BR" | wc -l)
    BR_GR=$(ls "$OUTPUT_DIR/binary_lenses/unified_npz/" | grep "^PRISM_lens_GR" | wc -l)
    BR_TOTAL=$(ls "$OUTPUT_DIR/binary_lenses/unified_npz/" | grep "^PRISM_lens_" | wc -l)
    
    echo "   SF files: $BR_SF"
    echo "   BR files: $BR_BR"
    echo "   GR files: $BR_GR"
    echo "   Total:    $BR_TOTAL"
    
    if [ $BR_SF -gt 0 ] || [ $BR_GR -gt 0 ]; then
        echo "   ❌ ERROR: Found non-BR lenses in binary_lenses directory!"
    else
        echo "   ✅ OK: Only BR lenses found"
    fi
else
    echo "   ⚠️  Directory doesn't exist yet"
fi
echo ""

# Check group_lenses directory
echo "📂 GROUP_LENSES Directory:"
if [ -d "$OUTPUT_DIR/group_lenses/unified_npz/" ]; then
    GR_SF=$(ls "$OUTPUT_DIR/group_lenses/unified_npz/" | grep "^PRISM_lens_SF" | wc -l)
    GR_BR=$(ls "$OUTPUT_DIR/group_lenses/unified_npz/" | grep "^PRISM_lens_BR" | wc -l)
    GR_GR=$(ls "$OUTPUT_DIR/group_lenses/unified_npz/" | grep "^PRISM_lens_GR" | wc -l)
    GR_TOTAL=$(ls "$OUTPUT_DIR/group_lenses/unified_npz/" | grep "^PRISM_lens_" | wc -l)
    
    echo "   SF files: $GR_SF"
    echo "   BR files: $GR_BR"
    echo "   GR files: $GR_GR"
    echo "   Total:    $GR_TOTAL"
    
    if [ $GR_SF -gt 0 ] || [ $GR_BR -gt 0 ]; then
        echo "   ❌ ERROR: Found non-GR lenses in group_lenses directory!"
    else
        echo "   ✅ OK: Only GR lenses found"
    fi
else
    echo "   ⚠️  Directory doesn't exist yet"
fi
echo ""

# Summary
echo "📊 Summary:"
if [ -d "$OUTPUT_DIR/single_field/unified_npz/" ] && [ -d "$OUTPUT_DIR/binary_lenses/unified_npz/" ] && [ -d "$OUTPUT_DIR/group_lenses/unified_npz/" ]; then
    if [ $SF_BR -eq 0 ] && [ $SF_GR -eq 0 ] && [ $BR_SF -eq 0 ] && [ $BR_GR -eq 0 ] && [ $GR_SF -eq 0 ] && [ $GR_BR -eq 0 ]; then
        echo "✅ ALL CHECKS PASSED - Batch separation is working correctly!"
    else
        echo "❌ BATCH SEPARATION FAILED - Misclassified lenses found"
    fi
else
    echo "⏳ Workflow still running - not all directories complete yet"
fi
