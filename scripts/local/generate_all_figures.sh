#!/bin/bash
#
# Generate All Intermediate Stage Figures
# Creates all three figure types for a given simulation output
#

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 <output_dir> [--lens-id <id>] [--all-lenses]"
    echo ""
    echo "Arguments:"
    echo "  output_dir          Path to simulation output directory"
    echo "  --lens-id <id>      Specific lens ID to visualize (e.g., 000001)"
    echo "  --all-lenses        Generate figures for multiple lenses (000001-000010)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/output"
    echo "  $0 /path/to/output --lens-id 000005"
    echo "  $0 /path/to/output --all-lenses"
}

# Parse arguments
if [[ $# -lt 1 ]]; then
    show_usage
    exit 1
fi

OUTPUT_DIR="$1"
LENS_ID="000001"
ALL_LENSES=false

# Parse optional arguments
while [[ $# -gt 1 ]]; do
    case $2 in
        --lens-id)
            LENS_ID="$3"
            shift 2
            ;;
        --all-lenses)
            ALL_LENSES=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $2"
            show_usage
            exit 1
            ;;
    esac
done

# Verify output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Find script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎨 JWST Lens Simulator: Intermediate Stage Figure Generator"
echo "==========================================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Lens ID(s): $LENS_ID"
echo ""

# Get Python executable
PYTHON=$(which python3 || which python)
if [ -z "$PYTHON" ]; then
    echo "❌ Error: Python not found"
    exit 1
fi

echo "✓ Using Python: $PYTHON"
echo ""

# Function to generate figures for a single lens ID
generate_figures() {
    local lens_id=$1
    echo "📊 Generating figures for lens ID: $lens_id"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Figure 1: Simple stages figure
    echo "  1️⃣  Creating intermediate stages figure..."
    if $PYTHON "$SCRIPT_DIR/create_intermediate_stages_figure.py" "$OUTPUT_DIR" --lens-id "$lens_id"; then
        echo "     ✓ Saved to: $OUTPUT_DIR/intermediate_stages_figure.png"
    else
        echo "     ⚠ Warning: Could not generate stages figure"
    fi
    
    # Figure 2: Advanced figure with statistics
    echo "  2️⃣  Creating advanced figure with statistics..."
    if $PYTHON "$SCRIPT_DIR/create_advanced_intermediate_figure.py" "$OUTPUT_DIR" --lens-id "$lens_id"; then
        echo "     ✓ Saved to: $OUTPUT_DIR/advanced_intermediate_figure.png"
    else
        echo "     ⚠ Warning: Could not generate advanced figure"
    fi
    
    # Figure 3: Flow diagram
    echo "  3️⃣  Creating flow diagram figure..."
    if $PYTHON "$SCRIPT_DIR/create_flow_diagram_figure.py" "$OUTPUT_DIR" --lens-id "$lens_id"; then
        echo "     ✓ Saved to: $OUTPUT_DIR/flow_diagram_figure.png"
    else
        echo "     ⚠ Warning: Could not generate flow diagram"
    fi
    
    echo ""
}

# Generate figures
if [ "$ALL_LENSES" = true ]; then
    echo "🔄 Generating figures for multiple lenses..."
    echo ""
    
    for i in {1..10}; do
        lens_id=$(printf "%06d" $i)
        generate_figures "$lens_id"
    done
else
    generate_figures "$LENS_ID"
fi

echo "✅ Figure generation complete!"
echo ""
echo "📁 Output files:"
echo "   - intermediate_stages_figure.png"
echo "   - advanced_intermediate_figure.png"
echo "   - flow_diagram_figure.png"
echo ""
echo "💡 Tips:"
echo "   - Use intermediate_stages_figure.png for your paper main text"
echo "   - Use advanced_intermediate_figure.png for supplementary material"
echo "   - Use flow_diagram_figure.png for presentations"
echo "   - Convert to PDF for submission: convert -density 300 figure.png figure.pdf"
