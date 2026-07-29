#!/usr/bin/env python
"""
Generate HTML index for browsing all treatment comparisons.
"""

import sys
import glob
from pathlib import Path

def generate_index(output_dir):
    """Generate HTML index file."""
    output_dir = Path(output_dir)
    
    # Find all comparison images
    comparison_files = sorted(glob.glob(str(output_dir / "comparison_matched_*.png")))
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Pair Galaxy Treatment Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }
        .treatment-box {
            background-color: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .treatment-title {
            font-size: 14px;
            font-weight: bold;
            color: #0066cc;
            margin: 10px 0;
        }
        .treatment-desc {
            font-size: 13px;
            color: #666;
            margin: 5px 0 15px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(800px, 1fr));
            gap: 20px;
        }
        .comparison-panel {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .filename {
            font-size: 12px;
            color: #888;
            font-family: monospace;
            margin-top: 5px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        .info-box {
            background: #f9f9f9;
            padding: 10px;
            border-left: 4px solid #0066cc;
            border-radius: 4px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h1>🔬 Pair Galaxy Treatment Comparison</h1>
    
    <div class="treatment-box">
        <div class="treatment-title">How to Interpret These Comparisons</div>
        <div class="treatment-desc">
            Each panel shows THE SAME LENS with the same source, but modeled with three different 
            pair galaxy treatments. This allows direct comparison of how different mass models 
            affect lensing morphology.
        </div>
    </div>
    
    <div class="info-grid">
        <div class="info-box">
            <strong>LEFT: SIE+SIE Binary Lens</strong><br>
            Both pair galaxies modeled as point masses (SIE). Creates sharp, angular arcs. 
            Fast computation (baseline speed).
        </div>
        <div class="info-box">
            <strong>CENTER: NFW+NFW Binary Lens</strong><br>
            Both pair galaxies modeled with dark matter halos (NFW). Creates softer arcs. 
            Realistic but slower (2-3× baseline).
        </div>
        <div class="info-box">
            <strong>RIGHT: Shear-Only</strong><br>
            Pair contributes only external shear, no binary lensing. Creates minimal lensing 
            effect. Fastest computation.
        </div>
        <div class="info-box">
            <strong>Key Observations</strong><br>
            • Arc sharpness varies significantly<br>
            • Image multiplicity changes (4 vs 2)<br>
            • Magnification patterns differ<br>
            • Lensed source visibility varies
        </div>
    </div>
    
    <h2>Matched Lens Comparisons</h2>
    <div class="grid-container">
"""
    
    for i, filepath in enumerate(comparison_files[:10]):  # Show first 10
        filename = Path(filepath).name
        rel_path = filepath.replace(str(output_dir), ".")
        
        # Extract properties from filename
        parts = filename.replace("comparison_matched_", "").replace(".png", "").split("_")
        
        html += f"""        <div class="comparison-panel">
            <img src="{rel_path}" alt="Comparison {i+1}">
            <div class="filename">{filename}</div>
        </div>
"""
    
    html += """    </div>
    
    <div class="treatment-box">
        <strong>Tips for Analysis:</strong>
        <ul>
            <li><strong>Arc morphology:</strong> Compare the shape and sharpness of arcs across panels</li>
            <li><strong>Image count:</strong> Binary models typically show 4 images, shear-only shows 2</li>
            <li><strong>Source visibility:</strong> Look for the faint lensed source in the arcs</li>
            <li><strong>Pair galaxy visibility:</strong> In binary models, you might see the pair affecting arc positions</li>
            <li><strong>Magnification:</strong> Notice how brightly the source appears - this varies by model</li>
        </ul>
    </div>
    
    <div class="treatment-box">
        <strong>For Your Paper:</strong>
        <p>These comparisons directly show the robustness of lens detection across different 
        pair galaxy modeling assumptions. You can argue that:</p>
        <ul>
            <li>NFW+NFW is more physically realistic (preferred for results)</li>
            <li>SIE+SIE is faster but produces sharper arcs (good for training)</li>
            <li>Shear-only tests robustness to environmental perturbations</li>
        </ul>
    </div>
    
</body>
</html>
"""
    
    output_file = output_dir / "COMPARISON_INDEX.html"
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Created HTML index: {output_file}")
    print(f"\nTo view: open {output_file}")
    
    return output_file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_comparison_index.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    generate_index(output_dir)
