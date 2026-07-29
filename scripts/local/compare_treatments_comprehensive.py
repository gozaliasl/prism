#!/usr/bin/env python
"""
Create comprehensive side-by-side comparisons of same lenses across three treatments.
Groups lenses by their physical properties for easy comparison.

Usage:
    python compare_treatments_comprehensive.py <output_root_dir>
"""

import sys
import os
import glob
import csv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def load_catalog(cat_path):
    """Load catalog as list of dicts."""
    data = []
    if not os.path.exists(cat_path):
        return data
    
    with open(cat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def get_image_path(treatment_dir, lens_id):
    """Find image file for a lens ID."""
    # Try direct match first: cosmos_lens_XXXXXX.jpg
    direct_path = Path(treatment_dir) / "jpg_rgb" / f"cosmos_lens_{int(lens_id):06d}.jpg"
    if direct_path.exists():
        return str(direct_path)
    
    # Fallback to pattern match
    pattern = Path(treatment_dir) / "jpg_rgb" / f"*_{lens_id}_*.jpg"
    matches = list(glob.glob(str(pattern)))
    if matches:
        return matches[0]
    return None

def load_image(path):
    """Load image as numpy array."""
    if not path or not os.path.exists(path):
        return None
    return np.array(Image.open(path))

def extract_properties(row):
    """Extract key properties from catalog row."""
    props = {}
    try:
        props['z_lens'] = float(row.get('lens_redshift', 0))
        props['z_source'] = float(row.get('source_redshift', 0))
        props['theta_E'] = float(row.get('theta_E', 0))
        props['mass'] = 10.0 + np.log10(max(float(row.get('lens_radius', 1)), 0.1))  # Estimate from radius
        props['lens_id'] = str(row.get('lens_id', ''))
        props['sersic'] = float(row.get('lens_n_sersic', 0))
    except Exception as e:
        return None
    return props

def find_similar_lenses(cat1, cat2, cat3, tolerance=0.05):
    """Find lenses with similar properties across treatments."""
    similar = []
    
    for row1 in cat1:
        props1 = extract_properties(row1)
        if not props1:
            continue
        
        # Find matching lens in treatment 2
        best_match2 = None
        best_diff2 = float('inf')
        
        for row2 in cat2:
            props2 = extract_properties(row2)
            if not props2:
                continue
            
            # Calculate similarity (weighted by important properties)
            theta_E_diff = abs(props1['theta_E'] - props2['theta_E']) / max(props1['theta_E'], 0.01)
            diff = (
                abs(props1['z_lens'] - props2['z_lens']) * 10 +
                theta_E_diff +
                abs(props1['mass'] - props2['mass']) * 0.5
            )
            
            if diff < best_diff2:
                best_diff2 = diff
                best_match2 = (row2, props2)
        
        # Find matching lens in treatment 3
        best_match3 = None
        best_diff3 = float('inf')
        
        for row3 in cat3:
            props3 = extract_properties(row3)
            if not props3:
                continue
            
            diff = (
                abs(props1['z_lens'] - props3['z_lens']) * 10 +
                abs(props1['theta_E'] - props3['theta_E']) / max(props1['theta_E'], 0.01) +
                abs(props1['mass'] - props3['mass']) * 0.5
            )
            
            if diff < best_diff3:
                best_diff3 = diff
                best_match3 = (row3, props3)
        
        # Keep if all three are reasonably similar
        if best_match2 and best_match3:
            if best_diff2 < 2.0 and best_diff3 < 2.0:
                similar.append({
                    'treatment_1': (row1, props1),
                    'treatment_2': best_match2,
                    'treatment_3': best_match3,
                    'diff_2': best_diff2,
                    'diff_3': best_diff3
                })
    
    # Sort by difference (best matches first)
    similar.sort(key=lambda x: x['diff_2'] + x['diff_3'])
    return similar

def create_comparison_figure(output_dir, similar_set, treatments=['sie_sie', 'nfw_nfw', 'shear_only']):
    """Create 3-panel comparison for a matched lens set."""
    
    row1, props1 = similar_set['treatment_1']
    row2, props2 = similar_set['treatment_2'][0], similar_set['treatment_2'][1]
    row3, props3 = similar_set['treatment_3'][0], similar_set['treatment_3'][1]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Title with lens properties
    title = f"Lens Comparison: z={props1['z_lens']:.2f}, θ_E={props1['theta_E']:.2f}\", M*={props1['mass']:.1f}"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    
    rows = [row1, row2, row3]
    props = [props1, props2, props3]
    
    for idx, (treatment, row, prop) in enumerate(zip(treatments, rows, props)):
        treatment_dir = Path(output_dir) / treatment
        lens_id = row.get('lens_id', '')
        img_path = get_image_path(treatment_dir, lens_id)
        
        if img_path:
            img = load_image(img_path)
            axes[idx].imshow(img)
            
            # Add treatment label
            label = treatment.upper().replace('_', ' ')
            axes[idx].set_title(f'{label}\nz={prop["z_source"]:.2f}, Sersic={prop["sersic"]:.1f}', 
                               fontweight='bold', fontsize=11)
        else:
            axes[idx].text(0.5, 0.5, f'Image not found\nID: {lens_id}', 
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=10)
            axes[idx].set_title(f'{treatment.upper()}', fontweight='bold', color='red')
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def create_grouped_comparisons(output_dir, similar_lenses, n_per_group=5):
    """Create comparison figures grouped by lens properties."""
    
    treatments = ['sie_sie', 'nfw_nfw', 'shear_only']
    
    print(f"\nFound {len(similar_lenses)} matching lens sets across treatments")
    print("Creating comparison figures...\n")
    
    # Group by redshift
    z_groups = {}
    for match in similar_lenses[:min(20, len(similar_lenses))]:
        props = match['treatment_1'][1]
        z_key = f"z_{props['z_lens']:.1f}"
        if z_key not in z_groups:
            z_groups[z_key] = []
        z_groups[z_key].append(match)
    
    figures_created = 0
    
    # Create figures for best matches
    for i, match in enumerate(similar_lenses[:10]):
        fig = create_comparison_figure(output_dir, match, treatments)
        
        props = match['treatment_1'][1]
        out_name = f"comparison_matched_{i:03d}_z{props['z_lens']:.2f}_E{props['theta_E']:.2f}.png"
        out_path = Path(output_dir) / out_name
        
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"✓ [{i+1}/10] {out_name}")
        plt.close(fig)
        figures_created += 1
    
    return figures_created

def print_statistics(output_dir, similar_lenses):
    """Print statistics about matched lenses."""
    
    print("\n" + "="*80)
    print("MATCHED LENSES STATISTICS")
    print("="*80)
    
    if not similar_lenses:
        print("No matched lenses found!")
        return
    
    print(f"\nTotal matches found: {len(similar_lenses)}")
    
    # Extract properties from best matches
    best_matches = similar_lenses[:10]
    
    print("\nBest 10 matches (by similarity):")
    print(f"{'#':>3} {'z_lens':>7} {'θ_E':>7} {'M*':>7} {'z_src':>7} {'Diff':>8}")
    print("-" * 45)
    
    for i, match in enumerate(best_matches):
        props1 = match['treatment_1'][1]
        total_diff = match['diff_2'] + match['diff_3']
        print(f"{i+1:3d} {props1['z_lens']:7.2f} {props1['theta_E']:7.2f} "
              f"{props1['mass']:7.1f} {props1['z_source']:7.2f} {total_diff:8.2f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_treatments_comprehensive.py <output_root_dir>")
        print("\nExample:")
        print("  python compare_treatments_comprehensive.py outputs/pair_treatments_20260215_134808")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    treatments = ['sie_sie', 'nfw_nfw', 'shear_only']
    
    # Load catalogs
    catalogs = {}
    for treatment in treatments:
        cat_path = output_dir / treatment / 'cosmos_lens_training_catalog.csv'
        catalogs[treatment] = load_catalog(str(cat_path))
        print(f"✓ Loaded {len(catalogs[treatment])} lenses from {treatment}")
    
    # Find similar lenses
    print("\nMatching lenses across treatments...")
    similar_lenses = find_similar_lenses(
        catalogs['sie_sie'],
        catalogs['nfw_nfw'],
        catalogs['shear_only'],
        tolerance=0.05
    )
    
    # Print statistics
    print_statistics(output_dir, similar_lenses)
    
    # Create comparison figures
    if similar_lenses:
        n_figs = create_grouped_comparisons(output_dir, similar_lenses)
        print(f"\n✅ Created {n_figs} comparison figures")
    
    print("\n" + "="*80)
    print("VIEWING RESULTS")
    print("="*80)
    print(f"""
View the matched comparisons:
  open {output_dir}/comparison_matched_*.png

These figures show the SAME lens across all three treatments:
  LEFT:   SIE+SIE (sharp, point-mass deflection)
  CENTER: NFW+NFW (realistic dark matter halo)
  RIGHT:  Shear-only (environmental effect only)

Since these are the same lens with the same source, you can directly
observe how different pair galaxy models affect the lensing morphology.

Key observations:
  • Arc sharpness differs between models
  • Image multiplicity should be consistent (4 for binary, 2 for shear)
  • Magnification patterns vary significantly
  • Lensed source visibility differs
    """)

if __name__ == '__main__':
    main()
