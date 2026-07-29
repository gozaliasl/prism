#!/usr/bin/env python
"""
Compare realistic lens simulations across three pair galaxy treatments.
Shows images side-by-side and extracts key lensing parameters.

Usage:
    python compare_treatments_visually.py <output_root_dir>
    python compare_treatments_visually.py /path/to/pair_treatments_20260215_134808
"""

import sys
import os
import glob
import csv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def get_catalogs(treatment_dir):
    """Load catalog for a treatment."""
    cat_path = Path(treatment_dir) / "cosmos_lens_training_catalog.csv"
    if not cat_path.exists():
        return None
    
    catalogs = []
    with open(cat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            catalogs.append(row)
    return catalogs

def get_image_path(treatment_dir, lens_id):
    """Find image file for a lens ID."""
    pattern = Path(treatment_dir) / "jpg_rgb" / f"*_{lens_id}_*.jpg"
    matches = list(glob.glob(str(pattern)))
    if matches:
        return matches[0]
    return None

def load_image(path):
    """Load image and convert to array."""
    if not path or not os.path.exists(path):
        return None
    return np.array(Image.open(path))

def create_comparison_panel(output_dir, lens_id, treatments=['sie_sie', 'nfw_nfw', 'shear_only']):
    """Create side-by-side comparison for a single lens across treatments."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Pair Galaxy Treatment Comparison: Lens ID {lens_id}', fontsize=14, fontweight='bold')
    
    for idx, treatment in enumerate(treatments):
        treatment_dir = Path(output_dir) / treatment
        img_path = get_image_path(treatment_dir, lens_id)
        
        if img_path:
            img = load_image(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{treatment.replace("_", " ").upper()}', fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, f'No image found\nfor {treatment}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{treatment.upper()}', fontweight='bold', color='red')
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def summarize_catalogs(output_dir, treatments=['sie_sie', 'nfw_nfw', 'shear_only']):
    """Print summary of each treatment."""
    print("\n" + "="*80)
    print("PAIR GALAXY TREATMENT COMPARISON SUMMARY")
    print("="*80)
    
    for treatment in treatments:
        treatment_dir = Path(output_dir) / treatment
        cat = get_catalogs(str(treatment_dir))
        
        if cat:
            print(f"\n{treatment.upper().replace('_', ' ')}")
            print("-" * 40)
            print(f"Total lenses: {len(cat)}")
            
            if len(cat) > 0:
                # Extract key parameters
                try:
                    theta_es = [float(row.get('theta_E_arcsec', 0)) for row in cat if row.get('theta_E_arcsec')]
                    z_lens = [float(row.get('z_lens', 0)) for row in cat if row.get('z_lens')]
                    z_source = [float(row.get('z_source', 0)) for row in cat if row.get('z_source')]
                    
                    if theta_es:
                        print(f"θ_E (arcsec): {np.mean(theta_es):.2f} ± {np.std(theta_es):.2f}")
                    if z_lens:
                        print(f"z_lens: {np.mean(z_lens):.2f} ± {np.std(z_lens):.2f}")
                    if z_source:
                        print(f"z_source: {np.mean(z_source):.2f} ± {np.std(z_source):.2f}")
                except:
                    pass
            
            # List image files
            img_dir = Path(treatment_dir) / "jpg_rgb"
            if img_dir.exists():
                n_images = len(list(img_dir.glob("*.jpg")))
                print(f"Images available: {n_images}")
        else:
            print(f"\n{treatment.upper()}: No catalog found")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_treatments_visually.py <output_root_dir>")
        print("\nExample:")
        print("  python compare_treatments_visually.py /path/to/pair_treatments_20260215_134808")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    # Print summary
    summarize_catalogs(output_dir)
    
    # Create comparison figures for first 5 lenses
    treatments = ['sie_sie', 'nfw_nfw', 'shear_only']
    
    # Find lens IDs from first treatment
    first_cat = get_catalogs(str(output_dir / treatments[0]))
    if first_cat:
        print("\n" + "="*80)
        print("CREATING COMPARISON FIGURES")
        print("="*80)
        
        sample_ids = [cat['lens_id'] if 'lens_id' in cat else str(i) 
                     for i, cat in enumerate(first_cat[:5])]
        
        for lens_id in sample_ids:
            fig = create_comparison_panel(output_dir, lens_id, treatments)
            out_path = output_dir / f"comparison_lens_{lens_id}.png"
            fig.savefig(out_path, dpi=100, bbox_inches='tight')
            print(f"✓ Saved: {out_path}")
            plt.close(fig)
    
    print("\n" + "="*80)
    print("VIEWING RESULTS")
    print("="*80)
    print("""
To view results:
  
1. Individual treatment images:
   open outputs/pair_treatments_*/sie_sie/jpg_rgb/*.jpg
   open outputs/pair_treatments_*/nfw_nfw/jpg_rgb/*.jpg
   open outputs/pair_treatments_*/shear_only/jpg_rgb/*.jpg

2. Comparison panels (created above):
   open outputs/pair_treatments_*/comparison_lens_*.png

3. Key differences to look for:
   - SIE+SIE: Sharp point-like mass deflections, clean arcs
   - NFW+NFW: Extended mass profiles, softer arcs, more extended images
   - Shear-only: Minimal pair galaxy effect, weaker lensing signals
    """)

if __name__ == '__main__':
    main()
