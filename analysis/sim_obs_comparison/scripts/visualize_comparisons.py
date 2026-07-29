#!/usr/bin/env python3
"""
Generate side-by-side visualizations of matched real vs simulated lenses.

For each match in matched_lens_pairs.csv, creates a figure showing:
- Top row: Real lens (FITS data or PNG if available, all 4 bands + RGB)
- Bottom row: Simulated lens equivalent (from .npz files, all 4 bands + RGB)

Output: visualizations/comparison_[real_name]_vs_[sim_id].png

Usage:
  python visualize_comparisons.py [--max-pairs 10] [--stretch log]
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_real_png(real_dir, lens_name, band):
    """Try to load a real lens band image (FITS or PNG preview)."""
    try:
        from astropy.io import fits
        fits_path = real_dir / f"{lens_name}_{band}.fits"
        if fits_path.exists():
            with fits.open(fits_path) as hdul:
                return hdul[0].data
    except Exception:
        pass
    return None


def load_sim_bands(sim_dir, sim_filename):
    """Load simulated lens bands from .npz file."""
    try:
        npz_path = sim_dir / "unified_npz" / f"{sim_filename}.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as data:
                # image_final is stored as (5, 300, 300): 4 bands + RGB
                img = data.get('image_final')
                if img is not None:
                    return img[:4]  # Return just the 4 bands
    except Exception as pass
    return None


def normalize_band(img, stretch='log', percentile=99.5):
    """Normalize band image for display."""
    if img is None:
        return None
    
    img_clean = img[~np.isnan(img)]
    if len(img_clean) == 0:
        return None
    
    vmax = np.percentile(img_clean, percentile)
    if vmax <= 0:
        return np.zeros_like(img)
    
    normalized = img / vmax
    
    if stretch == 'log':
        normalized = np.log10(1 + 9 * np.clip(normalized, 0, None))
    elif stretch == 'arcsinh':
        normalized = np.arcsinh(3 * np.clip(normalized, 0, None)) / np.arcsinh(3)
    
    return np.clip(normalized, 0, 1)


def make_rgb(band_stack):
    """Create RGB from 4-band stack (F115W, F150W, F277W, F444W)."""
    if band_stack is None or len(band_stack) < 4:
        return None
    
    f115, f150, f277, f444 = band_stack[:4]
    r = normalize_band(f444)
    g = normalize_band(0.5 * (f150 + f277))
    b = normalize_band(f115)
    
    if r is None or g is None or b is None:
        return None
    
    return np.stack([r, g, b], axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Visualize sim vs obs lenses")
    parser.add_argument("--max-pairs", type=int, default=10, help="Max pairs to visualize")
    parser.add_argument("--stretch", choices=['log', 'arcsinh', 'linear'], default='log')
    args = parser.parse_args()
    
    workspace_root = Path("/Users/gozalig1/Projects/jwst-mock-lens-simulator")
    
    matched_path = workspace_root / "analysis" / "sim_obs_comparison" / "catalogs" / "matched_lens_pairs.csv"
    real_dir = workspace_root / "data" / "real_lenses"
    sim_dir = workspace_root / "outputs" / "custom_20260213_155632"
    output_dir = workspace_root / "analysis" / "sim_obs_comparison" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not matched_path.exists():
        print(f"Error: {matched_path} not found. Run match_sim_to_obs.py first.")
        return
    
    print("Loading matched lens pairs...")
    matches = []
    with open(matched_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('match_rank') == '1':  # Only use top match per real lens
                matches.append(row)
    
    print(f"Found {len(matches)} top matches. Visualizing up to {args.max_pairs}...\n")
    
    for idx, match in enumerate(matches[:args.max_pairs]):
        real_name = match['real_name']
        sim_id = match['sim_lens_id']
        sim_filename = match['sim_filename_base']
        
        print(f"[{idx+1}/{min(len(matches), args.max_pairs)}] {real_name} <-> {sim_filename}")
        
        # Load real and simulated data
        real_bands = None  # Try F150W as representative
        for band in ['F150W', 'F115W', 'F277W', 'F444W']:
            img = load_real_png(real_dir, real_name, band)
            if img is not None:
                real_bands = img
                break
        
        sim_bands = load_sim_bands(sim_dir, sim_filename)
        
        if real_bands is None or sim_bands is None:
            print(f"  Skipping: missing data")
            continue
        
        # Create figure: 2 rows (real, sim) × 5 cols (4 bands + RGB)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        # Real lens row
        # Note: We only have real data as single band. For full implementation,
        # would need to load all 4 FITS bands
        for c_idx in range(4):
            img = normalize_band(real_bands, stretch=args.stretch)
            axes[0, c_idx].imshow(img, cmap='gray')
            axes[0, c_idx].set_title(f'Real (?)', fontsize=10)
            axes[0, c_idx].axis('off')
        
        # Real RGB
        axes[0, 4].text(0.5, 0.5, 'Real\n(Need 4 bands)', 
                       ha='center', va='center',
                       transform=axes[0, 4].transAxes)
        axes[0, 4].axis('off')
        
        # Simulated lens row
        bands = ['F115W', 'F150W', 'F277W', 'F444W']
        for c_idx in range(4):
            img = normalize_band(sim_bands[c_idx], stretch=args.stretch)
            axes[1, c_idx].imshow(img, cmap='gray')
            axes[1, c_idx].set_title(f'Sim {bands[c_idx]}', fontsize=10)
            axes[1, c_idx].axis('off')
        
        # Simulated RGB
        sim_rgb = make_rgb(sim_bands)
        if sim_rgb is not None:
            axes[1, 4].imshow(sim_rgb)
        axes[1, 4].set_title('Sim RGB', fontsize=10)
        axes[1, 4].axis('off')
        
        # Row labels
        axes[0, 0].text(-0.3, 0.5, 'REAL', ha='right', va='center',
                       transform=axes[0, 0].transAxes, fontsize=12, weight='bold')
        axes[1, 0].text(-0.3, 0.5, f'SIM ID {sim_id}', ha='right', va='center',
                       transform=axes[1, 0].transAxes, fontsize=12, weight='bold')
        
        fig.suptitle(f'{real_name} vs {sim_filename}', fontsize=14, weight='bold')
        fig.tight_layout()
        
        output_file = output_dir / f"comparison_{real_name}_vs_id{sim_id}.png"
        fig.savefig(output_file, dpi=120, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close(fig)
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
