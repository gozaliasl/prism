#!/usr/bin/env python3
"""
Extract properties of real observed lenses from FITS files and metadata.

Real lenses are in: /data/real_lenses/
Filename convention: COSJ[RA]+[DEC]_[BAND].fits (e.g., COSJ095846+020304_F115W.fits)

This script:
1. Discovers all unique lenses (by RA+DEC identifier)
2. Extracts FITS headers (photometry, astrometry if available)
3. Loads pixel data to compute basic statistics (min, max, mean, std)
4. Saves structured catalog to: catalogs/real_lens_properties.csv

Output columns:
- name (COSJ...)
- n_bands_available (how many of 4 bands have FITS files)
- f115w_available, f150w_available, f277w_available, f444w_available (boolean)
- f115w_sum, f150w_sum, f277w_sum, f444w_sum (total flux per band)
- f115w_max, f150w_max, f277w_max, f444w_max (peak brightness per band)
- f115w_rms, f150w_rms, f277w_rms, f444w_rms (background noise per band)
- notes (manual observations / challenges)
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from astropy.io import fits
except ImportError:
    print("Warning: astropy not available. Will skip FITS header extraction.")
    fits = None


def discover_real_lenses(real_lenses_dir):
    """
    Scan real_lenses directory for unique lens names.
    Returns dict: {name -> list of (band, filepath)}
    """
    real_lenses_dir = Path(real_lenses_dir)
    lenses = defaultdict(list)
    
    for fits_file in real_lenses_dir.glob("*_F115W.fits"):
        # Extract lens name (everything before _F115W)
        name = fits_file.stem.replace("_F115W", "")
        fits_path = real_lenses_dir / f"{name}_F115W.fits"
        if fits_path.exists():
            lenses[name].append(("F115W", fits_path))
    
    for band in ["F150W", "F277W", "F444W"]:
        for fits_file in real_lenses_dir.glob(f"*_{band}.fits"):
            name = fits_file.stem.replace(f"_{band}", "")
            fits_path = real_lenses_dir / f"{name}_{band}.fits"
            if fits_path.exists() and name in lenses:
                lenses[name].append((band, fits_path))
    
    return lenses


def extract_fits_statistics(fits_path):
    """Extract basic statistics from a FITS image."""
    try:
        if fits is None:
            return None
        
        with fits.open(fits_path) as hdul:
            img = hdul[0].data
            if img is None:
                return None
            
            # Compute statistics (excluding NaNs)
            img_clean = img[~np.isnan(img)]
            return {
                'sum': float(np.sum(img_clean)),
                'max': float(np.max(img_clean)),
                'min': float(np.min(img_clean)),
                'mean': float(np.mean(img_clean)),
                'std': float(np.std(img_clean)),
            }
    except Exception as e:
        print(f"  Error reading {fits_path}: {e}")
        return None


def main():
    workspace_root = Path("/Users/gozalig1/Projects/jwst-mock-lens-simulator")
    real_lenses_dir = workspace_root / "data" / "real_lenses"
    output_file = workspace_root / "analysis" / "sim_obs_comparison" / "catalogs" / "real_lens_properties.csv"
    
    print(f"Scanning real lenses in: {real_lenses_dir}")
    lenses = discover_real_lenses(real_lenses_dir)
    print(f"Found {len(lenses)} unique lenses with available bands\n")
    
    # Prepare output
    rows = []
    
    for lens_idx, (name, band_paths) in enumerate(sorted(lenses.items())):
        print(f"[{lens_idx+1}/{len(lenses)}] Processing {name}...")
        
        row = {'name': name}
        n_bands = 0
        
        # Collect all bands
        for band in ["F115W", "F150W", "F277W", "F444W"]:
            band_file = real_lenses_dir / f"{name}_{band}.fits"
            has_band = band_file.exists()
            row[f'{band.lower()}_available'] = 'Yes' if has_band else 'No'
            
            if has_band:
                n_bands += 1
                stats = extract_fits_statistics(band_file)
                if stats:
                    row[f'{band.lower()}_sum'] = round(stats['sum'], 2)
                    row[f'{band.lower()}_max'] = round(stats['max'], 4)
                    row[f'{band.lower()}_mean'] = round(stats['mean'], 6)
                    row[f'{band.lower()}_std'] = round(stats['std'], 6)
                else:
                    row[f'{band.lower()}_sum'] = 'N/A'
                    row[f'{band.lower()}_max'] = 'N/A'
                    row[f'{band.lower()}_mean'] = 'N/A'
                    row[f'{band.lower()}_std'] = 'N/A'
        
        row['n_bands_available'] = n_bands
        row['notes'] = ""  # For manual annotations
        rows.append(row)
    
    # Write CSV
    if rows:
        fieldnames = [
            'name', 'n_bands_available',
            'f115w_available', 'f115w_sum', 'f115w_max', 'f115w_mean', 'f115w_std',
            'f150w_available', 'f150w_sum', 'f150w_max', 'f150w_mean', 'f150w_std',
            'f277w_available', 'f277w_sum', 'f277w_max', 'f277w_mean', 'f277w_std',
            'f444w_available', 'f444w_sum', 'f444w_max', 'f444w_mean', 'f444w_std',
            'notes'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nSaved to: {output_file}")
        print(f"Total real lenses cataloged: {len(rows)}")
    else:
        print("No lenses found!")


if __name__ == "__main__":
    main()
