#!/usr/bin/env python3
"""Verify that identical lenses were generated at each resolution."""

import numpy as np
import pandas as pd
from pathlib import Path

output_dir = Path("outputs/test_identical_fix_20260220")

print("=" * 70)
print("VERIFICATION: Identical Lenses Across Resolutions")
print("=" * 70)

# Check main catalog
print("\n1. Main training catalog:")
catalog_file = output_dir / "cosmos_training_catalog_lens_and_nonlens.csv"
if catalog_file.exists():
    df = pd.read_csv(catalog_file)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)[:8]}")
    print(f"\n   First 10 rows (lens_id, filename_base, theta_E, lens_redshift):")
    cols_to_show = ['lens_id', 'filename_base', 'theta_E', 'lens_redshift']
    print(df[cols_to_show].head(10).to_string(index=False))
    
    # Check if same lenses appear multiple times (once per resolution)
    print(f"\n   Lens ID value counts:")
    print(df['lens_id'].value_counts().sort_index())

# Check NPZ files
print("\n2. NPZ file contents by resolution:")
for res in ["resolution_ground_based", "resolution_roman", "resolution_euclid", "resolution_jwst"]:
    npz_dir = output_dir / res / "unified_npz"
    npz_files = sorted(list(npz_dir.glob("*.npz")))
    if npz_files:
        print(f"\n   {res}:")
        for npz_file in npz_files[:2]:
            data = np.load(npz_file, allow_pickle=True)
            keys = list(data.keys())
            print(f"     {npz_file.name}: {len(keys)} keys")
            if 'deltapix' in data:
                print(f"       deltapix (pixel_scale): {float(data['deltapix']):.4f}")
            if 'lens_system_class' in data:
                print(f"       lens_system_class: {str(data['lens_system_class'])}")

# Check kappa maps
print("\n3. Kappa maps (convergence) by resolution:")
for res in ["resolution_ground_based", "resolution_roman", "resolution_euclid", "resolution_jwst"]:
    kappa_dir = output_dir / res / "kappa_maps"
    kappa_files = sorted(list(kappa_dir.glob("*_kappa_convergence.npy")))
    if kappa_files:
        print(f"\n   {res}:")
        for kappa_file in kappa_files[:2]:
            kappa = np.load(kappa_file)
            print(f"     {kappa_file.name}: shape={kappa.shape}, max={kappa.max():.4f}, min={kappa.min():.4f}")

# Verify identical lens parameters across resolutions
print("\n4. Verifying identical lens parameters across resolutions:")
# Load NPZ files for same lens ID at each resolution
for lens_id in [0, 1]:  # Check first 2 lenses
    print(f"\n   Lens ID {lens_id}:")
    lens_params = {}
    for res in ["resolution_ground_based", "roman", "euclid", "jwst"]:
        npz_file = output_dir / f"resolution_{res}" / "unified_npz" / f"PRISM_lens_*_{lens_id:06d}.npz"
        # Use glob to find the file (name might vary)
        npz_files = list((output_dir / f"resolution_{res}" / "unified_npz").glob(f"*_00000{lens_id}.npz"))
        if npz_files:
            npz_file = npz_files[0]
            data = np.load(npz_file, allow_pickle=True)
            if 'deltapix' in data:
                pixel_scale = float(data['deltapix'])
                lens_params[res] = {'pixel_scale': pixel_scale, 'shape': data['image'].shape}
                print(f"     {res:15s}: pixel_scale={pixel_scale:.4f}\"/pix, image_shape={data['image'].shape}")

print("\n" + "=" * 70)
print("Summary: If all lenses show DIFFERENT pixel_scales but SAME image_shapes,")
print("then the identical lens mechanism is working correctly!")
print("=" * 70)
