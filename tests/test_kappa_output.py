#!/usr/bin/env python3
"""
Test script for PRISM kappa output module.

Runs a single SIE+SHEAR system end-to-end and verifies all output files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel

from src.prism_kappa_output import compute_kappa_products, save_kappa_outputs


def test_sie_shear_kappa_output():
    """Test kappa computation and output for a simple SIE+SHEAR system."""
    
    print("=" * 70)
    print("PRISM Kappa Output Test: SIE + SHEAR")
    print("=" * 70)
    
    # Create a simple SIE + SHEAR lens model
    lens_model = LensModel(lens_model_list=["SIE", "SHEAR"])
    
    # SIE parameters: Einstein radius 1.2 arcsec, ellipticity 0.3
    # SHEAR parameters: gamma1=0.05, gamma2=0.02
    kwargs_lens = [
        {"theta_E": 1.2, "e1": 0.15, "e2": 0.1, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": 0.05, "gamma2": 0.02},
    ]
    
    # Compute kappa products
    print("\n[1] Computing kappa products (300×300, 0.031\"/pixel)...")
    kappa_dict = compute_kappa_products(
        lens_model,
        kwargs_lens,
        num_pix=300,
        delta_pix=0.031,
    )
    
    print(f"  ✓ Kappa shape: {kappa_dict['kappa'].shape}")
    print(f"  ✓ Kappa range: [{kappa_dict['kappa'].min():.4f}, {kappa_dict['kappa'].max():.4f}]")
    print(f"  ✓ theta_E_eff: {kappa_dict['theta_E_eff']:.3f} arcsec")
    print(f"  ✓ critical_area: {kappa_dict['critical_area']*100:.2f}%")
    print(f"  ✓ mu_max: {kappa_dict['mu_max']:.2f}")
    
    # Save outputs to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[2] Saving outputs to {tmpdir}...")
        success = save_kappa_outputs(
            kappa_dict,
            out_dir=tmpdir,
            lens_id="test_sie_shear_001",
            category="single",
            sub_type="SIE+SHEAR",
        )
        
        if not success:
            print("  ✗ FAILED: save_kappa_outputs returned False")
            return False
        
        # Verify all output files exist
        print("\n[3] Verifying output files...")
        expected_files = [
            "test_sie_shear_001_kappa.npy",
            "test_sie_shear_001_kappa_data.npz",
            "test_sie_shear_001_kappa.jpg",
            "test_sie_shear_001_kappa_panel.jpg",
        ]
        
        all_exist = True
        for fname in expected_files:
            fpath = Path(tmpdir) / fname
            exists = fpath.exists()
            size_mb = fpath.stat().st_size / (1024**2) if exists else 0
            status = "✓" if exists else "✗"
            print(f"  {status} {fname:40s} ({size_mb:.2f} MB)")
            all_exist = all_exist and exists
        
        if not all_exist:
            print("\n  FAILED: Not all files were created")
            return False
        
        # Verify NPZ contents
        print("\n[4] Verifying NPZ contents...")
        npz = np.load(Path(tmpdir) / "test_sie_shear_001_kappa_data.npz")
        required_keys = ["kappa", "gamma_mag", "mu", "lens_id", "category", "sub_type"]
        for key in required_keys:
            if key in npz:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ {key} MISSING")
                all_exist = False
        
        # Verify NPY shape
        print("\n[5] Verifying NPY contents...")
        kappa_npy = np.load(Path(tmpdir) / "test_sie_shear_001_kappa.npy")
        print(f"  ✓ Shape: {kappa_npy.shape}")
        print(f"  ✓ Dtype: {kappa_npy.dtype}")
        
        if kappa_npy.shape != (300, 300):
            print(f"  ✗ Unexpected shape: {kappa_npy.shape}")
            return False
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_sie_shear_kappa_output()
    raise SystemExit(0 if success else 1)
