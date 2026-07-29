#!/usr/bin/env python3
"""
Test CSV Environment Training

Quick test to verify the CSV-based environment training works.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

def test_csv_data_loading():
    """Test loading the CSV data"""
    print("=== Testing CSV Data Loading ===")
    
    data_dir = Path("../data")
    
    # Test loading merged catalog
    print("Loading merged catalog...")
    try:
        merged_df = pd.read_csv(data_dir / "merged_lens_field_catalog.csv")
        print(f"Loaded {len(merged_df)} galaxies from merged catalog")
        print(f"Columns: {len(merged_df.columns)}")
        
        # Check for required columns
        required_cols = ['LP_mass_med_PDF', 'LP_zfinal', 'LP_warn_fl', 'RA_DETEC', 'DEC_DETEC']
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return False
        else:
            print("All required columns found!")
            
            # Test filtering for massive galaxies
            print("\nTesting massive galaxy filtering...")
            massive_mask = (
                (merged_df['LP_mass_med_PDF'] >= 10.0) &
                (merged_df['LP_zfinal'] > 0) &
                (merged_df['LP_zfinal'] < 10) &
                (merged_df['LP_warn_fl'] == 0) &
                (merged_df['LP_mass_med_PDF'] > 0)
            )
            
            massive_galaxies = merged_df[massive_mask]
            print(f"Found {len(massive_galaxies)} massive galaxies")
            print(f"Mass range: {massive_galaxies['LP_mass_med_PDF'].min():.2f} - {massive_galaxies['LP_mass_med_PDF'].max():.2f}")
            print(f"Redshift range: {massive_galaxies['LP_zfinal'].min():.2f} - {massive_galaxies['LP_zfinal'].max():.2f}")
            
            return True
            
    except Exception as e:
        print(f"Error loading merged catalog: {e}")
        return False

def test_lens_catalog():
    """Test loading lens catalog"""
    print("\n=== Testing Lens Catalog ===")
    
    data_dir = Path("../data")
    
    try:
        lens_df = pd.read_csv(data_dir / "cosmos_web_lens_structural_properties.csv")
        print(f"Loaded {len(lens_df)} lens systems")
        print(f"Sample lens IDs: {lens_df['ASSOC_ID'].head().tolist()}")
        return True
    except Exception as e:
        print(f"Error loading lens catalog: {e}")
        return False

def test_environment_analysis():
    """Test a small environment analysis"""
    print("\n=== Testing Environment Analysis ===")
    
    data_dir = Path("../data")
    
    try:
        # Load data
        merged_df = pd.read_csv(data_dir / "merged_lens_field_catalog.csv")
        lens_df = pd.read_csv(data_dir / "cosmos_web_lens_structural_properties.csv")
        lens_ids = set(lens_df['ASSOC_ID'].values)
        
        # Filter massive galaxies
        massive_mask = (
            (merged_df['LP_mass_med_PDF'] >= 10.0) &
            (merged_df['LP_zfinal'] > 0) &
            (merged_df['LP_zfinal'] < 10) &
            (merged_df['LP_warn_fl'] == 0) &
            (merged_df['LP_mass_med_PDF'] > 0)
        )
        
        massive_galaxies = merged_df[massive_mask].copy()
        print(f"Found {len(massive_galaxies)} massive galaxies")
        
        # Test environment analysis for a few galaxies
        sample_galaxies = massive_galaxies.head(5)
        print(f"Testing environment analysis for {len(sample_galaxies)} galaxies...")
        
        for idx, (_, central_gal) in enumerate(sample_galaxies.iterrows()):
            central_ra = central_gal['RA_DETEC']
            central_dec = central_gal['DEC_DETEC']
            central_z = central_gal['LP_zfinal']
            central_mass = central_gal['LP_mass_med_PDF']
            
            # Find surrounding galaxies within 5 arcsec
            ra_diff = merged_df['RA_DETEC'] - central_ra
            dec_diff = merged_df['DEC_DETEC'] - central_dec
            separation_arcsec = np.sqrt(ra_diff**2 + dec_diff**2) * 3600
            
            nearby_mask = (separation_arcsec <= 5.0) & (separation_arcsec > 0)
            nearby_galaxies = merged_df[nearby_mask]
            
            n_nearby = len(nearby_galaxies)
            if n_nearby <= 2:
                env_type = 'isolated_field'
            elif n_nearby <= 5:
                env_type = 'galaxy_pair'
            else:
                env_type = 'group'
            
            print(f"  Galaxy {idx+1}: z={central_z:.2f}, log10(M)={central_mass:.2f}")
            print(f"    Environment: {env_type}, {n_nearby} nearby galaxies")
        
        return True
        
    except Exception as e:
        print(f"Error in environment analysis: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing CSV Environment Training Setup")
    print("=" * 50)
    
    # Test data loading
    csv_ok = test_csv_data_loading()
    lens_ok = test_lens_catalog()
    env_ok = test_environment_analysis()
    
    print("\n=== Test Results ===")
    print(f"CSV catalog: {'✓' if csv_ok else '✗'}")
    print(f"Lens catalog: {'✓' if lens_ok else '✗'}")
    print(f"Environment analysis: {'✓' if env_ok else '✗'}")
    
    if csv_ok and lens_ok and env_ok:
        print("\n✓ All tests passed!")
        print("Ready to run CSV-based environment training.")
        print("\nTo train models, run:")
        print("python train_environment_models_csv.py")
    else:
        print("\n✗ Some tests failed.")
        print("Please check the data files and try again.")

if __name__ == "__main__":
    main()
