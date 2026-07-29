#!/usr/bin/env python3
import numpy as np
import os
import glob

def check_bands(output_dir, config_name):
    npz_files = glob.glob(os.path.join(output_dir, "**/PRISM*.npz"), recursive=True)
    
    if not npz_files:
        print(f"{config_name}: No NPZ files found")
        return False
    
    npz_file = npz_files[0]
    data = np.load(npz_file)
    
    print(f"{config_name}:")
    print(f"  File: {os.path.basename(npz_file)}")
    print(f"  Keys: {list(data.keys())}")
    
    if 'image_final' in data:
        shape = data['image_final'].shape
        print(f"  image_final shape: {shape}")
        print(f"  ✓ Filters: {shape[0]}")
        return True
    
    return False

# Check outputs
print("=== CONFIGURATION VERIFICATION ===\n")
check_bands("/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs/test_4band_default", 
            "4-BAND (DEFAULT)")

print()

check_bands("/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs/test_5band_config", 
            "5-BAND (THESIS)")
