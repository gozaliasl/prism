#!/usr/bin/env python3
"""
Utility script to load and inspect unified .npz storage files

Usage:
    python load_unified_sample.py <npz_file>
    python load_unified_sample.py <npz_file> --show-all
    python load_unified_sample.py <npz_file> --extract-intermediate lens_sources
"""

import numpy as np
import argparse
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def load_unified_sample(npz_path):
    """Load a unified .npz sample file
    
    Returns:
        dict: Contains image arrays and metadata
    """
    with np.load(npz_path, allow_pickle=True) as data:
        sample = {}
        
        # Load all arrays
        for key in data.files:
            if key == 'metadata':
                # Parse JSON metadata
                sample[key] = json.loads(str(data[key]))
            else:
                sample[key] = data[key]
        
        return sample

def display_sample(sample, show_intermediate=False):
    """Display sample contents"""
    print("=" * 60)
    print("UNIFIED SAMPLE CONTENTS")
    print("=" * 60)
    
    # Metadata
    if 'metadata' in sample:
        print("\nMetadata:")
        for key, val in sample['metadata'].items():
            print(f"  {key}: {val}")
    
    # Image components
    print("\nImage Components:")
    for key in sorted(sample.keys()):
        if key.startswith('image_'):
            arr = sample[key]
            print(f"  {key}:")
            print(f"    Shape: {arr.shape}")
            print(f"    Dtype: {arr.dtype}")
            print(f"    Range: [{arr.min():.3e}, {arr.max():.3e}]")
            print(f"    Size: {arr.nbytes / (1024**2):.2f} MB")
    
    # RGB visualization
    if 'rgb_visualization' in sample:
        rgb = sample['rgb_visualization']
        print(f"\n  RGB Visualization:")
        print(f"    Shape: {rgb.shape}")
        print(f"    Dtype: {rgb.dtype}")
        print(f"    Size: {rgb.nbytes / (1024**2):.2f} MB")
    
    # Show intermediate images if requested
    if show_intermediate:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        intermediate_keys = [k for k in sample.keys() if k.startswith('image_') and k != 'image_final']
        
        # Show final image
        if 'image_final' in sample:
            img_final = sample['image_final']
            # Create RGB from bands (F444W, F277W+F150W, F115W)
            R = img_final[3]  # F444W
            G = (img_final[1] + img_final[2]) / 2  # F150W + F277W
            B = img_final[0]  # F115W
            
            rgb_final = np.stack([R, G, B], axis=-1)
            rgb_final = np.clip(rgb_final / np.percentile(rgb_final, 99.5), 0, 1)
            
            axes[0].imshow(rgb_final)
            axes[0].set_title('Final Composite')
            axes[0].axis('off')
        
        # Show intermediate images
        for idx, key in enumerate(intermediate_keys[:5], start=1):
            img = sample[key]
            R = img[3]
            G = (img[1] + img[2]) / 2
            B = img[0]
            
            rgb = np.stack([R, G, B], axis=-1)
            rgb = np.clip(rgb / np.percentile(rgb, 99.5), 0, 1)
            
            axes[idx].imshow(rgb)
            axes[idx].set_title(key.replace('image_', '').replace('_', ' ').title())
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(intermediate_keys) + 1, 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

def extract_intermediate(sample, component_name, output_path=None):
    """Extract a specific intermediate image component
    
    Args:
        sample: Loaded sample dict
        component_name: e.g., 'lens_sources', 'sources_only'
        output_path: Optional path to save extracted .npy file
    """
    key = f'image_{component_name}'
    
    if key not in sample:
        print(f"Error: Component '{component_name}' not found")
        print(f"Available: {[k.replace('image_', '') for k in sample.keys() if k.startswith('image_')]}")
        return None
    
    arr = sample[key]
    print(f"Extracted {key}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    
    if output_path:
        np.save(output_path, arr)
        print(f"Saved to: {output_path}")
    
    return arr

def compare_storage_efficiency(npz_path):
    """Compare unified vs legacy storage size"""
    npz_size = Path(npz_path).stat().st_size
    
    # Load to count components
    sample = load_unified_sample(npz_path)
    n_components = sum(1 for k in sample.keys() if k.startswith('image_'))
    
    # Estimate legacy size (uncompressed .npy files)
    legacy_size = 0
    for key in sample.keys():
        if key.startswith('image_'):
            arr = sample[key]
            # .npy has small header overhead (~128 bytes)
            legacy_size += arr.nbytes + 128
    
    print("\n" + "=" * 60)
    print("STORAGE EFFICIENCY COMPARISON")
    print("=" * 60)
    print(f"\nUnified .npz:     {npz_size / (1024**2):.2f} MB")
    print(f"Legacy .npy est:  {legacy_size / (1024**2):.2f} MB")
    print(f"Compression:      {100 * (1 - npz_size / legacy_size):.1f}% savings")
    print(f"File count:       1 file vs {n_components} files")
    print(f"Reduction:        {n_components}x fewer files")

def main():
    parser = argparse.ArgumentParser(description='Load and inspect unified .npz sample files')
    parser.add_argument('npz_file', help='Path to .npz file')
    parser.add_argument('--show-all', action='store_true', help='Display all components')
    parser.add_argument('--extract-intermediate', help='Extract specific intermediate component')
    parser.add_argument('--output', help='Output path for extracted component')
    parser.add_argument('--compare-efficiency', action='store_true', help='Compare storage efficiency')
    
    args = parser.parse_args()
    
    if not Path(args.npz_file).exists():
        print(f"Error: File not found: {args.npz_file}")
        return 1
    
    # Load sample
    sample = load_unified_sample(args.npz_file)
    
    # Display contents
    display_sample(sample, show_intermediate=args.show_all)
    
    # Extract intermediate if requested
    if args.extract_intermediate:
        extract_intermediate(sample, args.extract_intermediate, args.output)
    
    # Compare efficiency if requested
    if args.compare_efficiency:
        compare_storage_efficiency(args.npz_file)
    
    print("\n" + "=" * 60)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
