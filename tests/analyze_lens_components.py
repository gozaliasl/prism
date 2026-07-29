#!/usr/bin/env python3
"""Analyze whether generated lenses are actually binary pairs."""

from PIL import Image
import numpy as np
from scipy import ndimage
from pathlib import Path

base_dir = Path("/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs/pair_treatments_20260215_134808")

print("=" * 80)
print("LENS COMPONENT ANALYSIS")
print("=" * 80)

for treatment in ['sie_sie', 'nfw_nfw', 'shear_only']:
    print(f"\n{treatment.upper()}:")
    print("-" * 40)
    
    jpg_dir = base_dir / treatment / "jpg_rgb"
    
    # Analyze first 5 images
    blob_counts = []
    for i in range(5):
        jpg_path = jpg_dir / f"cosmos_lens_{i:06d}.jpg"
        if not jpg_path.exists():
            continue
        
        img = Image.open(jpg_path)
        arr = np.array(img)
        
        # Convert to grayscale
        if len(arr.shape) == 3:
            gray = np.mean(arr, axis=2)
        else:
            gray = arr
        
        # Find bright regions
        threshold = np.percentile(gray, 93)
        bright = gray > threshold
        
        # Count connected components
        labeled, n_blobs = ndimage.label(bright)
        blob_counts.append(n_blobs)
        
        print(f"  Image {i:06d}: {n_blobs} bright blobs/components")
    
    if blob_counts:
        avg_blobs = np.mean(blob_counts)
        print(f"  Average: {avg_blobs:.1f} components per image")
        if avg_blobs < 1.5:
            print(f"  ⚠️  Single galaxy (NOT a pair lens)")
        elif avg_blobs < 2.5:
            print(f"  ✓ Appears to be binary/pair lens")
        else:
            print(f"  ? Multiple components detected")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("""
If you see ~1 component on average: Single galaxy lenses (NOT binary pairs)
If you see ~2 components on average: True binary pair lenses
If you see ~3+ components: Multiple galaxies (pair + field contamination)
""")
