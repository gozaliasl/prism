#!/usr/bin/env python3
"""Better analysis: Look for major galaxy peaks instead of noise."""

from PIL import Image
import numpy as np
from scipy import ndimage
from pathlib import Path

base_dir = Path("/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs/pair_treatments_20260215_134808")

print("=" * 80)
print("LENS MORPHOLOGY ANALYSIS - Galaxy Peaks Detection")
print("=" * 80)

for treatment in ['sie_sie', 'nfw_nfw', 'shear_only']:
    print(f"\n{treatment.upper()}:")
    print("-" * 60)
    
    jpg_dir = base_dir / treatment / "jpg_rgb"
    
    galaxy_counts = []
    
    for i in range(5):
        jpg_path = jpg_dir / f"cosmos_lens_{i:06d}.jpg"
        if not jpg_path.exists():
            continue
        
        img = Image.open(jpg_path)
        arr = np.array(img, dtype=np.float32)
        
        # Convert to grayscale
        if len(arr.shape) == 3:
            gray = np.mean(arr, axis=2)
        else:
            gray = arr
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
        
        # Apply morphological closing to connect nearby bright regions
        from scipy.ndimage import binary_closing, binary_opening
        
        # Use higher threshold to focus on galaxy-scale features (not noise)
        threshold = 0.5  # Strong peaks only
        strong_features = gray > threshold
        
        # Apply morphological closing to merge nearby peaks
        closed = binary_closing(strong_features, structure=np.ones((5, 5)))
        
        # Label connected components
        labeled, n_galaxies = ndimage.label(closed)
        
        # Filter by size - real galaxies should have some area
        galaxy_sizes = np.bincount(labeled.flatten())
        major_galaxies = np.sum(galaxy_sizes[1:] > 10)  # Ignore tiny speckles
        
        galaxy_counts.append(major_galaxies)
        
        print(f"  Image {i:06d}: {major_galaxies} major galaxy peaks (threshold=0.5)")
    
    if galaxy_counts:
        avg = np.mean(galaxy_counts)
        print(f"\n  Average: {avg:.1f} galaxy peaks per image")
        
        if avg < 1.5:
            status = "⚠️  SINGLE GALAXY (NOT BINARY)"
        elif 1.5 <= avg < 2.5:
            status = "✓ BINARY PAIR (as expected)"
        else:
            status = "? MULTIPLE GALAXIES + FIELD CONTAMINATION"
        
        print(f"  Status: {status}")

print("\n" + "=" * 80)
print("KEY FINDING:")
print("=" * 80)
print("""
The analysis checks if lenses show 1 or 2 major galaxy peaks:
  - SIE+SIE should show: 2 peaks (pair of SIE galaxies)
  - NFW+NFW should show: 2 peaks (pair of NFW halos)
  - Shear-only should show: 1 peak (single galaxy + shear)

Actual results above show whether binary pair generation was successful.
""")
