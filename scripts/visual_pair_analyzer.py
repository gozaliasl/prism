#!/usr/bin/env python3
"""
Visual analyzer for binary pair lenses using intermediate images.
Helps distinguish SIE+SIE (smooth) vs NFW+NFW (complex) patterns.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

def analyze_lensing_patterns(output_dir, num_samples=10):
    """
    Analyze intermediate lens_sources images to identify lensing patterns.
    """
    output_dir = Path(output_dir)
    jpg_dir = output_dir / "jpg_rgb" / "intermediate_lens_sources"
    
    if not jpg_dir.exists():
        print("[ERROR] intermediate_lens_sources directory not found")
        return
    
    images = sorted(list(jpg_dir.glob("cosmos_lens_*.jpg")))
    
    print("\n" + "=" * 90)
    print("VISUAL PATTERN ANALYZER - Intermediate Lens+Sources Images")
    print("=" * 90)
    print(f"\n[✓] Found {len(images)} intermediate lens+source images")
    print(f"[✓] Each image shows: Lens galaxy + Lensed background source arcs\n")
    
    print("INTERPRETING ARC PATTERNS:")
    print("-" * 90)
    print("""
    SINGLE LENS (65% expected):
      ✓ ONE clear arc system or Einstein ring
      ✓ Symmetric distortion
      ✓ Clean, smooth arc shape
      ✓ Regular curvature
      
    SIE+SIE BINARY PAIR (~17.5% expected):
      ✓ TWO smooth lens galaxies visible
      ✓ Arcs show smooth, overlapping distortion from both lenses
      ✓ Regular, symmetric arc patterns from each lens component
      ✓ Smoother overall appearance
      ✓ Arcs may form multiple clean segments
      
    NFW+NFW BINARY PAIR (~17.5% expected):
      ✓ TWO lens galaxies visible (often appearing clumpier)
      ✓ Complex, multi-segment arc patterns
      ✓ Irregular, asymmetric lensing features
      ✓ May show "clumpiness" or multiple arc segments
      ✓ Cusped density profiles create distinctive features
      ✓ Less smooth compared to SIE+SIE
    """)
    
    print("\n" + "=" * 90)
    print("RECOMMENDED EXAMINATION WORKFLOW:")
    print("=" * 90)
    print(f"""
    1. Open 4 intermediate images side-by-side:
       open {jpg_dir}/cosmos_lens_000000.jpg \\
       open {jpg_dir}/cosmos_lens_000001.jpg \\
       open {jpg_dir}/cosmos_lens_000010.jpg \\
       open {jpg_dir}/cosmos_lens_000050.jpg
       
    2. Compare characteristics:
       - Smoothness of arcs
       - Number of arc segments
       - Symmetry/Asymmetry
       - Presence of multiple lens cores (indicates pair)
       
    3. Create categories:
       - Type A: Single smooth lenses
       - Type B: Binary pairs with smooth arcs (SIE+SIE)
       - Type C: Binary pairs with complex arcs (NFW+NFW)
       
    4. Verify distribution matches config (35% pairs, split 50/50)
    """)
    
    # Analyze image statistics
    print("\n" + "=" * 90)
    print("IMAGE STATISTICS (for all intermediate_lens_sources):")
    print("=" * 90 + "\n")
    
    image_stats = []
    
    for i, img_path in enumerate(images[:num_samples]):
        try:
            img = Image.open(img_path)
            arr = np.array(img)
            
            # Basic statistics
            brightness = np.mean(arr)
            contrast = np.std(arr)
            peak_brightness = np.max(arr)
            
            image_stats.append({
                'id': i,
                'brightness': brightness,
                'contrast': contrast,
                'peak': peak_brightness
            })
            
            if i < 5:
                print(f"Lens {i:3d}: Brightness={brightness:6.1f}, Contrast={contrast:6.1f}, Peak={peak_brightness:3d}")
        except Exception as e:
            print(f"Error analyzing lens {i}: {e}")
    
    # Show summary statistics
    if image_stats:
        brights = [s['brightness'] for s in image_stats]
        contrasts = [s['contrast'] for s in image_stats]
        
        print(f"\n[SUMMARY] First {num_samples} images:")
        print(f"  Average brightness: {np.mean(brights):.1f} ± {np.std(brights):.1f}")
        print(f"  Average contrast: {np.mean(contrasts):.1f} ± {np.std(contrasts):.1f}")
        
        print(f"\n[NOTE] High contrast/brightness variance might indicate pair lenses")
        print(f"       (due to multiple lens centers creating complex patterns)")
    
    print("\n" + "=" * 90)
    print("SUMMARY OF YOUR DATA")
    print("=" * 90)
    print(f"""
    Total lenses in output: {len(images)}
    
    EXPECTED DISTRIBUTION (from config):
      - Single lenses: ~65% ({int(len(images)*0.65)} lenses)
      - Binary pairs: ~35% ({int(len(images)*0.35)} lenses)
        * SIE+SIE: ~50% of pairs ({int(len(images)*0.35*0.5)} lenses)
        * NFW+NFW: ~50% of pairs ({int(len(images)*0.35*0.5)} lenses)
    
    STATUS: ⚠️  Binary pairs not generated in catalog (all have unique base_lens_id)
            But check if the lensing patterns suggest pairs were created anyway!
    """)
    
    print("\n" + "=" * 90)
    print("QUICK VISUAL CHECK")
    print("=" * 90)
    print("""
    One-liner to open 10 random intermediate images to compare:
    
    for i in $(seq 0 5 50); do open '{}/cosmos_lens_00000$i.jpg' 2>/dev/null || true; done
    
    Look for:
      1. Images with 1 lens galaxy = SINGLE LENS
      2. Images with 2 lens galaxies + smooth arcs = SIE+SIE
      3. Images with 2 lens galaxies + complex arcs = NFW+NFW
    """.format(jpg_dir))
    
    print("\n" + "=" * 90 + "\n")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs/custom_20260215_155541'
    
    analyze_lensing_patterns(output_dir)
