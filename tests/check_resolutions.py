#!/usr/bin/env python3
"""Check if multi-resolution images are actually different"""

from PIL import Image
import numpy as np
import os

res_dirs = {
    'resolution_jwst': 0.031,
    'resolution_euclid': 0.100, 
    'resolution_roman': 0.110,
    'resolution_ground_based': 0.200
}
base_path = 'outputs/test_rgb_fix'

print('Checking if images differ between resolutions:')
print('='*70)

images = {}
for res_dir, pixel_scale in res_dirs.items():
    jpg_dir = os.path.join(base_path, res_dir, 'jpg_rgb')
    if os.path.exists(jpg_dir):
        jpgs = [f for f in os.listdir(jpg_dir) if f.endswith('.jpg')]
        if jpgs:
            img_path = os.path.join(jpg_dir, jpgs[0])
            img = Image.open(img_path)
            arr = np.array(img)
            images[res_dir] = arr
            
            # Calculate statistics
            mean = arr.mean()
            std = arr.std()
            nonzero = np.count_nonzero(arr) / arr.size * 100
            
            field_size = 300 * pixel_scale
            print(f'{res_dir:25s} ({pixel_scale:.3f}"/pix, {field_size:.1f}" field)')
            print(f'  Mean: {mean:.2f}, Std: {std:.2f}, Non-zero: {nonzero:.1f}%')

# Compare images
if len(images) > 1:
    print('\nImage Differences:')
    print('-'*70)
    keys = list(images.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            diff = np.abs(images[keys[i]].astype(float) - images[keys[j]].astype(float))
            mean_diff = diff.mean()
            max_diff = diff.max()
            pct_different = (diff > 1).sum() / diff.size * 100
            print(f'{keys[i]} vs {keys[j]}:')
            print(f'  Mean diff: {mean_diff:.2f}, Max diff: {max_diff:.0f}, {pct_different:.1f}% pixels differ')
