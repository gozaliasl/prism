#!/usr/bin/env python3
"""
Prepare Training Data for ML Arc Detection

Creates training dataset from simulation outputs by:
1. Loading images with known lensed image positions
2. Extracting patches around lensed images (positive samples)
3. Extracting patches from random locations (negative samples)
4. Saving as CSV for training
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def extract_patches_from_image(
    image_path: Path,
    positions: list,
    patch_size: int = 64,
    num_negative: int = 10
) -> tuple:
    """
    Extract positive (arc) and negative (non-arc) patches from image.
    
    Args:
        image_path: Path to image
        positions: List of (x, y) positions of lensed images
        patch_size: Size of patches to extract
        num_negative: Number of negative samples to extract
    
    Returns:
        Tuple of (patches, labels) where labels are 1 for arcs, 0 for non-arcs
    """
    # Load image
    img = Image.open(image_path)
    if img.mode == 'RGB':
        img_array = np.array(img.convert('L')) / 255.0
    else:
        img_array = np.array(img) / 255.0
    
    h, w = img_array.shape
    half_size = patch_size // 2
    patches = []
    labels = []
    
    # Positive samples (around lensed images)
    for x, y in positions:
        x, y = int(x), int(y)
        y_min = max(0, y - half_size)
        y_max = min(h, y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(w, x + half_size)
        
        patch = img_array[y_min:y_max, x_min:x_max]
        
        # Pad if necessary
        if patch.shape != (patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size))
            pad_y = (patch_size - patch.shape[0]) // 2
            pad_x = (patch_size - patch.shape[1]) // 2
            padded[pad_y:pad_y+patch.shape[0], pad_x:pad_x+patch.shape[1]] = patch
            patch = padded
        
        patches.append(patch)
        labels.append(1)  # Arc
    
    # Negative samples (random locations, avoiding lensed images)
    np.random.seed(42)
    negative_count = 0
    max_attempts = num_negative * 10
    
    for attempt in range(max_attempts):
        if negative_count >= num_negative:
            break
        
        # Random position
        x = np.random.randint(half_size, w - half_size)
        y = np.random.randint(half_size, h - half_size)
        
        # Check if too close to any lensed image
        too_close = False
        for lx, ly in positions:
            dist = np.sqrt((x - lx)**2 + (y - ly)**2)
            if dist < patch_size:  # Too close
                too_close = True
                break
        
        if too_close:
            continue
        
        # Extract patch
        y_min = max(0, y - half_size)
        y_max = min(h, y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(w, x + half_size)
        
        patch = img_array[y_min:y_max, x_min:x_max]
        
        # Pad if necessary
        if patch.shape != (patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size))
            pad_y = (patch_size - patch.shape[0]) // 2
            pad_x = (patch_size - patch.shape[1]) // 2
            padded[pad_y:pad_y+patch.shape[0], pad_x:pad_x+patch.shape[1]] = patch
            patch = padded
        
        patches.append(patch)
        labels.append(0)  # Non-arc
        negative_count += 1
    
    return np.array(patches), np.array(labels)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for ML arc detection"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Simulation output directory'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='training_data.csv',
        help='Output CSV file for training data'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=64,
        help='Size of patches to extract'
    )
    parser.add_argument(
        '--num-negative',
        type=int,
        default=10,
        help='Number of negative samples per image'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    # Read time delay catalog for positions
    catalog_path = output_dir / 'time_delay_catalog.csv'
    if not catalog_path.exists():
        print(f"❌ Error: Time delay catalog not found: {catalog_path}")
        sys.exit(1)
    
    catalog = pd.read_csv(catalog_path)
    
    # Image directory
    jpg_dir = output_dir / 'jpg_rgb'
    if not jpg_dir.exists():
        print(f"❌ Error: RGB images directory not found: {jpg_dir}")
        sys.exit(1)
    
    # Prepare training data
    all_patches = []
    all_labels = []
    all_metadata = []
    
    print("📊 Preparing training data from simulation outputs...")
    
    for idx, row in catalog.iterrows():
        lens_id = int(row['lens_id'])
        
        # Try to find image (epoch 0)
        image_file = jpg_dir / f"cosmos_lens_{lens_id:06d}_epoch00.jpg"
        if not image_file.exists():
            continue
        
        # Get image positions from catalog
        image_positions_str = row.get('image_positions_arcsec', '[]')
        if pd.isna(image_positions_str) or image_positions_str == '[]':
            continue
        
        # Parse positions (format: [(x1, y1), (x2, y2), ...])
        try:
            import ast
            positions_arcsec = ast.literal_eval(image_positions_str)
            if not positions_arcsec:
                continue
        except:
            continue
        
        # Convert arcsec to pixels
        pixel_scale = 0.03
        numpix = 300
        center_pix = numpix // 2
        positions_pix = []
        for x_arcsec, y_arcsec in positions_arcsec:
            x_pix = center_pix + x_arcsec / pixel_scale
            y_pix = center_pix + y_arcsec / pixel_scale
            positions_pix.append((x_pix, y_pix))
        
        # Extract patches
        patches, labels = extract_patches_from_image(
            image_file,
            positions_pix,
            patch_size=args.patch_size,
            num_negative=args.num_negative
        )
        
        all_patches.append(patches)
        all_labels.append(labels)
        
        # Metadata
        for i, (patch, label) in enumerate(zip(patches, labels)):
            all_metadata.append({
                'lens_id': lens_id,
                'image_file': str(image_file.relative_to(output_dir)),
                'patch_idx': i,
                'is_arc': int(label),
                'n_images': len(positions_pix)
            })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(catalog)} lens systems...")
    
    # Combine all patches
    if len(all_patches) == 0:
        print("❌ No training data extracted")
        sys.exit(1)
    
    all_patches = np.vstack(all_patches)
    all_labels = np.hstack(all_labels)
    
    print(f"✅ Extracted {len(all_patches)} patches ({np.sum(all_labels)} arcs, {len(all_labels) - np.sum(all_labels)} non-arcs)")
    
    # Save as CSV (flatten patches)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Flatten patches for CSV
    patch_data = []
    for i, patch in enumerate(all_patches):
        patch_flat = patch.flatten()
        patch_data.append(patch_flat)
    
    patch_df = pd.DataFrame(patch_data)
    patch_df.columns = [f'pixel_{i}' for i in range(patch_df.shape[1])]
    
    # Combine
    training_df = pd.concat([metadata_df, patch_df], axis=1)
    
    # Save
    output_csv = output_dir / args.output_csv
    training_df.to_csv(output_csv, index=False)
    print(f"✅ Saved training data to: {output_csv}")
    print(f"   Shape: {training_df.shape}")
    print(f"   Arcs: {np.sum(all_labels)}, Non-arcs: {len(all_labels) - np.sum(all_labels)}")


if __name__ == '__main__':
    main()

