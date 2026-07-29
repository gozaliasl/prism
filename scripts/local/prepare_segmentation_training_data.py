#!/usr/bin/env python3
"""
Prepare Training Data for Lensed Source Segmentation

This script creates training data for ML-based lensed source detection by:
1. Loading simulation outputs with known lensed image positions
2. Creating segmentation masks (only lensed sources, excluding lens galaxy and field galaxies)
3. Extracting patches around lensed sources
4. Preparing data for U-Net or similar segmentation models

This enables inverse engineering: train on what we know (simulations) to detect in real images.
"""

import numpy as np
import pandas as pd
import ast
from pathlib import Path
from PIL import Image
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from photutils.segmentation import detect_sources, SourceCatalog
    from photutils.background import Background2D, MedianBackground
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False
    print("[WARNING] photutils not available")

try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    LENSTRONOMY_AVAILABLE = False
    print("[WARNING] lenstronomy not available - cannot calculate image positions")

try:
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage import measure, morphology
    SCIPY_AVAILABLE = True
    SKIMAGE_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    SKIMAGE_AVAILABLE = False


def create_segmentation_mask(
    image_array: np.ndarray,
    lensed_image_positions: list,
    lens_center: tuple,
    einstein_radius_pix: float,
    pixel_scale: float = 0.03,
    numpix: int = 300,
    mask_radius_pix: float = 10.0  # Increased default radius
) -> np.ndarray:
    """
    Create a segmentation mask where only lensed source pixels are marked.
    
    Args:
        image_array: 2D image array
        lensed_image_positions: List of (x, y) positions in arcsec
        lens_center: (x, y) lens center in pixels
        einstein_radius_pix: Einstein radius in pixels
        pixel_scale: Pixel scale
        numpix: Image size
        mask_radius_pix: Radius around each position to include in mask
    
    Returns:
        2D mask array (1 = lensed source, 0 = everything else)
    """
    h, w = image_array.shape
    mask = np.zeros((h, w), dtype=bool)
    
    # Use provided lens_center (don't recalculate)
    center_x, center_y = lens_center
    
    # Debug: track if any masks were created
    masks_created = 0
    
    # For each lensed image position, find the actual source and mark it
    for x_arcsec, y_arcsec in lensed_image_positions:
        # Convert arcsec to pixels (positions are relative to lens center)
        x_pix = center_x + x_arcsec / pixel_scale
        y_pix = center_y + y_arcsec / pixel_scale
        
        # Ensure positions are within bounds
        x_pix = max(0, min(w - 1, x_pix))
        y_pix = max(0, min(h - 1, y_pix))
        
        # Always create a circular mask at the position (fallback)
        # This ensures we have training data even if source detection fails
        y_int, x_int = int(y_pix), int(x_pix)
        if 0 <= y_int < h and 0 <= x_int < w:
            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((x_coords - x_int)**2 + (y_coords - y_int)**2)
            # Create circular mask with reasonable radius
            circular_mask = dist < mask_radius_pix * 2  # 20 pixels radius
            mask |= circular_mask
            masks_created += 1
        
        # Try to find actual source using photutils (optional enhancement)
        # But we already created a circular mask above, so this is just for refinement
        if PHOTUTILS_AVAILABLE:
            try:
                # Detect sources
                bkg_estimator = MedianBackground()
                bkg = Background2D(image_array, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
                threshold = bkg.background + 2.0 * bkg.background_rms
                
                segm = detect_sources(image_array, threshold, npixels=5)
                if segm is None or segm.nlabels == 0:
                    # Already created circular mask above, continue to next position
                    continue
                
                # Find closest source
                cat = SourceCatalog(image_array, segm)
                sources = cat.to_table()
                
                min_dist = float('inf')
                best_source = None
                
                center_x, center_y = lens_center
                
                for source in sources:
                    try:
                        sx = float(source['xcentroid'])
                        sy = float(source['ycentroid'])
                    except:
                        sx, sy = source['xcentroid'], source['ycentroid']
                    
                    dist = np.sqrt((sx - x_pix)**2 + (sy - y_pix)**2)
                    dist_from_center = np.sqrt((sx - center_x)**2 + (sy - center_y)**2)
                    
                    # Must be around Einstein radius (not lens galaxy)
                    if einstein_radius_pix:
                        if dist_from_center < einstein_radius_pix * 0.7:
                            continue
                        if dist_from_center > einstein_radius_pix * 2.5:
                            continue
                    
                    if dist < min_dist and dist < mask_radius_pix * 5:  # Increased search radius
                        min_dist = dist
                        best_source = source
                
                if best_source:
                    # Get segmentation region for this source (refine the mask)
                    label_id = int(best_source['label'])
                    source_mask = segm.data == label_id
                    
                    # Replace circular mask with actual source shape
                    # Remove the circular mask at this position first
                    y_int, x_int = int(y_pix), int(x_pix)
                    y_coords, x_coords = np.ogrid[:h, :w]
                    dist = np.sqrt((x_coords - x_int)**2 + (y_coords - y_int)**2)
                    circular_region = dist < mask_radius_pix * 2
                    mask[circular_region] = False  # Remove circular mask
                    
                    # Add actual source mask
                    mask |= source_mask
            except Exception as e:
                # Error in source detection, but circular mask already created above
                pass
    
    # Debug: check final mask
    final_pixels = np.sum(mask)
    if final_pixels == 0 and masks_created > 0:
        print(f"    ERROR: Created {masks_created} masks but final mask has 0 pixels!")
    
    return mask.astype(np.uint8)


def extract_training_patches(
    image_array: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    min_source_pixels: int = 5  # Lowered from 10 to 5
) -> list:
    """
    Extract image patches and corresponding mask patches for training.
    
    Returns:
        List of (image_patch, mask_patch) tuples
    """
    h, w = image_array.shape
    patches = []
    
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            # Extract patch
            img_patch = image_array[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Only include patches with lensed sources
            if np.sum(mask_patch) >= min_source_pixels:
                patches.append((img_patch, mask_patch))
    
    return patches


def prepare_training_data_from_simulation(
    output_dir: Path,
    output_training_dir: Path,
    patch_size: int = 128,
    stride: int = 64,
    pixel_scale: float = 0.03
):
    """
    Prepare training data from simulation outputs.
    
    Creates:
    - images/: Image patches
    - masks/: Segmentation masks (1 = lensed source, 0 = everything else)
    - metadata.csv: Training metadata
    """
    # Read catalog - try multiple possible locations and names
    possible_catalogs = [
        output_dir / 'time_delay_catalog.csv',
        output_dir / 'cosmos_lens_training_catalog.csv',
        output_dir / 'cosmos_training_catalog_lens_and_nonlens.csv',
    ]
    
    # Also check for date subdirectories (some simulations create date subdirs)
    date_dirs = list(output_dir.glob('*_date_*'))
    for date_dir in date_dirs:
        if date_dir.is_dir():
            possible_catalogs.extend([
                date_dir / 'time_delay_catalog.csv',
                date_dir / 'cosmos_lens_training_catalog.csv',
                date_dir / 'cosmos_training_catalog_lens_and_nonlens.csv',
            ])
    
    catalog_path = None
    for cat_path in possible_catalogs:
        if cat_path.exists():
            catalog_path = cat_path
            break
    
    if catalog_path is None:
        print(f"❌ Error: No catalog found!")
        print(f"   Searched in: {output_dir}")
        if date_dirs:
            print(f"   Also searched in date subdirectories: {[str(d) for d in date_dirs]}")
        print(f"   Available CSV files: {list(output_dir.glob('*.csv'))}")
        if date_dirs:
            for date_dir in date_dirs:
                csvs = list(date_dir.glob('*.csv'))
                if csvs:
                    print(f"   In {date_dir.name}: {[f.name for f in csvs]}")
        return
    
    # Update jpg_dir if catalog was found in a date subdirectory
    if catalog_path.parent != output_dir:
        print(f"ℹ️  Found catalog in subdirectory: {catalog_path.parent.name}")
        # Check if jpg_rgb is also in the subdirectory
        subdir_jpg = catalog_path.parent / 'jpg_rgb'
        if subdir_jpg.exists():
            jpg_dir = subdir_jpg
        else:
            jpg_dir = output_dir / 'jpg_rgb'
    else:
        jpg_dir = output_dir / 'jpg_rgb'
    
    print(f"ℹ️  Using catalog: {catalog_path.name}")
    
    catalog = pd.read_csv(catalog_path)
    
    # Image directory (already set above if catalog was in subdirectory)
    if not jpg_dir.exists():
        print(f"❌ Error: RGB images directory not found: {jpg_dir}")
        print(f"   Tried: {jpg_dir}")
        # Try alternative locations
        alt_jpg = output_dir / 'jpg_rgb'
        if alt_jpg.exists() and alt_jpg != jpg_dir:
            jpg_dir = alt_jpg
            print(f"   Found images in: {jpg_dir}")
        else:
            return
    
    # Create output directories
    images_dir = output_training_dir / 'images'
    masks_dir = output_training_dir / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Processing {len(catalog)} lens systems...")
    print(f"   Checking catalog columns: {list(catalog.columns)}")
    
    # Check if we have the required column
    if 'image_positions_arcsec' not in catalog.columns:
        print(f"\n⚠️  WARNING: 'image_positions_arcsec' column not found in catalog!")
        print(f"   Available columns: {list(catalog.columns)}")
        print(f"   This column is needed to extract lensed source positions.")
        print(f"   The script will try to use other methods, but may extract 0 patches.")
    
    all_metadata = []
    patch_idx = 0
    
    skipped_no_positions = 0
    skipped_no_image = 0
    skipped_no_valid_positions = 0
    
    for idx, row in catalog.iterrows():
        lens_id = int(row['lens_id'])
        
        # Get image positions - try multiple column names
        image_positions = []
        
        # Try 'image_positions_arcsec' first (from time delay catalog)
        if 'image_positions_arcsec' in row and pd.notna(row['image_positions_arcsec']):
            try:
                positions = ast.literal_eval(str(row['image_positions_arcsec']))
                if isinstance(positions, list) and len(positions) > 0:
                    if isinstance(positions[0], (list, tuple)):
                        image_positions = [(float(x), float(y)) for x, y in positions]
            except Exception as e:
                pass
        
        # If no positions from catalog, we'll calculate using lenstronomy below
        
        # If no positions, try to calculate from lens model using lenstronomy
        if len(image_positions) == 0:
            # Check if we have enough info to calculate positions
            # For main training catalog, we may not have n_images, so try to calculate anyway
            n_images_known = False
            if 'n_images' in row and pd.notna(row['n_images']):
                n_images = int(row['n_images'])
                n_images_known = n_images > 0
            
            # Try to calculate positions using lenstronomy
            if LENSTRONOMY_AVAILABLE:
                # Try to calculate image positions from lens model
                try:
                    theta_E = float(row.get('theta_E', 1.0))
                    lens_z = float(row.get('lens_redshift', 0.6))
                    source_z = float(row.get('source_redshift', 2.0))
                    
                    # Use simple SIE model to calculate image positions
                    # This is approximate but should work for most cases
                    lens_model_list = ['SIE']
                    kwargs_lens = [{
                        'theta_E': theta_E,
                        'center_x': 0.0,
                        'center_y': 0.0,
                        'e1': 0.0,  # Assume circular for simplicity
                        'e2': 0.0
                    }]
                    
                    lens_model = LensModel(lens_model_list)
                    solver = LensEquationSolver(lens_model)
                    
                    # Try multiple source positions to find images
                    # We'll try a few offsets to find the best configuration
                    best_positions = []
                    best_n_images = 0
                    
                    for offset in [0.05, 0.1, 0.15, 0.2, 0.3]:
                        for angle in [0, 45, 90, 135]:
                            source_x = offset * np.cos(np.radians(angle))
                            source_y = offset * np.sin(np.radians(angle))
                            
                            try:
                                x_image, y_image = solver.image_position_from_source(
                                    source_x, source_y, kwargs_lens
                                )
                                
                                if len(x_image) > best_n_images:
                                    best_n_images = len(x_image)
                                    best_positions = [(float(x), float(y)) for x, y in zip(x_image, y_image)]
                                    
                                    # If we found the expected number of images, use this
                                    if n_images_known and len(x_image) == n_images:
                                        break
                            except:
                                continue
                        
                        if n_images_known and len(best_positions) == n_images:
                            break
                    
                    if len(best_positions) > 0:
                        image_positions = best_positions
                        if (idx + 1) % 100 == 0 or idx < 5:
                            print(f"  Lens {lens_id}: Calculated {len(image_positions)} image positions from lens model")
                except Exception as e:
                    # Calculation failed, skip
                    skipped_no_valid_positions += 1
                    if (idx + 1) % 100 == 0:
                        print(f"  Processed {idx + 1}/{len(catalog)}: {patch_idx} patches, {skipped_no_image} missing images, {skipped_no_positions} no positions")
                    continue
            else:
                skipped_no_positions += 1
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(catalog)}: {patch_idx} patches, {skipped_no_image} missing images, {skipped_no_positions} no positions")
                continue  # Skip if no positions available
        
        if len(image_positions) == 0:
            skipped_no_valid_positions += 1
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(catalog)}: {patch_idx} patches, {skipped_no_image} missing images, {skipped_no_positions} no positions")
            continue
        
        # Load image - try multiple naming patterns
        image_file = None
        for pattern in [
            f"cosmos_lens_{lens_id:06d}_epoch00.jpg",
            f"cosmos_lens_{lens_id:06d}.jpg",
            f"lens_{lens_id:06d}_epoch00.jpg",
            f"lens_{lens_id:06d}.jpg"
        ]:
            candidate = jpg_dir / pattern
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file is None:
            skipped_no_image += 1
            if (idx + 1) % 100 == 0 or idx < 5:  # Debug first few
                print(f"  Lens {lens_id}: Image not found (tried: cosmos_lens_{lens_id:06d}_epoch00.jpg, etc.)")
            continue
        
        img = Image.open(image_file)
        if img.mode == 'RGB':
            img_array = np.array(img.convert('L')) / 255.0
        else:
            img_array = np.array(img.convert('L')) / 255.0
        
        h, w = img_array.shape
        
        # Determine lens center
        if w > h * 2:  # Panoramic
            lens_center = (w // 6, h // 2)
        else:
            lens_center = (w // 2, h // 2)
        
        # Get Einstein radius
        theta_E = float(row.get('theta_E', 1.0))
        einstein_radius_pix = theta_E / pixel_scale
        
        # Debug: Check if positions are within bounds
        if idx < 3:  # Debug first few
            print(f"  Lens {lens_id}: Image size {w}x{h}, lens_center={lens_center}, theta_E={theta_E:.3f} arcsec ({einstein_radius_pix:.1f} pix)")
            for i, (x_arcsec, y_arcsec) in enumerate(image_positions):
                x_pix = lens_center[0] + x_arcsec / pixel_scale
                y_pix = lens_center[1] + y_arcsec / pixel_scale
                in_bounds = (0 <= x_pix < w and 0 <= y_pix < h)
                print(f"    Position {i}: ({x_arcsec:.3f}, {y_arcsec:.3f}) arcsec -> ({x_pix:.1f}, {y_pix:.1f}) pix, in_bounds={in_bounds}")
        
        # Create segmentation mask (only lensed sources)
        mask = create_segmentation_mask(
            img_array, image_positions, lens_center, einstein_radius_pix,
            pixel_scale, max(h, w)
        )
        
        # Check if mask has any sources
        n_source_pixels = np.sum(mask)
        if n_source_pixels < 5:  # Too few source pixels, skip
            skipped_no_valid_positions += 1
            if (idx + 1) % 100 == 0 or idx < 3:  # Debug first few
                print(f"  Lens {lens_id}: Mask too small ({n_source_pixels} pixels), positions: {image_positions}")
            continue
        
        # Extract patches
        patches = extract_training_patches(img_array, mask, patch_size, stride)
        
        if len(patches) == 0:
            # If no patches extracted, create at least one patch centered on the sources
            # Find center of mass of mask
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0:
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                
                # Extract patch centered on sources
                y_start = max(0, center_y - patch_size // 2)
                y_end = min(h, center_y + patch_size // 2)
                x_start = max(0, center_x - patch_size // 2)
                x_end = min(w, center_x + patch_size // 2)
                
                # Pad if needed
                img_patch = np.zeros((patch_size, patch_size))
                mask_patch = np.zeros((patch_size, patch_size))
                
                patch_y_start = max(0, patch_size // 2 - center_y)
                patch_y_end = patch_y_start + (y_end - y_start)
                patch_x_start = max(0, patch_size // 2 - center_x)
                patch_x_end = patch_x_start + (x_end - x_start)
                
                img_patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = img_array[y_start:y_end, x_start:x_end]
                mask_patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = mask[y_start:y_end, x_start:x_end]
                
                patches = [(img_patch, mask_patch)]
        
        # Save patches
        for img_patch, mask_patch in patches:
            # Save image patch
            img_patch_file = images_dir / f"patch_{patch_idx:06d}.npy"
            np.save(img_patch_file, img_patch)
            
            # Save mask patch
            mask_patch_file = masks_dir / f"patch_{patch_idx:06d}.npy"
            np.save(mask_patch_file, mask_patch)
            
            # Metadata
            all_metadata.append({
                'patch_id': patch_idx,
                'lens_id': lens_id,
                'n_images': len(image_positions),
                'theta_E': theta_E,
                'source_pixels': int(np.sum(mask_patch)),
                'image_file': str(img_patch_file.relative_to(output_training_dir)),
                'mask_file': str(mask_patch_file.relative_to(output_training_dir))
            })
            
            patch_idx += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(catalog)}: {patch_idx} patches, {skipped_no_image} missing images, {skipped_no_positions} no positions, {skipped_no_valid_positions} invalid positions")
    
    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_file = output_training_dir / 'training_metadata.csv'
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"\n✅ Prepared {patch_idx} training patches")
    print(f"   Images: {images_dir}")
    print(f"   Masks: {masks_dir}")
    print(f"   Metadata: {metadata_file}")
    
    if len(metadata_df) > 0:
        print(f"\n📊 Statistics:")
        print(f"   Average source pixels per patch: {metadata_df['source_pixels'].mean():.1f}")
        print(f"   Total patches: {len(metadata_df)}")
    else:
        print(f"\n⚠️  WARNING: No training patches were extracted!")
        print(f"   Summary:")
        print(f"   - Skipped (no positions): {skipped_no_positions}")
        print(f"   - Skipped (invalid positions): {skipped_no_valid_positions}")
        print(f"   - Skipped (missing images): {skipped_no_image}")
        print(f"   Possible reasons:")
        print(f"   - Catalog missing 'image_positions_arcsec' column")
        print(f"   - Image files not found (checked: {jpg_dir})")
        print(f"   - No valid lensed image positions in catalog")
        print(f"   - lenstronomy not available to calculate positions")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare segmentation training data from simulation outputs"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Simulation output directory'
    )
    parser.add_argument(
        '--training-dir',
        type=str,
        default=None,
        help='Output directory for training data (default: output_dir/segmentation_training)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=128,
        help='Size of image patches (default: 128)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=64,
        help='Stride for patch extraction (default: 64)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    if args.training_dir:
        training_dir = Path(args.training_dir)
    else:
        training_dir = output_dir / 'segmentation_training'
    
    training_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Preparing segmentation training data...")
    print(f"   Source: {output_dir}")
    print(f"   Output: {training_dir}")
    
    prepare_training_data_from_simulation(
        output_dir, training_dir, args.patch_size, args.stride
    )


if __name__ == '__main__':
    main()

