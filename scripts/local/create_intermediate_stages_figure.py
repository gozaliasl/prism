#!/usr/bin/env python3
"""
Create Publication-Ready Figure of Intermediate Image Stages

This script generates a comprehensive figure showing the different stages
of the simulation pipeline:
1. Lens only
2. Lens + lensed sources (arcs)
3. Lensed sources only
4. Field galaxies only
5. Final composite image

Suitable for paper figures showing simulation workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from PIL import Image
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def normalize_image(image, vmin=None, vmax=None, stretch='log'):
    """
    Normalize image for visualization.
    
    Args:
        image: 2D or 3D numpy array
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        stretch: 'linear', 'log', 'sqrt', or 'asinh'
    
    Returns:
        Normalized image in [0, 1] range
    """
    if image.size == 0:
        return image
    
    # Handle 3D images (e.g., multi-band)
    if len(image.shape) == 3:
        return np.array([normalize_image(img, vmin, vmax, stretch) for img in image])
    
    # Avoid division by zero
    img = image.copy().astype(float)
    
    if vmin is None:
        vmin = np.nanpercentile(img[img > 0], 1) if np.any(img > 0) else 0
    if vmax is None:
        vmax = np.nanpercentile(img, 99)
    
    # Clip to range
    img = np.clip(img, vmin, vmax)
    
    # Apply stretch
    if stretch == 'log':
        # Avoid log(0)
        img = np.log1p(img - vmin + 1)
        vmax = np.log1p(vmax - vmin + 1)
        vmin = 0
    elif stretch == 'sqrt':
        img = np.sqrt(np.maximum(img - vmin, 0))
        vmax = np.sqrt(vmax - vmin)
        vmin = 0
    elif stretch == 'asinh':
        # Asinh scaling
        sinh_scale = 0.1 * (vmax - vmin)
        img = np.arcsinh((img - vmin) / sinh_scale)
        vmax = np.arcsinh((vmax - vmin) / sinh_scale)
        vmin = 0
    
    # Normalize to [0, 1]
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    
    return np.clip(img, 0, 1)


def create_rgb_from_bands(bands_dict, band_names=['F115W', 'F150W', 'F277W', 'F444W']):
    """
    Create RGB composite from multi-band image.
    
    Args:
        bands_dict: Dictionary with band names as keys and 2D arrays as values
        band_names: List of band names in order
    
    Returns:
        RGB image (height, width, 3)
    """
    # Use three bands for RGB: F150W, F277W, F444W -> B, G, R
    if isinstance(bands_dict, dict):
        try:
            # Extract specific bands
            b_band = normalize_image(bands_dict.get('F115W', bands_dict.get('F150W', np.zeros((1, 1)))))
            g_band = normalize_image(bands_dict.get('F277W', np.zeros((1, 1))))
            r_band = normalize_image(bands_dict.get('F444W', np.zeros((1, 1))))
            
            # Handle 3D bands (shouldn't happen but just in case)
            if len(b_band.shape) == 3:
                b_band = b_band[0]
            if len(g_band.shape) == 3:
                g_band = g_band[0]
            if len(r_band.shape) == 3:
                r_band = r_band[0]
            
            rgb = np.stack([r_band, g_band, b_band], axis=2)
            return np.clip(rgb, 0, 1)
        except Exception as e:
            print(f"Warning: Could not create RGB from bands: {e}")
            return np.zeros((*bands_dict['F115W'].shape, 3))
    else:
        # Assume single band
        gray = normalize_image(bands_dict)
        return np.stack([gray, gray, gray], axis=2)


def load_image_data(image_path):
    """
    Load image data from .npy or .jpg file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image data as numpy array
    """
    image_path = Path(image_path)
    
    if image_path.suffix == '.npy':
        data = np.load(image_path)
        # If 4-band, return as dict
        if len(data.shape) == 3 and data.shape[0] == 4:
            bands = ['F115W', 'F150W', 'F277W', 'F444W']
            return {bands[i]: data[i] for i in range(4)}
        return data
    elif image_path.suffix == '.jpg':
        img = Image.open(image_path)
        return np.array(img) / 255.0
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")


def create_intermediate_stages_figure(output_dir, lens_id='000001', save_path=None, figsize=(16, 12)):
    """
    Create figure showing intermediate stages of image generation.
    
    Args:
        output_dir: Path to simulation output directory
        lens_id: Lens ID to use (e.g., '000001')
        save_path: Path to save figure (default: output_dir/intermediate_stages_figure.png)
        figsize: Figure size in inches
    
    Returns:
        Figure object
    """
    output_dir = Path(output_dir)
    
    # Define intermediate stages
    stages = {
        'Lens Only': 'intermediate_lens_only',
        'Lens + Sources\n(Lensed Arcs)': 'intermediate_lens_sources',
        'Sources Only\n(No Lens Light)': 'intermediate_sources_only',
        'Field Galaxies\n(Contamination)': 'intermediate_field_only',
        'Final Image\n(Complete System)': None  # Use jpg_rgb/ directly
    }
    
    # Load images
    images_data = {}
    missing_stages = []
    
    for stage_name, stage_dir in stages.items():
        try:
            if stage_dir is None:
                # Final image from jpg_rgb
                jpg_path = output_dir / 'jpg_rgb' / f'cosmos_lens_{lens_id}.jpg'
            else:
                # Intermediate images from jpg_rgb subdirectories
                jpg_path = output_dir / 'jpg_rgb' / stage_dir / f'cosmos_lens_{lens_id}.jpg'
            
            if jpg_path.exists():
                images_data[stage_name] = load_image_data(jpg_path)
                print(f"✓ Loaded {stage_name}: {jpg_path}")
            else:
                print(f"✗ Not found: {jpg_path}")
                missing_stages.append(stage_name)
        except Exception as e:
            print(f"✗ Error loading {stage_name}: {e}")
            missing_stages.append(stage_name)
    
    if missing_stages:
        print(f"\nWarning: Could not load {len(missing_stages)} stage(s)")
        # Remove missing stages
        for stage in missing_stages:
            if stage in images_data:
                del images_data[stage]
    
    # Create figure
    n_stages = len(images_data)
    if n_stages == 0:
        print("Error: No image data could be loaded!")
        return None
    
    # Determine layout
    if n_stages <= 3:
        ncols = n_stages
        nrows = 1
    elif n_stages <= 5:
        ncols = 3
        nrows = 2
    else:
        ncols = 3
        nrows = (n_stages + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes for easier iteration
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each stage
    for idx, (stage_name, image_data) in enumerate(images_data.items()):
        ax = axes[idx]
        
        # Handle different image shapes
        if isinstance(image_data, dict):
            # Multi-band image
            rgb = create_rgb_from_bands(image_data)
        elif len(image_data.shape) == 3:
            # Already RGB
            rgb = image_data
        else:
            # Single band - convert to gray RGB
            gray = normalize_image(image_data)
            rgb = np.stack([gray, gray, gray], axis=2)
        
        # Display image
        ax.imshow(rgb, origin='upper', cmap='gray')
        ax.set_title(stage_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Pixels', fontsize=10)
        ax.set_ylabel('Pixels', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add subtle text annotations
        if 'Lens Only' in stage_name:
            ax.text(0.5, 0.95, 'Step 1', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        elif 'Lens + Sources' in stage_name:
            ax.text(0.5, 0.95, 'Step 2a', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        elif 'Sources Only' in stage_name:
            ax.text(0.5, 0.95, 'Step 2b', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
        elif 'Field Galaxies' in stage_name:
            ax.text(0.5, 0.95, 'Step 3a', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        elif 'Final' in stage_name:
            ax.text(0.5, 0.95, 'Step 3b', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(n_stages, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title and description
    fig.suptitle('JWST Mock Lens Simulator: Intermediate Image Stages', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add description text
    description = (
        'Progressive stages of lens system image generation:\n'
        '1. Lens-only image (lens galaxy without sources)\n'
        '2a. Combined lens and lensed sources (Einstein ring/arcs)\n'
        '2b. Sources only (lensed arcs without lens light)\n'
        '3a. Field galaxies only (environmental contamination)\n'
        '3b. Final composite image (all components combined)\n\n'
        'All intermediate images include PSF convolution, realistic noise, '
        'sky background, and JWST artifacts.'
    )
    fig.text(0.5, 0.01, description, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # Save figure
    if save_path is None:
        save_path = output_dir / 'intermediate_stages_figure.png'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create figure showing intermediate stages of lens system simulation'
    )
    parser.add_argument('output_dir', help='Path to simulation output directory')
    parser.add_argument('--lens-id', default='000001', help='Lens ID to visualize (default: 000001)')
    parser.add_argument('--save-path', help='Path to save figure (default: output_dir/intermediate_stages_figure.png)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 12],
                       help='Figure size in inches (width height)')
    
    args = parser.parse_args()
    
    # Create figure
    fig = create_intermediate_stages_figure(
        args.output_dir,
        lens_id=args.lens_id,
        save_path=args.save_path,
        figsize=tuple(args.figsize)
    )
    
    if fig is not None:
        plt.show()


if __name__ == '__main__':
    main()
