#!/usr/bin/env python3
"""
Create Advanced Publication Figure with Scientific Annotations

This script creates a more sophisticated figure suitable for a scientific paper,
including component decomposition, flux measurements, and detailed annotations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from PIL import Image
import sys
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def normalize_image(image, vmin=None, vmax=None, stretch='log'):
    """Normalize image for visualization."""
    if image.size == 0:
        return image
    
    if len(image.shape) == 3:
        return np.array([normalize_image(img, vmin, vmax, stretch) for img in image])
    
    img = image.copy().astype(float)
    
    if vmin is None:
        vmin = np.nanpercentile(img[img > 0], 1) if np.any(img > 0) else 0
    if vmax is None:
        vmax = np.nanpercentile(img, 99)
    
    img = np.clip(img, vmin, vmax)
    
    if stretch == 'log':
        img = np.log1p(img - vmin + 1)
        vmax = np.log1p(vmax - vmin + 1)
        vmin = 0
    elif stretch == 'sqrt':
        img = np.sqrt(np.maximum(img - vmin, 0))
        vmax = np.sqrt(vmax - vmin)
        vmin = 0
    elif stretch == 'asinh':
        sinh_scale = 0.1 * (vmax - vmin)
        img = np.arcsinh((img - vmin) / sinh_scale)
        vmax = np.arcsinh((vmax - vmin) / sinh_scale)
        vmin = 0
    
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    
    return np.clip(img, 0, 1)


def create_rgb_from_bands(bands_dict):
    """Create RGB composite from multi-band image."""
    if isinstance(bands_dict, dict):
        try:
            b_band = normalize_image(bands_dict.get('F115W', bands_dict.get('F150W', np.zeros((1, 1)))))
            g_band = normalize_image(bands_dict.get('F277W', np.zeros((1, 1))))
            r_band = normalize_image(bands_dict.get('F444W', np.zeros((1, 1))))
            
            if len(b_band.shape) == 3:
                b_band = b_band[0]
            if len(g_band.shape) == 3:
                g_band = g_band[0]
            if len(r_band.shape) == 3:
                r_band = r_band[0]
            
            rgb = np.stack([r_band, g_band, b_band], axis=2)
            return np.clip(rgb, 0, 1)
        except Exception as e:
            print(f"Warning: Could not create RGB: {e}")
            return np.zeros((*list(bands_dict.values())[0].shape, 3))
    else:
        gray = normalize_image(bands_dict)
        return np.stack([gray, gray, gray], axis=2)


def load_image_data(image_path):
    """Load image data from .npy or .jpg file."""
    image_path = Path(image_path)
    
    if image_path.suffix == '.npy':
        data = np.load(image_path)
        if len(data.shape) == 3 and data.shape[0] == 4:
            bands = ['F115W', 'F150W', 'F277W', 'F444W']
            return {bands[i]: data[i] for i in range(4)}
        return data
    elif image_path.suffix == '.jpg':
        img = Image.open(image_path)
        return np.array(img) / 255.0
    else:
        raise ValueError(f"Unsupported format: {image_path.suffix}")


def calculate_image_stats(image_data):
    """Calculate statistics from image data."""
    if isinstance(image_data, dict):
        # Use F150W as reference
        img = image_data.get('F150W', list(image_data.values())[0])
    else:
        img = image_data
    
    if len(img.shape) == 3:
        img = img[0]
    
    # Flatten and remove zeros
    flat = img[img > 0].flatten()
    
    if len(flat) > 0:
        return {
            'mean': np.mean(flat),
            'median': np.median(flat),
            'std': np.std(flat),
            'max': np.max(flat),
            'total_flux': np.sum(img)
        }
    return {'mean': 0, 'median': 0, 'std': 0, 'max': 0, 'total_flux': 0}


def create_advanced_figure(output_dir, lens_id='000001', save_path=None):
    """
    Create advanced scientific figure with multiple panels and annotations.
    """
    output_dir = Path(output_dir)
    
    stages = {
        'Lens Only': 'intermediate_lens_only',
        'Lens + Sources': 'intermediate_lens_sources',
        'Sources Only': 'intermediate_sources_only',
        'Field Only': 'intermediate_field_only',
        'Final': None
    }
    
    # Load images and stats
    images_data = {}
    stats_data = {}
    
    for stage_name, stage_dir in stages.items():
        try:
            if stage_dir is None:
                jpg_path = output_dir / 'jpg_rgb' / f'cosmos_lens_{lens_id}.jpg'
            else:
                jpg_path = output_dir / 'jpg_rgb' / stage_dir / f'cosmos_lens_{lens_id}.jpg'
            
            if jpg_path.exists():
                images_data[stage_name] = load_image_data(jpg_path)
                stats_data[stage_name] = calculate_image_stats(images_data[stage_name])
                print(f"✓ Loaded {stage_name}")
        except Exception as e:
            print(f"✗ Error loading {stage_name}: {e}")
    
    if not images_data:
        print("Error: No images loaded!")
        return None
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.3)
    
    # Main image panels (top row and part of middle)
    stage_order = ['Lens Only', 'Lens + Sources', 'Sources Only', 'Field Only', 'Final']
    
    for idx, stage_name in enumerate(stage_order):
        if stage_name not in images_data:
            continue
        
        # Main image subplot
        ax = fig.add_subplot(gs[0:2, idx])
        
        image_data = images_data[stage_name]
        if isinstance(image_data, dict):
            rgb = create_rgb_from_bands(image_data)
        elif len(image_data.shape) == 3:
            rgb = image_data
        else:
            gray = normalize_image(image_data)
            rgb = np.stack([gray, gray, gray], axis=2)
        
        ax.imshow(rgb, origin='upper')
        ax.set_title(stage_name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
    
    # Statistics panels (bottom row)
    for idx, stage_name in enumerate(stage_order):
        if stage_name not in images_data:
            continue
        
        ax = fig.add_subplot(gs[2, idx])
        ax.axis('off')
        
        stats = stats_data[stage_name]
        
        # Create stats text
        stats_text = (
            f"{stage_name}\n"
            f"━━━━━━━━━\n"
            f"Max: {stats['max']:.2e}\n"
            f"Mean: {stats['mean']:.2e}\n"
            f"Flux: {stats['total_flux']:.2e}"
        )
        
        ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
               fontsize=9, ha='center', va='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Title and description
    fig.suptitle('JWST Lens Simulation: Component Decomposition Pipeline',
                fontsize=16, fontweight='bold', y=0.98)
    
    description = (
        'Complete pipeline showing how realistic lens system images are built through progressive stages:\n'
        '(1) Lens galaxy only | (2) Lens + lensed sources | (3) Sources only | (4) Field galaxies | (5) Final composite\n'
        'Each image includes PSF convolution, realistic JWST noise, sky background, and detector artifacts.'
    )
    fig.text(0.5, 0.01, description, ha='center', fontsize=10, style='italic')
    
    # Save
    if save_path is None:
        save_path = output_dir / 'advanced_intermediate_figure.png'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Create advanced intermediate stages figure')
    parser.add_argument('output_dir', help='Path to simulation output directory')
    parser.add_argument('--lens-id', default='000001', help='Lens ID to visualize')
    parser.add_argument('--save-path', help='Path to save figure')
    
    args = parser.parse_args()
    
    fig = create_advanced_figure(args.output_dir, lens_id=args.lens_id, save_path=args.save_path)
    
    if fig is not None:
        plt.show()


if __name__ == '__main__':
    main()
