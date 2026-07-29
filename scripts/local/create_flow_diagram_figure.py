#!/usr/bin/env python3
"""
Create Schematic Flow Diagram Figure

This script creates a publication-quality figure that shows the complete
simulation workflow with visual connections between stages.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import argparse
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def normalize_image(image, vmin=None, vmax=None):
    """Normalize image for visualization."""
    if len(image.shape) == 3:
        return np.array([normalize_image(img, vmin, vmax) for img in image])
    
    img = image.copy().astype(float)
    
    if vmin is None:
        vmin = np.nanpercentile(img[img > 0], 2) if np.any(img > 0) else 0
    if vmax is None:
        vmax = np.nanpercentile(img, 98)
    
    img = np.clip(img, vmin, vmax)
    img = np.log1p(img - vmin + 1)
    vmax = np.log1p(vmax - vmin + 1)
    
    if vmax > 0:
        img = img / vmax
    
    return np.clip(img, 0, 1)


def create_rgb_from_bands(bands_dict):
    """Create RGB composite from multi-band image."""
    if isinstance(bands_dict, dict):
        try:
            b = normalize_image(bands_dict.get('F115W', bands_dict.get('F150W', np.zeros((1, 1)))))
            g = normalize_image(bands_dict.get('F277W', np.zeros((1, 1))))
            r = normalize_image(bands_dict.get('F444W', np.zeros((1, 1))))
            
            if len(b.shape) == 3:
                b = b[0]
            if len(g.shape) == 3:
                g = g[0]
            if len(r.shape) == 3:
                r = r[0]
            
            rgb = np.stack([r, g, b], axis=2)
            return np.clip(rgb, 0, 1)
        except:
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


def create_flow_diagram_figure(output_dir, lens_id='000001', save_path=None):
    """
    Create a detailed flow diagram showing the simulation pipeline.
    """
    output_dir = Path(output_dir)
    
    # Load images
    stages = {
        'Lens': 'intermediate_lens_only',
        'Lens\n+ Sources': 'intermediate_lens_sources',
        'Sources\nOnly': 'intermediate_sources_only',
        'Field\nGalaxies': 'intermediate_field_only',
        'Final': None
    }
    
    images = {}
    for name, stage_dir in stages.items():
        try:
            if stage_dir is None:
                jpg_path = output_dir / 'jpg_rgb' / f'cosmos_lens_{lens_id}.jpg'
            else:
                jpg_path = output_dir / 'jpg_rgb' / stage_dir / f'cosmos_lens_{lens_id}.jpg'
            
            if jpg_path.exists():
                images[name] = load_image_data(jpg_path)
        except Exception as e:
            print(f"Could not load {name}: {e}")
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.25, top=0.92, bottom=0.1, left=0.05, right=0.95)
    
    # Stage positions: (row, col)
    stage_positions = {
        'Lens': (0, 0),
        'Lens\n+ Sources': (0, 2),
        'Sources\nOnly': (0, 4),
        'Field\nGalaxies': (1, 1),
        'Final': (1, 3)
    }
    
    # Plot images
    for stage_name, (row, col) in stage_positions.items():
        if stage_name not in images:
            ax = axes[row, col]
            ax.axis('off')
            continue
        
        ax = axes[row, col]
        
        image_data = images[stage_name]
        if isinstance(image_data, dict):
            rgb = create_rgb_from_bands(image_data)
        else:
            rgb = image_data
        
        ax.imshow(rgb, origin='upper', cmap='gray')
        ax.set_title(stage_name, fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
    
    # Hide unused subplots
    for row in range(2):
        for col in range(5):
            if (row, col) not in stage_positions.values():
                axes[row, col].axis('off')
    
    # Add description boxes in bottom-left area
    ax_desc = fig.add_axes([0.05, 0.02, 0.4, 0.06])
    ax_desc.axis('off')
    
    description = (
        'Pipeline: Galaxy lens + lensed sources + field contamination → Realistic JWST observation\n'
        'All stages include: PSF convolution, realistic noise, sky background, detector artifacts'
    )
    ax_desc.text(0, 0.5, description, fontsize=9, style='italic', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add title
    fig.suptitle('JWST Mock Lens Simulator: Image Generation Pipeline', 
                fontsize=16, fontweight='bold', y=0.97)
    
    # Add annotations for each component
    ax_annotations = fig.add_axes([0.5, 0.02, 0.45, 0.06])
    ax_annotations.axis('off')
    
    annotations = (
        'Step 1: Lens galaxy morphology\n'
        'Step 2a: Add lensed source(s)\n'
        'Step 2b: Sources only (diagnostic)\n'
        'Step 3a: Field galaxy contamination\n'
        'Step 3b: Combine all components'
    )
    ax_annotations.text(0, 0.5, annotations, fontsize=9, va='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save figure
    if save_path is None:
        save_path = output_dir / 'flow_diagram_figure.png'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Create flow diagram figure')
    parser.add_argument('output_dir', help='Path to simulation output directory')
    parser.add_argument('--lens-id', default='000001', help='Lens ID to visualize')
    parser.add_argument('--save-path', help='Path to save figure')
    
    args = parser.parse_args()
    
    fig = create_flow_diagram_figure(args.output_dir, lens_id=args.lens_id, 
                                     save_path=args.save_path)
    
    if fig is not None:
        plt.show()


if __name__ == '__main__':
    main()
