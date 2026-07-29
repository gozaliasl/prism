#!/usr/bin/env python3
"""
Create Composite Figure Comparing Multiple Lens Examples

This script creates a figure showing intermediate stages for multiple
different lens systems side-by-side, useful for showing diversity
and quality of simulated lenses.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def normalize_image(image, vmin=None, vmax=None, stretch='log'):
    """Normalize image for visualization."""
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
    
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    
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


def create_composite_lenses_figure(output_dir, lens_ids=None, save_path=None, 
                                   n_cols=3, stages=None):
    """
    Create figure showing multiple lenses with their intermediate stages.
    
    Args:
        output_dir: Path to simulation output
        lens_ids: List of lens IDs to include (default: 000001-000006)
        save_path: Path to save figure
        n_cols: Number of columns (3 = 3 lenses wide)
        stages: List of stages to show (default: all 5)
    """
    output_dir = Path(output_dir)
    
    # Default lens IDs
    if lens_ids is None:
        lens_ids = [f'{i:06d}' for i in range(1, 7)]
    
    # Default stages
    if stages is None:
        stages = ['lens_only', 'lens_sources', 'sources_only', 'field_only', None]
    
    stage_names = {
        'lens_only': 'Lens Only',
        'lens_sources': 'Lens+Sources',
        'sources_only': 'Sources Only',
        'field_only': 'Field Only',
        None: 'Final'
    }
    
    n_stages = len(stages)
    n_lenses = len(lens_ids)
    n_cols = min(n_cols, n_lenses)
    n_rows = (n_lenses + n_cols - 1) // n_cols
    
    # Create figure with subplots for each (lens, stage) combination
    fig, axes = plt.subplots(n_rows, n_cols * n_stages, 
                             figsize=(4 * n_cols * n_stages, 4 * n_rows))
    
    if n_rows == 1 and n_cols * n_stages == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or (n_cols * n_stages) == 1:
        axes = axes.reshape(n_rows, n_cols * n_stages)
    else:
        axes = axes.reshape(n_rows, n_cols * n_stages)
    
    # Load and plot images
    for lens_idx, lens_id in enumerate(lens_ids):
        row = lens_idx // n_cols
        
        for stage_idx, stage_dir in enumerate(stages):
            col = (lens_idx % n_cols) * n_stages + stage_idx
            
            ax = axes[row, col]
            
            try:
                if stage_dir is None:
                    jpg_path = output_dir / 'jpg_rgb' / f'cosmos_lens_{lens_id}.jpg'
                else:
                    jpg_path = output_dir / 'jpg_rgb' / f'intermediate_{stage_dir}' / f'cosmos_lens_{lens_id}.jpg'
                
                if jpg_path.exists():
                    image_data = load_image_data(jpg_path)
                    
                    if isinstance(image_data, dict):
                        rgb = create_rgb_from_bands(image_data)
                    else:
                        rgb = image_data
                    
                    ax.imshow(rgb, origin='upper')
                else:
                    ax.text(0.5, 0.5, 'Not Found', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_facecolor('lightgray')
            
            except Exception as e:
                ax.text(0.5, 0.5, f'Error', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_facecolor('pink')
            
            # Label axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add stage name at top of first row
            if lens_idx == 0:
                ax.set_title(stage_names[stage_dir], fontsize=11, fontweight='bold', pad=8)
            
            # Add lens ID on left
            if stage_idx == 0:
                ax.set_ylabel(f'Lens {lens_id}', fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for row in range(n_rows):
        for col in range(n_cols * n_stages):
            if row >= n_rows or col >= n_cols * n_stages:
                axes[row, col].axis('off')
    
    # Add overall title
    fig.suptitle('JWST Mock Lens Simulator: Multiple Example Systems', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Add description
    description = (
        'Showing diversity of simulated lens systems across all intermediate stages. '
        'Each row shows a different lens, each column shows a different component stage. '
        'Final column shows complete composite image.'
    )
    fig.text(0.5, 0.01, description, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    # Save
    if save_path is None:
        save_path = output_dir / 'composite_lenses_figure.png'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create composite figure comparing multiple lens systems'
    )
    parser.add_argument('output_dir', help='Path to simulation output directory')
    parser.add_argument('--lens-ids', nargs='+', help='Lens IDs to include')
    parser.add_argument('--n-lenses', type=int, default=6, help='Number of lenses to show')
    parser.add_argument('--n-cols', type=int, default=3, help='Number of columns')
    parser.add_argument('--save-path', help='Path to save figure')
    
    args = parser.parse_args()
    
    # Generate lens IDs if not provided
    if args.lens_ids is None:
        lens_ids = [f'{i:06d}' for i in range(1, args.n_lenses + 1)]
    else:
        lens_ids = args.lens_ids
    
    fig = create_composite_lenses_figure(
        args.output_dir,
        lens_ids=lens_ids,
        save_path=args.save_path,
        n_cols=args.n_cols
    )
    
    if fig is not None:
        plt.show()


if __name__ == '__main__':
    main()
