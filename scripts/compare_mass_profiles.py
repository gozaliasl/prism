#!/usr/bin/env python3
"""
Compare the same lens across three mass profile configurations (SIE+SIE, NFW+NFW, Shear-only)
Shows how different mass distributions create different arc patterns for identical sources.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import json

def load_lens_image(output_dir, lens_id, stage='final'):
    """
    Load lens image from output directory.
    
    Args:
        output_dir: Path to the output directory (e.g., 'outputs/pair_treatments_*/sie_sie')
        lens_id: Lens ID (e.g., 0, 1, 2, ...)
        stage: 'final', 'lens_only', 'lens_sources', 'sources_only', or 'field_only'
    
    Returns:
        RGB image as numpy array (0-1 range) or None if not found
    """
    output_dir = Path(output_dir)
    
    if stage == 'final':
        # Try both naming conventions
        jpg_path = output_dir / 'jpg_rgb' / f'cosmos_lens_{lens_id:06d}_rgb.jpg'
        if not jpg_path.exists():
            jpg_path = output_dir / 'jpg_rgb' / f'cosmos_lens_{lens_id:06d}.jpg'
    else:
        jpg_path = output_dir / 'jpg_rgb' / f'intermediate_{stage}' / f'cosmos_lens_{lens_id:06d}.jpg'
    
    if jpg_path.exists():
        img = Image.open(jpg_path)
        return np.array(img) / 255.0  # Normalize to 0-1
    else:
        print(f"[WARNING] Image not found: {jpg_path}")
        return None

def load_lens_metadata(output_dir, lens_id):
    """Load lens metadata (parameters, redshifts, etc.)"""
    output_dir = Path(output_dir)
    diag_path = output_dir / 'diagnostics' / f'cosmos_structural_{lens_id:06d}_diag.json'
    
    if diag_path.exists():
        with open(diag_path, 'r') as f:
            return json.load(f)
    return None

def create_comparison_figure(output_dirs, lens_ids, stage='final', save_path=None):
    """
    Create a comparison figure showing the same lenses across three configurations.
    
    Args:
        output_dirs: Dict with keys 'sie_sie', 'nfw_nfw', 'shear_only' pointing to output dirs
        lens_ids: List of lens IDs to compare (e.g., [0, 1, 2])
        stage: Which stage to show ('final', 'lens_only', 'lens_sources', 'sources_only')
        save_path: Path to save the figure (optional)
    """
    
    configs = ['SIE+SIE', 'NFW+NFW', 'Shear Only']
    config_keys = ['sie_sie', 'nfw_nfw', 'shear_only']
    
    n_lenses = len(lens_ids)
    fig, axes = plt.subplots(n_lenses, 3, figsize=(15, 5*n_lenses))
    
    if n_lenses == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Mass Profile Comparison - {stage.replace("_", " ").title()} Stage', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for row_idx, lens_id in enumerate(lens_ids):
        for col_idx, (config_name, config_key) in enumerate(zip(configs, config_keys)):
            ax = axes[row_idx, col_idx]
            
            # Load image
            output_dir = output_dirs.get(config_key)
            if output_dir is None:
                ax.text(0.5, 0.5, f'{config_name}\nNot Found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{config_name}', fontweight='bold')
                ax.axis('off')
                continue
            
            img = load_lens_image(output_dir, lens_id, stage)
            
            if img is not None:
                ax.imshow(img, origin='lower', interpolation='nearest')
                
                # Load and display metadata
                metadata = load_lens_metadata(output_dir, lens_id)
                if metadata:
                    theta_E = metadata.get('parameters', {}).get('theta_E', '?')
                    z_lens = metadata.get('parameters', {}).get('lens_redshift', '?')
                    z_source = metadata.get('parameters', {}).get('source_redshift', '?')
                    
                    title = f'{config_name}\n'
                    title += f'θ_E = {theta_E:.2f}"\n'
                    title += f'z_lens = {z_lens:.2f}, z_src = {z_source:.2f}'
                else:
                    title = f'{config_name}\nLens {lens_id:06d}'
                
                ax.set_title(title, fontweight='bold', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Image\nNot Found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{config_name}', fontweight='bold')
            
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")
    
    plt.show()

def find_latest_pair_treatment_outputs():
    """Find the latest pair_treatments output directory"""
    outputs_dir = Path('/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs')
    pair_dirs = sorted(outputs_dir.glob('pair_treatments_*'))
    
    if not pair_dirs:
        print("[ERROR] No pair_treatments directories found!")
        return None
    
    latest_dir = pair_dirs[-1]
    print(f"[INFO] Using latest pair_treatments: {latest_dir.name}")
    
    return {
        'sie_sie': latest_dir / 'sie_sie',
        'nfw_nfw': latest_dir / 'nfw_nfw',
        'shear_only': latest_dir / 'shear_only'
    }

if __name__ == '__main__':
    print("=" * 70)
    print("MASS PROFILE COMPARISON FIGURE GENERATOR")
    print("=" * 70)
    
    # Find latest pair treatment outputs
    output_dirs = find_latest_pair_treatment_outputs()
    if output_dirs is None:
        exit(1)
    
    # Verify directories exist and have images
    print("\n[INFO] Checking available images in each configuration...")
    for config_name, config_path in output_dirs.items():
        jpg_dir = config_path / 'jpg_rgb'
        if jpg_dir.exists():
            n_images = len(list(jpg_dir.glob('cosmos_lens_*.jpg')))
            print(f"  {config_name}: {n_images} images found")
        else:
            print(f"  {config_name}: NOT FOUND")
    
    # Create comparison figures for different stages
    print("\n[INFO] Creating comparison figures...")
    
    # Compare first 3 lenses, final stage
    lens_ids = [0, 1, 2]
    
    stages = ['final', 'lens_only', 'lens_sources', 'sources_only', 'field_only']
    
    for stage in stages:
        print(f"\n  Creating {stage.upper()} stage comparison...")
        
        save_path = Path('/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs') / \
                   f'comparison_{stage}_lenses_0-2.png'
        
        try:
            create_comparison_figure(output_dirs, lens_ids, stage=stage, save_path=save_path)
        except Exception as e:
            print(f"  [ERROR] Failed to create {stage} comparison: {e}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
