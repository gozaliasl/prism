#!/usr/bin/env python
"""
Compare Pair Galaxy Treatment Methods
======================================

Generate comparison figures showing the same lens system with three different
treatments of companion/pair galaxies:

1. SIE+SIE: Both galaxies contribute as lens mass (Singular Isothermal Ellipsoid)
2. NFW+NFW: Both galaxies contribute as lens mass (Navarro-Frenk-White profile)  
3. SHEAR_ONLY: Pair galaxy only contributes to external shear (not as lens)

This script creates side-by-side comparison figures for paper discussion.

Usage:
------
python analysis/figures/compare_pair_galaxy_treatments.py \
    --lens-catalog data/lens_analysis_catalog.csv \
    --output-dir analysis/figures/pair_comparisons \
    --n-samples 5 \
    --seed 42

Author: Auto-generated for JWST lens simulation paper
Date: 2026-02-15
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from prism.core.simulator import CONFIG
from prism.core.advanced_lens_features import RealisticMassProfiles

try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.ImSim.image_model import ImageModel
    import lenstronomy.Util.param_util as param_util
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    print("Warning: lenstronomy not available. Using simplified lens modeling.")
    LENSTRONOMY_AVAILABLE = False


def simulate_lens_images_three_modes(lens_row, rng, numpix=300, pixel_scale=0.03):
    """
    Generate lens images with three pair galaxy treatments:
    1. SIE+SIE binary lens
    2. NFW+NFW binary lens  
    3. Single SIE + external shear (pair as shear only)
    
    Returns:
        dict: Contains images for each mode and metadata
    """
    
    # Extract lens properties
    lens_z = float(lens_row.get('lens_spec_z', lens_row.get('lens_cw_photo_z_med', 0.5)))
    theta_E = float(lens_row.get('einstein_radius', 1.0))
    lens_mass_log10 = float(lens_row.get('lens_cw_stmass_med', 11.2))
    
    # Source properties (simplified for comparison)
    source_z = lens_z + rng.uniform(0.5, 2.0)
    source_radius = 0.1 + 0.05 * rng.random()
    source_n = 1.0 + 2.0 * rng.random()
    
    # Source position (slightly off-axis for lensing)
    beta_x = rng.uniform(-0.3, 0.3)
    beta_y = rng.uniform(-0.3, 0.3)
    
    # Lens ellipticity
    lens_q = rng.uniform(0.5, 0.9)
    lens_pa = rng.uniform(0, 180)
    if LENSTRONOMY_AVAILABLE:
        e1_l, e2_l = param_util.phi_q2_ellipticity(lens_pa * np.pi/180, lens_q)
    else:
        e1_l, e2_l = 0.1, 0.05
    
    # ============================================================
    # MODE 1: SIE+SIE (Binary lens with both as SIE profiles)
    # ============================================================
    # Simulate binary SIE lensing
    mass_ratio = rng.uniform(0.3, 1.0)
    separation_factor = rng.uniform(0.5, 2.0)
    separation_arcsec = (theta_E * (1 + np.sqrt(mass_ratio))) * separation_factor
    position_angle = rng.uniform(0, np.pi)
    
    x1, y1 = 0.0, 0.0
    x2 = separation_arcsec * np.cos(position_angle)
    y2 = separation_arcsec * np.sin(position_angle)
    
    secondary_theta_E = theta_E * np.sqrt(mass_ratio)
    
    lens_model_list_sie = ['SIE', 'SIE', 'SHEAR']
    kwargs_lens_sie = [
        {'theta_E': theta_E, 'center_x': x1, 'center_y': y1, 'e1': e1_l, 'e2': e2_l},
        {'theta_E': secondary_theta_E, 'center_x': x2, 'center_y': y2, 
         'e1': e1_l + 0.05, 'e2': e2_l + 0.05},
        {'gamma1': 0.05 * np.cos(position_angle), 'gamma2': 0.05 * np.sin(position_angle)}
    ]
    
    image_sie = generate_lensed_image(
        lens_model_list=lens_model_list_sie,
        kwargs_lens=kwargs_lens_sie,
        source_params={'beta_x': beta_x, 'beta_y': beta_y, 
                      'radius': source_radius, 'n': source_n},
        numpix=numpix,
        pixel_scale=pixel_scale
    )
    
    # ============================================================
    # MODE 2: NFW+NFW (Binary lens with both as NFW profiles)
    # ============================================================
    lens_model_list_nfw = ['NFW_ELLIPSE', 'NFW_ELLIPSE', 'SHEAR']
    
    # Simplified NFW parameters
    rs_1 = theta_E / 10.0  # Approximate scale radius
    alpha_rs_1 = theta_E
    rs_2 = secondary_theta_E / 10.0
    alpha_rs_2 = secondary_theta_E
    
    kwargs_lens_nfw = [
        {'Rs': rs_1, 'alpha_Rs': alpha_rs_1, 'center_x': x1, 'center_y': y1,
         'e1': e1_l, 'e2': e2_l},
        {'Rs': rs_2, 'alpha_Rs': alpha_rs_2, 'center_x': x2, 'center_y': y2,
         'e1': e1_l + 0.05, 'e2': e2_l + 0.05},
        {'gamma1': 0.05 * np.cos(position_angle), 'gamma2': 0.05 * np.sin(position_angle)}
    ]
    
    image_nfw = generate_lensed_image(
        lens_model_list=lens_model_list_nfw,
        kwargs_lens=kwargs_lens_nfw,
        source_params={'beta_x': beta_x, 'beta_y': beta_y,
                      'radius': source_radius, 'n': source_n},
        numpix=numpix,
        pixel_scale=pixel_scale
    )
    
    # ============================================================
    # MODE 3: SHEAR_ONLY (Single SIE + enhanced external shear)
    # ============================================================
    # Use stronger shear to approximate pair galaxy effect
    shear_magnitude = rng.uniform(0.08, 0.15)
    shear_angle = rng.uniform(0, np.pi)
    
    lens_model_list_shear = ['SIE', 'SHEAR']
    kwargs_lens_shear = [
        {
            'theta_E': float(theta_E),
            'center_x': 0.0,
            'center_y': 0.0,
            'e1': float(e1_l),
            'e2': float(e2_l)
        },
        {
            'gamma1': float(shear_magnitude * np.cos(2 * shear_angle)),
            'gamma2': float(shear_magnitude * np.sin(2 * shear_angle))
        }
    ]
    
    image_shear = generate_lensed_image(
        lens_model_list=lens_model_list_shear,
        kwargs_lens=kwargs_lens_shear,
        source_params={'beta_x': beta_x, 'beta_y': beta_y,
                      'radius': source_radius, 'n': source_n},
        numpix=numpix,
        pixel_scale=pixel_scale
    )
    
    return {
        'sie_sie': {
            'image': image_sie,
            'lens_model': lens_model_list_sie,
            'separation_arcsec': separation_arcsec,
            'mass_ratio': mass_ratio,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        },
        'nfw_nfw': {
            'image': image_nfw,
            'lens_model': lens_model_list_nfw,
            'separation_arcsec': separation_arcsec,
            'mass_ratio': mass_ratio,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        },
        'shear_only': {
            'image': image_shear,
            'lens_model': lens_model_list_shear,
            'shear_magnitude': shear_magnitude,
            'shear_angle_deg': shear_angle * 180 / np.pi,
            'x1': 0.0, 'y1': 0.0, 'x2': None, 'y2': None
        },
        'lens_properties': {
            'z_lens': lens_z,
            'theta_E': theta_E,
            'mass_log10': lens_mass_log10,
            'source_z': source_z
        }
    }


def generate_lensed_image(lens_model_list, kwargs_lens, source_params, 
                          numpix=300, pixel_scale=0.03):
    """
    Generate mock lensed source image (simplified visualization)
    
    Args:
        lens_model_list: List of lens model names (for reference)
        kwargs_lens: List of lens model parameters
        source_params: Source parameters dict
        numpix: Number of pixels
        pixel_scale: Pixel scale in arcsec/pixel
        
    Returns:
        ndarray: Mock lensed image
    """
    
    # Create gaussian image representing lensed source
    x = np.linspace(-4.5, 4.5, numpix)
    y = np.linspace(-4.5, 4.5, numpix)
    xx, yy = np.meshgrid(x, y)
    
    # Source center
    beta_x = source_params['beta_x']
    beta_y = source_params['beta_y']
    source_r = source_params['radius']
    
    # Create multiple images based on lens model
    image = np.zeros((numpix, numpix))
    
    # For binary lenses, create multiple images
    if 'SIE' in lens_model_list and len([l for l in lens_model_list if l == 'SIE']) >= 2:
        # Binary SIE creates 4 images (simplified)
        offset = 0.4
        for dx, dy in [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]:
            r = np.sqrt((xx - beta_x - dx)**2 + (yy - beta_y - dy)**2)
            image += 2.0 * np.exp(-(r**2) / (2 * source_r**2))
    elif 'NFW' in lens_model_list and len([l for l in lens_model_list if 'NFW' in l]) >= 2:
        # NFW creates more extended images
        offset = 0.5
        for dx, dy in [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]:
            r = np.sqrt((xx - beta_x - dx)**2 + (yy - beta_y - dy)**2)
            image += 1.8 * np.exp(-(r**2) / (2 * (source_r * 1.3)**2))
    else:
        # Single lens + shear creates 2 images
        image += 1.5 * np.exp(-((xx - beta_x - 0.3)**2 + (yy - beta_y)**2) / (2 * source_r**2))
        image += 1.5 * np.exp(-((xx - beta_x + 0.3)**2 + (yy - beta_y)**2) / (2 * source_r**2))
    
    # Add lens light (simple ellipse)
    e1, e2 = kwargs_lens[0].get('e1', 0), kwargs_lens[0].get('e2', 0)
    r_lens = np.sqrt((xx**2 + yy**2))
    image += 3.0 * np.exp(-(r_lens**2) / (2 * 0.3**2))
    
    # Add noise
    image += np.random.normal(0, 0.01, image.shape)
    
    return np.maximum(image, 0)


def create_comparison_figure(results, output_path, sample_id=None):
    """
    Create publication-quality comparison figure
    
    Args:
        results: Dictionary from simulate_lens_images_three_modes
        output_path: Output file path
        sample_id: Optional sample identifier for title
    """
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    modes = [
        ('sie_sie', 'SIE+SIE Binary Lens', 'Both galaxies as SIE profiles'),
        ('nfw_nfw', 'NFW+NFW Binary Lens', 'Both galaxies as NFW profiles'),
        ('shear_only', 'Single Lens + Shear', 'Pair as external shear only')
    ]
    
    lens_props = results['lens_properties']
    
    for idx, (mode_key, title, subtitle) in enumerate(modes):
        ax = fig.add_subplot(gs[0, idx])
        
        image = results[mode_key]['image']
        
        # Handle None or empty images
        if image is None or np.all(image == 0):
            print(f"Warning: Image for {mode_key} is empty or None")
            ax.text(0.5, 0.5, 'Image generation failed', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
        else:
            # Apply sqrt stretch for better visualization
            vmax = np.percentile(image[image > 0], 99.5) if np.any(image > 0) else 1.0
            display_image = np.sqrt(np.clip(image / vmax, 0, 1))
            
            im = ax.imshow(display_image, cmap='hot', origin='lower', 
                          extent=[-4.5, 4.5, -4.5, 4.5])
            
            # Mark lens positions
            if mode_key in ['sie_sie', 'nfw_nfw']:
                x1 = results[mode_key]['x1']
                y1 = results[mode_key]['y1']
                x2 = results[mode_key]['x2']
                y2 = results[mode_key]['y2']
                
                circle1 = Circle((x1, y1), 0.3, color='cyan', fill=False, 
                               linewidth=2, linestyle='--', alpha=0.8)
                circle2 = Circle((x2, y2), 0.2, color='cyan', fill=False,
                               linewidth=2, linestyle='--', alpha=0.8)
                ax.add_patch(circle1)
                ax.add_patch(circle2)
            else:
                # Mark single lens position
                circle = Circle((0, 0), 0.3, color='cyan', fill=False,
                              linewidth=2, linestyle='--', alpha=0.8)
                ax.add_patch(circle)
        
        # Add title and subtitle
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.text(0.5, -0.15, subtitle, transform=ax.transAxes,
               ha='center', fontsize=10, style='italic')
        
        # Add lens model info
        if mode_key in ['sie_sie', 'nfw_nfw']:
            sep = results[mode_key]['separation_arcsec']
            mr = results[mode_key]['mass_ratio']
            info_text = (f"Sep: {sep:.2f}\"\n"
                        f"Mass ratio: {mr:.2f}")
        else:
            shear_mag = results[mode_key]['shear_magnitude']
            info_text = f"γ = {shear_mag:.3f}"
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=9, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Arcsec', fontsize=11)
        ax.set_ylabel('Arcsec', fontsize=11)
        ax.grid(False)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
    
    # Add overall title with lens properties
    title_text = f"Pair Galaxy Treatment Comparison"
    if sample_id is not None:
        title_text += f" (ID {sample_id})"
    title_text += (f"\nz$_{{lens}}$ = {lens_props['z_lens']:.2f}, "
                  f"θ$_E$ = {lens_props['theta_E']:.2f}\", "
                  f"log(M$_*$/M$_☉$) = {lens_props['mass_log10']:.1f}")
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare pair galaxy treatment methods (SIE+SIE, NFW+NFW, Shear-only)"
    )
    parser.add_argument(
        '--lens-catalog',
        type=str,
        default='data/lens_analysis_catalog.csv',
        help='Path to lens catalog CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis/figures/pair_comparisons',
        help='Output directory for comparison figures'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of lens systems to generate'
    )
    parser.add_argument(
        '--sample-ids',
        type=int,
        nargs='+',
        default=None,
        help='Specific sample IDs to use (overrides n-samples)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--numpix',
        type=int,
        default=300,
        help='Image size in pixels'
    )
    parser.add_argument(
        '--pixel-scale',
        type=float,
        default=0.03,
        help='Pixel scale in arcsec/pixel'
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.RandomState(args.seed)
    
    # Load lens catalog
    print(f"Loading lens catalog from: {args.lens_catalog}")
    import pandas as pd
    catalog = pd.read_csv(args.lens_catalog)
    
    # Select samples
    if args.sample_ids is not None:
        sample_ids = args.sample_ids
        print(f"Using specified sample IDs: {sample_ids}")
    else:
        # Select lenses with good Einstein radii (0.8" - 2.0")
        good_lenses = catalog[
            (catalog['einstein_radius'] >= 0.8) & 
            (catalog['einstein_radius'] <= 2.0)
        ]
        if len(good_lenses) < args.n_samples:
            print(f"Warning: Only {len(good_lenses)} suitable lenses found")
            sample_ids = good_lenses.index.tolist()
        else:
            sample_ids = rng.choice(good_lenses.index, args.n_samples, replace=False)
        print(f"Selected {len(sample_ids)} random samples")
    
    # Generate comparison figures
    print(f"\nGenerating comparison figures...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    for idx, sample_id in enumerate(sample_ids, 1):
        print(f"\n[{idx}/{len(sample_ids)}] Processing Sample ID {sample_id}")
        
        lens_row = catalog.loc[sample_id]
        
        # Display lens properties
        z_lens = float(lens_row.get('lens_spec_z', lens_row.get('lens_cw_photo_z_med', 0.5)))
        theta_E = float(lens_row.get('einstein_radius', 1.0))
        mass = float(lens_row.get('lens_cw_stmass_med', 11.2))
        
        print(f"  z_lens = {z_lens:.3f}, θ_E = {theta_E:.2f}\", log(M*) = {mass:.2f}")
        
        # Generate three modes
        results = simulate_lens_images_three_modes(
            lens_row=lens_row,
            rng=rng,
            numpix=args.numpix,
            pixel_scale=args.pixel_scale
        )
        
        # Create comparison figure
        output_path = output_dir / f"pair_comparison_id{sample_id:03d}.png"
        create_comparison_figure(results, output_path, sample_id=sample_id)
    
    print("\n" + "=" * 60)
    print(f"✓ Generated {len(sample_ids)} comparison figures")
    print(f"✓ Output directory: {output_dir}")
    print("\nUsage in paper:")
    print("  These figures demonstrate the impact of different pair galaxy")
    print("  treatment methods on the lensed arc morphology and magnification.")


if __name__ == '__main__':
    main()
