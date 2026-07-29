#!/usr/bin/env python3
"""
Create a demo figure showing a time delay system with configurable epochs.

Layout:
- 4 rows (one per epoch)
- Column 1: Light curve with epoch marked
- Columns 2-5: Individual band images (F115W, F150W, F277W, F444W)
- Column 6: RGB composite image
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
import yaml

# Add src to path (project root/src)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.lensing.time_delay_integration import (
    generate_light_curve_for_source,
    apply_time_delay_to_source_magnitude
)

def load_time_delay_data(output_dir, lens_id):
    """Load time delay data for a specific lens."""
    output_path = Path(output_dir)
    
    # Find time delay catalog
    td_catalog = output_path / "time_delay_catalog.csv"
    if not td_catalog.exists():
        # Try in batch directory
        td_catalog = output_path / "batchA_empirical" / "time_delay_catalog.csv"
    
    if td_catalog.exists():
        df = pd.read_csv(td_catalog)
        lens_data = df[df['lens_id'] == lens_id]
        if len(lens_data) > 0:
            return lens_data.iloc[0].to_dict()
    
    return None

def load_epoch_images(output_dir, lens_id, max_epochs=20):
    """Load images for all available epochs."""
    output_path = Path(output_dir)
    
    # Try different possible locations
    npy_dir = output_path / "npy"
    if not npy_dir.exists():
        npy_dir = output_path / "batchA_empirical" / "npy"
    
    rgb_dir = output_path / "jpg_rgb"
    if not rgb_dir.exists():
        rgb_dir = output_path / "batchA_empirical" / "jpg_rgb"
    
    images_4band = {}
    images_rgb = {}
    
    # Find all available epochs
    for epoch in range(max_epochs):
        epoch_str = f"epoch{epoch:02d}"
        
        # Load 4-band stack
        npy_file = npy_dir / f"cosmos_lens_{lens_id:06d}_{epoch_str}.npy"
        if npy_file.exists():
            stack = np.load(npy_file)
            # Stack is (4, H, W) for F115W, F150W, F277W, F444W
            images_4band[epoch] = {
                'F115W': stack[0],
                'F150W': stack[1],
                'F277W': stack[2],
                'F444W': stack[3]
            }
        
        # Load RGB image
        rgb_file = rgb_dir / f"cosmos_lens_{lens_id:06d}_{epoch_str}.jpg"
        if rgb_file.exists():
            rgb_img = Image.open(rgb_file)
            images_rgb[epoch] = np.array(rgb_img) / 255.0
    
    return images_4band, images_rgb

def select_phase_epoch(light_curve, time_array, phase):
    """Select epoch time for a given phase (max/min) of the light curve."""
    finite_mask = np.isfinite(light_curve)
    if not np.any(finite_mask):
        return time_array[0]

    if phase == 'max':
        idx = np.argmin(light_curve[finite_mask])  # brightest (lowest mag)
        return time_array[np.where(finite_mask)[0][idx]]
    if phase == 'min':
        idx = np.argmax(light_curve[finite_mask])  # faintest (highest mag)
        return time_array[np.where(finite_mask)[0][idx]]

    return time_array[0]

def create_time_delay_demo_figure(output_dir, lens_id, source_type=None, n_epochs=4):
    """Create demo figure for time delay system."""
    
    # Load time delay metadata
    td_data = load_time_delay_data(output_dir, lens_id)
    if td_data is None:
        print(f"Error: No time delay catalog found for lens {lens_id}")
        return None
    
    # Parse time delays and get source type from simulation output
    import ast
    time_delays = np.array(ast.literal_eval(td_data.get('time_delays_days', '[0.0, 5.0, 10.0, 15.0]')))
    base_magnitude = 20.5
    # Get source type: prioritize command-line argument if provided, otherwise use catalog value
    if source_type is None:
        source_type = td_data.get('source_type', 'quasar')
    # source_type now uses command-line value if provided, or catalog value if not
    # Get redshifts
    lens_z = td_data.get('lens_redshift', None)
    source_z = td_data.get('source_redshift', None)
    
    # Generate light curve first to determine epoch times
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    rng = np.random.default_rng(42)
    time_array = np.linspace(0, 200, 1000)  # Fixed time range for light curve
    
    # Get redshift from time delay data if available
    source_z = td_data.get('source_redshift', 2.0) if td_data else 2.0
    black_hole_mass = None  # Could be extracted from catalog if available
    
    light_curve = generate_light_curve_for_source(
        time_array, source_type, config, rng, base_magnitude,
        redshift=source_z, black_hole_mass=black_hole_mass
    )
    
    # Select epochs where magnitude differences are most visible
    # Strategy: For 2 epochs use min/max; otherwise space by magnitude range
    finite_mask = np.isfinite(light_curve)
    if np.any(finite_mask):
        mag_min = np.min(light_curve[finite_mask])  # Brightest (lowest magnitude)
        mag_max = np.max(light_curve[finite_mask])  # Faintest (highest magnitude)
    else:
        mag_min = base_magnitude
        mag_max = base_magnitude
    mag_range = mag_max - mag_min
    if not np.isfinite(mag_range) or mag_range == 0:
        mag_range = 1.0
    
    if n_epochs <= 1:
        target_mags = [mag_min]
    elif n_epochs == 2:
        target_mags = [mag_min, mag_max]
    else:
        fractions = np.linspace(0.0, 1.0, n_epochs)
        target_mags = list(mag_min + fractions * mag_range)
    
    # Find closest times in light curve for each target magnitude
    selected_epoch_times = []
    for target_mag in target_mags:
        # Find time where light curve is closest to target magnitude
        idx = np.argmin(np.abs(light_curve - target_mag))
        selected_epoch_times.append(time_array[idx])
    
    selected_epoch_times = np.array(selected_epoch_times)
    # Sort by time to ensure chronological order
    sort_idx = np.argsort(selected_epoch_times)
    selected_epoch_times = selected_epoch_times[sort_idx]
    
    # Now load images - we need to find which saved epochs are closest to our selected times
    # Load all available epochs first
    all_images_4band, all_images_rgb = load_epoch_images(output_dir, lens_id, 20)  # Load up to 20 epochs
    
    if not all_images_4band:
        print(f"Error: No images found for lens {lens_id}")
        return None
    
    # Find which saved epochs are closest to our selected times
    # We'll need to load the time delay catalog to get actual epoch times
    td_data_reload = load_time_delay_data(output_dir, lens_id)
    if td_data_reload is not None:
        import ast
        try:
            saved_epoch_times_str = td_data_reload.get('epoch_times_days', '[]')
            if saved_epoch_times_str and saved_epoch_times_str != '[]':
                saved_epoch_times = np.array(ast.literal_eval(saved_epoch_times_str))
            else:
                saved_epoch_times = None
        except:
            saved_epoch_times = None
    else:
        saved_epoch_times = None
    
    # If we don't have saved epoch times, use the selected times directly
    # and match to available epoch indices
    if saved_epoch_times is None or len(saved_epoch_times) == 0:
        # Use selected times and match to available epochs by index
        available_epochs = sorted(list(all_images_4band.keys()))
        if len(available_epochs) >= n_epochs:
            # Use first n_epochs available epochs
            matched_epoch_indices = available_epochs[:n_epochs]
            epoch_times = selected_epoch_times
        else:
            # Use all available epochs
            matched_epoch_indices = available_epochs
            epoch_times = selected_epoch_times[:len(available_epochs)]
    else:
        # Match selected times to saved epochs
        matched_epoch_indices = []
        for target_time in selected_epoch_times:
            # Find closest saved epoch
            closest_idx = np.argmin(np.abs(saved_epoch_times - target_time))
            matched_epoch_indices.append(closest_idx)
        epoch_times = saved_epoch_times[matched_epoch_indices] if len(matched_epoch_indices) > 0 else selected_epoch_times
    
    # Map matched indices to actual epoch numbers and load images
    images_4band = {}
    images_rgb = {}
    for display_idx, epoch_idx in enumerate(matched_epoch_indices):
        if epoch_idx in all_images_4band:
            images_4band[display_idx] = all_images_4band[epoch_idx]
        if epoch_idx in all_images_rgb:
            images_rgb[display_idx] = all_images_rgb[epoch_idx]
    
    # Use the matched epoch times when available (aligns with displayed images)
    # Fallback to selected_epoch_times if no matched times exist
    if len(epoch_times) == 0:
        epoch_times = selected_epoch_times
    
    # Calculate consistent y-axis limits for all light curve panels
    if np.any(finite_mask):
        mag_min_global = np.min(light_curve[finite_mask])
        mag_max_global = np.max(light_curve[finite_mask])
    else:
        mag_min_global = base_magnitude
        mag_max_global = base_magnitude
    mag_range = mag_max_global - mag_min_global
    if not np.isfinite(mag_range) or mag_range == 0:
        mag_range = 1.0
    y_min = mag_max_global + 0.3 * mag_range  # Add padding at top (fainter)
    y_max = mag_min_global - 0.3 * mag_range  # Add padding at bottom (brighter)
    
    # Create figure: 4 rows (epochs) x 2 columns (light curve + RGB composite)
    # Ensure equal heights for light curve and RGB in each row
    fig = plt.figure(figsize=(12, 10))
    # Calculate total available height for panels
    top_margin = 0.95
    bottom_margin = 0.06
    total_height = top_margin - bottom_margin
    row_height = total_height / n_epochs
    
    # Calculate positions for each row (from top to bottom)
    row_positions = []
    for i in range(n_epochs):
        y0 = top_margin - (i + 1) * row_height
        row_positions.append(y0)
    
    for epoch_idx in range(n_epochs):
        obs_time = epoch_times[epoch_idx]
        
        # Calculate exact positions for this row
        y0 = row_positions[epoch_idx]
        left_lc = 0.08
        width_lc = 0.35  # Light curve width
        left_rgb = left_lc + width_lc + 0.01  # Small gap
        width_rgb = 0.56  # RGB width
        
        # Column 1: Light curve
        ax_lc = fig.add_axes([left_lc, y0, width_lc, row_height])
        
        # Column 2: RGB composite (includes all 4 bands)
        ax_rgb = fig.add_axes([left_rgb, y0, width_rgb, row_height])
        
        # Set up RGB image first to get its dimensions and position
        if epoch_idx in images_rgb:
            img_shape = images_rgb[epoch_idx].shape
            img_height, img_width = img_shape[0], img_shape[1]
            
            # Display image with correct aspect ratio (not stretched)
            # Use origin='upper' (default for images) to match PIL/Image convention
            # where first row is at the top
            ax_rgb.imshow(images_rgb[epoch_idx], origin='upper', aspect='equal', 
                         extent=[0, img_width, 0, img_height])
        else:
            ax_rgb.text(0.5, 0.5, 'No RGB data', ha='center', va='center', 
                       transform=ax_rgb.transAxes, fontsize=8)
        
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        ax_rgb.set_aspect('equal')  # Keep image aspect ratio correct (not stretched)
        
        # Plot full light curve
        ax_lc.plot(time_array, light_curve, 'k-', alpha=0.4, linewidth=1.5, label='Light curve', zorder=1)
        
        # Mark all epochs
        for i, t in enumerate(epoch_times):
            mag_at_t = np.interp(t, time_array, light_curve)
            if i == epoch_idx:
                ax_lc.plot(t, mag_at_t, 'ro', markersize=10, zorder=3)
                ax_lc.axvline(t, color='r', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
            else:
                ax_lc.plot(t, mag_at_t, 'ko', markersize=5, alpha=0.3, zorder=2)
        
        # Calculate magnitude at this epoch (with time delays)
        image_magnitudes = apply_time_delay_to_source_magnitude(
            base_magnitude, light_curve, time_array,
            time_delays, obs_time
        )
        avg_mag = np.mean(image_magnitudes)
        mag_at_obs = np.interp(obs_time, time_array, light_curve)
        
        # Set light curve limits and labels
        # Use consistent y-axis limits across all panels (calculated above)
        ax_lc.set_xlim(0, max(time_array))
        ax_lc.set_ylim(y_min, y_max)  # Inverted for magnitude (higher y = fainter)
        
        # Only show xlabel and ticks on bottom panel
        if epoch_idx == n_epochs - 1:
            ax_lc.set_xlabel('Time (days)', fontsize=9)
        else:
            ax_lc.set_xticklabels([])  # Remove x-axis tick labels for upper panels
            ax_lc.set_xlabel('')  # Remove x-axis label for upper panels
        
        # Show y-axis label only on first panel, but keep tick labels on all panels
        if epoch_idx == 0:
            ax_lc.set_ylabel('Magnitude', fontsize=9)
        else:
            ax_lc.set_ylabel('')  # Remove y-axis label text, but keep tick labels
        
        # Add info inside the plot
        info_text = f't={obs_time:.1f} d, mag={mag_at_obs:.2f}'
        if lens_z is not None and source_z is not None:
            info_text += f'\nz_l={lens_z:.2f}, z_s={source_z:.2f}'
        
        ax_lc.text(0.02, 0.98, info_text, 
                  transform=ax_lc.transAxes, fontsize=9, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax_lc.grid(True, alpha=0.2, linestyle=':')
        
        # Remove top and bottom spines for upper panels to connect them visually
        if epoch_idx < n_epochs - 1:
            ax_lc.spines['bottom'].set_visible(False)
            ax_rgb.spines['bottom'].set_visible(False)
            ax_lc.tick_params(bottom=False)
            ax_rgb.tick_params(bottom=False)
        if epoch_idx > 0:
            ax_lc.spines['top'].set_visible(False)
            ax_rgb.spines['top'].set_visible(False)
            ax_lc.tick_params(top=False)
            ax_rgb.tick_params(top=False)
    
    # Add overall title (simplified)
    fig.suptitle(f'Time Delay System: Lens {lens_id:06d} ({source_type.capitalize()})',
                fontsize=12, fontweight='bold', y=0.98)
    
    return fig

def create_time_delay_combo_figure(output_dir, lens_ids, phase='max'):
    """Create a single figure with AGN, quasar, and supernova rows for one phase."""
    source_order = [
        ('agn', lens_ids.get('agn')),
        ('quasar', lens_ids.get('quasar')),
        ('supernova', lens_ids.get('supernova')),
    ]

    # Basic figure setup
    fig = plt.figure(figsize=(12, 7))
    top_margin = 0.93
    bottom_margin = 0.08
    total_height = top_margin - bottom_margin
    n_rows = len(source_order)
    row_height = total_height / n_rows

    # Load config once
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(42)
    time_array = np.linspace(0, 200, 1000)

    for row_idx, (source_type, lens_id) in enumerate(source_order):
        if lens_id is None:
            continue

        td_data = load_time_delay_data(output_dir, lens_id)
        lens_z = td_data.get('lens_redshift', None) if td_data else None
        source_z = td_data.get('source_redshift', 2.0) if td_data else 2.0

        light_curve = generate_light_curve_for_source(
            time_array, source_type, config, rng, 20.5,
            redshift=source_z, black_hole_mass=None
        )

        obs_time = select_phase_epoch(light_curve, time_array, phase)

        # Load images and pick closest saved epoch with available RGB data
        all_images_4band, all_images_rgb = load_epoch_images(output_dir, lens_id, 20)
        if not all_images_rgb:
            images_rgb = None
        else:
            td_data_reload = load_time_delay_data(output_dir, lens_id)
            saved_epoch_times = None
            if td_data_reload is not None:
                try:
                    import ast
                    saved_epoch_times_str = td_data_reload.get('epoch_times_days', '[]')
                    if saved_epoch_times_str and saved_epoch_times_str != '[]':
                        saved_epoch_times = np.array(ast.literal_eval(saved_epoch_times_str))
                except Exception:
                    saved_epoch_times = None

            available_epochs = sorted(list(all_images_rgb.keys()))
            if not available_epochs:
                images_rgb = None
            elif saved_epoch_times is None or len(saved_epoch_times) == 0:
                epoch_idx = available_epochs[0]
                images_rgb = all_images_rgb.get(epoch_idx)
            else:
                # Choose closest available epoch to desired obs_time
                candidate_times = saved_epoch_times[available_epochs]
                closest_pos = int(np.argmin(np.abs(candidate_times - obs_time)))
                epoch_idx = available_epochs[closest_pos]
                images_rgb = all_images_rgb.get(epoch_idx)

        # Axis positions
        y0 = top_margin - (row_idx + 1) * row_height
        left_lc = 0.08
        width_lc = 0.35
        left_rgb = left_lc + width_lc + 0.01
        width_rgb = 0.56

        ax_lc = fig.add_axes([left_lc, y0, width_lc, row_height])
        ax_rgb = fig.add_axes([left_rgb, y0, width_rgb, row_height])

        # Light curve plot
        ax_lc.plot(time_array, light_curve, 'k-', alpha=0.4, linewidth=1.5, zorder=1)
        mag_at_obs = np.interp(obs_time, time_array, light_curve)
        ax_lc.plot(obs_time, mag_at_obs, 'ro', markersize=9, zorder=3)
        ax_lc.axvline(obs_time, color='r', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)

        finite_mask = np.isfinite(light_curve)
        if np.any(finite_mask):
            mag_min_global = np.min(light_curve[finite_mask])
            mag_max_global = np.max(light_curve[finite_mask])
        else:
            mag_min_global = 20.5
            mag_max_global = 20.5
        mag_range = mag_max_global - mag_min_global
        if not np.isfinite(mag_range) or mag_range == 0:
            mag_range = 1.0
        y_min = mag_max_global + 0.3 * mag_range
        y_max = mag_min_global - 0.3 * mag_range
        ax_lc.set_xlim(0, max(time_array))
        ax_lc.set_ylim(y_min, y_max)

        if row_idx == n_rows - 1:
            ax_lc.set_xlabel('Time (days)', fontsize=9)
        else:
            ax_lc.set_xticklabels([])
            ax_lc.set_xlabel('')

        if row_idx == 0:
            ax_lc.set_ylabel('Magnitude', fontsize=9)
        else:
            ax_lc.set_ylabel('')

        info_text = f'{source_type.upper()} (Lens {lens_id:06d})\n{phase.upper()} t={obs_time:.1f} d'
        if lens_z is not None and source_z is not None:
            info_text += f'\nz_l={lens_z:.2f}, z_s={source_z:.2f}'
        ax_lc.grid(True, alpha=0.2, linestyle=':')

        # RGB image
        if images_rgb is not None:
            img_height, img_width = images_rgb.shape[0], images_rgb.shape[1]
            ax_rgb.imshow(images_rgb, origin='upper', aspect='equal',
                         extent=[0, img_width, 0, img_height])
        else:
            ax_rgb.text(0.5, 0.5, 'No RGB data', ha='center', va='center',
                        transform=ax_rgb.transAxes, fontsize=8)
        # Label on light curve panel (bottom) with yellow background
        ax_lc.text(0.02, 0.02, info_text,
              transform=ax_lc.transAxes, fontsize=9,
              verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        ax_rgb.set_aspect('equal')

        if row_idx < n_rows - 1:
            ax_lc.spines['bottom'].set_visible(False)
            ax_rgb.spines['bottom'].set_visible(False)
            ax_lc.tick_params(bottom=False)
            ax_rgb.tick_params(bottom=False)
        if row_idx > 0:
            ax_lc.spines['top'].set_visible(False)
            ax_rgb.spines['top'].set_visible(False)
            ax_lc.tick_params(top=False)
            ax_rgb.tick_params(top=False)

    fig.suptitle(f'Time Delay System (Phase: {phase.upper()})',
                fontsize=12, fontweight='bold', y=0.98)

    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create time delay demo figure')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory containing time delay images')
    parser.add_argument('--lens-id', type=int, default=58,
                       help='Lens ID to use for demo (default: 58)')
    parser.add_argument('--source-type', type=str, default=None,
                       choices=['quasar', 'supernova', 'agn', None],
                       help='Source type (default: read from simulation output)')
    parser.add_argument('--n-epochs', type=int, default=4,
                       help='Number of epochs to show (default: 4)')
    parser.add_argument('--combo', action='store_true',
                       help='Create combined AGN/Quasar/SN figure (single phase)')
    parser.add_argument('--phase', type=str, default='max', choices=['max', 'min'],
                       help='Phase for combined figure (max or min)')
    parser.add_argument('--agn-id', type=int, default=None,
                       help='Lens ID for AGN row in combined figure')
    parser.add_argument('--quasar-id', type=int, default=None,
                       help='Lens ID for Quasar row in combined figure')
    parser.add_argument('--supernova-id', type=int, default=None,
                       help='Lens ID for Supernova row in combined figure')
    parser.add_argument('--output-fig', type=str, default=None,
                       help='Output figure filename (default: time_delay_demo_lens{id}.png)')
    
    args = parser.parse_args()
    
    # Create figure
    if args.combo:
        fig = create_time_delay_combo_figure(
            args.output_dir,
            {
                'agn': args.agn_id,
                'quasar': args.quasar_id,
                'supernova': args.supernova_id,
            },
            phase=args.phase
        )
    else:
        fig = create_time_delay_demo_figure(
            args.output_dir, args.lens_id, args.source_type, args.n_epochs
        )
    
    if fig is None:
        print("Failed to create figure")
        return 1
    
    # Save figure
    if args.output_fig is None:
        if args.combo:
            output_fig = f"time_delay_demo_combo_{args.phase}.png"
        else:
            output_fig = f"time_delay_demo_lens{args.lens_id:06d}.png"
    else:
        output_fig = args.output_fig
    
    fig.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_fig}")
    
    plt.close(fig)
    return 0

if __name__ == '__main__':
    sys.exit(main())

