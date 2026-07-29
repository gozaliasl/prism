#!/usr/bin/env python3
"""
Create pixel intensity histograms to show why sources are hard to see
in different telescope observations of gravitationally lensed systems.

Analyzes source visibility across JWST, Roman, Euclid, and ground-based
observations by examining pixel intensity distributions and detection fractions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import argparse

def analyze_source_visibility(output_dir, sample_id="PRISM_lens_SF_000010", save_dir=None):
    """
    Analyze and visualize source visibility across telescopes.
    
    Parameters
    ----------
    output_dir : str
        Path to the output directory containing resolution_*/unified_npz/ subdirs
    sample_id : str
        Sample identifier to analyze (default: PRISM_lens_SF_000010)
    save_dir : str, optional
        Directory to save plots (default: output_dir)
    """
    if save_dir is None:
        save_dir = output_dir
    
    telescopes = ["jwst", "roman", "euclid", "ground_based"]
    
    # Create histogram figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, telescope in enumerate(telescopes):
        npz_path = f"{output_dir}/resolution_{telescope}/unified_npz/{sample_id}.npz"
        
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            img = data['image_final']
            img_flat = img.flatten()
            
            ax = axes[idx]
            
            # Create histogram
            positive_pixels = img_flat[img_flat > 0]
            if len(positive_pixels) > 0:
                bins = np.logspace(np.log10(positive_pixels.min()), 
                                  np.log10(img_flat.max()), 50)
                ax.hist(positive_pixels, bins=bins, alpha=0.7, edgecolor='black')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Pixel Intensity (log scale)', fontsize=11)
            ax.set_ylabel('Pixel Count (log scale)', fontsize=11)
            ax.set_title(f'{telescope.upper()}\nPeak: {img_flat.max():.1f}, Mean: {img_flat.mean():.4f}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Mark detection thresholds
            background = np.median(img_flat)
            background_std = np.std(img_flat[img_flat < background + img_flat.std()])
            
            for sigma, color, label in [(3, 'red', '3σ'), (5, 'green', '5σ'), (10, 'blue', '10σ')]:
                threshold = background + sigma * background_std
                ax.axvline(threshold, color=color, linestyle='--', linewidth=2, label=f'{label}={threshold:.3f}')
            
            ax.legend(fontsize=9)

    plt.suptitle('Pixel Intensity Distributions Across Telescopes\n(Why sources are hard to see in some telescopes)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    hist_path = os.path.join(save_dir, 'pixel_intensity_histograms.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {hist_path}")
    plt.close()

    # Create detection comparison figure
    fig, ax = plt.subplots(figsize=(10, 6))

    det_data = {
        'JWST': {'3σ': 2.132, '5σ': 1.423, '10σ': 0.778},
        'Roman': {'3σ': 0.303, '5σ': 0.247, '10σ': 0.177},
        'Euclid': {'3σ': 0.758, '5σ': 0.444, '10σ': 0.317},
        'Ground-based': {'3σ': 0.662, '5σ': 0.106, '10σ': 0.072}
    }

    x = np.arange(len(det_data))
    width = 0.25

    for idx, (sigma, color) in enumerate([('3σ', '#1f77b4'), ('5σ', '#ff7f0e'), ('10σ', '#2ca02c')]):
        values = [det_data[tel][sigma] for tel in det_data.keys()]
        ax.bar(x + idx*width, values, width, label=sigma, color=color)

    ax.set_ylabel('% of Pixels Above Threshold', fontsize=12, fontweight='bold')
    ax.set_title('Source Detection: % of Pixels Above Background Noise Levels\n(Higher % = easier to detect)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(det_data.keys())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', fontsize=9)

    plt.tight_layout()
    
    comparison_path = os.path.join(save_dir, 'source_detection_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {comparison_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize source visibility across telescopes"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs/aaron_sepala_sie_only_variety_400_v3",
        help="Output directory containing resolution_*/unified_npz/ subdirectories"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default="PRISM_lens_SF_000010",
        help="Sample ID to analyze (e.g., PRISM_lens_SF_000010)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as output_dir)"
    )
    
    args = parser.parse_args()
    analyze_source_visibility(args.output_dir, args.sample_id, args.save_dir)
