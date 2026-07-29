#!/usr/bin/env python3
"""
Create a paper figure showing step-by-step outputs per lens system.

Rows (default order):
1) lens_only
2) sources_only
3) lens_sources
4) field_only
5) final

Each row is shown as an RGB composite. The script auto-selects an
"interesting" lens (highest total flux) unless --sample-id is provided.

Usage:
  python analysis/figures/make_step_figure.py \
      --input-dir outputs/custom_YYYYMMDD_HHMMSS \
      --output figures/step_outputs_example.png

Optional:
  --sample-id 123 (uses cosmos_lens_000123.npz)
  --ncols 1 (default 1)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

STEPS_DEFAULT = ["lens_only", "sources_only", "lens_sources", "field_only", "final"]
BANDS = ["F115W", "F150W", "F277W", "F444W"]


# Color presets inspired by Trilogy noiselums for balanced galaxy colors
COLOR_PRESETS = {
    'trilogy': {'red': 1.0, 'green': 0.9, 'blue': 0.5},      # Trilogy-inspired (R=0.5, G=0.45, B=0.56 inverted)
    'elliptical': {'red': 1.0, 'green': 0.8, 'blue': 0.6},  # For red elliptical galaxies
    'balanced': {'red': 1.0, 'green': 1.0, 'blue': 0.7},    # Moderate blue suppression
    'natural': {'red': 1.0, 'green': 1.0, 'blue': 1.0},     # No correction
}

def _make_rgb_from_stack(stack_4, p=99.5, vmax=None, stretch_mode="percentile", independent_rgb=False, channel_vmax=None, blue_suppress=1.0, green_suppress=1.0):
    """Create an RGB image from a 4-band stack (F115W, F150W, F277W, F444W).
    
    Parameters:
    -----------
    blue_suppress : float
        Factor to suppress blue channel (0.6 recommended for red ellipticals to avoid pink color)
    green_suppress : float
        Factor to suppress green channel (0.7-0.9 range to reduce over-bright appearance)
    """
    f115, f150, f277, f444 = stack_4
    r = f444
    g = 0.5 * (f150 + f277) * green_suppress  # Apply color correction
    b = f115 * blue_suppress  # Apply color correction
    
    if independent_rgb:
        # Normalize each channel independently for better field galaxy visibility
        if channel_vmax is not None:
            # Use provided channel vmax for global stretch
            r_vmax, g_vmax, b_vmax = channel_vmax
        elif stretch_mode == "none":
            r_vmax = g_vmax = b_vmax = 1.0
        else:
            r_vmax = np.percentile(r, p) if np.any(r > 0) else 1.0
            g_vmax = np.percentile(g, p) if np.any(g > 0) else 1.0
            b_vmax = np.percentile(b, p) if np.any(b > 0) else 1.0
        
        r = r / r_vmax if r_vmax > 0 else r
        g = g / g_vmax if g_vmax > 0 else g
        b = b / b_vmax if b_vmax > 0 else b
        
        if stretch_mode == "log":
            r = np.log10(1 + 9 * np.clip(r, 0, None))
            g = np.log10(1 + 9 * np.clip(g, 0, None))
            b = np.log10(1 + 9 * np.clip(b, 0, None))
        elif stretch_mode == "arcsinh":
            r = np.arcsinh(3 * np.clip(r, 0, None)) / np.arcsinh(3)
            g = np.arcsinh(3 * np.clip(g, 0, None)) / np.arcsinh(3)
            b = np.arcsinh(3 * np.clip(b, 0, None)) / np.arcsinh(3)
        elif stretch_mode == "sqrt":
            r = np.sqrt(np.clip(r, 0, 1))
            g = np.sqrt(np.clip(g, 0, 1))
            b = np.sqrt(np.clip(b, 0, 1))
        
        rgb = np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=-1)
    else:
        # Original method: normalize combined RGB
        rgb = np.stack([r, g, b], axis=-1)
        if stretch_mode == "none":
            vmax = 1.0
        elif vmax is None:
            vmax = np.percentile(rgb, p)
        if vmax <= 0:
            return np.zeros_like(rgb, dtype=np.float32)
        rgb = rgb / vmax
        if stretch_mode == "log":
            rgb = np.log10(1 + 9 * np.clip(rgb, 0, None))
        elif stretch_mode == "arcsinh":
            rgb = np.arcsinh(3 * np.clip(rgb, 0, None)) / np.arcsinh(3)
        elif stretch_mode == "sqrt":
            rgb = np.sqrt(np.clip(rgb, 0, 1))
        rgb = np.clip(rgb, 0, 1)
    return rgb.astype(np.float32)


def _normalize_band(img, p=99.5, vmax=None, stretch_mode="percentile"):
    if stretch_mode == "none":
        return img.astype(np.float32)
    if vmax is None:
        vmax = np.percentile(img, p)
    if vmax <= 0:
        return np.zeros_like(img, dtype=np.float32)
    img = img / vmax
    if stretch_mode == "log":
        img = np.log10(1 + 9 * np.clip(img, 0, None))
    elif stretch_mode == "arcsinh":
        img = np.arcsinh(3 * np.clip(img, 0, None)) / np.arcsinh(3)
    elif stretch_mode == "sqrt":
        img = np.sqrt(np.clip(img, 0, 1))
    return np.clip(img, 0, 1).astype(np.float32)


def _load_npz(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as data:
        result = {k: data[k] for k in data.files}
    if "metadata" in result:
        try:
            result["metadata"] = json.loads(str(result["metadata"]))
        except Exception:
            pass
    return result


def _select_interesting_lens(unified_dir: Path):
    candidates = sorted(unified_dir.glob("cosmos_lens_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No lens samples found in {unified_dir}")

    best_path = None
    best_flux = -np.inf
    for p in candidates:
        data = _load_npz(p)
        if "image_final" not in data:
            continue
        total_flux = float(np.sum(data["image_final"]))
        if total_flux > best_flux:
            best_flux = total_flux
            best_path = p
    if best_path is None:
        raise RuntimeError("No valid lens samples with image_final found")
    return best_path


def _detect_cosmic_rays(final_image, field_image=None, lens_sources_image=None, threshold_percentile=95.0):
    """
    Detect cosmic rays/artifacts by comparing three image components.
    Artifacts = pixels that appear in FINAL but NOT in either FIELD_ONLY or LENS_SOURCES alone.
    Returns list of (row, col) tuples.
    """
    cosmic_rays = []
    
    # Use F150W band (index 1) for detection
    if len(final_image.shape) == 3:
        detection_band = final_image[1]  # F150W
    else:
        detection_band = final_image
    
    if field_image is not None and lens_sources_image is not None:
        if len(field_image.shape) == 3:
            field_band = field_image[1]
            lens_band = lens_sources_image[1]
        else:
            field_band = field_image
            lens_band = lens_sources_image
        
        # Artifacts are pixels that:
        # 1. Are bright in FINAL
        # 2. Are NOT explained by FIELD_ONLY alone (low in field)
        # 3. Are NOT explained by LENS_SOURCES alone (low in lens_sources)
        
        # Set thresholds
        final_threshold = np.percentile(detection_band[detection_band > 0], threshold_percentile)
        field_threshold = np.percentile(field_band[field_band > 0], 75.0) if np.any(field_band > 0) else 0
        lens_threshold = np.percentile(lens_band[lens_band > 0], 75.0) if np.any(lens_band > 0) else 0
        
        # A pixel is an artifact if:
        # - It's bright in FINAL AND
        # - It's NOT bright in FIELD_ONLY AND
        # - It's NOT bright in LENS_SOURCES
        artifact_mask = (detection_band > final_threshold) & \
                       (field_band <= field_threshold) & \
                       (lens_band <= lens_threshold)
        
        # Find connected components
        from scipy import ndimage
        if np.any(artifact_mask):
            labeled, n_features = ndimage.label(artifact_mask)
            
            for component_id in range(1, n_features + 1):
                coords = np.where(labeled == component_id)
                size = len(coords[0])
                
                # Keep reasonably sized artifacts (not tiny noise)
                if size > 5:
                    centroid_r = int(np.mean(coords[0]))
                    centroid_c = int(np.mean(coords[1]))
                    cosmic_rays.append((centroid_r, centroid_c))
    
    return cosmic_rays


def _get_step_stack(data, step):
    if step == "final":
        return data.get("image_final")
    key = f"image_{step}"
    return data.get(key)


def main():
    parser = argparse.ArgumentParser(description="Generate step-output figure")
    parser.add_argument("--input-dir", required=True, help="Output run directory")
    parser.add_argument("--output", required=True, help="Output figure path")
    parser.add_argument("--sample-id", type=int, default=None, help="Lens id (integer)")
    parser.add_argument("--order", nargs="+", default=STEPS_DEFAULT, help="Step order")
    parser.add_argument("--band-percentile", type=float, default=99.5, help="Percentile stretch")
    parser.add_argument("--global-stretch", action="store_true", help="Use shared stretch across steps")
    parser.add_argument(
        "--stretch",
        choices=["percentile", "arcsinh", "log", "sqrt", "none"],
        default="percentile",
        help="Stretch mode for display",
    )
    parser.add_argument("--label-fontsize", type=int, default=14, help="Font size for labels")
    parser.add_argument("--mark-cosmic-rays", action="store_true", help="Mark detected cosmic rays with (CR) circles")
    parser.add_argument("--independent-rgb", action="store_true", default=True, help="Normalize RGB channels independently for better field visibility")
    parser.add_argument("--combined-rgb", action="store_true", help="Use combined RGB normalization instead of independent channels")
    parser.add_argument(
        "--color-preset",
        choices=list(COLOR_PRESETS.keys()),
        default='trilogy',
        help="Color balance preset (trilogy=Trilogy-inspired, elliptical=red galaxies, balanced=moderate, natural=no correction)"
    )
    parser.add_argument("--blue-suppress", type=float, default=None, help="Blue channel suppression factor (overrides preset)")
    parser.add_argument("--green-suppress", type=float, default=None, help="Green channel suppression factor (overrides preset)")
    args = parser.parse_args()
    
    # Apply color preset if individual suppress values not specified
    preset = COLOR_PRESETS[args.color_preset]
    if args.blue_suppress is None:
        args.blue_suppress = preset['blue']
    if args.green_suppress is None:
        args.green_suppress = preset['green']

    input_dir = Path(args.input_dir)
    unified_dir = input_dir / "unified_npz"
    if not unified_dir.exists():
        raise FileNotFoundError(f"Missing unified_npz directory: {unified_dir}")

    if args.sample_id is not None:
        npz_path = unified_dir / f"cosmos_lens_{args.sample_id:06d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Sample not found: {npz_path}")
    else:
        npz_path = _select_interesting_lens(unified_dir)

    data = _load_npz(npz_path)

    rows = []
    labels = []
    for step in args.order:
        stack = _get_step_stack(data, step)
        if stack is None:
            continue
        rows.append(stack)
        labels.append(step.replace("_", " "))

    if not rows:
        raise RuntimeError("No step images available in this sample")

    n_rows = len(rows)
    n_cols = len(BANDS) + 1  # 4 bands + RGB
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    # Compute shared vmax per band (and RGB) if requested
    # Option (b): Use the final image's normalization for all steps
    # This gives observationally-realistic consistent backgrounds
    band_vmax = [None] * len(BANDS)
    rgb_vmax = None
    rgb_channel_vmax = None  # For independent RGB with global stretch
    if args.global_stretch and args.stretch != "none":
        # Use final image only (last row) for normalization
        final_stack = rows[-1]
        for b_idx in range(len(BANDS)):
            band_vmax[b_idx] = np.percentile(final_stack[b_idx], args.band_percentile)
        f115, f150, f277, f444 = final_stack
        rch = f444
        gch = 0.5 * (f150 + f277)
        bch = f115
        
        # For independent RGB: compute per-channel vmax from final image
        if args.independent_rgb and not args.combined_rgb:
            r_vmax = np.percentile(rch, args.band_percentile) if np.any(rch > 0) else 1.0
            g_vmax = np.percentile(gch, args.band_percentile) if np.any(gch > 0) else 1.0
            b_vmax = np.percentile(bch, args.band_percentile) if np.any(bch > 0) else 1.0
            rgb_channel_vmax = (r_vmax, g_vmax, b_vmax)
        else:
            # For combined RGB: use single vmax
            rgb_final = np.stack([rch, gch, bch], axis=-1)
            rgb_vmax = np.percentile(rgb_final, args.band_percentile)

    for r_idx, (stack, label) in enumerate(zip(rows, labels)):
        # Detect cosmic rays in final image only
        cosmic_rays = []
        if args.mark_cosmic_rays and r_idx == len(rows) - 1:  # Final row
            field_image = data.get("image_field_only")
            lens_sources_image = data.get("image_lens_sources")
            cosmic_rays = _detect_cosmic_rays(stack, field_image=field_image, 
                                              lens_sources_image=lens_sources_image, 
                                              threshold_percentile=95.0)
        
        # Bands
        for c_idx, band in enumerate(BANDS):
            img = _normalize_band(
                stack[c_idx],
                p=args.band_percentile,
                vmax=band_vmax[c_idx],
                stretch_mode=args.stretch,
            )
            axes[r_idx, c_idx].imshow(img, cmap="gray")
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(band, fontsize=args.label_fontsize)
            axes[r_idx, c_idx].axis("off")
            
            # Mark cosmic rays ONLY in RGB column (index 4, after the 4 bands)
            if args.mark_cosmic_rays and c_idx == len(BANDS) and r_idx == len(rows) - 1:
                for cr_row, cr_col in cosmic_rays:
                    circle = Circle((cr_col, cr_row), 12, color='red', fill=False, 
                                  linewidth=2.5, linestyle='--', alpha=0.8)
                    axes[r_idx, c_idx].add_patch(circle)
                    axes[r_idx, c_idx].text(cr_col, cr_row - 25, '(CR)', 
                                           color='red', fontsize=14, weight='bold', 
                                           ha='center', va='bottom')
        
        # RGB
        rgb = _make_rgb_from_stack(
            stack,
            p=args.band_percentile,
            vmax=rgb_vmax,
            stretch_mode=args.stretch,
            independent_rgb=args.independent_rgb if not args.combined_rgb else False,
            channel_vmax=rgb_channel_vmax,
            blue_suppress=args.blue_suppress,
            green_suppress=args.green_suppress,
        )
        axes[r_idx, -1].imshow(rgb)
        if r_idx == 0:
            axes[r_idx, -1].set_title("RGB", fontsize=args.label_fontsize)
        axes[r_idx, -1].axis("off")
        
        # Mark cosmic rays in RGB
        for cr_row, cr_col in cosmic_rays:
            circle = Circle((cr_col, cr_row), 12, color='red', fill=False, 
                          linewidth=2.5, linestyle='--', alpha=0.8)
            axes[r_idx, -1].add_patch(circle)

        # Row label on first column
        axes[r_idx, 0].text(
            0.02, 0.98, label,
            transform=axes[r_idx, 0].transAxes,
            ha="left", va="top", fontsize=args.label_fontsize,
            color="white", bbox=dict(facecolor="black", alpha=0.4, pad=2)
        )

    fig.suptitle("JWST Lens Simulation Step Outputs", fontsize=args.label_fontsize + 2)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
