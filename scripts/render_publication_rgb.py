#!/usr/bin/env python3
"""
Render publication-quality RGB composites from the full-precision per-band
arrays stored in unified_npz/<name>.npz (image_final, shape (4,300,300)).

Unlike the quick-look jpg_rgb/ previews (300x300, JPEG q=95, tuned for fast
ML-dataset visualization), this renderer:
  - works directly on the float32 science arrays (no re-reading lossy JPEGs)
  - uses a per-channel asinh stretch with percentile-based black/white points
    tuned for a clean, high-contrast look
  - mildly smooths the noise floor for display only (does not touch the data
    used for training)
  - upsamples with high-quality Lanczos interpolation for crisp print output
  - saves as quality=100, no-chroma-subsampling JPEG (or PNG, lossless)

Usage:
    python3 scripts/render_publication_rgb.py <unified_npz_path> <out.jpg> [--scale 4] [--png]
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

# Red=longest wavelength, Blue=shortest; bands ordered F115W,F150W,F277W,F444W
BAND_ORDER = ['F115W', 'F150W', 'F277W', 'F444W']


def stretch_channel(im, white_pct=99.85, black_sigma=1.6, smooth_sigma=0.9):
    """Per-channel asinh stretch with a robust, noise-adaptive black point.

    The black point is set from sigma-clipped background statistics
    (median + black_sigma * MAD-based sigma) rather than a fixed percentile,
    so it tracks the actual per-image noise floor -- a fixed percentile sits
    too close to the noise for faint/low-SNR systems (amplifying graininess)
    and too far from it for bright ones. A mild Gaussian smoothing
    (display-only, computed after black/white points so it doesn't bias them)
    suppresses pixel-to-pixel shot noise for a clean print-quality look.
    """
    finite = np.isfinite(im)
    data = np.where(finite, im, 0.0)

    median = np.median(data)
    mad = np.median(np.abs(data - median))
    sigma = 1.4826 * mad if mad > 0 else np.std(data)

    black = median + black_sigma * sigma
    white = np.percentile(data, white_pct)
    if white <= black:
        white = black + max(np.max(data) - black, 1e-12) * 0.01

    softening = max((white - black) * 0.05, 1e-12)

    smoothed = gaussian_filter(data, sigma=smooth_sigma)
    scaled = (smoothed - black) / softening
    stretched = np.arcsinh(np.clip(scaled, -1.0, None))

    s_black = np.arcsinh(0.0)
    s_white = np.arcsinh((white - black) / softening)
    out = (stretched - s_black) / (s_white - s_black)
    return np.clip(out, 0.0, 1.0)


def build_rgb(image_final, bands=BAND_ORDER):
    """image_final: (4, H, W) float32, ordered per `bands`."""
    band_idx = {b: i for i, b in enumerate(bands)}

    # Use all 4 NIRCam bands: longest -> R, two middle bands blended -> G,
    # shortest -> B (Trilogy-style multi-band composite).
    R = stretch_channel(image_final[band_idx['F444W']], white_pct=99.85)
    G = 0.5 * (stretch_channel(image_final[band_idx['F277W']], white_pct=99.8)
               + stretch_channel(image_final[band_idx['F150W']], white_pct=99.8))
    B = stretch_channel(image_final[band_idx['F115W']], white_pct=99.7)

    rgb = np.stack([R, G, B], axis=-1)

    # Gentle global color-balance: slightly cool the blue channel, warm the red,
    # matching the look of real JWST NIRCam color composites.
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * 1.05, 0, 1)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * 0.97, 0, 1)

    # Mild saturation boost (push colors away from gray) for a punchier composite.
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    rgb = np.clip(luma[..., None] + (rgb - luma[..., None]) * 1.25, 0, 1)

    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz_path')
    ap.add_argument('out_path')
    ap.add_argument('--scale', type=int, default=4, help='Upsampling factor (Lanczos)')
    ap.add_argument('--png', action='store_true', help='Save as lossless PNG instead of JPEG')
    args = ap.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)
    image_final = d['image_final']  # (4, H, W)

    rgb = build_rgb(image_final)
    img = Image.fromarray((rgb * 255).astype(np.uint8))

    if args.scale > 1:
        w, h = img.size
        img = img.resize((w * args.scale, h * args.scale), Image.LANCZOS)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.png or out_path.suffix.lower() == '.png':
        img.save(out_path)
    else:
        img.save(out_path, quality=100, subsampling=0)

    print(f"Saved {out_path} ({img.size[0]}x{img.size[1]})")


if __name__ == '__main__':
    main()
