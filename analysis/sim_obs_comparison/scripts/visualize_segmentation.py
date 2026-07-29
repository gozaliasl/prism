#!/usr/bin/env python3
"""
Diagnostic visualization of the photutils segmentation used by
morphology_metrics.measure_lens_arrays() -- plots the SNR-combined
detection image alongside the resulting segmentation map for a real
COSMOS-Web lens (FITS cutouts) and/or a simulated lens (unified npz).

Used in the Phase 2 multiplicity investigation
(analysis/sim_obs_comparison/reports/phase2_noise_and_color_fixes.md,
section 11) to compare how real vs. simulated lens systems fragment (or
don't) into separate n_components.

Usage:
    conda run -n astro-clean python visualize_segmentation.py \
        --real COSJ095856+015821 \
        --real_dir data/real_lenses \
        --sim /tmp/phase2_multfix2/unified_npz/PRISM_lens_BR_000003.npz \
        --outdir /tmp
"""

import argparse
import json
import glob

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from morphology_metrics import measure_lens_arrays, PIXEL_SCALE

BANDS = ["F115W", "F150W", "F277W", "F444W"]


def load_real(name, real_dir):
    images = {}
    for band in BANDS:
        files = glob.glob(f"{real_dir}/{name}_{band}.fits")
        if files:
            images[band] = fits.getdata(files[0]).astype(float)
    return images


def load_sim(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(d["metadata"].item())
    bands = meta["bands"]
    img = d["image_final"]
    return {b: img[i] for i, b in enumerate(bands) if b in BANDS}, meta


def plot_segmentation(images, pixel_scale, name, outdir):
    result, det, segm, rows = measure_lens_arrays(name, images, pixel_scale, return_seg=True)
    print(f"{name}: n_components={result['n_components']}, "
          f"max_sep_arcsec={result.get('max_sep_arcsec'):.3f}, "
          f"lens_reff_arcsec={result.get('lens_reff_arcsec'):.3f}")
    for src in rows:
        a = float(src.semimajor_sigma.value) * pixel_scale * 2.355
        b = float(src.semiminor_sigma.value) * pixel_scale * 2.355
        print(f"  label={src.label} area={src.area.value:.0f}px "
              f"centroid=({src.xcentroid:.1f},{src.ycentroid:.1f}) "
              f"semimajor={a:.3f}\" semiminor={b:.3f}\"")

    ny, nx = det.shape
    cy, cx = ny // 2, nx // 2
    half_box = int(3.0 / pixel_scale)
    sub_det = det[cy - half_box:cy + half_box, cx - half_box:cx + half_box]
    sub_segm = segm.data[cy - half_box:cy + half_box, cx - half_box:cx + half_box] if segm is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.arcsinh(sub_det), origin="lower", cmap="gray")
    axes[0].set_title(f"{name} detection image")
    if sub_segm is not None:
        axes[1].imshow(sub_segm, origin="lower", cmap="tab20")
    axes[1].set_title(f"{name} segmentation (n={result['n_components']})")
    plt.tight_layout()
    outpath = f"{outdir}/seg_{name}.png"
    plt.savefig(outpath, dpi=100)
    plt.close()
    print(f"  saved {outpath}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", help="Real lens name, e.g. COSJ095856+015821")
    p.add_argument("--real_dir", default="data/real_lenses")
    p.add_argument("--sim", action="append", default=[], help="Path to a unified npz file (repeatable)")
    p.add_argument("--outdir", default="/tmp")
    args = p.parse_args()

    if args.real:
        images = load_real(args.real, args.real_dir)
        plot_segmentation(images, PIXEL_SCALE, args.real, args.outdir)

    for sim_path in args.sim:
        images, meta = load_sim(sim_path)
        name = sim_path.split("/")[-1].replace(".npz", "")
        plot_segmentation(images, PIXEL_SCALE, name, args.outdir)


if __name__ == "__main__":
    main()
