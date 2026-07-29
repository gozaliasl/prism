#!/usr/bin/env python3
"""
Regenerate F277W_kernel.fits and F444W_kernel.fits for every tile in
data/psf_v5_30mas/tiles/ directly from the PSFEx .psf models, fixing the
~1.7-1.8x FWHM excess found in
analysis/sim_obs_comparison/reports/phase2_noise_and_color_fixes.md
(item 15): the existing LW kernels were ~1.7-1.8x broader (in FWHM) than
the empirical PSF measured from the same real COSMOS-Web mosaics
(PSF_FWHM in the .psf header), while the SW kernels (F115W/F150W) were
close to correct (~1.0-1.1x).

For each tile/band, take the PSFEx model's constant component
(PSF_MASK[0], the tile-center PSF, sampled at PSF_SAMP image-pixels per
PSF-pixel), resample it from PSF_SAMP sampling to 1 image-pixel (0.03"/px)
sampling via spline zoom, crop/pad to 101x101 centered on the peak, and
normalize to sum=1 -- matching the existing kernel.fits format
(float32 PrimaryHDU, 101x101, sum=1).

Existing kernels are backed up to <band>_kernel.fits.bak_v5 before being
overwritten.

Usage:
    conda run -n astro-clean python scripts/local/regen_lw_psf_kernels.py
"""
import glob
import shutil

import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.ndimage import zoom

TILES_DIR = "data/psf_v5_30mas/tiles"
BANDS = ["F277W", "F444W"]
KERNEL_SIZE = 101
PIXEL_SCALE = 0.03


def gaussian_fwhm(psf, pixel_scale=PIXEL_SCALE):
    psf = np.asarray(psf, dtype=float)
    psf = psf / psf.sum()
    ny, nx = psf.shape
    y, x = np.mgrid[0:ny, 0:nx]
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    g_init = models.Gaussian2D(amplitude=psf.max(), x_mean=cx, y_mean=cy,
                                 x_stddev=2, y_stddev=2)
    fitter = fitting.LevMarLSQFitter()
    g = fitter(g_init, x, y, psf)
    sigma = (abs(g.x_stddev.value) + abs(g.y_stddev.value)) / 2
    return 2.355 * sigma * pixel_scale


def build_kernel(psf_path):
    with fits.open(psf_path) as hdul:
        h = hdul[1].header
        mask = hdul[1].data['PSF_MASK'][0]  # (3, N, N)
        samp = h['PSF_SAMP']
        fwhm_target = h['PSF_FWHM'] * PIXEL_SCALE

    comp0 = np.asarray(mask[0], dtype=float)
    comp0 = np.clip(comp0, 0, None)

    # Resample from PSF-pixel grid (1 PSF-px = samp image-px) to image-px grid
    resampled = zoom(comp0, samp, order=3)
    resampled = np.clip(resampled, 0, None)

    # Crop/pad to KERNEL_SIZE x KERNEL_SIZE, centered on the peak
    ny, nx = resampled.shape
    cy, cx = np.unravel_index(np.argmax(resampled), resampled.shape)
    half = KERNEL_SIZE // 2

    out = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float64)
    for oy in range(KERNEL_SIZE):
        sy = cy + (oy - half)
        if not (0 <= sy < ny):
            continue
        for ox in range(KERNEL_SIZE):
            sx = cx + (ox - half)
            if not (0 <= sx < nx):
                continue
            out[oy, ox] = resampled[sy, sx]

    out /= out.sum()
    return out.astype(np.float32), fwhm_target


def main():
    tile_dirs = sorted(glob.glob(f"{TILES_DIR}/*/"))
    print(f"Found {len(tile_dirs)} tile directories")

    for band in BANDS:
        print(f"\n=== {band} ===")
        ratios = []
        for tile_dir in tile_dirs:
            tile = tile_dir.rstrip("/").split("/")[-1]
            psf_path = f"{tile_dir}{band}.psf"
            kernel_path = f"{tile_dir}{band}_kernel.fits"
            backup_path = f"{tile_dir}{band}_kernel.fits.bak_v5"

            kernel, fwhm_target = build_kernel(psf_path)
            fwhm_new = gaussian_fwhm(kernel)

            old = fits.getdata(kernel_path)
            fwhm_old = gaussian_fwhm(old)

            shutil.copy2(kernel_path, backup_path)
            hdu = fits.PrimaryHDU(data=kernel)
            hdu.writeto(kernel_path, overwrite=True)

            ratios.append(fwhm_new / fwhm_target)
            print(f"  {tile}: old FWHM={fwhm_old:.4f}\"  new FWHM={fwhm_new:.4f}\"  "
                  f"target(PSFEx)={fwhm_target:.4f}\"  new/target={fwhm_new/fwhm_target:.3f}")

        ratios = np.array(ratios)
        print(f"  -> {band} new/target ratio: median={np.median(ratios):.3f}, "
              f"range=[{ratios.min():.3f},{ratios.max():.3f}]")

    print("\nDone. Old kernels backed up as <band>_kernel.fits.bak_v5")


if __name__ == "__main__":
    main()
