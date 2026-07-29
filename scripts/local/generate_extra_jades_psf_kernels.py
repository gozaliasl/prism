#!/usr/bin/env python3
"""
Generate analytic (Gaussian) PSF kernels for the additional JADES NIRCam
filters that lack real WebbPSF-derived kernels in data/psf_v5_30mas/tiles/
(only F115W, F150W, F277W, F444W have real kernels).

New bands: F070W, F090W, F200W, F356W
FWHM (arcsec, empirical NIRCam values, Eisenstein+2023 / JDox):
  F070W ~0.023   F090W ~0.034   F200W ~0.066   F356W ~0.114

Pixel scale: 0.03"/px, kernel size matches existing kernels (101x101),
normalized to sum=1, written as float32 FITS, same naming convention
(<BAND>_kernel.fits) into every existing tile directory.

This is a fallback approximation (Gaussian core only, no diffraction
rings/spikes) -- adequate for noise/depth-level dataset generation but
not for PSF-sensitive morphology studies. Replace with WebbPSF-derived
kernels if available.
"""
import numpy as np
from astropy.io import fits
from pathlib import Path

PIXEL_SCALE = 0.03  # arcsec/px
KERNEL_SIZE = 101

FWHM_ARCSEC = {
    "F070W": 0.023,
    "F090W": 0.034,
    "F200W": 0.066,
    "F356W": 0.114,
}

TILES_DIR = Path("data/psf_v5_30mas/tiles")


def gaussian_kernel(fwhm_arcsec, size, pixel_scale):
    sigma_px = (fwhm_arcsec / pixel_scale) / 2.3548200450309493
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size]
    r2 = (x - c) ** 2 + (y - c) ** 2
    kernel = np.exp(-0.5 * r2 / sigma_px ** 2).astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def main():
    tiles = sorted(p for p in TILES_DIR.iterdir() if p.is_dir())
    print(f"Found {len(tiles)} tile directories")

    for band, fwhm in FWHM_ARCSEC.items():
        kernel = gaussian_kernel(fwhm, KERNEL_SIZE, PIXEL_SCALE)
        print(f"{band}: FWHM={fwhm}\" -> sigma={fwhm/PIXEL_SCALE/2.3548:.3f} px, "
              f"sum={kernel.sum():.6f}, max={kernel.max():.6f}")
        for tile in tiles:
            out_path = tile / f"{band}_kernel.fits"
            hdu = fits.PrimaryHDU(data=kernel)
            hdu.writeto(out_path, overwrite=True)

    print("Done.")


if __name__ == "__main__":
    main()
