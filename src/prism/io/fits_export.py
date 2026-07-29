"""Phase 8: multi-extension FITS data product export.

Writes one `.fits` file per sample alongside the existing `.npz`/JPEG
outputs, with extensions modeled loosely on COSMOS-Web/JADES/CEERS-style
public data releases:

- Primary HDU: per-band image cube (`image_final`, shape ``[n_bands, H, W]``).
- ``PSF``: per-band PSF kernel cube (from `field_info['psf_arrays']`).
- ``NOISE``: per-band noise-sigma maps, computed analytically as
  ``sqrt(bg_rms^2 + read_noise^2 + max(image, 0) / gain)``. This is an
  approximation -- the detector chain applies noise in-place during
  rendering rather than tracking per-pixel contributions, so this is a
  standard CCD/IR noise-model estimate from the final image, not a measured
  propagation.
- ``SEGMENTATION``: integer segmentation map from
  `photutils.segmentation.detect_sources` + `deblend_sources` on a
  combined-band detection image.
- ``TRUTH_CATALOG``: one-row `BinTableHDU` flattened from the existing
  per-object metadata dict (lens/source/field/TNG properties), i.e. the same
  information already written to `unified_npz`'s ``metadata`` JSON.

Gated by ``config['output']['save_fits']`` (default ``False``); called from
`save_outputs_unified` in `jwst_lens_simulator.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def compute_noise_sigma_maps(image_final, band_noise_cfgs, band_names):
    """Per-band noise-sigma maps: sqrt(bg_rms^2 + read_noise^2 + max(image,0)/gain).

    Args:
        image_final: array, shape (n_bands, H, W)
        band_noise_cfgs: dict band -> dict with 'bg_rms', 'read_noise', optional 'gain'
        band_names: list of band names matching image_final's first axis

    Returns:
        array, shape (n_bands, H, W), float32
    """
    n_bands, h, w = image_final.shape
    noise_maps = np.zeros((n_bands, h, w), dtype=np.float32)
    for i, band in enumerate(band_names):
        cfg = band_noise_cfgs.get(band, {})
        bg_rms = float(cfg.get('bg_rms', cfg.get('_bg_rms', 0.02)))
        read_noise = float(cfg.get('read_noise', 0.0))
        gain = float(cfg.get('gain', 1.0)) or 1.0
        signal_term = np.clip(image_final[i], 0.0, None) / gain
        noise_maps[i] = np.sqrt(bg_rms ** 2 + read_noise ** 2 + signal_term)
    return noise_maps


def compute_segmentation_map(image_final, band_noise_cfgs, band_names, npixels=8, threshold_sigma=3.0):
    """Combined-band detection segmentation map (photutils).

    Returns an int32 array (H, W) with 0 = background, >0 = source labels
    (deblended where possible), or an all-zero array if no sources are
    detected.
    """
    from photutils.segmentation import detect_sources, deblend_sources

    n_bands, h, w = image_final.shape
    det = np.zeros((h, w), dtype=np.float64)
    for i, band in enumerate(band_names):
        cfg = band_noise_cfgs.get(band, {})
        bg_rms = float(cfg.get('bg_rms', cfg.get('_bg_rms', 0.02)))
        if bg_rms <= 0:
            bg_rms = 1e-6
        det += image_final[i] / bg_rms

    threshold = threshold_sigma * np.sqrt(max(n_bands, 1))
    segm = detect_sources(det, threshold, npixels=npixels)
    if segm is None:
        return np.zeros((h, w), dtype=np.int32)

    try:
        segm = deblend_sources(det, segm, npixels=npixels)
    except Exception:
        pass

    return segm.data.astype(np.int32)


def _to_fits_value(value):
    """Coerce a metadata value to something a FITS table column / header can hold."""
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    return str(value)


def _truth_catalog_hdu(truth_catalog_row):
    """Build a one-row BinTableHDU from a flat metadata dict."""
    columns = []
    for key, raw_value in truth_catalog_row.items():
        value = _to_fits_value(raw_value)
        if isinstance(value, bool):
            fmt = 'L'
            arr = np.array([value], dtype=bool)
        elif isinstance(value, int):
            fmt = 'K'
            arr = np.array([value], dtype=np.int64)
        elif isinstance(value, float):
            fmt = 'D'
            arr = np.array([value], dtype=np.float64)
        else:
            value = str(value)
            fmt = f'{max(len(value), 1)}A'
            arr = np.array([value])
        # FITS column names must be <= 68 chars and are uppercased by convention
        columns.append(fits.Column(name=str(key)[:68], format=fmt, array=arr))

    table_hdu = fits.BinTableHDU.from_columns(columns, name='TRUTH_CATALOG')
    return table_hdu


def write_lens_fits(base_path, image_final, psf_arrays, noise_sigma_maps,
                     segmentation_map, truth_catalog_row, band_names,
                     wcs_header=None, pixel_scale=None, exposure_time=None,
                     magnitude_zero_point=None):
    """Write a multi-extension FITS file for one simulated sample.

    Args:
        base_path: Path (without extension) to write `<base_path>.fits`.
        image_final: array, shape (n_bands, H, W) -- final composite image.
        psf_arrays: dict band -> 2D array (or None) -- PSF kernels.
        noise_sigma_maps: array, shape (n_bands, H, W) -- from
            `compute_noise_sigma_maps`.
        segmentation_map: int array, shape (H, W) -- from
            `compute_segmentation_map`.
        truth_catalog_row: flat dict of per-object metadata.
        band_names: list of band names matching image_final's first axis.
        wcs_header: optional `astropy.io.fits.Header` (or dict) of WCS
            keywords to merge into the primary header.
        pixel_scale, exposure_time, magnitude_zero_point: optional values
            recorded in the primary header.

    Returns:
        Path to the written `.fits` file.
    """
    image_final = np.asarray(image_final, dtype=np.float32)

    primary_header = fits.Header()
    primary_header['NBANDS'] = (len(band_names), 'Number of bands in image cube')
    for i, band in enumerate(band_names):
        primary_header[f'BAND{i}'] = (band, f'Band name for plane {i}')
    if pixel_scale is not None:
        primary_header['PIXSCALE'] = (float(pixel_scale), 'arcsec/pixel')
    if exposure_time is not None:
        primary_header['EXPTIME'] = (float(exposure_time), 'seconds')
    if magnitude_zero_point is not None:
        primary_header['MAGZP'] = (float(magnitude_zero_point), 'AB magnitude zero point')
    if wcs_header is not None:
        primary_header.update(wcs_header)

    hdul = [fits.PrimaryHDU(data=image_final, header=primary_header)]

    # PSF extension: stack available per-band PSF kernels into a cube.
    if psf_arrays:
        psf_bands = [b for b in band_names if psf_arrays.get(b) is not None]
        if psf_bands:
            shapes = {np.asarray(psf_arrays[b]).shape for b in psf_bands}
            if len(shapes) == 1:
                psf_cube = np.stack(
                    [np.asarray(psf_arrays[b], dtype=np.float32) for b in psf_bands], axis=0
                )
                psf_header = fits.Header()
                psf_header['NBANDS'] = len(psf_bands)
                for i, band in enumerate(psf_bands):
                    psf_header[f'BAND{i}'] = band
                hdul.append(fits.ImageHDU(data=psf_cube, header=psf_header, name='PSF'))
            else:
                # Mismatched PSF shapes across bands: write one extension per band.
                for band in psf_bands:
                    arr = np.asarray(psf_arrays[band], dtype=np.float32)
                    hdul.append(fits.ImageHDU(data=arr, name=f'PSF_{band}'))

    # NOISE extension
    if noise_sigma_maps is not None:
        hdul.append(fits.ImageHDU(data=np.asarray(noise_sigma_maps, dtype=np.float32), name='NOISE'))

    # SEGMENTATION extension
    if segmentation_map is not None:
        hdul.append(fits.ImageHDU(data=np.asarray(segmentation_map, dtype=np.int32), name='SEGMENTATION'))

    # TRUTH_CATALOG extension
    if truth_catalog_row:
        hdul.append(_truth_catalog_hdu(truth_catalog_row))

    fits_path = Path(str(base_path) + '.fits')
    fits.HDUList(hdul).writeto(fits_path, overwrite=True)
    return fits_path
