"""
Synthetic PSF generator for non-JWST telescopes.

Generates realistic PSF kernels for:
  - Roman WFI  : stpsf (space-based diffraction-limited)
  - Euclid VIS/NISP : GalSim OpticalPSF (space-based diffraction-limited)
  - Subaru HSC / ground_based : GalSim Kolmogorov (atmospheric, seeing-limited)

All output kernels are (psf_size x psf_size) float32, normalised to sum = 1.

Usage
-----
from prism.io.synthetic_psf_generator import build_resolution_psf_cache  # noqa: E402

cache = build_resolution_psf_cache(
    res_name   = 'roman',
    bands      = ['ROMAN_F087', 'ROMAN_F106', ...],
    pixel_scale= 0.11,
    psf_size   = 101,
    cache_dir  = Path('data/psf_cache'),
)
# cache is {'synthetic': {band: 2-D numpy array or None}}
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    total = arr.sum()
    return (arr / total).astype(np.float32) if total > 0 else arr.astype(np.float32)


def _gaussian_fallback(psf_size: int, pixel_scale: float, fwhm_arcsec: float) -> np.ndarray:
    """Simple symmetric Gaussian — used when no toolkit is available."""
    sigma_pix = fwhm_arcsec / (2.355 * pixel_scale)
    cy, cx = psf_size // 2, psf_size // 2
    y, x = np.mgrid[:psf_size, :psf_size]
    kernel = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma_pix ** 2))
    return _normalize(kernel)


def _trim_or_pad(arr: np.ndarray, target: int) -> np.ndarray:
    """Ensure array is exactly (target × target)."""
    h, w = arr.shape
    # Crop
    if h > target:
        trim = (h - target) // 2
        arr = arr[trim: trim + target, :]
    if w > target:
        trim = (w - target) // 2
        arr = arr[:, trim: trim + target]
    # Pad
    if arr.shape[0] < target or arr.shape[1] < target:
        pad_h = max(0, target - arr.shape[0])
        pad_w = max(0, target - arr.shape[1])
        arr = np.pad(arr, ((pad_h // 2, pad_h - pad_h // 2),
                           (pad_w // 2, pad_w - pad_w // 2)))
    return arr


# ---------------------------------------------------------------------------
# Roman WFI  (stpsf)
# ---------------------------------------------------------------------------

# Map our band names → stpsf filter strings
_ROMAN_BAND_MAP = {
    'ROMAN_F062': 'F062',
    'ROMAN_F087': 'F087',
    'ROMAN_F106': 'F106',
    'ROMAN_F129': 'F129',
    'ROMAN_F146': 'F146',
    'ROMAN_F158': 'F158',
    'ROMAN_F184': 'F184',
    'ROMAN_F213': 'F213',
}


def _ensure_stpsf_data_path() -> None:
    """Point STPSF to the local reference data directory when available."""
    if os.environ.get('STPSF_PATH'):
        return

    repo_data_dir = Path(__file__).resolve().parent.parent / 'data' / 'stpsf-data'
    cwd_data_dir = Path.cwd() / 'data' / 'stpsf-data'

    for candidate in (repo_data_dir, cwd_data_dir):
        if candidate.exists() and candidate.is_dir():
            os.environ['STPSF_PATH'] = str(candidate)
            print(f"  [PSF] Using local STPSF_PATH={candidate}")
            return


def generate_roman_psf(band: str,
                       pixel_scale: float = 0.11,
                       psf_size: int = 101,
                       detector: str = 'SCA01') -> np.ndarray | None:
    """Generate a Roman WFI PSF with stpsf."""
    stpsf_band = _ROMAN_BAND_MAP.get(band)
    if stpsf_band is None:
        return None
    try:
        _ensure_stpsf_data_path()
        import stpsf  # noqa: F401 – checked at call site
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            roman = stpsf.roman.WFI()
            roman.filter = stpsf_band
            roman.detector = detector
            roman.detector_position = (512, 512)
            psf_hdul = roman.calc_psf(
                fov_pixels=psf_size,
                oversample=4,
            )
        # Extension 1 = detector-sampled PSF
        kernel = psf_hdul[1].data.copy()
        kernel = _trim_or_pad(kernel, psf_size)
        print(f"  [PSF] Roman {band}: generated via stpsf ({kernel.shape})")
        return _normalize(kernel)
    except Exception as exc:
        print(f"  [PSF] stpsf Roman {band} failed: {exc} — using Gaussian fallback")
        return _gaussian_fallback(psf_size, pixel_scale, fwhm_arcsec=0.14)


# ---------------------------------------------------------------------------
# Euclid VIS / NISP  (GalSim OpticalPSF)
# ---------------------------------------------------------------------------

# Central wavelengths in nm for Euclid bands
_EUCLID_LAMBDA_NM = {
    'EUCLID_VIS': 700,
    'EUCLID_Y':   1021,
    'EUCLID_J':   1254,
    'EUCLID_H':   1647,
}

# Euclid primary mirror diameter  [m]
_EUCLID_DIAM_M = 1.2


def generate_euclid_psf(band: str,
                        pixel_scale: float = 0.10,
                        psf_size: int = 101) -> np.ndarray | None:
    """Generate a Euclid-like space PSF with GalSim OpticalPSF."""
    lam_nm = _EUCLID_LAMBDA_NM.get(band, 800)
    try:
        import galsim  # noqa: F401
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # OpticalPSF wants lam in nm, diam in metres
            opt_psf = galsim.OpticalPSF(
                lam=float(lam_nm),
                diam=_EUCLID_DIAM_M,
                obscuration=0.30,       # central obscuration ratio
                nstruts=3,              # three support struts
                strut_thick=0.012,
                defocus=0.025,          # slight defocus [waves]
                astig1=0.01,
                astig2=0.01,
                spher=0.005,
            )
            img = galsim.Image(psf_size, psf_size, scale=pixel_scale)
            opt_psf.drawImage(image=img, method='auto')
        kernel = img.array.copy()
        kernel = _trim_or_pad(kernel, psf_size)
        print(f"  [PSF] Euclid {band}: generated via GalSim OpticalPSF ({kernel.shape})")
        return _normalize(kernel)
    except Exception as exc:
        print(f"  [PSF] GalSim Euclid {band} failed: {exc} — using Gaussian fallback")
        # Euclid is diffraction-limited; FWHM ~ lambda/D in arcsec
        fwhm = (lam_nm * 1e-9) / _EUCLID_DIAM_M * (180 * 3600 / np.pi)
        return _gaussian_fallback(psf_size, pixel_scale, fwhm_arcsec=fwhm)


# ---------------------------------------------------------------------------
# Ground-based / Subaru HSC  (GalSim Kolmogorov + optics)
# ---------------------------------------------------------------------------

# Central wavelengths in nm for Subaru HSC bands
_SUBARU_LAMBDA_NM = {
    'SUBARU_B': 440,
    'SUBARU_V': 550,
    'SUBARU_G': 477,
    'SUBARU_R': 622,
    'SUBARU_I': 763,
    'SUBARU_Z': 905,
    'SUBARU_Y': 1000,
}

# Subaru primary mirror diameter  [m]
_SUBARU_DIAM_M = 8.2

# Reference seeing FWHM at 700 nm [arcsec]  (median HSC seeing)
_HSC_SEEING_FWHM_700NM = 0.65


def generate_ground_psf(band: str,
                        pixel_scale: float = 0.20,
                        psf_size: int = 101,
                        seeing_fwhm: float | None = None) -> np.ndarray | None:
    """
    Generate a ground-based (Subaru/HSC-like) PSF.

    Combines:
      * Kolmogorov atmospheric turbulence (FWHM scales as λ^−0.2)
      * Diffraction-limited Airy disc for the 8.2 m primary

    Parameters
    ----------
    seeing_fwhm : FWHM at the reference wavelength (700 nm) in arcsec.
                  If None, the median HSC value (0.65") is used.
    """
    if seeing_fwhm is None:
        seeing_fwhm = _HSC_SEEING_FWHM_700NM

    lam_nm = _SUBARU_LAMBDA_NM.get(band, 700)

    try:
        import galsim
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Chromatic scaling of Kolmogorov FWHM  (λ/λ_ref)^{-0.2}
            lam_ref = 700.0
            fwhm_band = seeing_fwhm * (lam_nm / lam_ref) ** (-0.2)

            atm = galsim.Kolmogorov(fwhm=fwhm_band)

            # Diffraction limit (tiny compared with seeing, but correct)
            lam_over_D = (lam_nm * 1e-9) / _SUBARU_DIAM_M * (180 * 3600 / np.pi)
            optics = galsim.Airy(lam_over_diam=lam_over_D, obscuration=0.13)

            total_psf = galsim.Convolve([atm, optics])

            img = galsim.Image(psf_size, psf_size, scale=pixel_scale)
            total_psf.drawImage(image=img, method='auto')

        kernel = img.array.copy()
        kernel = _trim_or_pad(kernel, psf_size)
        print(f"  [PSF] Ground {band}: generated via GalSim Kolmogorov "
              f"(FWHM={fwhm_band:.3f}\", λ={lam_nm}nm, {kernel.shape})")
        return _normalize(kernel)

    except Exception as exc:
        print(f"  [PSF] GalSim ground {band} failed: {exc} — using Gaussian fallback")
        fwhm_band = seeing_fwhm * (lam_nm / 700.0) ** (-0.2)
        return _gaussian_fallback(psf_size, pixel_scale, fwhm_arcsec=fwhm_band)


# ---------------------------------------------------------------------------
# LSST / Rubin Observatory  (GalSim Kolmogorov + optics)
# ---------------------------------------------------------------------------

# Central wavelengths in nm for LSST bands (Ivezić et al. 2019)
_LSST_LAMBDA_NM = {
    'LSST_U': 367,
    'LSST_G': 482,
    'LSST_R': 622,
    'LSST_I': 754,
    'LSST_Z': 869,
    'LSST_Y': 971,
}

# Rubin primary mirror effective diameter  [m]  (8.36 m primary, 6.68 m effective)
_LSST_DIAM_M = 6.68

# Median single-visit seeing FWHM at 700 nm [arcsec]  (LSST SRD: 0.7")
_LSST_SEEING_FWHM_700NM = 0.70


def generate_lsst_psf(band: str,
                      pixel_scale: float = 0.200,
                      psf_size: int = 101,
                      seeing_fwhm: float | None = None) -> np.ndarray | None:
    """
    Generate a Rubin/LSST-like PSF.

    Combines Kolmogorov atmospheric turbulence (FWHM ∝ λ^−0.2) with the
    8.36 m Rubin primary (effective aperture 6.68 m due to central obscuration).
    The Rubin mirror has a large central obscuration (~0.61) from the secondary.

    Parameters
    ----------
    seeing_fwhm : FWHM at the reference wavelength (700 nm) in arcsec.
                  If None, uses the LSST SRD median value (0.70").
    """
    if seeing_fwhm is None:
        seeing_fwhm = _LSST_SEEING_FWHM_700NM

    lam_nm = _LSST_LAMBDA_NM.get(band, 622)   # default to r-band

    try:
        import galsim
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Chromatic Kolmogorov seeing: FWHM ∝ λ^{-0.2}
            lam_ref = 700.0
            fwhm_band = seeing_fwhm * (lam_nm / lam_ref) ** (-0.2)

            atm = galsim.Kolmogorov(fwhm=fwhm_band)

            # Rubin diffraction limit (very small vs. seeing but physically correct)
            lam_over_D = (lam_nm * 1e-9) / _LSST_DIAM_M * (180 * 3600 / np.pi)
            optics = galsim.Airy(lam_over_diam=lam_over_D, obscuration=0.61)

            total_psf = galsim.Convolve([atm, optics])

            img = galsim.Image(psf_size, psf_size, scale=pixel_scale)
            total_psf.drawImage(image=img, method='auto')

        kernel = img.array.copy()
        kernel = _trim_or_pad(kernel, psf_size)
        print(f"  [PSF] LSST {band}: generated via GalSim Kolmogorov "
              f"(FWHM={fwhm_band:.3f}\", λ={lam_nm}nm, {kernel.shape})")
        return _normalize(kernel)

    except Exception as exc:
        print(f"  [PSF] GalSim LSST {band} failed: {exc} — using Gaussian fallback")
        fwhm_band = seeing_fwhm * (lam_nm / 700.0) ** (-0.2)
        return _gaussian_fallback(psf_size, pixel_scale, fwhm_arcsec=fwhm_band)


# ---------------------------------------------------------------------------
# Euclid Q1 empirical PSFs  (from real cutouts)
# ---------------------------------------------------------------------------

_EUCLID_Q1_BANDS = ('EUCLID_VIS', 'EUCLID_Y', 'EUCLID_J', 'EUCLID_H')


def load_euclid_q1_psf_data(
        psf_dir: Path | str | None = None,
        bands: list[str] | None = None,
) -> dict:
    """
    Load empirical Euclid Q1 PSF kernels from ``data/euclid_q1_psf/tiles/``.

    Returns the same ``{tile_name: {band: array}}`` dict as ``load_psf_data()``
    so it can be passed directly to ``get_psf_for_simulation()``.

    Each tile holds 101×101 normalised kernels padded from the native Q1 stamps
    (VIS 21×21, NIR 33×33) embedded in the lenstronomy cutout FITS files.
    """
    from astropy.io import fits

    if psf_dir is None:
        psf_dir = Path(__file__).resolve().parent.parent / 'data' / 'euclid_q1_psf' / 'tiles'
    else:
        psf_dir = Path(psf_dir)

    if bands is None:
        bands = list(_EUCLID_Q1_BANDS)

    psf_data: dict[str, dict[str, np.ndarray | None]] = {}
    if not psf_dir.is_dir():
        print(f"[PSF] Euclid Q1 empirical PSF dir not found: {psf_dir}")
        return psf_data

    for tile_dir in sorted(psf_dir.iterdir()):
        if not tile_dir.is_dir():
            continue
        tile_name = tile_dir.name
        psf_data[tile_name] = {}
        for band in bands:
            kernel_file = tile_dir / f"{band}_kernel.fits"
            if kernel_file.exists():
                try:
                    psf_data[tile_name][band] = fits.getdata(kernel_file).astype(np.float32)
                except Exception as exc:
                    print(f"[PSF] Error loading {kernel_file}: {exc}")
                    psf_data[tile_name][band] = None
            else:
                psf_data[tile_name][band] = None

    n_tiles = len(psf_data)
    if n_tiles:
        print(f"[PSF] Loaded Euclid Q1 empirical PSFs: {n_tiles} tiles from {psf_dir}")
    return psf_data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_resolution_psf_cache(
        res_name: str,
        bands: list[str],
        pixel_scale: float,
        psf_size: int = 101,
        rng=None,
        cache_dir: Path | str | None = None,
        q1_psf_dir: Path | str | None = None,
) -> dict:
    """
    Build a PSF data dictionary for a given telescope/resolution.

    Returns a dict in the same format as ``load_psf_data()`` in the
    simulator — i.e. ``{tile_name: {band: array_or_None}}`` — so it
    can be passed directly to ``get_psf_for_simulation()``.

    For synthetic PSFs there is only one "tile", keyed ``'synthetic'``.

    Parameters
    ----------
    res_name   : 'roman', 'euclid', 'ground_based', or 'jwst'
    bands      : list of band names (telescope-specific)
    pixel_scale: arcsec/pixel
    psf_size   : output kernel size (pixels, square)
    cache_dir  : if given, save/load kernels from an .npz cache file
    """
    # Euclid: prefer empirical Q1 kernels when available (before disk cache)
    if res_name == 'euclid':
        # FIX (adversarial audit finding, 2026-08-01): load_euclid_q1_psf_data
        # was always called with psf_dir=None, which falls back to a
        # package-relative default path (src/prism/data/euclid_q1_psf/tiles)
        # that does NOT exist in this package layout -- the configured
        # euclid_q1.data_dir (e.g. pointing at the real PSF tiles on an
        # external data volume) was silently ignored, and empirical PSFs
        # silently fell back to a Gaussian/analytical approximation instead
        # (confirmed by execution: "[PSF] Euclid Q1 empirical PSF dir not
        # found" printed even when a working data_dir WAS configured and
        # successfully used by the separate euclid_q1 population-prior
        # loading code elsewhere in the pipeline -- i.e. two different
        # code paths for the same config value disagreed on where to find
        # the data). q1_psf_dir is now threaded through from CONFIG at the
        # call site in simulator.py.
        empirical = load_euclid_q1_psf_data(psf_dir=q1_psf_dir, bands=bands)
        if empirical:
            return empirical

    # Check disk cache (GalSim / analytical fallbacks)
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"psf_{res_name}_{pixel_scale:.4f}_{psf_size}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            psf_dict = {b: data[b] for b in bands if b in data}
            missing = [b for b in bands if b not in psf_dict]
            if not missing:
                print(f"[PSF] Loaded cached {res_name} PSFs from {cache_path}")
                return {'synthetic': psf_dict}
            print(f"[PSF] Cache incomplete for {res_name} (missing {missing}), regenerating")

    print(f"[PSF] Generating {res_name} PSFs: {bands}  pixel_scale={pixel_scale}\"/pix")
    psf_dict: dict[str, np.ndarray | None] = {}

    for band in bands:
        kernel: np.ndarray | None = None

        if res_name == 'roman':
            kernel = generate_roman_psf(band, pixel_scale, psf_size)

        elif res_name == 'euclid':
            kernel = generate_euclid_psf(band, pixel_scale, psf_size)

        elif res_name in ('ground_based', 'subaru'):
            kernel = generate_ground_psf(band, pixel_scale, psf_size)

        elif res_name == 'lsst':
            kernel = generate_lsst_psf(band, pixel_scale, psf_size)

        elif res_name == 'jwst':
            # JWST uses empirical tiles loaded separately — skip here
            kernel = None

        psf_dict[band] = kernel

    # Write cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"psf_{res_name}_{pixel_scale:.4f}_{psf_size}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        saveable = {b: arr for b, arr in psf_dict.items() if arr is not None}
        if saveable:
            np.savez_compressed(cache_path, **saveable)
            print(f"[PSF] Cached {res_name} PSFs → {cache_path}")

    return {'synthetic': psf_dict}
