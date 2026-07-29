"""
Mock observation module following GalaxyGenius Section 2.4 (Zhou et al. 2025).

Implements bandpass integration (Eq. 5–6), PSF convolution, and noise
(Eq. 7–12) with JWST NIRCam parameters from Appendix A Table 6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import constants
from scipy.ndimage import convolve

# JWST NIRCam Table 6 (Zhou et al. 2025, Appendix A)
JWST_NIRCAM_TABLE6: dict[str, dict[str, float | str]] = {
    "F070W": {
        "channel": "short", "lambda_5": 6241.0, "lambda_95": 7751.0,
        "B_sky": 0.3307, "B_dark": 0.0019, "sigma_RN": 15.77, "pixel_scale": 0.031,
    },
    "F150W": {
        "channel": "short", "lambda_5": 13477.0, "lambda_95": 16547.0,
        "B_sky": 0.4721, "B_dark": 0.0019, "sigma_RN": 15.77, "pixel_scale": 0.031,
    },
    "F200W": {
        "channel": "short", "lambda_5": 17784.0, "lambda_95": 22054.0,
        "B_sky": 0.4326, "B_dark": 0.0019, "sigma_RN": 15.77, "pixel_scale": 0.031,
    },
    "F182M": {
        "channel": "short", "lambda_5": 17335.0, "lambda_95": 19575.0,
        "B_sky": 0.2502, "B_dark": 0.0019, "sigma_RN": 15.77, "pixel_scale": 0.031,
    },
    "F356W": {
        "channel": "long", "lambda_5": 31767.0, "lambda_95": 39456.0,
        "B_sky": 1.1707, "B_dark": 0.0342, "sigma_RN": 13.25, "pixel_scale": 0.063,
    },
    "F444W": {
        "channel": "long", "lambda_5": 39320.0, "lambda_95": 49374.0,
        "B_sky": 3.2627, "B_dark": 0.0342, "sigma_RN": 13.25, "pixel_scale": 0.063,
    },
}

JWST_APERTURE_M = 6.5
JWST_T_EXP_S = 600.0
JWST_N_EXP = 1


@dataclass(frozen=True)
class JWSTMockParams:
    band: str
    t_exp_s: float = JWST_T_EXP_S
    n_exp: int = JWST_N_EXP
    aperture_m: float = JWST_APERTURE_M

    @property
    def table(self) -> dict[str, float | str]:
        if self.band not in JWST_NIRCAM_TABLE6:
            raise KeyError(f"Unknown JWST band {self.band!r}")
        return JWST_NIRCAM_TABLE6[self.band]

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(self.table["pixel_scale"])

    @property
    def B_sky(self) -> float:
        return float(self.table["B_sky"])

    @property
    def B_dark(self) -> float:
        return float(self.table["B_dark"])

    @property
    def sigma_RN(self) -> float:
        return float(self.table["sigma_RN"])

    @property
    def N_mean(self) -> float:
        """Eq. 9: per-pixel mean background + dark current (electrons)."""
        return (self.B_sky + self.B_dark) * self.t_exp_s * self.n_exp


def flux_nu_to_flux_lambda(f_nu_mjy_sr: np.ndarray, wavelength_angstrom: np.ndarray) -> np.ndarray:
    """Eq. 6: MJy/sr per Hz -> MJy/sr per Angstrom."""
    lam_m = wavelength_angstrom * 1e-10
    return f_nu_mjy_sr * constants.c / (lam_m ** 2)


def integrate_bandpass_electrons(
    wavelength_angstrom: np.ndarray,
    flux_mjy_sr: np.ndarray,
    throughput: np.ndarray,
    params: JWSTMockParams,
) -> float:
    """
    Eq. 5: integrate a single spectrum to total electron counts in one pixel.

    flux_mjy_sr : MJy/sr vs wavelength (same length as wavelength_angstrom)
    throughput : dimensionless filter throughput T(lambda)
    """
    f_lambda = flux_nu_to_flux_lambda(flux_mjy_sr, wavelength_angstrom)
    lam_m = wavelength_angstrom * 1e-10
    lp_m = params.pixel_scale_arcsec * (np.pi / (180.0 * 3600.0))
    integrand = lam_m * f_lambda * throughput
    # MJy/sr -> Jy/sr -> SI for Ryon (2023) convention used in paper
    # 1 MJy = 1e-6 Jy; electron conversion factor from Eq. 5
    integral = np.trapz(integrand, wavelength_angstrom * 1e-10)
    prefactor = (
        params.t_exp_s * params.n_exp * params.aperture_m * (lp_m ** 2)
        / (constants.h * constants.c)
    )
    # MJy/sr -> Jy/sr
    return prefactor * integral * 1e-6


def gaussian_psf_kernel(fwhm_arcsec: float, pixel_scale_arcsec: float, size: int = 31) -> np.ndarray:
    """Circular Gaussian PSF normalized to unit sum."""
    sigma = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pixel_scale_arcsec
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    kernel /= kernel.sum()
    return kernel


def convolve_psf(image_electrons: np.ndarray, psf: np.ndarray) -> np.ndarray:
    return convolve(image_electrons, psf, mode="constant")


def add_instrumental_noise(
    image_electrons: np.ndarray,
    params: JWSTMockParams,
    rng: np.random.Generator | None = None,
    *,
    include_noise: bool = True,
) -> np.ndarray:
    """
    Eq. 7–12: Poisson shot noise, Gaussian read noise, subtract N_mean.

    If include_noise=False, returns the noiseless PSF-convolved electron image.
    """
    if rng is None:
        rng = np.random.default_rng()

    e = np.asarray(image_electrons, dtype=np.float64)
    n_mean = params.N_mean
    e_bar = e + n_mean

    if not include_noise:
        return e

    e_hat = rng.poisson(np.clip(e_bar, 0, None).astype(np.float64))
    read = rng.normal(0.0, params.sigma_RN, size=e.shape)
    e_tilde = e_hat + params.n_exp * read
    return e_tilde - n_mean


def mock_observe_image(
    ideal_electrons: np.ndarray,
    band: str,
    *,
    psf: np.ndarray | None = None,
    psf_fwhm_arcsec: float | None = None,
    t_exp_s: float = JWST_T_EXP_S,
    n_exp: int = JWST_N_EXP,
    rng: np.random.Generator | None = None,
    include_noise: bool = True,
) -> np.ndarray:
    """
    Full mock observation on a 2D electron-count image (already bandpass-integrated).

    Provide either `psf` kernel or `psf_fwhm_arcsec` (Gaussian fallback).
    """
    params = JWSTMockParams(band=band, t_exp_s=t_exp_s, n_exp=n_exp)
    if psf is None:
        if psf_fwhm_arcsec is None:
            psf_fwhm_arcsec = 0.064 if params.table["channel"] == "short" else 0.13
        psf = gaussian_psf_kernel(psf_fwhm_arcsec, params.pixel_scale_arcsec)
    convolved = convolve_psf(ideal_electrons, psf)
    return add_instrumental_noise(convolved, params, rng, include_noise=include_noise)


def display_log_image(
    electrons: np.ndarray,
    clip: float = 1e-5,
) -> np.ndarray:
    """Paper display: log stretch with lower clip (Section 3.1)."""
    data = np.asarray(electrons, dtype=np.float64)
    pos = np.clip(data, clip, None)
    return np.log10(pos)
