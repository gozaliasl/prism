"""
Strong-lens mock with GalaxyGenius deflector light via lenstronomy INTERPOL.

Mass model: SIE + external shear.
Source: Sersic (gravitationally lensed through ImageModel.source_surface_brightness).
Lens light: GalaxyGenius stamp as INTERPOL profile (image-plane, unlensed).

All components are rendered in one ImageModel.image() call — no overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .stamp_compositor import build_galaxygenius_interpol_kwargs


@dataclass(frozen=True)
class LensStampSimConfig:
    numpix: int = 300
    pixel_scale: float = 0.031
    band: str = "F150W"
    theta_E: float = 1.15
    lens_q: float = 0.75
    lens_pa_deg: float = 25.0
    lens_mag: float = 21.0
    lens_re_arcsec: float = 0.85
    source_mag: float = 22.0
    source_re_arcsec: float = 0.35
    source_n: float = 1.2
    source_x_frac: float = 0.38
    source_y_frac: float = 0.10
    shear_g1: float = 0.04
    shear_g2: float = 0.02
    psf_fwhm: float = 0.064
    zp: float = 28.0
    supersampling_factor: int = 4


def _q_pa_to_e1e2(q: float, pa_deg: float) -> tuple[float, float]:
    e = (1.0 - q) / (1.0 + q)
    pa = np.radians(pa_deg)
    return e * np.cos(2 * pa), e * np.sin(2 * pa)


def _convolve_gaussian(image: np.ndarray, fwhm: float, pixel_scale: float) -> np.ndarray:
    sigma = fwhm / (2.355 * pixel_scale)
    return gaussian_filter(image, sigma=sigma)


def simulate_galaxygenius_lens_system(
    morph_type: str,
    cfg: LensStampSimConfig | None = None,
    *,
    figures_dir: Path | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Unified lenstronomy render: lensed source + INTERPOL GalaxyGenius lens light.

    Returns
    -------
    image : ndarray
        Combined model image from a single lenstronomy ray-tracing pass (+ PSF).
    meta : dict
        GalaxyGenius catalog metadata and render settings.
    """
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    cfg = cfg or LensStampSimConfig()
    interpol_kwargs, meta = build_galaxygenius_interpol_kwargs(
        morph_type,
        re_arcsec=cfg.lens_re_arcsec,
        magnitude=cfg.lens_mag,
        pa_deg=cfg.lens_pa_deg,
        figures_dir=figures_dir,
    )

    e1_l, e2_l = _q_pa_to_e1e2(cfg.lens_q, cfg.lens_pa_deg)
    source_x = cfg.source_x_frac * cfg.theta_E
    source_y = cfg.source_y_frac * cfg.theta_E

    kwargs_lens = [
        dict(theta_E=cfg.theta_E, center_x=0.0, center_y=0.0, e1=e1_l, e2=e2_l),
        dict(gamma1=cfg.shear_g1, gamma2=cfg.shear_g2),
    ]
    kwargs_source_mag = [dict(
        R_sersic=cfg.source_re_arcsec,
        n_sersic=cfg.source_n,
        center_x=source_x,
        center_y=source_y,
        e1=0.08,
        e2=0.04,
        magnitude=cfg.source_mag,
    )]
    kwargs_lens_light_mag = [interpol_kwargs]

    kwargs_band = dict(
        read_noise=0,
        ccd_gain=1.0,
        sky_brightness=30.0,
        exposure_time=3600,
        magnitude_zero_point=cfg.zp,
        num_exposures=1,
        seeing=0.01,
        pixel_scale=cfg.pixel_scale,
        psf_type="NONE",
    )
    model_lists = dict(
        lens_model_list=["SIE", "SHEAR"],
        lens_light_model_list=["INTERPOL"],
        source_light_model_list=["SERSIC_ELLIPSE"],
    )
    sim = SimAPI(
        numpix=int(cfg.numpix),
        kwargs_single_band=kwargs_band,
        kwargs_model=model_lists,
    )
    kw_lens_light, kw_source, _ = sim.magnitude2amplitude(
        kwargs_lens_light_mag, kwargs_source_mag,
    )
    im_model = sim.image_model_class(dict(
        supersampling_factor=int(cfg.supersampling_factor),
        supersampling_convolution=False,
    ))
    image = im_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kw_source,
        kwargs_lens_light=kw_lens_light,
    )
    image = np.clip(image, 0, None)
    image = _convolve_gaussian(image, cfg.psf_fwhm, cfg.pixel_scale)
    return image, meta
