"""Lenstronomy morphology showcase renderers (single- and multi-component)."""
from __future__ import annotations

import numpy as np

from showcase_common import BAND, PSF_FWHM, TOTAL_MAG, q_to_e1e2


KWARGS_NO_PSF = dict(
    read_noise=0, ccd_gain=1.0, sky_brightness=30.0, exposure_time=3600,
    magnitude_zero_point=28.0, num_exposures=1, seeing=0.01,
    pixel_scale=0.031, psf_type="NONE",
)

NUMPIX = 256


def _convolve_gaussian(image, fwhm, pixel_scale):
    from scipy.ndimage import gaussian_filter

    sigma = fwhm / (2.355 * pixel_scale)
    out = gaussian_filter(image, sigma=sigma)
    if np.sum(out) > 0 and np.sum(image) > 0:
        out *= np.sum(image) / np.sum(out)
    return out


def _render_lenstronomy(fragment, kw_list):
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    model_lists = dict(
        lens_model_list=[],
        lens_light_model_list=fragment,
        source_light_model_list=[],
    )
    sim = SimAPI(
        numpix=NUMPIX,
        kwargs_single_band=KWARGS_NO_PSF,
        kwargs_model=model_lists,
    )
    kw_amp, _, _ = sim.magnitude2amplitude(kw_list, [])
    im_model = sim.image_model_class(dict(supersampling_factor=4, supersampling_convolution=False))
    image = im_model.image(kwargs_lens=[], kwargs_source=[], kwargs_lens_light=kw_amp)
    image = _convolve_gaussian(np.clip(image, 0, None), PSF_FWHM, 0.031)
    return image


def render_lenstronomy_single(morph_type: str, params: dict, idx: int) -> tuple[np.ndarray, str]:
    e1, e2 = q_to_e1e2(params["q"])
    kw = dict(
        R_sersic=params["R_eff"],
        n_sersic=params["n_sersic"],
        e1=e1, e2=e2,
        center_x=0.0, center_y=0.0,
        magnitude=TOTAL_MAG,
    )
    image = _render_lenstronomy(["SERSIC_ELLIPSE"], [kw])
    text = (
        f"Lenstronomy single Sersic\n"
        f"n={params['n_sersic']:.1f}  q={params['q']:.2f}  Re={params['R_eff']:.2f}\""
    )
    return image, text


def render_lenstronomy_multicomponent(
    morph_type: str,
    params: dict,
    morph_cfg: dict,
    rng,
    idx: int,
) -> tuple[np.ndarray, str]:
    from prism.morphology import band_flux_fractions, build_components, fractions_to_magnitudes

    e1, e2 = q_to_e1e2(params["q"])
    base = dict(
        R_sersic=params["R_eff"],
        n_sersic=params["n_sersic"],
        e1=e1, e2=e2,
        center_x=0.0, center_y=0.0,
    )
    components = build_components(base, morph_type, rng, morph_cfg, role="lens")
    fractions = band_flux_fractions(components, BAND, morph_type, morph_cfg)
    mags = fractions_to_magnitudes(TOTAL_MAG, fractions)

    fragment = ["SERSIC_ELLIPSE"] * len(components)
    kw_list = []
    for comp, mag in zip(components, mags):
        kw = {k: v for k, v in comp.items() if k not in ("name", "flux_fraction_ref")}
        kw["magnitude"] = mag
        kw_list.append(kw)

    image = _render_lenstronomy(fragment, kw_list)
    comp_names = ", ".join(c["name"] for c in components)
    text = f"Lenstronomy multi-component\ncomponents: {comp_names}"
    return image, text
