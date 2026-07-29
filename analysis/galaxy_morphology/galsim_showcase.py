"""GalSim morphology showcase renderers (single- and multi-component)."""
from __future__ import annotations

import numpy as np

from showcase_common import (
    BAND,
    MORPH_PARAMS,
    PANEL_SIZE,
    PIXEL_SCALE,
    PSF_FWHM,
    TOTAL_MAG,
    mag_to_flux,
    q_to_e1e2,
)


def _apply_ellipticity(prof, e1: float, e2: float):
    if np.hypot(e1, e2) > 1e-8:
        prof = prof.shear(g1=float(e1), g2=float(e2))
    return prof


def _component_profile(comp: dict, flux: float):
    import galsim

    prof = galsim.Sersic(
        n=float(comp["n_sersic"]),
        half_light_radius=float(comp["R_sersic"]),
        flux=float(flux),
    )
    prof = _apply_ellipticity(prof, comp["e1"], comp["e2"])
    dx = float(comp.get("center_x", 0.0))
    dy = float(comp.get("center_y", 0.0))
    if abs(dx) > 1e-9 or abs(dy) > 1e-9:
        prof = prof.shift(dx, dy)
    return prof


def _draw_galsim_profile(gal, numpix: int = PANEL_SIZE, pixel_scale: float = PIXEL_SCALE) -> np.ndarray:
    import galsim

    psf = galsim.Gaussian(fwhm=PSF_FWHM)
    obj = galsim.Convolve(gal, psf)
    img = galsim.Image(numpix, numpix, scale=pixel_scale)
    obj.drawImage(img, method="auto")
    return np.array(img.array, dtype=np.float64)


def render_galsim_single(morph_type: str, params: dict, idx: int) -> tuple[np.ndarray, str]:
    import galsim

    e1, e2 = q_to_e1e2(params["q"])
    flux = mag_to_flux(TOTAL_MAG)
    gal = galsim.Sersic(
        n=float(params["n_sersic"]),
        half_light_radius=float(params["R_eff"]),
        flux=flux,
    )
    gal = _apply_ellipticity(gal, e1, e2)

    image = _draw_galsim_profile(gal)
    text = (
        f"GalSim single Sersic\n"
        f"n={params['n_sersic']:.1f}  q={params['q']:.2f}  Re={params['R_eff']:.2f}\""
    )
    return image, text


def render_galsim_multicomponent(
    morph_type: str,
    params: dict,
    morph_cfg: dict,
    rng,
    idx: int,
) -> tuple[np.ndarray, str]:
    import galsim

    from prism.morphology import band_flux_fractions, build_components

    e1, e2 = q_to_e1e2(params["q"])
    base = dict(
        R_sersic=params["R_eff"],
        n_sersic=params["n_sersic"],
        e1=e1, e2=e2,
        center_x=0.0, center_y=0.0,
    )
    components = build_components(base, morph_type, rng, morph_cfg, role="lens")
    fractions = band_flux_fractions(components, BAND, morph_type, morph_cfg)
    total_flux = mag_to_flux(TOTAL_MAG)

    profiles = []
    for comp, frac in zip(components, fractions):
        profiles.append(_component_profile(comp, total_flux * frac))

    gal = galsim.Add(profiles) if len(profiles) > 1 else profiles[0]
    image = _draw_galsim_profile(gal)
    comp_names = ", ".join(c["name"] for c in components)
    text = f"GalSim multi-component\ncomponents: {comp_names}"
    return image, text
