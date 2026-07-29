"""
Single strong-lens mock with GalaxyGenius elliptical INTERPOL lens light,
following jwst_lens_simulator conventions for mass model, source, and bands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .stamp_compositor import build_galaxygenius_interpol_kwargs


@dataclass(frozen=True)
class PipelineGalaxyGeniusLensConfig:
    morph_type: str = "elliptical"
    re_arcsec: float = 0.8
    numpix: int = 300
    pixel_scale: float = 0.031
    seed: int = 42
    source_morph_type: str = "spiral"
    include_field_galaxies: bool = False


def make_pipeline_demo_row(
    rng: np.random.Generator,
    config: dict[str, Any],
    *,
    theta_E: float = 1.15,
    lens_z: float = 0.45,
    source_z: float = 2.0,
    lens_q: float = 0.75,
) -> tuple[dict[str, Any], float]:
    """Build a catalog-style row dict matching jwst_lens_simulator fields."""
    from prism.core.jwst_lens_simulator import UPPER_BANDS, BAND_TO_LOWER

    phot = config.get("photometry", {})
    offset_ratio = 0.42
    source_angle = float(rng.uniform(0.0, 2 * np.pi))
    source_x = float(np.clip(offset_ratio * theta_E * np.cos(source_angle), -0.4, 0.4))
    source_y = float(np.clip(offset_ratio * theta_E * np.sin(source_angle), -0.4, 0.4))
    lens_pa = float(rng.uniform(-180.0, 180.0))

    lens_mag = float(phot.get("lens_base_mag_zero", 21.0))
    src_mag = float(phot.get("source_base_mag", 20.5)) + float(
        phot.get("min_source_fainter_than_lens_mag", 1.2)
    )

    row: dict[str, Any] = {
        "lens_id": "galaxygenius_elliptical_pipeline_demo",
        "theta_E": float(theta_E),
        "lens_axis_ratio": float(lens_q),
        "lens_redshift": float(lens_z),
        "source_redshift": float(source_z),
        "source_axis_ratio": 0.65,
        "source_x": source_x,
        "source_y": source_y,
        "source_offset_ratio": offset_ratio,
        "source_angle": source_angle,
        "lens_mass_log10": 11.3,
        "n_rest": 4.0,
    }
    for band in UPPER_BANDS:
        bl = BAND_TO_LOWER[band]
        row[f"lens_mag_{bl}"] = lens_mag + float(rng.normal(0.0, 0.03))
        row[f"source_mag_{bl}"] = src_mag + float(rng.normal(0.0, 0.05))
    return row, lens_pa


def simulate_galaxygenius_pipeline_lens(
    row: dict[str, Any],
    band_cfgs: dict[str, dict],
    rng: np.random.Generator,
    config: dict[str, Any],
    *,
    lens_pa: float,
    gg_cfg: PipelineGalaxyGeniusLensConfig | None = None,
    psf_data: dict | None = None,
    figures_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Render a lens system like ``simulate_complete_lens_system_with_real_fields``,
    but with GalaxyGenius INTERPOL replacing parametric lens light.

    Returns
    -------
    images : dict[band -> ndarray]
    metadata : dict
    """
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    from prism.morphology import build_light_model as gm_build_light_model
    from prism.core.jwst_lens_simulator import (
        UPPER_BANDS,
        apply_psf_convolution,
        ellipticity,
        filter_lenstronomy_params,
        get_psf_for_simulation,
    )

    gg_cfg = gg_cfg or PipelineGalaxyGeniusLensConfig()
    phot = config.get("photometry", {})
    geo = config.get("geometry", {})
    morph_cfg = dict(config.get("morphology", {}))

    theta_E = float(np.clip(row.get("theta_E", 1.15), 0.3, 5.0))
    lens_q = float(np.clip(row.get("lens_axis_ratio", 0.75), 0.2, 1.0))
    source_q = float(np.clip(row.get("source_axis_ratio", 0.65), 0.2, 1.0))
    lens_z = float(row.get("lens_redshift", 0.45))
    source_z = float(row.get("source_redshift", 2.0))

    e1_l, e2_l = ellipticity(lens_q, lens_pa)
    e1_l, e2_l = float(np.clip(e1_l, -0.8, 0.8)), float(np.clip(e2_l, -0.8, 0.8))
    source_pa = float(row.get("source_pa", rng.uniform(-180, 180)))
    e1_s, e2_s = ellipticity(source_q, source_pa)
    e1_s, e2_s = float(np.clip(e1_s, -0.8, 0.8)), float(np.clip(e2_s, -0.8, 0.8))

    env = config.get("environment", {}).get("types", {}).get("isolated_field", {})
    shear_g = float(rng.uniform(env.get("shear_min", 0.01), env.get("shear_max", 0.05)))
    shear_phi = float(rng.uniform(0.0, np.pi))
    kwargs_lens = [
        dict(theta_E=theta_E, center_x=0.0, center_y=0.0, e1=e1_l, e2=e2_l),
        dict(gamma1=shear_g * np.cos(2 * shear_phi), gamma2=shear_g * np.sin(2 * shear_phi)),
    ]

    lens_radius = float(np.clip(
        geo.get("lens_radius_min", 0.35) + 0.35,
        geo.get("lens_radius_min", 0.35),
        geo.get("lens_radius_max", 2.2),
    ))
    size_fraction = float(np.clip(
        rng.lognormal(np.log(geo.get("source_fraction_mean", 0.12)),
                      geo.get("source_fraction_sigma", 0.45)),
        geo.get("source_fraction_min", 0.02),
        geo.get("source_fraction_max", 0.4),
    ))
    source_radius = float(np.clip(
        theta_E * size_fraction * np.clip((1 + source_z) ** (-1.2), 0.1, 1.0),
        0.05,
        lens_radius * geo.get("max_source_to_lens_ratio", 0.6),
    ))
    source_n = float(np.clip(rng.lognormal(np.log(1.2), 0.4), 0.3, 8.0))

    lensed_source = dict(
        R_sersic=source_radius,
        n_sersic=source_n,
        center_x=float(row.get("source_x", 0.0)),
        center_y=float(row.get("source_y", 0.0)),
        e1=e1_s,
        e2=e2_s,
    )

    _min_delta = float(phot.get("min_source_fainter_than_lens_mag", 1.2))
    lens_mag_by_band = {}
    src_mag_by_band = {}
    for band in UPPER_BANDS:
        from prism.core.jwst_lens_simulator import BAND_TO_LOWER
        _lm = float(row.get(f"lens_mag_{BAND_TO_LOWER[band]}", phot.get("lens_base_mag_zero", 21.0)))
        _sm = float(row.get(f"source_mag_{BAND_TO_LOWER[band]}", phot.get("source_base_mag", 20.5)))
        _lm = float(np.clip(_lm, phot.get("lens_mag_min", 18.0), phot.get("lens_mag_max", 27.0)))
        _sm = float(np.clip(_sm, phot.get("source_mag_min", 19.0), phot.get("source_mag_max", 27.5)))
        if _sm < _lm + _min_delta:
            _sm = _lm + _min_delta
        lens_mag_by_band[band] = _lm
        src_mag_by_band[band] = _sm

    _lens_id = str(row.get("lens_id", 0))
    _source_morph_seed = int(abs(hash(_lens_id + ":source")) % (2 ** 32))
    source_fragment, source_kw_by_band, source_morph_type = gm_build_light_model(
        "source",
        lensed_source,
        src_mag_by_band,
        UPPER_BANDS,
        rng,
        config,
        morph_type=gg_cfg.source_morph_type,
        morph_seed=_source_morph_seed,
    )

    interpol_template, gg_meta = build_galaxygenius_interpol_kwargs(
        gg_cfg.morph_type,
        re_arcsec=float(gg_cfg.re_arcsec),
        magnitude=lens_mag_by_band["F150W"],
        pa_deg=lens_pa,
        figures_dir=figures_dir,
    )

    model_lists = dict(
        lens_model_list=["SIE", "SHEAR"],
        lens_light_model_list=["INTERPOL"],
        source_light_model_list=list(source_fragment),
    )
    numerics = dict(supersampling_factor=4, supersampling_convolution=False)

    psf_arrays = None
    if psf_data is not None:
        psf_arrays = get_psf_for_simulation(psf_data, row.get("lens_id"), rng)

    images: dict[str, np.ndarray] = {}
    for band in UPPER_BANDS:
        cfg = dict(band_cfgs[band])
        cfg["pixel_scale"] = float(gg_cfg.pixel_scale)

        kw_lens_light_mag = [dict(
            interpol_template,
            magnitude=lens_mag_by_band[band],
        )]
        kw_src = list(source_kw_by_band[band])

        sim = SimAPI(
            numpix=int(gg_cfg.numpix),
            kwargs_single_band=filter_lenstronomy_params(cfg),
            kwargs_model=model_lists,
        )
        kw_lens_light, kw_src_amp, _ = sim.magnitude2amplitude(kw_lens_light_mag, kw_src)
        im_model = sim.image_model_class(numerics)
        image = im_model.image(
            kwargs_lens=kwargs_lens,
            kwargs_source=kw_src_amp,
            kwargs_lens_light=kw_lens_light,
        )
        image = np.clip(image, 0, None)

        if psf_arrays and band in psf_arrays and psf_arrays[band] is not None:
            image = apply_psf_convolution(image, psf_arrays[band])

        images[band] = image.astype(np.float32)

    metadata = {
        "row": row,
        "galaxygenius": gg_meta,
        "lens_pa_deg": lens_pa,
        "lens_radius_arcsec": lens_radius,
        "source_morph_type": source_morph_type,
        "source_radius_arcsec": source_radius,
        "source_n_sersic": source_n,
        "theta_E": theta_E,
        "lens_model_list": ["SIE", "SHEAR"],
        "lens_light_model": "INTERPOL",
        "source_components": list(source_fragment),
        "render_mode": "lenstronomy_unified",
    }
    return images, metadata
