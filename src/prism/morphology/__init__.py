"""
galaxy_morphology — native multi-component light models for the
strong-lens pipeline.

Public API: build_light_model(...)
"""
import numpy as np

from .taxonomy import classify_morphology, MORPH_COMPONENTS, _axis_ratio_from_e
from .components import build_components, fractions_to_magnitudes
from .band_variation import band_flux_fractions
from .ring_merger_profiles import build_ring_interpol_kwargs, build_tidal_tail_interpol_kwargs
from .galfit_prescriptions import build_galfit_components, galfit_config_lines
from .galaxygenius_assets import load_morphology_panel, load_catalog, morphology_order
from .mock_observation import JWSTMockParams, mock_observe_image, JWST_NIRCAM_TABLE6
from .stamp_compositor import (
    build_galaxygenius_interpol_kwargs,
    build_interpol_lens_kwargs,
)
from .lens_stamp_sim import LensStampSimConfig, simulate_galaxygenius_lens_system
from .pipeline_lens_mock import (
    PipelineGalaxyGeniusLensConfig,
    make_pipeline_demo_row,
    simulate_galaxygenius_pipeline_lens,
)

__all__ = [
    "classify_morphology",
    "MORPH_COMPONENTS",
    "build_components",
    "build_galfit_components",
    "galfit_config_lines",
    "fractions_to_magnitudes",
    "band_flux_fractions",
    "build_light_model",
    "load_morphology_panel",
    "load_catalog",
    "morphology_order",
    "JWSTMockParams",
    "mock_observe_image",
    "JWST_NIRCAM_TABLE6",
    "build_galaxygenius_interpol_kwargs",
    "build_interpol_lens_kwargs",
    "LensStampSimConfig",
    "simulate_galaxygenius_lens_system",
    "PipelineGalaxyGeniusLensConfig",
    "make_pipeline_demo_row",
    "simulate_galaxygenius_pipeline_lens",
]


def build_light_model(role, base_params, total_mag_by_band, bands, rng, config,
                       morph_type=None, morph_seed=None):
    """
    Construct a native multi-component lenstronomy light-model fragment for
    a single galaxy (lens, source, or field member).

    Parameters
    ----------
    role : str
        'lens' | 'source' | 'field' (currently informational only -- the
        same component recipes are used for all roles).
    base_params : dict
        Existing single-SERSIC_ELLIPSE kwargs: R_sersic, n_sersic, e1, e2,
        center_x, center_y (no 'magnitude' key).
    total_mag_by_band : dict
        band -> total magnitude for this galaxy (as currently computed).
    bands : list[str]
        Bands to build per-band kwargs for (e.g. UPPER_BANDS).
    rng : np.random.Generator
    config : dict
        Full CONFIG dict (or at least the 'morphology' sub-block).
    morph_type : str, optional
        Pre-classified morphology type. If None, classified from
        base_params['n_sersic'] / axis ratio derived from e1/e2.
    morph_seed : int, optional
        If given, morphology classification and component placement
        (bar angle, bulge_secondary offset, etc.) draw from an independent
        `np.random.default_rng(morph_seed)` stream instead of `rng`, so
        enabling multicomponent mode does not perturb `rng`'s draw
        sequence (and therefore does not change which downstream
        mass-model/source-position values get sampled). If omitted, falls
        back to drawing from `rng` directly (legacy behavior).

    Returns
    -------
    light_model_list_fragment : list[str]
        e.g. ['SERSIC_ELLIPSE'] or ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
    kwargs_fragment_by_band : dict[str, list[dict]]
        band -> list of per-component kwargs dicts (with 'magnitude').
    morph_type : str
        The (possibly classified) morphology type.
    """
    morph_cfg = config.get('morphology', {}) if isinstance(config, dict) else {}

    role_enabled = {
        'lens': morph_cfg.get('apply_to_lens', True),
        'source': morph_cfg.get('apply_to_source', True),
    }.get(role, True)

    if not morph_cfg.get('multicomponent_enabled', False) or not role_enabled:
        # Drop-in single-component fallback (unchanged behavior, including
        # RNG draw sequence -- does NOT classify morph_type if not given).
        fragment = ["SERSIC_ELLIPSE"]
        kwargs_by_band = {
            b: [dict(base_params, magnitude=float(total_mag_by_band[b]))]
            for b in bands
        }
        return fragment, kwargs_by_band, morph_type

    morph_rng = np.random.default_rng(morph_seed) if morph_seed is not None else rng

    if morph_type is None:
        q = _axis_ratio_from_e(base_params.get('e1', 0.0), base_params.get('e2', 0.0))
        # Ring morphology for lensed sources produces nested donut artifacts;
        # keep ring disabled for all roles.
        allow_ring = False
        morph_type = classify_morphology(base_params.get('n_sersic', 2.0), q, morph_rng,
                                          allow_ring=allow_ring)

    components = build_components(base_params, morph_type, morph_rng, morph_cfg, role=role)
    fragment = ["SERSIC_ELLIPSE"] * len(components)

    # Native ring / tidal-tail geometry: replace the thin-Sersic 'ring'
    # (morph_type == 'ring') and 'bulge_secondary' (morph_type ==
    # 'post_merger') component slots with procedural INTERPOL annular /
    # one-sided-tail surface-brightness maps (ring_merger_profiles.py),
    # rendered once here and reused across bands (only the per-band
    # magnitude differs). Gated by 'native_ring_merger_geometry' (default
    # True once multicomponent_enabled is True).
    interpol_image_kwargs = {}
    if morph_cfg.get('native_ring_merger_geometry', True):
        cx_t = float(base_params.get('center_x', 0.0))
        cy_t = float(base_params.get('center_y', 0.0))
        e1_t = float(base_params.get('e1', 0.0))
        e2_t = float(base_params.get('e2', 0.0))
        pa_total = 0.5 * np.arctan2(e2_t, e1_t)
        q_total = _axis_ratio_from_e(e1_t, e2_t)

        for idx, comp in enumerate(components):
            if comp['name'] == 'ring' and morph_type == 'ring':
                fragment[idx] = "INTERPOL"
                interpol_image_kwargs[idx] = build_ring_interpol_kwargs(
                    magnitude_ref=0.0,
                    center_x=comp['center_x'], center_y=comp['center_y'],
                    phi_G=pa_total,
                    r_ring_arcsec=float(comp['R_sersic']),
                    ring_width_arcsec=float(comp['R_sersic']) * 0.25,
                    axis_ratio=q_total,
                    target_size_arcsec=float(comp['R_sersic']) * 3.0,
                    rng=morph_rng,
                )
            elif comp['name'] == 'bulge_secondary' and morph_type == 'post_merger':
                fragment[idx] = "INTERPOL"
                dx = comp['center_x'] - cx_t
                dy = comp['center_y'] - cy_t
                dist = float(np.hypot(dx, dy))
                tail_angle = float(np.arctan2(dy, dx))
                length_arcsec = max(dist + float(comp['R_sersic']), 1e-3)
                interpol_image_kwargs[idx] = build_tidal_tail_interpol_kwargs(
                    magnitude_ref=0.0,
                    center_x=cx_t, center_y=cy_t,
                    phi_G=0.0,
                    tail_angle_rad=tail_angle,
                    offset_arcsec=max(dist * 0.2, 0.02),
                    length_arcsec=length_arcsec,
                    width_arcsec=float(comp['R_sersic']) * 0.5,
                    target_size_arcsec=length_arcsec * 3.0,
                    rng=morph_rng,
                )

    kwargs_by_band = {}
    for b in bands:
        fractions = band_flux_fractions(components, b, morph_type, morph_cfg)
        mags = fractions_to_magnitudes(float(total_mag_by_band[b]), fractions)
        kw_list = []
        for idx, (comp, mag) in enumerate(zip(components, mags)):
            if idx in interpol_image_kwargs:
                kw = dict(interpol_image_kwargs[idx])
                kw['magnitude'] = mag
            else:
                kw = {k: v for k, v in comp.items() if k not in ('name', 'flux_fraction_ref')}
                kw['magnitude'] = mag
            kw_list.append(kw)
        kwargs_by_band[b] = kw_list

    return fragment, kwargs_by_band, morph_type
