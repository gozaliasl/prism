#!/usr/bin/env python3
"""
Single elliptical GalaxyGenius lens mock using jwst_lens_simulator conventions.

Mass: SIE + shear (isolated field)
Source: pipeline build_light_model (multicomponent per default_config.yaml)
Lens light: GalaxyGenius elliptical via lenstronomy INTERPOL (one ImageModel pass)

Usage:
  conda run -n astro-clean python analysis/galaxy_morphology/plot_galaxygenius_single_lens.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import PowerNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prism.morphology.pipeline_lens_mock import (
    PipelineGalaxyGeniusLensConfig,
    make_pipeline_demo_row,
    simulate_galaxygenius_pipeline_lens,
)


def _load_config():
    from prism.core.jwst_lens_simulator import load_config, create_jwst_band_configs, CONFIG

    cfg_path = REPO_ROOT / "configs" / "default_config.yaml"
    load_config(str(cfg_path))
    return CONFIG


def _render_component_panels(metadata, band_cfgs, rng, config, gg_cfg, row, lens_pa, figures_dir):
    """Lens-only and source-only panels via lenstronomy toggles."""
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    from prism.morphology import build_light_model as gm_build_light_model
    from prism.morphology.stamp_compositor import build_galaxygenius_interpol_kwargs
    from prism.core.jwst_lens_simulator import (
        UPPER_BANDS,
        BAND_TO_LOWER,
        apply_psf_convolution,
        ellipticity,
        filter_lenstronomy_params,
    )

    band = "F150W"
    cfg = dict(band_cfgs[band])
    cfg["pixel_scale"] = gg_cfg.pixel_scale

    theta_E = metadata["theta_E"]
    lens_q = row["lens_axis_ratio"]
    e1_l, e2_l = ellipticity(lens_q, lens_pa)
    env = config.get("environment", {}).get("types", {}).get("isolated_field", {})
    shear_g = 0.035
    kwargs_lens = [
        dict(theta_E=theta_E, center_x=0.0, center_y=0.0, e1=e1_l, e2=e2_l),
        dict(gamma1=shear_g, gamma2=0.01),
    ]

    phot = config["photometry"]
    lens_mag = float(row[f"lens_mag_{BAND_TO_LOWER[band]}"])
    src_mag = float(row[f"source_mag_{BAND_TO_LOWER[band]}"])

    interpol_template, _ = build_galaxygenius_interpol_kwargs(
        gg_cfg.morph_type,
        re_arcsec=gg_cfg.re_arcsec,
        magnitude=lens_mag,
        pa_deg=lens_pa,
        figures_dir=figures_dir,
    )

    lensed_source = dict(
        R_sersic=metadata["source_radius_arcsec"],
        n_sersic=metadata["source_n_sersic"],
        center_x=row["source_x"],
        center_y=row["source_y"],
        e1=0.08,
        e2=0.04,
    )
    src_mag_by_band = {b: float(row.get(f"source_mag_{BAND_TO_LOWER[b]}", src_mag)) for b in UPPER_BANDS}
    lens_mag_by_band = {b: float(row.get(f"lens_mag_{BAND_TO_LOWER[b]}", lens_mag)) for b in UPPER_BANDS}
    source_fragment, source_kw_by_band, _ = gm_build_light_model(
        "source", lensed_source, src_mag_by_band, UPPER_BANDS, rng, config,
        morph_type=gg_cfg.source_morph_type,
        morph_seed=42,
    )

    model_lists = dict(
        lens_model_list=["SIE", "SHEAR"],
        lens_light_model_list=["INTERPOL"],
        source_light_model_list=list(source_fragment),
    )
    numerics = dict(supersampling_factor=4, supersampling_convolution=False)
    sim = SimAPI(numpix=gg_cfg.numpix, kwargs_single_band=filter_lenstronomy_params(cfg), kwargs_model=model_lists)
    im_model = sim.image_model_class(numerics)

    kw_lens_light, kw_src, _ = sim.magnitude2amplitude(
        [dict(interpol_template, magnitude=lens_mag)],
        list(source_kw_by_band[band]),
    )

    lens_only = im_model.image(
        kwargs_lens=kwargs_lens, kwargs_source=[], kwargs_lens_light=kw_lens_light,
        source_add=False, lens_light_add=True,
    )
    source_only = im_model.image(
        kwargs_lens=kwargs_lens, kwargs_source=kw_src, kwargs_lens_light=[],
        source_add=True, lens_light_add=False,
    )
    sigma = 0.064 / (2.355 * gg_cfg.pixel_scale)
    from scipy.ndimage import gaussian_filter
    lens_only = gaussian_filter(np.clip(lens_only, 0, None), sigma=sigma)
    source_only = gaussian_filter(np.clip(source_only, 0, None), sigma=sigma)
    return lens_only, source_only


def _show(ax, image, title, fov, note=None):
    pos = np.clip(image, 0, None)
    vmax = max(float(np.percentile(pos, 99.7)), 1e-12)
    ax.imshow(
        image, origin="lower", cmap="magma",
        extent=[-fov, fov, -fov, fov],
        norm=PowerNorm(gamma=0.42, vmin=0, vmax=vmax),
        aspect="equal",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("arcsec", fontsize=9)
    ax.set_ylabel("arcsec", fontsize=9)
    if note:
        ax.text(
            0.03, 0.97, note, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
            va="top", fontsize=7, family="monospace",
        )


def main():
    config = _load_config()
    rng = np.random.default_rng(42)
    from prism.core.jwst_lens_simulator import create_jwst_band_configs

    band_cfgs = create_jwst_band_configs(rng=rng, use_distribution=False)
    figures_dir = REPO_ROOT / "analysis" / "galaxy_morphology" / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    morph_cfg = dict(config.get("morphology", {}))
    morph_cfg["multicomponent_enabled"] = True
    config = dict(config)
    config["morphology"] = morph_cfg

    gg_cfg = PipelineGalaxyGeniusLensConfig(
        morph_type="elliptical",
        source_morph_type="spiral",
        numpix=int(config.get("image_size", 300)),
        pixel_scale=float(config.get("pixel_scale", 0.031)),
        seed=42,
    )

    row, lens_pa = make_pipeline_demo_row(rng, config)
    images, meta = simulate_galaxygenius_pipeline_lens(
        row, band_cfgs, rng, config,
        lens_pa=lens_pa,
        gg_cfg=gg_cfg,
        figures_dir=figures_dir,
    )

    band = "F150W"
    fov = gg_cfg.numpix * gg_cfg.pixel_scale / 2.0
    full = images[band]
    lens_only, source_only = _render_component_panels(
        meta, band_cfgs, rng, config, gg_cfg, row, lens_pa, figures_dir,
    )

    note = (
        f"TNG100 subhalo {meta['galaxygenius'].get('subhalo_id', 0)}\n"
        f"θ_E={meta['theta_E']:.2f}\"  z_l={row['lens_redshift']:.2f}  "
        f"z_s={row['source_redshift']:.1f}\n"
        f"source: {meta['source_morph_type']}  "
        f"({', '.join(meta['source_components'])})\n"
        "lenstronomy INTERPOL + pipeline source"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    _show(axes[0], lens_only, "Lens light (GalaxyGenius elliptical)", fov,
          "INTERPOL lens plane\n(unlensed)")
    _show(axes[1], source_only, f"Lensed source ({meta['source_morph_type']})", fov,
          "build_light_model\n(gravitationally lensed)")
    _show(axes[2], full, "Combined (pipeline-style F150W)", fov, note)

    fig.suptitle(
        "Single elliptical GalaxyGenius lens + pipeline lensed source (jwst_lens_simulator conventions)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = figures_dir / "galaxygenius_single_elliptical_lens.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
