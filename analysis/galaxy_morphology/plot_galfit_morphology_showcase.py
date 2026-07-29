#!/usr/bin/env python3
"""
Render 12 GALFIT-style morphology models as a 3x4 model-only showcase.

Disk galaxies (spiral, late_spiral, barred_spiral, edge_on) use an analytic
bulge + exponential disk + log-spiral / bar / dust-lane model built at native
resolution, then convolved with a single JWST PSF.  All other types use
lenstronomy GALFIT component renders (also pre-PSF), with texture applied
before the one PSF pass.

Usage: conda run -n astro-clean python analysis/galaxy_morphology/plot_galfit_morphology_showcase.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prism.morphology.galfit_prescriptions import build_galfit_components
from prism.morphology import band_flux_fractions, fractions_to_magnitudes
from lenstronomy.SimulationAPI.sim_api import SimAPI
from showcase_enhancements import (
    ANALYTIC_MORPH,
    build_analytic_showcase,
    convolve_gaussian_psf,
    enhance_showcase_model,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BAND = 'F150W'
TOTAL_MAG = 21.0
PIXEL_SCALE = 0.03
NUMPIX = 260
SEEING = 0.064

KWARGS_NO_PSF = dict(
    read_noise=0, ccd_gain=1.0, sky_brightness=30.0, exposure_time=3600,
    magnitude_zero_point=28.0, num_exposures=1, seeing=0.01,
    pixel_scale=PIXEL_SCALE, psf_type='NONE',
)

MORPH_PARAMS = {
    'elliptical':    dict(n_sersic=4.2, q=0.75, R_eff=0.8,  label='Elliptical'),
    's0':            dict(n_sersic=2.8, q=0.85, R_eff=0.6,  label='S0 / Early-type'),
    'spiral':        dict(n_sersic=1.5, q=0.70, R_eff=1.0,  label='Spiral'),
    'late_spiral':   dict(n_sersic=1.0, q=0.45, R_eff=1.2,  label='Late Spiral (Sc/Sd)'),
    'edge_on':       dict(n_sersic=1.0, q=0.20, R_eff=1.1,  label='Edge-on Disk'),
    'barred_spiral': dict(n_sersic=1.5, q=0.75, R_eff=1.0,  label='Barred Spiral'),
    'ring':          dict(n_sersic=2.0, q=0.85, R_eff=1.3,  label='Ring Galaxy'),
    'post_merger':   dict(n_sersic=3.2, q=0.65, R_eff=0.9,  label='Post-Merger'),
    'irregular':     dict(n_sersic=0.6, q=0.55, R_eff=0.9,  label='Irregular'),
    'primordial':    dict(n_sersic=0.3, q=0.60, R_eff=0.7,  label='Primordial'),
    'clumpy':        dict(n_sersic=0.5, q=0.35, R_eff=1.1,  label='Clumpy'),
    'starburst':     dict(n_sersic=0.6, q=0.50, R_eff=0.8,  label='Starburst'),
}

_SHOWCASE_STRIP_COMPONENTS = {'barred_spiral': {'bar'}}


def q_to_e1e2(q):
    e = (1.0 - q) / (1.0 + q)
    return e, 0.0


def galfit_summary_text(components):
    parts = []
    for c in components:
        prof = c.get('profile', 'SERSIC')
        if prof == 'EXP':
            h = c['R_sersic'] / 1.678
            parts.append(f"{c['name']}:Exp(h={h:.2f}\")")
        else:
            parts.append(f"{c['name']}:Sersic(n={c['n_sersic']:.1f},Re={c['R_sersic']:.2f}\")")
    return '\n'.join(parts)


def _normalize_flux(components):
    total = sum(c['flux_fraction_ref'] for c in components)
    if total <= 0:
        return components
    for c in components:
        c['flux_fraction_ref'] /= total
    return components


def _galfit_components(morph_type, params, morph_cfg, rng):
    e1, e2 = q_to_e1e2(params['q'])
    base = dict(
        R_sersic=params['R_eff'], n_sersic=params['n_sersic'],
        e1=e1, e2=e2, center_x=0.0, center_y=0.0,
    )
    components = build_galfit_components(base, morph_type, rng, morph_cfg, role='lens')
    strip = _SHOWCASE_STRIP_COMPONENTS.get(morph_type, set())
    if strip:
        components = _normalize_flux([c for c in components if c['name'] not in strip])
    return components


def render_lenstronomy_no_psf(morph_type, params, morph_cfg, rng):
    components = _galfit_components(morph_type, params, morph_cfg, rng)
    fractions = band_flux_fractions(components, BAND, morph_type, morph_cfg)
    mags = fractions_to_magnitudes(TOTAL_MAG, fractions)

    fragment = ['SERSIC_ELLIPSE'] * len(components)
    kw_list = []
    for comp, mag in zip(components, mags):
        kw = {k: v for k, v in comp.items() if k not in ('name', 'flux_fraction_ref', 'profile')}
        kw['magnitude'] = mag
        kw_list.append(kw)

    sim = SimAPI(
        numpix=NUMPIX,
        kwargs_single_band=KWARGS_NO_PSF,
        kwargs_model=dict(lens_model_list=[], lens_light_model_list=fragment, source_light_model_list=[]),
    )
    kw_amp, _, _ = sim.magnitude2amplitude(kw_list, [])
    im_model = sim.image_model_class(dict(supersampling_factor=4, supersampling_convolution=False))
    image = im_model.image(kwargs_lens=[], kwargs_source=[], kwargs_lens_light=kw_amp)
    return np.clip(image, 0, None), components


def render_showcase_model(morph_type, params, morph_cfg, rng, idx):
    if morph_type in ANALYTIC_MORPH:
        sharp = build_analytic_showcase(morph_type, params, NUMPIX, PIXEL_SCALE, TOTAL_MAG, seed=500 + idx)
        components = _galfit_components(morph_type, params, morph_cfg, rng)
    else:
        sharp, components = render_lenstronomy_no_psf(morph_type, params, morph_cfg, rng)
        sharp = enhance_showcase_model(sharp, morph_type, params, idx, NUMPIX, PIXEL_SCALE)

    model = convolve_gaussian_psf(sharp, SEEING, PIXEL_SCALE)
    model = np.clip(model, 0, None)
    # Preserve total flux through PSF convolution
    if np.sum(model) > 0 and np.sum(sharp) > 0:
        model *= np.sum(sharp) / np.sum(model)
    return model, components


def render_panel(ax, image, fov, title, param_text=None):
    vmax = np.percentile(image, 99.8)
    ax.imshow(
        image, origin='lower', cmap='magma',
        extent=[-fov, fov, -fov, fov],
        norm=PowerNorm(gamma=0.42, vmin=0, vmax=max(vmax, 1e-12)),
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('arcsec', fontsize=10)
    ax.set_ylabel('arcsec', fontsize=10)
    ax.tick_params(labelsize=8)
    if param_text:
        ax.text(
            0.03, 0.97, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.88),
            verticalalignment='top', fontsize=8, family='monospace',
        )


def main():
    cfg_path = REPO_ROOT / 'configs' / 'default_config.yaml'
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    morph_cfg = dict(full_cfg.get('morphology', {}))
    morph_cfg['multicomponent_enabled'] = True

    rng = np.random.default_rng(42)
    fov = NUMPIX * PIXEL_SCALE / 2.0
    out_dir = REPO_ROOT / 'analysis' / 'galaxy_morphology' / 'reports' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    axes = axes.ravel()

    for idx, (morph_type, params) in enumerate(MORPH_PARAMS.items()):
        model, components = render_showcase_model(morph_type, params, morph_cfg, rng, idx)
        title = f"{params['label']}\n({morph_type})"
        param_text = f"GALFIT B+D, PSF={SEEING}\" FWHM\n{galfit_summary_text(components)}"
        render_panel(axes[idx], model, fov, title, param_text)

    fig.suptitle(
        f'GALFIT-style morphology models ({BAND}, structure pre-PSF, model only)',
        fontsize=16, fontweight='bold', y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / 'galfit_morphology_showcase.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
