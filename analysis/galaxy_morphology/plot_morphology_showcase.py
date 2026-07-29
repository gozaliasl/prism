#!/usr/bin/env python3
"""
Render the 12 morphology types from src/galaxy_morphology's native
bulge/disk/bar/ring/bulge_secondary multi-component models, using the
tuned configs/default_config.yaml morphology block (item24: r_frac_source
1.15, neutral band_weight_bulge_like, item20 bt_fractions).

Produces a 3x4 grid figure analogous to the paper's
fig04_morphology_showcase, but driven by the actual lenstronomy
SERSIC_ELLIPSE component fragments + band_flux_fractions, instead of the
synthetic matplotlib-only profiles used in that figure.

Usage: conda run -n astro-clean python analysis/galaxy_morphology/plot_morphology_showcase.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.morphology import build_components, band_flux_fractions, fractions_to_magnitudes, MORPH_COMPONENTS
from lenstronomy.SimulationAPI.sim_api import SimAPI
from prism.core.simulator import apply_morphological_enhancements


REPO_ROOT = Path(__file__).resolve().parents[2]
BAND = 'F150W'
TOTAL_MAG = 21.0

KWARGS_SINGLE_BAND = dict(
    read_noise=4, ccd_gain=1.0, sky_brightness=22.0, exposure_time=1000,
    magnitude_zero_point=28.0, num_exposures=1, seeing=0.1,
    pixel_scale=0.03, psf_type='NONE',
)

NUMPIX = 200  # 6.0 arcsec FOV at 0.03"/pix -- gives low-n (irregular/primordial/
              # clumpy/starburst) profiles enough room to fall off within frame

# Representative structural parameters per morphology type (n_sersic,
# axis ratio q, R_eff in arcsec), chosen to span the range used in the
# paper's morphology showcase figure.
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


def q_to_e1e2(q):
    e = (1.0 - q) / (1.0 + q)
    return e, 0.0


def render_morph(morph_type, params, morph_cfg, rng):
    e1, e2 = q_to_e1e2(params['q'])
    base = dict(R_sersic=params['R_eff'], n_sersic=params['n_sersic'],
                 e1=e1, e2=e2, center_x=0.0, center_y=0.0)

    components = build_components(base, morph_type, rng, morph_cfg, role='lens')
    fractions = band_flux_fractions(components, BAND, morph_type, morph_cfg)
    mags = fractions_to_magnitudes(TOTAL_MAG, fractions)

    fragment = ["SERSIC_ELLIPSE"] * len(components)
    kw_list = []
    for comp, mag in zip(components, mags):
        kw = {k: v for k, v in comp.items() if k not in ('name', 'flux_fraction_ref')}
        kw['magnitude'] = mag
        kw_list.append(kw)

    model_lists = dict(lens_model_list=[], lens_light_model_list=fragment,
                        source_light_model_list=[])
    sim = SimAPI(numpix=NUMPIX, kwargs_single_band=KWARGS_SINGLE_BAND, kwargs_model=model_lists)
    kw_amp, _, _ = sim.magnitude2amplitude(kw_list, [])
    im_model = sim.image_model_class(dict(supersampling_factor=3, supersampling_convolution=False))
    image = im_model.image(kwargs_lens=[], kwargs_source=[], kwargs_lens_light=kw_amp)

    return image, components


def render_panel(ax, image, fov, title, param_text):
    image = np.clip(image, 0, None)
    vmax = np.percentile(image, 99.9)
    ax.imshow(image, origin='lower', cmap='inferno',
              extent=[-fov, fov, -fov, fov],
              norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax))
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('arcsec', fontsize=10)
    ax.set_ylabel('arcsec', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.text(0.03, 0.97, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.85),
            verticalalignment='top', fontsize=9)


def main():
    cfg_path = REPO_ROOT / "configs" / "default_config.yaml"
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    morph_cfg = dict(full_cfg.get('morphology', {}))
    morph_cfg['multicomponent_enabled'] = True

    rng = np.random.default_rng(42)
    fov = NUMPIX * 0.03 / 2.0  # half-width in arcsec
    out_dir = REPO_ROOT / "analysis" / "galaxy_morphology" / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: native multi-component only
    fig1, axes1 = plt.subplots(3, 4, figsize=(18, 13.5))
    axes1 = axes1.ravel()

    # Figure 2: native multi-component + pixel-space texture
    # (spiral arms, clumps, bar, ring, dust lane -- the same
    # apply_morphological_enhancements used in the production pipeline)
    fig2, axes2 = plt.subplots(3, 4, figsize=(18, 13.5))
    axes2 = axes2.ravel()

    for idx, (ax1, ax2, (morph_type, params)) in enumerate(zip(axes1, axes2, MORPH_PARAMS.items())):
        image, components = render_morph(morph_type, params, morph_cfg, rng)
        comp_names = [c['name'] for c in components]

        title = f"{params['label']}\n({morph_type})"
        param_text = (f"n={params['n_sersic']:.1f}  q={params['q']:.2f}  "
                       f"R_e={params['R_eff']:.1f}\"\ncomponents: {', '.join(comp_names)}")
        render_panel(ax1, image, fov, title, param_text)

        # Native bar/ring components already present for barred_spiral/ring
        # -- skip the matching pixel-space feature to avoid double-rendering.
        skip_native = morph_type in ('barred_spiral', 'ring')
        r_eff_pix = params['R_eff'] / 0.03
        # Several pixel-space features (e.g. add_bar_structure_to_image's
        # bar_profile amplitude of 0.3) are written in absolute-flux units
        # tuned for images normalized to a peak of ~O(1); our lenstronomy
        # amplitudes are on an arbitrary zeropoint-dependent scale, so
        # normalize to peak=1 before enhancing (display is rescaled anyway).
        image_norm = image / np.max(image)
        enhanced = apply_morphological_enhancements(
            {BAND: image_norm}, n_sersic=params['n_sersic'], q_ratio=params['q'],
            morph_type=morph_type, numpix=NUMPIX, pixel_scale=0.03, seed=100 + idx,
            position_angle=0.0, context='field',
            skip_native_bar_ring=skip_native, r_eff_pix=r_eff_pix,
        )[BAND]
        render_panel(ax2, enhanced, fov, title, param_text)

    fig1.suptitle(
        f"Native multi-component morphology ONLY ({BAND}, default_config.yaml tuned settings)",
        fontsize=16, fontweight='bold', y=1.0)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = out_dir / "multicomponent_morphology_showcase.png"
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved: {out1}")
    plt.close(fig1)

    fig2.suptitle(
        f"Native multi-component + pixel-space texture ({BAND}, default_config.yaml tuned settings)",
        fontsize=16, fontweight='bold', y=1.0)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = out_dir / "multicomponent_plus_texture_showcase.png"
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
