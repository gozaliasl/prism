"""
GALFIT-style bulge + exponential-disk decomposition prescriptions.

Implements the generic batch-fitting recipe described by Peng for automated
GALFIT fitting of distant galaxies: a de Vaucouleurs (n=4) bulge plus an
exponential disk (n=1 Sersic with R_eff = 1.678 h) for disk-dominated types,
with a single-component fallback for smooth ellipticals and low-n irregulars.

Reference: https://users.obs.carnegiescience.edu/peng/work/galfit/distant.html

Lenstronomy has no native EXP profile; disk components use SERSIC_ELLIPSE with
n=1 and R_sersic = 1.678 * h, which matches an exponential scale length h.
"""
import numpy as np

from .taxonomy import MORPH_COMPONENTS
from .components import _q_to_e1e2, _component_cfg

# n=1 Sersic effective radius / exponential scale length (3.915 +/- 0.015 for
# standard Sersic definition used in lenstronomy/GALFIT).
_EXP_RE_OVER_H = 1.678

# Per-type GALFIT batch initial-guess recipe.  B/T is the bulge flux fraction
# in the reference band; re_bulge_frac and h_disk_frac scale the total R_eff.
_DEFAULT_GALFIT_CFG = {
    'elliptical': {
        'bt': 0.95, 'n_bulge': 4.0, 're_bulge_frac': 1.0,
        'h_disk_frac': None, 'n_disk': 1.0, 'single': 'bulge',
    },
    's0': {
        'bt': 0.82, 'n_bulge': 4.0, 're_bulge_frac': 0.22,
        'h_disk_frac': 0.65, 'n_disk': 1.0,
    },
    'spiral': {
        'bt': 0.22, 'n_bulge': 4.0, 're_bulge_frac': 0.18,
        'h_disk_frac': 1.05, 'n_disk': 1.0,
    },
    'late_spiral': {
        'bt': 0.12, 'n_bulge': 2.5, 're_bulge_frac': 0.12,
        'h_disk_frac': 1.35, 'n_disk': 1.0,
    },
    'edge_on': {
        'bt': 0.28, 'n_bulge': 4.0, 're_bulge_frac': 0.15,
        'h_disk_frac': 1.25, 'n_disk': 1.0,
    },
    'barred_spiral': {
        'bt': 0.18, 'n_bulge': 4.0, 're_bulge_frac': 0.16,
        'h_disk_frac': 1.0, 'n_disk': 1.0, 'bar_frac_of_disk': 0.10,
    },
    'ring': {
        'bt': 0.15, 'n_bulge': 4.0, 're_bulge_frac': 0.14,
        'h_disk_frac': 0.9, 'n_disk': 1.0, 'ring_frac_of_disk': 0.14,
    },
    'post_merger': {
        'bt': 0.30, 'n_bulge': 4.0, 're_bulge_frac': 0.20,
        'h_disk_frac': 0.85, 'n_disk': 1.0, 'secondary_frac_of_bulge': 0.25,
    },
    'irregular': {
        'bt': 0.0, 'n_bulge': 1.0, 're_bulge_frac': 1.0,
        'h_disk_frac': 1.1, 'n_disk': 0.8, 'single': 'disk',
    },
    'primordial': {
        'bt': 0.0, 'n_bulge': 1.0, 're_bulge_frac': 1.0,
        'h_disk_frac': 0.85, 'n_disk': 0.5, 'single': 'disk',
    },
    'clumpy': {
        'bt': 0.0, 'n_bulge': 1.0, 're_bulge_frac': 1.0,
        'h_disk_frac': 1.2, 'n_disk': 0.7, 'single': 'disk',
    },
    'starburst': {
        'bt': 0.08, 'n_bulge': 2.0, 're_bulge_frac': 0.10,
        'h_disk_frac': 0.95, 'n_disk': 0.9,
    },
}


def _galfit_cfg(config, morph_type):
    cfg = dict(_DEFAULT_GALFIT_CFG.get(morph_type, _DEFAULT_GALFIT_CFG['spiral']))
    overrides = (config.get('galfit', {}) or {}).get(morph_type, {})
    cfg.update(overrides)
    return cfg


def _exp_r_sersic(h_scale):
    """Convert exponential scale length h to n=1 Sersic R_sersic."""
    return float(h_scale) * _EXP_RE_OVER_H


def build_galfit_components(base_params, morph_type, rng, config, role=None):
    """
    Build GALFIT-style component specs for lenstronomy rendering.

    Returns the same dict structure as ``build_components`` so downstream
    band_flux_fractions / SimAPI code can be reused unchanged.
    """
    morph_cfg = config if isinstance(config, dict) else {}
    gcfg = _galfit_cfg(morph_cfg, morph_type)
    bar_cfg = _component_cfg(morph_cfg, 'bar')
    ring_cfg = _component_cfg(morph_cfg, 'ring')
    sec_cfg = _component_cfg(morph_cfg, 'bulge_secondary')

    R_total = float(base_params['R_sersic'])
    e1_t = float(base_params.get('e1', 0.0))
    e2_t = float(base_params.get('e2', 0.0))
    cx = float(base_params.get('center_x', 0.0))
    cy = float(base_params.get('center_y', 0.0))
    q_total = max((1.0 - np.sqrt(e1_t ** 2 + e2_t ** 2)) / (1.0 + np.sqrt(e1_t ** 2 + e2_t ** 2)), 0.05)
    pa_total = 0.5 * np.arctan2(e2_t, e1_t)
    q_bulge = float(np.clip(q_total + (1.0 - q_total) * 0.35, 0.05, 1.0))
    e1_b, e2_b = _q_to_e1e2(q_bulge, pa_total)

    names = MORPH_COMPONENTS.get(morph_type, ['bulge', 'disk'])
    single = gcfg.get('single')

    if single == 'bulge' or names == ['bulge']:
        return [{
            'name': 'bulge',
            'profile': 'SERSIC',
            'R_sersic': R_total * float(gcfg.get('re_bulge_frac', 1.0)),
            'n_sersic': float(gcfg.get('n_bulge', 4.0)),
            'e1': e1_t, 'e2': e2_t,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': 1.0,
        }]

    if single == 'disk' or names == ['disk']:
        h = R_total * float(gcfg.get('h_disk_frac', 1.0))
        n_disk = float(gcfg.get('n_disk', base_params.get('n_sersic', 1.0)))
        return [{
            'name': 'disk',
            'profile': 'EXP' if abs(n_disk - 1.0) < 0.05 else 'SERSIC',
            'R_sersic': _exp_r_sersic(h) if abs(n_disk - 1.0) < 0.05 else R_total,
            'n_sersic': n_disk,
            'e1': e1_t, 'e2': e2_t,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': 1.0,
        }]

    bt = float(np.clip(gcfg.get('bt', 0.25), 0.0, 0.99))
    h = R_total * float(gcfg.get('h_disk_frac', 1.0))
    if role == 'source':
        h *= float((morph_cfg.get('components', {}) or {}).get('disk', {}).get('r_frac_source', 1.0))

    bulge = {
        'name': 'bulge',
        'profile': 'SERSIC',
        'R_sersic': R_total * float(gcfg.get('re_bulge_frac', 0.2)),
        'n_sersic': float(gcfg.get('n_bulge', 4.0)),
        'e1': e1_b, 'e2': e2_b,
        'center_x': cx, 'center_y': cy,
        'flux_fraction_ref': bt,
    }
    disk = {
        'name': 'disk',
        'profile': 'EXP',
        'R_sersic': _exp_r_sersic(h),
        'n_sersic': float(gcfg.get('n_disk', 1.0)),
        'e1': e1_t, 'e2': e2_t,
        'center_x': cx, 'center_y': cy,
        'flux_fraction_ref': 1.0 - bt,
    }
    components = [bulge, disk]

    if 'bar' in names:
        bar_frac = min(float(gcfg.get('bar_frac_of_disk', 0.08)), disk['flux_fraction_ref'])
        disk['flux_fraction_ref'] -= bar_frac
        bar_offset_deg = float(rng.uniform(15.0, 45.0))
        pa_bar = pa_total + np.radians(bar_offset_deg)
        q_bar = float(bar_cfg.get('q', 0.22))
        e1_bar, e2_bar = _q_to_e1e2(q_bar, pa_bar)
        components.append({
            'name': 'bar',
            'profile': 'SERSIC',
            'R_sersic': _exp_r_sersic(h) * float(bar_cfg.get('r_frac', 0.35)),
            'n_sersic': float(bar_cfg.get('n_sersic', 0.5)),
            'e1': e1_bar, 'e2': e2_bar,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': bar_frac,
        })

    if 'ring' in names:
        ring_frac = min(float(gcfg.get('ring_frac_of_disk', 0.12)), disk['flux_fraction_ref'])
        disk['flux_fraction_ref'] -= ring_frac
        components.append({
            'name': 'ring',
            'profile': 'SERSIC',
            'R_sersic': _exp_r_sersic(h) * float(ring_cfg.get('r_frac', 0.75)),
            'n_sersic': float(ring_cfg.get('n_sersic', 0.5)),
            'e1': e1_t, 'e2': e2_t,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': ring_frac,
        })

    if 'bulge_secondary' in names:
        sec_frac = min(
            float(gcfg.get('secondary_frac_of_bulge', 0.2)) * bulge['flux_fraction_ref'],
            bulge['flux_fraction_ref'],
        )
        bulge['flux_fraction_ref'] -= sec_frac
        off_lo, off_hi = sec_cfg.get('offset_frac', [0.3, 0.8])
        offset_r = R_total * float(rng.uniform(off_lo, off_hi))
        offset_angle = float(rng.uniform(0, 2 * np.pi))
        components.append({
            'name': 'bulge_secondary',
            'profile': 'SERSIC',
            'R_sersic': bulge['R_sersic'] * float(sec_cfg.get('r_frac', 0.55)),
            'n_sersic': float(sec_cfg.get('n_sersic', 4.0)),
            'e1': e1_b, 'e2': e2_b,
            'center_x': cx + offset_r * np.cos(offset_angle),
            'center_y': cy + offset_r * np.sin(offset_angle),
            'flux_fraction_ref': sec_frac,
        })

    total = sum(c['flux_fraction_ref'] for c in components)
    if total > 0:
        for c in components:
            c['flux_fraction_ref'] /= total

    return components


def galfit_config_lines(components, total_mag, pixel_scale, numpix, zpt=28.0):
    """
    Emit GALFIT ``config`` file lines for the given component list.

    Coordinates are in 1-based pixel units (GALFIT convention) with the
    galaxy centered on the image.
    """
    cx = cy = numpix / 2.0 + 0.5
    lines = [
        '================================================================================',
        '# GALFIT CONFIG (auto-generated; bulge=Sersic fn=1, disk=Exp fn=2)',
        '================================================================================',
        '0)  30  30  5  30  5',
    ]
    comp_id = 1
    for comp in components:
        mag = total_mag - 2.5 * np.log10(max(comp['flux_fraction_ref'], 1e-6))
        q = max((1.0 - np.sqrt(comp['e1'] ** 2 + comp['e2'] ** 2))
                / (1.0 + np.sqrt(comp['e1'] ** 2 + comp['e2'] ** 2)), 0.05)
        pa = np.degrees(0.5 * np.arctan2(comp['e2'], comp['e1'])) % 180.0
        profile = comp.get('profile', 'SERSIC')
        if profile == 'EXP':
            h_pix = comp['R_sersic'] / _EXP_RE_OVER_H / pixel_scale
            lines.append(
                f'{comp_id}) {cx:.2f} {cy:.2f} {mag:.3f} {h_pix:.2f} {q:.3f} {pa:.1f}  '
                f'0 0 0 0 0 0  # {comp["name"]} exponential'
            )
        else:
            re_pix = comp['R_sersic'] / pixel_scale
            lines.append(
                f'{comp_id}) {cx:.2f} {cy:.2f} {mag:.3f} {comp["n_sersic"]:.2f} '
                f'{re_pix:.2f} {q:.3f} {pa:.1f}  0 0 0 0 0 0  # {comp["name"]} sersic'
            )
        comp_id += 1
    lines.extend([
        'J) 0.74 0.09 0.08 0.09',
        f'K) {pixel_scale:.4f} 0.0000 {zpt:.2f}',
        'O) regular',
        'P) 0',
        'R) 0',
        'S) 0',
        'T) stop',
        'U) 0',
    ])
    return lines
