"""
Native multi-component (bulge/disk/bar/ring/bulge_secondary) SERSIC_ELLIPSE
component builder.
"""
import numpy as np

from .taxonomy import MORPH_COMPONENTS, _axis_ratio_from_e


# Default per-component shape recipes, overridable via
# config['morphology']['components'] (CONFIG['morphology']['components']).
_DEFAULT_COMPONENT_CFG = {
    'bulge': {'n_sersic': 4.0, 'r_frac': 0.25, 'q_boost': 0.3},
    # 'r_frac_source' overrides 'r_frac' for the disk component when
    # build_components is called with role='source'. A source disk
    # extended beyond the total Sersic radius produces more extended
    # lensed disk images (more/larger detected components, larger
    # max_sep_arcsec) without changing the lens-side rendering.
    'disk': {'n_sersic': 1.0, 'r_frac': 1.0, 'r_frac_source': 1.0},
    'bar': {'n_sersic': 0.5, 'r_frac': 0.4, 'q': 0.25},
    'ring': {'n_sersic': 0.5, 'r_frac': 0.8},
    'bulge_secondary': {'n_sersic': 4.0, 'r_frac': 0.6, 'offset_frac': [0.3, 0.8]},
}

# Default reference-band (F150W-like) flux fraction taken by the
# "bulge-like" component for each morphology type. The remaining flux goes
# to the disk (or is the total for bulge-only / disk-only types).
_DEFAULT_BT_FRACTIONS = {
    'elliptical': 0.9,
    's0': 0.85,
    'spiral': 0.25,
    'late_spiral': 0.2,
    'barred_spiral': 0.2,
    'edge_on': 0.3,
    'ring': 0.2,
    'post_merger': 0.25,
}

# Fraction of the disk flux taken by a bar / ring component (subtracted
# from the disk's share).
_BAR_FRAC_OF_DISK = 0.08
_RING_FRAC_OF_DISK = 0.12
# Fraction of the primary bulge flux taken by the secondary (post-merger)
# nucleus.
_SECONDARY_FRAC_OF_BULGE = 0.2


def _q_to_e1e2(q, pa_rad):
    e = (1.0 - q) / (1.0 + q)
    return e * np.cos(2 * pa_rad), e * np.sin(2 * pa_rad)


def _component_cfg(config, name):
    cfg = dict(_DEFAULT_COMPONENT_CFG.get(name, {}))
    cfg.update((config.get('components', {}) or {}).get(name, {}))
    return cfg


def build_components(base_params, morph_type, rng, config, role=None):
    """
    Build a list of native SERSIC_ELLIPSE component specs for a galaxy.

    Parameters
    ----------
    base_params : dict
        Existing single-Sersic kwargs: R_sersic, n_sersic, e1, e2,
        center_x, center_y.
    morph_type : str
        Key into MORPH_COMPONENTS.
    rng : np.random.Generator
    config : dict
        config['morphology'] sub-dict (components, bt_fractions, ...).
    role : str, optional
        'lens' | 'source' | 'field'. When 'source', the disk component's
        R_sersic uses 'r_frac_source' instead of 'r_frac' (see
        _DEFAULT_COMPONENT_CFG).

    Returns
    -------
    list[dict]
        Each dict has keys: name, R_sersic, n_sersic, e1, e2, center_x,
        center_y, flux_fraction_ref. flux_fraction_ref values sum to 1.
    """
    names = MORPH_COMPONENTS.get(morph_type, ['bulge', 'disk'])

    R_total = float(base_params['R_sersic'])
    e1_t = float(base_params.get('e1', 0.0))
    e2_t = float(base_params.get('e2', 0.0))
    cx = float(base_params.get('center_x', 0.0))
    cy = float(base_params.get('center_y', 0.0))

    q_total = _axis_ratio_from_e(e1_t, e2_t)
    pa_total = 0.5 * np.arctan2(e2_t, e1_t)

    bt_table = dict(_DEFAULT_BT_FRACTIONS)
    bt_table.update(config.get('bt_fractions', {}) or {})

    components = []

    if names == ['bulge']:
        cfg = _component_cfg(config, 'bulge')
        components.append({
            'name': 'bulge',
            'R_sersic': R_total,
            'n_sersic': float(cfg.get('n_sersic', 4.0)),
            'e1': e1_t, 'e2': e2_t,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': 1.0,
        })
        return components

    if names == ['disk']:
        cfg = _component_cfg(config, 'disk')
        components.append({
            'name': 'disk',
            'R_sersic': R_total,
            'n_sersic': float(base_params.get('n_sersic', cfg.get('n_sersic', 1.0))),
            'e1': e1_t, 'e2': e2_t,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': 1.0,
        })
        return components

    # bulge + disk (+ bar / ring / bulge_secondary)
    bulge_cfg = _component_cfg(config, 'bulge')
    disk_cfg = _component_cfg(config, 'disk')

    bt_frac = float(np.clip(bt_table.get(morph_type, 0.3), 0.01, 0.99))
    q_bulge = float(np.clip(q_total + (1.0 - q_total) * float(bulge_cfg.get('q_boost', 0.3)), 0.0, 1.0))
    e1_b, e2_b = _q_to_e1e2(q_bulge, pa_total)

    bulge = {
        'name': 'bulge',
        'R_sersic': R_total * float(bulge_cfg.get('r_frac', 0.25)),
        'n_sersic': float(bulge_cfg.get('n_sersic', 4.0)),
        'e1': e1_b, 'e2': e2_b,
        'center_x': cx, 'center_y': cy,
        'flux_fraction_ref': bt_frac,
    }
    if role == 'source':
        disk_r_frac = float(disk_cfg.get('r_frac_source', disk_cfg.get('r_frac', 1.0)))
    else:
        disk_r_frac = float(disk_cfg.get('r_frac', 1.0))

    disk = {
        'name': 'disk',
        'R_sersic': R_total * disk_r_frac,
        'n_sersic': float(disk_cfg.get('n_sersic', 1.0)),
        'e1': e1_t, 'e2': e2_t,
        'center_x': cx, 'center_y': cy,
        'flux_fraction_ref': 1.0 - bt_frac,
    }
    components = [bulge, disk]

    if 'bar' in names:
        bar_cfg = _component_cfg(config, 'bar')
        bar_offset_deg = float(rng.uniform(20.0, 70.0))
        pa_bar = pa_total + np.radians(bar_offset_deg)
        q_bar = float(bar_cfg.get('q', 0.25))
        e1_bar, e2_bar = _q_to_e1e2(q_bar, pa_bar)
        bar_frac = min(_BAR_FRAC_OF_DISK, disk['flux_fraction_ref'])
        disk['flux_fraction_ref'] -= bar_frac
        components.append({
            'name': 'bar',
            'R_sersic': disk['R_sersic'] * float(bar_cfg.get('r_frac', 0.4)),
            'n_sersic': float(bar_cfg.get('n_sersic', 0.5)),
            'e1': e1_bar, 'e2': e2_bar,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': bar_frac,
        })

    if 'ring' in names:
        ring_cfg = _component_cfg(config, 'ring')
        ring_frac = min(_RING_FRAC_OF_DISK, disk['flux_fraction_ref'])
        disk['flux_fraction_ref'] -= ring_frac
        components.append({
            'name': 'ring',
            'R_sersic': disk['R_sersic'] * float(ring_cfg.get('r_frac', 0.8)),
            'n_sersic': float(ring_cfg.get('n_sersic', 0.5)),
            'e1': e1_t, 'e2': e2_t,
            'center_x': cx, 'center_y': cy,
            'flux_fraction_ref': ring_frac,
        })

    if 'bulge_secondary' in names:
        sec_cfg = _component_cfg(config, 'bulge_secondary')
        off_lo, off_hi = sec_cfg.get('offset_frac', [0.3, 0.8])
        offset_r = R_total * float(rng.uniform(off_lo, off_hi))
        offset_angle = float(rng.uniform(0, 2 * np.pi))
        sec_frac = min(_SECONDARY_FRAC_OF_BULGE * bulge['flux_fraction_ref'], bulge['flux_fraction_ref'])
        bulge['flux_fraction_ref'] -= sec_frac
        components.append({
            'name': 'bulge_secondary',
            'R_sersic': bulge['R_sersic'] * float(sec_cfg.get('r_frac', 0.6)),
            'n_sersic': float(sec_cfg.get('n_sersic', 4.0)),
            'e1': e1_b, 'e2': e2_b,
            'center_x': cx + offset_r * np.cos(offset_angle),
            'center_y': cy + offset_r * np.sin(offset_angle),
            'flux_fraction_ref': sec_frac,
        })

    # Normalize (defensive -- should already sum to ~1)
    total = sum(c['flux_fraction_ref'] for c in components)
    if total > 0:
        for c in components:
            c['flux_fraction_ref'] /= total

    return components


def fractions_to_magnitudes(total_mag, fractions):
    """Convert flux fractions (summing to 1) to per-component magnitudes."""
    return [total_mag - 2.5 * np.log10(max(f, 1e-6)) for f in fractions]
