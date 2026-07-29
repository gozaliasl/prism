"""
Morphology classification and per-type native-component recipes.
"""
import numpy as np


# Which native SERSIC_ELLIPSE components to generate for each morphology
# type when morphology.multicomponent_enabled is True. 'irregular',
# 'primordial', 'clumpy', 'starburst' stay single-component (low-n,
# bulge-less); clump/spiral/dust texture is still applied as pixel-space
# post-processing on top via apply_morphological_enhancements.
MORPH_COMPONENTS = {
    'elliptical': ['bulge'],
    's0': ['bulge'],
    'spiral': ['bulge', 'disk'],
    'late_spiral': ['bulge', 'disk'],
    'edge_on': ['bulge', 'disk'],
    'barred_spiral': ['bulge', 'disk', 'bar'],
    'ring': ['bulge', 'disk', 'ring'],
    'irregular': ['disk'],
    'primordial': ['disk'],
    'clumpy': ['disk'],
    'starburst': ['disk'],
    'post_merger': ['bulge', 'disk', 'bulge_secondary'],
}


def _axis_ratio_from_e(e1, e2):
    """Convert lenstronomy (e1, e2) ellipticity to axis ratio q."""
    e = np.sqrt(e1 ** 2 + e2 ** 2)
    e = min(e, 0.99)
    return (1.0 - e) / (1.0 + e)


def classify_morphology(n_sersic, q_ratio, rng=None, allow_ring=True):
    """
    Morphological classification of a galaxy from its Sersic index and
    axis ratio.

    This is the canonical implementation, moved from
    src/jwst_lens_simulator.py:classify_galaxy_morphology_enhanced (kept
    there as a thin backward-compatible wrapper).

    Parameters
    ----------
    allow_ring : bool
        If False, never return 'ring' (falls back to 'spiral'). Used for
        lens-deflector / non-lens central galaxies, where a pixel-space
        collisional-ring feature can be visually confused with an Einstein
        ring; 'ring' remains available for lensed source galaxies.

    Returns one of the keys of MORPH_COMPONENTS.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Edge-on detection (highly flattened)
    if q_ratio < 0.35:
        return 'edge_on'

    # Very low n: Irregular/Clumpy
    if n_sersic < 0.7:
        # Chance of being primordial high-z
        if rng.random() < 0.3:
            return 'primordial'
        else:
            return 'irregular'

    # Low n: Late spirals
    elif n_sersic < 1.2:
        # Chance of being barred
        if rng.random() < 0.15:
            return 'barred_spiral'
        else:
            return 'late_spiral'

    # Intermediate n: Early spirals or S0
    elif n_sersic < 2.0:
        # Chance of being barred
        if rng.random() < 0.20:
            return 'barred_spiral'
        # Small chance of being ring galaxy (draw consumed regardless of
        # allow_ring, to keep the RNG sequence identical either way)
        elif rng.random() < 0.05:
            return 'ring' if allow_ring else 'spiral'
        else:
            return 'spiral'

    # Intermediate-high n: S0 lenticulars
    elif n_sersic < 3.0:
        # Small chance of being post-merger
        if rng.random() < 0.10:
            return 'post_merger'
        else:
            return 's0'

    # High n: Ellipticals
    else:
        return 'elliptical'
