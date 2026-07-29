"""
Per-band flux-fraction adjustment for native multi-component light models.

Bulge-like components (bulge, bar, ring, bulge_secondary -- older stellar
populations) are made relatively redder than the disk by applying a
per-band multiplicative weight, then renormalizing. This is what makes the
multi-component morphology wavelength-dependent (the audit's gap #4)
without needing per-band pixel-texture re-rendering.
"""

# Default per-band weight applied to bulge-like components relative to the
# disk. >1 => bulge-like components relatively brighter (redder galaxy
# core) at longer wavelengths; <1 => relatively fainter at shorter
# wavelengths (bluer disk dominates in the blue).
_DEFAULT_BAND_WEIGHT_BULGE_LIKE = {
    'F115W': 0.8,
    'F150W': 0.9,
    'F200W': 1.0,
    'F277W': 1.1,
    'F356W': 1.15,
    'F444W': 1.2,
}

# Independent per-band weight applied to the disk component, relative to
# the other components. Defaults to neutral (1.0 everywhere), i.e. no
# change from the original single-knob (bulge-like vs. disk) behavior.
# Configurable via config['band_weight_disk'] -- gives a second,
# independent degree of freedom per band so the F115W-F444W and
# F277W-F444W colors can be tuned without being fully coupled through a
# single bulge-like weight curve.
_DEFAULT_BAND_WEIGHT_DISK = {
    'F115W': 1.0,
    'F150W': 1.0,
    'F200W': 1.0,
    'F277W': 1.0,
    'F356W': 1.0,
    'F444W': 1.0,
}

_BULGE_LIKE = {'bulge', 'bar', 'ring', 'bulge_secondary'}


def band_flux_fractions(components, band, morph_type, config):
    """
    Compute per-band flux fractions for a list of component specs.

    Parameters
    ----------
    components : list[dict]
        As returned by build_components (each has 'name' and
        'flux_fraction_ref').
    band : str
        JWST band name (e.g. 'F150W').
    morph_type : str
        Morphology type (currently unused but kept for future
        type-dependent band weighting).
    config : dict
        config['morphology'] sub-dict; may contain
        'band_weight_bulge_like'.

    Returns
    -------
    list[float]
        Per-component flux fractions for this band, summing to 1, in the
        same order as `components`.
    """
    weight_table = dict(_DEFAULT_BAND_WEIGHT_BULGE_LIKE)
    weight_table.update(config.get('band_weight_bulge_like', {}) or {})
    w = float(weight_table.get(band, 1.0))

    disk_weight_table = dict(_DEFAULT_BAND_WEIGHT_DISK)
    disk_weight_table.update(config.get('band_weight_disk', {}) or {})
    w_disk = float(disk_weight_table.get(band, 1.0))

    weighted = []
    for comp in components:
        f = comp['flux_fraction_ref']
        if comp['name'] in _BULGE_LIKE:
            f *= w
        elif comp['name'] == 'disk':
            f *= w_disk
        weighted.append(f)

    total = sum(weighted)
    if total <= 0:
        n = len(components)
        return [1.0 / n] * n
    return [f / total for f in weighted]
