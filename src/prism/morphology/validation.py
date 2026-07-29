"""
Non-parametric morphology statistics for real-vs-simulated validation:
CAS (Concentration, Asymmetry, Smoothness -- Conselice 2003), Gini and M20
(Lotz et al. 2004), and clumpiness.

Designed to slot into
analysis/sim_obs_comparison/scripts/morphology_metrics.measure_lens_arrays()
as additional output columns.
"""
import numpy as np
from scipy import ndimage


def _galaxy_pixels(image, segm):
    """Return (values, y-coords, x-coords) for segm > 0 pixels."""
    mask = segm > 0
    ys, xs = np.where(mask)
    return image[mask].astype(float), ys.astype(float), xs.astype(float)


def _centroid(image, segm):
    vals, ys, xs = _galaxy_pixels(image, segm)
    total = vals.sum()
    if total <= 0:
        return float(np.mean(ys)), float(np.mean(xs))
    return float((vals * ys).sum() / total), float((vals * xs).sum() / total)


def concentration(image, segm, frac_inner=0.2, frac_outer=0.8):
    """
    Concentration index C = 5 * log10(r_80 / r_20), where r_20/r_80 are the
    circular-aperture radii (centered on the flux centroid) containing
    20%/80% of the total segmented flux.
    """
    vals, ys, xs = _galaxy_pixels(image, segm)
    total = vals.sum()
    if total <= 0 or len(vals) == 0:
        return np.nan

    cy, cx = _centroid(image, segm)
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    order = np.argsort(r)
    r_sorted = r[order]
    cumflux = np.cumsum(vals[order]) / total

    r_20 = np.interp(frac_inner, cumflux, r_sorted)
    r_80 = np.interp(frac_outer, cumflux, r_sorted)
    if r_20 <= 0:
        return np.nan
    return float(5.0 * np.log10(r_80 / r_20))


def asymmetry(image, segm):
    """
    Asymmetry index A = sum(|I - I_180|) / sum(|I|), where I_180 is the
    image rotated by 180 degrees about the flux centroid (Conselice 2003).
    """
    mask = (segm > 0).astype(float)
    img = image * mask
    cy, cx = _centroid(image, segm)

    shift = (2 * cy - (image.shape[0] - 1), 2 * cx - (image.shape[1] - 1))
    rotated = np.rot90(img, 2)
    rotated = ndimage.shift(rotated, shift=(-shift[0], -shift[1]), order=1,
                             mode='constant', cval=0.0)

    denom = np.sum(np.abs(img))
    if denom <= 0:
        return np.nan
    return float(np.sum(np.abs(img - rotated)) / denom)


def smoothness(image, segm, smoothing_factor=0.25):
    """
    Smoothness (clumpiness-precursor) index S = sum(|I - I_smooth|) /
    sum(|I|), where I_smooth is a boxcar-smoothed version of I with kernel
    size = smoothing_factor * r_petrosian-like scale (here approximated by
    sqrt(segm area)).
    """
    mask = segm > 0
    img = image * mask
    area = mask.sum()
    if area <= 0:
        return np.nan

    sigma = max(1.0, smoothing_factor * np.sqrt(area))
    smoothed = ndimage.uniform_filter(img, size=int(round(sigma)))

    denom = np.sum(np.abs(img))
    if denom <= 0:
        return np.nan
    return float(np.sum(np.abs(img - smoothed) * mask) / denom)


def clumpiness(image, segm, smoothing_factor=0.25):
    """
    Clumpiness S = 10 * sum(|I - I_smooth|) / sum(I), restricted to
    positive residuals (high-frequency power), per Conselice (2003).
    """
    mask = segm > 0
    img = image * mask
    area = mask.sum()
    total = img.sum()
    if area <= 0 or total <= 0:
        return np.nan

    sigma = max(1.0, smoothing_factor * np.sqrt(area))
    smoothed = ndimage.uniform_filter(img, size=int(round(sigma)))
    residual = (img - smoothed) * mask
    residual = np.clip(residual, 0, None)
    return float(10.0 * residual.sum() / total)


def gini(image, segm):
    """
    Gini coefficient of the flux distribution within the segmentation map
    (Lotz et al. 2004).
    """
    vals, _, _ = _galaxy_pixels(image, segm)
    vals = np.abs(vals)
    n = len(vals)
    if n == 0:
        return np.nan
    vals_sorted = np.sort(vals)
    mean_val = vals_sorted.mean()
    if mean_val == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    g = (2 * np.sum(idx * vals_sorted) - (n + 1) * np.sum(vals_sorted)) / (n * np.sum(vals_sorted))
    return float(g)


def m20(image, segm, frac=0.2):
    """
    M20: normalized second-order moment of the brightest 20% of the
    segmented flux (Lotz et al. 2004).
    """
    vals, ys, xs = _galaxy_pixels(image, segm)
    total = vals.sum()
    if total <= 0 or len(vals) == 0:
        return np.nan

    cy, cx = _centroid(image, segm)
    mtot = np.sum(vals * ((ys - cy) ** 2 + (xs - cx) ** 2))
    if mtot <= 0:
        return np.nan

    order = np.argsort(vals)[::-1]
    cumflux = np.cumsum(vals[order]) / total
    n_bright = np.searchsorted(cumflux, frac) + 1
    bright_idx = order[:n_bright]
    mbright = np.sum(vals[bright_idx] * ((ys[bright_idx] - cy) ** 2 + (xs[bright_idx] - cx) ** 2))

    return float(np.log10(mbright / mtot))


def compute_cas_gini_m20(image, segm):
    """
    Compute all morphology statistics for a single image + segmentation
    map. Returns a dict with keys: concentration, asymmetry, smoothness,
    clumpiness, gini, m20. Falls back to the numpy/scipy implementations
    above (statmorph is not required).
    """
    image = np.asarray(image, dtype=float)
    segm = np.asarray(segm)
    return {
        'concentration': concentration(image, segm),
        'asymmetry': asymmetry(image, segm),
        'smoothness': smoothness(image, segm),
        'clumpiness': clumpiness(image, segm),
        'gini': gini(image, segm),
        'm20': m20(image, segm),
    }
