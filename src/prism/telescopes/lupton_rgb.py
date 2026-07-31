"""Standard Lupton+2004 asinh RGB composites (astropy make_lupton_rgb).

Preferred over the Trilogy log-stretch compositor when faint field galaxies
need to stay visible without saturating the bright lens/source: the asinh
stretch is linear near zero (preserves faint low-S/N structure) and
logarithmic at high flux (compresses bright cores), with a single global
(stretch, Q) pair applied consistently across R/G/B so color is preserved.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_lupton_rgb


def _lift_black_point(rgb_u8: np.ndarray, floor: float) -> np.ndarray:
    """Blend uint8 image toward a grey floor so the sky isn't flat black.

    A pure asinh/log stretch drives near-zero sky pixels to 0,0,0 -- fine
    for signal-to-noise but reads as an unnaturally "dead" black void.
    Real deep-field images (COSMOS-Web, HUDF) show faint grey sky texture.
    `floor` in [0, 255]; ~10-25 is a subtle lift, matches ~ratio 0.04-0.10.
    """
    if floor <= 0:
        return rgb_u8
    arr = rgb_u8.astype(np.float32)
    out = floor + arr * (255.0 - floor) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _boost_saturation(rgb_u8: np.ndarray, factor: float) -> np.ndarray:
    """Scale color saturation around each pixel's luminance, in RGB space."""
    if abs(factor - 1.0) < 1e-6:
        return rgb_u8
    arr = rgb_u8.astype(np.float32)
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., None]
    out = lum + (arr - lum) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _sky_level_and_sigma(arr: np.ndarray) -> tuple[float, float]:
    """Robust sigma-clipped sky level and noise sigma (excludes bright sources)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    _, median, std = sigma_clipped_stats(finite, sigma=3.0, maxiters=5)
    return float(median), float(std)


def _clip_low_noise_outliers(arr: np.ndarray, n_sigma: float = 4.0) -> np.ndarray:
    """Clamp only the LOW (negative-excursion) tail of sky noise back to
    sky - n_sigma*sigma; the high tail is left untouched (real sources
    live there, and clipping both directions would clamp bright objects).

    Suppresses excess random-noise spikes in an unusually noisy band
    (confirmed: one band with ~3x the sigma of the others produced a
    visible colour-tinted sky, since Lupton's shared stretch amplifies
    whichever channel has the widest noise excursions -- most visibly via
    the negative tail interacting with the asinh "minimum" floor).
    """
    sky, sigma = _sky_level_and_sigma(arr)
    if sigma <= 0:
        return arr
    lo = sky - n_sigma * sigma
    return np.where(arr < lo, lo, arr)


def _band_norm_scale(arr: np.ndarray, pct: float = 99.5) -> float:
    """Robust per-band brightness scale (percentile above sky), used to
    equalize bands before RGB combination.

    Lupton's algorithm stretches TOTAL intensity (R+G+B)/3 and preserves
    the input color RATIOS exactly -- it does NOT independently normalize
    each channel's absolute scale. If one band is intrinsically brighter
    for a given source (e.g. F277W for a red quiescent galaxy) and it's
    averaged into a channel with a dimmer band (e.g. G = avg(F150W,
    F277W)), that channel gets dragged toward the brighter band's scale,
    producing a systematic color-cast (confirmed: F277W peaked ~175sigma
    vs F115W/F150W's ~93-97sigma and F444W's ~132sigma on a real render,
    directly causing the persistent green/yellow cast seen throughout
    testing). Dividing each band by its own robust scale before combining
    lets color differences reflect genuine per-pixel SED shape rather than
    one band's overall greater dynamic range.
    """
    sky, sigma = _sky_level_and_sigma(arr)
    scale = np.percentile(arr, pct) - sky
    return float(scale) if scale > 1e-8 else 1.0


def jwst_lupton_rgb(
    images: Mapping[str, np.ndarray],
    *,
    stretch: float = 1.0,
    Q: float = 15.0,
    saturation_boost: float = 1.3,
    minimum: float = "sky",
    sky_pedestal_sigma: float = 0.6,
    black_point_lift: float = 20.0,
    normalize_bands: bool = True,
) -> np.ndarray:
    """R=F444W, G=(F150W+F277W)/2, B=F115W Lupton asinh composite, uint8 HxWx3.

    `minimum="sky"` (default) sets the black point a fraction of a sky-noise
    sigma *below* the median background per channel, instead of clipping at
    0 -- this keeps the sky as a faint grey noise texture instead of a flat
    black void, and keeps faint sources above it visible.

    `normalize_bands=True` (default) divides each of the 4 input bands by
    its own robust brightness scale (99.5th percentile above sky) before
    combining into R/G/B -- see _band_norm_scale for why this matters.
    """
    f115 = _clip_low_noise_outliers(np.asarray(images["F115W"], dtype=np.float32))
    f150 = _clip_low_noise_outliers(np.asarray(images["F150W"], dtype=np.float32))
    f277 = _clip_low_noise_outliers(np.asarray(images["F277W"], dtype=np.float32))
    f444 = _clip_low_noise_outliers(np.asarray(images["F444W"], dtype=np.float32))

    if normalize_bands:
        f115 = f115 / _band_norm_scale(f115)
        f150 = f150 / _band_norm_scale(f150)
        f277 = f277 / _band_norm_scale(f277)
        f444 = f444 / _band_norm_scale(f444)

    R = f444
    G = 0.5 * (f150 + f277)
    B = f115

    if minimum == "sky":
        # Per-channel-independent floors go unstable when one band has
        # noticeably more noise than the others for a given render (e.g.
        # a bad PSF/detector-noise realization) -- the noisier channel's
        # floor swings much more negative, and since Lupton preserves
        # input color ratios, that alone produces a visible colour tint
        # in the "empty" sky (confirmed: a render with F115W sigma ~3x
        # the other bands showed a blue-tinted background). Using one
        # shared, most-conservative (least negative) floor across R/G/B
        # avoids this instability.
        candidates = []
        for arr in (R, G, B):
            sky, sigma = _sky_level_and_sigma(arr)
            candidates.append(sky - sky_pedestal_sigma * sigma)
        shared_min = max(candidates)
        mins = [shared_min] * 3
    else:
        mins = [float(minimum)] * 3

    rgb = make_lupton_rgb(R, G, B, stretch=stretch, Q=Q, minimum=mins)
    rgb = _lift_black_point(rgb, black_point_lift)
    return _boost_saturation(rgb, saturation_boost)


def euclid_lupton_rgb(
    images: Mapping[str, np.ndarray],
    *,
    stretch: float = 1.0,
    Q: float = 15.0,
    saturation_boost: float = 1.3,
    minimum: float = "sky",
    sky_pedestal_sigma: float = 0.6,
    black_point_lift: float = 20.0,
    normalize_bands: bool = True,
) -> np.ndarray:
    """R=H, G=(Y+J)/2, B=VIS Lupton asinh composite, uint8 HxWx3.

    `normalize_bands=True` (default) -- see jwst_lupton_rgb's docstring /
    _band_norm_scale for why this matters (prevents one band's overall
    brightness dominating an averaged channel and causing a color cast).
    """
    def pick(names):
        lower = {k.lower(): k for k in images}
        for n in names:
            if n.lower() in lower:
                return np.asarray(images[lower[n.lower()]], dtype=np.float32)
        raise KeyError(names)

    H = _clip_low_noise_outliers(pick(("EUCLID_H", "H")))
    Y = _clip_low_noise_outliers(pick(("EUCLID_Y", "Y")))
    J = _clip_low_noise_outliers(pick(("EUCLID_J", "J")))
    I = _clip_low_noise_outliers(pick(("EUCLID_VIS", "VIS", "I")))

    if normalize_bands:
        H = H / _band_norm_scale(H)
        Y = Y / _band_norm_scale(Y)
        J = J / _band_norm_scale(J)
        I = I / _band_norm_scale(I)

    R = H
    G = 0.5 * (Y + J)
    B = I

    if minimum == "sky":
        # Per-channel-independent floors go unstable when one band has
        # noticeably more noise than the others for a given render (e.g.
        # a bad PSF/detector-noise realization) -- the noisier channel's
        # floor swings much more negative, and since Lupton preserves
        # input color ratios, that alone produces a visible colour tint
        # in the "empty" sky (confirmed: a render with F115W sigma ~3x
        # the other bands showed a blue-tinted background). Using one
        # shared, most-conservative (least negative) floor across R/G/B
        # avoids this instability.
        candidates = []
        for arr in (R, G, B):
            sky, sigma = _sky_level_and_sigma(arr)
            candidates.append(sky - sky_pedestal_sigma * sigma)
        shared_min = max(candidates)
        mins = [shared_min] * 3
    else:
        mins = [float(minimum)] * 3

    rgb = make_lupton_rgb(R, G, B, stretch=stretch, Q=Q, minimum=mins)
    rgb = _lift_black_point(rgb, black_point_lift)
    return _boost_saturation(rgb, saturation_boost)
