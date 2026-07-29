"""Procedural INTERPOL profiles for ring and post-merger (tidal-tail)
structural components.

lenstronomy has no native annular/torus or tidal-tail light profile, and the
``ring``/``bulge_secondary`` components in :mod:`components` are
approximated as thin Sersic blobs. This module renders true annular and
one-sided asymmetric-tail surface-brightness maps procedurally (numpy, on a
small grid) and exposes them with the same ``INTERPOL`` kwargs shape used by
:mod:`tng_particle_light` / :mod:`src.galaxygenius_stamps`
(``image``, ``center_x``, ``center_y``, ``phi_G``, ``scale``, ``magnitude``),
so they can be substituted for the corresponding ``SERSIC_ELLIPSE`` fragment
slot without changing the surrounding lenstronomy wiring.

The image is rendered in the profile's own (unrotated) frame -- ``phi_G`` is
applied by lenstronomy's ``INTERPOL`` at evaluation time, exactly as for the
other procedural builders.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

NPIX = 100


def build_ring_interpol_kwargs(
    magnitude_ref: float,
    ref_band: str = "F150W",
    center_x: float = 0.0,
    center_y: float = 0.0,
    phi_G: float = 0.0,
    r_ring_arcsec: float = 0.5,
    ring_width_arcsec: float = 0.1,
    axis_ratio: float = 0.8,
    target_size_arcsec: float = 2.0,
    rng=None,
) -> dict:
    """Procedural annular (Cartwheel-like) surface-brightness map.

    Renders ``exp(-((r - r_ring) / width)^2 / 2)`` on an elliptical radius
    grid (major axis along x, flattened by ``axis_ratio`` along y -- the
    overall orientation is then set by ``phi_G`` as usual), with mild
    per-pixel clump noise for visual realism.

    ``magnitude_ref`` / ``ref_band`` are carried through unchanged (no
    internal color correction, like the field-galaxy Sersic fallback) --
    callers that need a band-dependent magnitude should pass the
    already-corrected value.
    """
    rng = rng if rng is not None else np.random.default_rng()
    scale = float(target_size_arcsec) / NPIX

    idx = (np.arange(NPIX) - (NPIX - 1) / 2.0) * scale
    x, y = np.meshgrid(idx, idx)
    axis_ratio = float(np.clip(axis_ratio, 0.1, 1.0))
    r_ell = np.sqrt(x ** 2 + (y / axis_ratio) ** 2)

    profile = np.exp(-0.5 * ((r_ell - float(r_ring_arcsec)) / float(ring_width_arcsec)) ** 2)

    clump_noise = 1.0 + 0.3 * rng.standard_normal(profile.shape)
    clump_noise = gaussian_filter(np.abs(clump_noise), sigma=1.5)
    image_per_pixel = np.clip(profile * clump_noise, 0.0, None)

    image_sb = image_per_pixel / scale ** 2

    return dict(
        image=image_sb,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(phi_G),
        scale=float(scale),
        magnitude=float(magnitude_ref),
    )


def build_tidal_tail_interpol_kwargs(
    magnitude_ref: float,
    ref_band: str = "F150W",
    center_x: float = 0.0,
    center_y: float = 0.0,
    phi_G: float = 0.0,
    tail_angle_rad: float = 0.0,
    offset_arcsec: float = 0.1,
    length_arcsec: float = 0.4,
    width_arcsec: float = 0.12,
    target_size_arcsec: float = 2.0,
    rng=None,
) -> dict:
    """Procedural one-sided asymmetric "tidal tail" surface-brightness map.

    An elongated, exponentially-decaying blob extending from
    ``offset_arcsec`` to ``offset_arcsec + length_arcsec`` along
    ``tail_angle_rad`` (measured in the profile's own, pre-``phi_G`` frame),
    with Gaussian cross-tail width ``width_arcsec`` and mild clump noise --
    representing a post-merger tidal feature / displaced secondary nucleus
    rather than a symmetric Sersic blob.
    """
    rng = rng if rng is not None else np.random.default_rng()
    scale = float(target_size_arcsec) / NPIX

    idx = (np.arange(NPIX) - (NPIX - 1) / 2.0) * scale
    x, y = np.meshgrid(idx, idx)

    cos_a, sin_a = np.cos(tail_angle_rad), np.sin(tail_angle_rad)
    u = x * cos_a + y * sin_a       # along-tail coordinate
    v = -x * sin_a + y * cos_a      # cross-tail coordinate

    along = np.clip(u - float(offset_arcsec), 0.0, None)
    profile = np.exp(-along / float(length_arcsec)) * np.exp(-0.5 * (v / float(width_arcsec)) ** 2)
    profile = np.where(u >= float(offset_arcsec), profile, 0.0)

    clump_noise = 1.0 + 0.3 * rng.standard_normal(profile.shape)
    clump_noise = gaussian_filter(np.abs(clump_noise), sigma=1.5)
    image_per_pixel = np.clip(profile * clump_noise, 0.0, None)

    image_sb = image_per_pixel / scale ** 2

    return dict(
        image=image_sb,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(phi_G),
        scale=float(scale),
        magnitude=float(magnitude_ref),
    )
