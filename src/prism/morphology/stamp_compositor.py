"""
GalaxyGenius stamps as lenstronomy INTERPOL lens-light profiles.

Lens deflector light is modeled natively via LightModel INTERPOL (surface
brightness in flux/arcsec^2), rendered together with the gravitationally lensed
source in a single ImageModel.image() call — not pasted as a post-hoc overlay.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from .galaxygenius_assets import load_morphology_panel

DEFAULT_STAMP_SIZE = 256


def estimate_half_light_radius(stamp: np.ndarray) -> float:
    """Half-light radius in pixels (circular aperture on positive flux)."""
    data = np.clip(np.asarray(stamp, dtype=np.float64), 0, None)
    total = data.sum()
    if total <= 0:
        return max(data.shape) * 0.25

    cy = (data.shape[0] - 1) / 2.0
    cx = (data.shape[1] - 1) / 2.0
    yy, xx = np.indices(data.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    flux = data.ravel()
    order = np.argsort(r)
    cumulative = np.cumsum(flux[order])
    idx = int(np.searchsorted(cumulative, 0.5 * total))
    idx = min(idx, len(order) - 1)
    return max(float(r[order][idx]), 1.0)


def _square_crop(stamp: np.ndarray) -> np.ndarray:
    h, w = stamp.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return stamp[y0:y0 + side, x0:x0 + side]


def prepare_morphology_stamp(
    stamp: np.ndarray,
    *,
    target_size: int = DEFAULT_STAMP_SIZE,
    stamp_q: float = 1.0,
) -> np.ndarray:
    """
    Prepare a GalaxyGenius panel like the morphology showcase.

    Center-crop to square, resize to ``target_size``, optionally flatten along
    y for explicit axis-ratio overrides.  By default ``stamp_q=1.0`` preserves
    the intrinsic SKIRT morphology from the published figure.
    """
    data = np.clip(_square_crop(np.asarray(stamp, dtype=np.float64)), 0, None)
    side = data.shape[0]
    if side != target_size:
        data = zoom(data, target_size / side, order=1)

    stamp_q = float(np.clip(stamp_q, 0.12, 1.0))
    if stamp_q < 0.995:
        data = zoom(data, (stamp_q, 1.0), order=1)
        # Re-center after axis-ratio zoom changes the grid extent.
        data = _square_crop(data)
        if data.shape[0] != target_size:
            data = zoom(data, target_size / data.shape[0], order=1)

    return np.clip(data, 0, None)


def _soft_edge_apodize(
    positive: np.ndarray,
    *,
    edge_start: float = 0.90,
    feather: float = 0.10,
) -> np.ndarray:
    """
    Feather only the outer edge of the square stamp.

    Avoids hard square INTERPOL boundaries without truncating the galaxy body
    (the old circular mask at radius_frac=0.46 removed most of the light).
    """
    h, w = positive.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.indices(positive.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_edge = min(h, w) * 0.5
    r_norm = r / max(r_edge, 1e-6)
    mask = np.clip((1.0 + feather - r_norm) / feather, 0.0, 1.0)
    mask = np.where(r_norm <= edge_start, 1.0, mask)
    return positive * mask


def build_interpol_surface_brightness(
    stamp: np.ndarray,
    *,
    re_arcsec: float,
    stamp_q: float = 1.0,
    target_size: int = DEFAULT_STAMP_SIZE,
) -> tuple[np.ndarray, float]:
    """
    Build an INTERPOL ``image`` array and per-pixel ``scale`` (arcsec/pix).

    Returns surface brightness in flux/arcsec^2 normalized so that integrated
    flux is 1 at amp=1 (lenstronomy magnitude2amplitude convention).
    """
    positive = prepare_morphology_stamp(stamp, target_size=target_size, stamp_q=stamp_q)
    positive = _soft_edge_apodize(np.clip(positive, 0, None))
    if positive.sum() <= 0:
        raise ValueError("Stamp has no positive flux after shaping.")

    r_half = estimate_half_light_radius(positive)
    scale = max(float(re_arcsec) / r_half, 1e-4)
    image = positive / positive.sum() / (scale ** 2)
    return image, scale


def build_interpol_lens_kwargs(
    stamp: np.ndarray,
    *,
    re_arcsec: float,
    magnitude: float,
    pa_deg: float = 0.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    stamp_q: float | None = None,
    target_size: int = DEFAULT_STAMP_SIZE,
) -> dict:
    """
    Keyword arguments for lenstronomy LightModel profile ``INTERPOL``.

    Position angle is applied via ``phi_G``.  Axis ratio is *not* taken from the
    lens mass model by default — the GalaxyGenius stamp already carries the
    observed morphology.
    """
    q = 1.0 if stamp_q is None else float(stamp_q)
    image, scale = build_interpol_surface_brightness(
        stamp,
        re_arcsec=re_arcsec,
        stamp_q=q,
        target_size=target_size,
    )
    return dict(
        image=image,
        center_x=float(center_x),
        center_y=float(center_y),
        phi_G=float(np.radians(pa_deg)),
        scale=float(scale),
        magnitude=float(magnitude),
    )


def build_galaxygenius_interpol_kwargs(
    morph_type: str,
    *,
    re_arcsec: float,
    magnitude: float = 21.0,
    pa_deg: float = 0.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    stamp_q: float | None = None,
    figures_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Load a GalaxyGenius panel and return INTERPOL kwargs + metadata."""
    raw, meta = load_morphology_panel(morph_type, figures_dir=figures_dir)
    kwargs = build_interpol_lens_kwargs(
        raw,
        re_arcsec=re_arcsec,
        magnitude=magnitude,
        pa_deg=pa_deg,
        center_x=center_x,
        center_y=center_y,
        stamp_q=stamp_q,
    )
    meta = dict(meta)
    meta["render_mode"] = "lenstronomy_INTERPOL"
    meta["interpol_scale_arcsec"] = kwargs["scale"]
    meta["stamp_q"] = 1.0 if stamp_q is None else float(stamp_q)
    return kwargs, meta
