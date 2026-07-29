"""Euclid RGB composites via Dan Coe's Trilogy (oliveirara/trilogy v1.1+).

Reference: https://github.com/oliveirara/trilogy  (PyPI: trilogy)
Algorithm: log10 stretch with (x0, x1, x2) → (0, noiselum, 1),
per-channel independent scaling, optional colour saturation boost.

Channel mapping matches eummy / Euclid MER convention:
  R = H,  G = (Y+J)/2,  B = VIS
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

try:
    from trilogy.trilogy import (
        TrilogyConfig,
        adjust_saturation,
        apply_scaling,
        auto_adjust_parameters,
        compute_robust_stats,
        compute_scaling_k,
        _levels_from_stats,
    )
    _TRILOGY_OK = True
except Exception:  # noqa: BLE001
    _TRILOGY_OK = False


def _pick_band(images: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    lower = {k.lower(): k for k in images}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return np.asanyarray(images[key], dtype=np.float32).copy()
    raise KeyError(f"None of {list(names)} found in images keys={list(images)}")


def _scale_channel(data: np.ndarray, config: "TrilogyConfig", channel: str) -> np.ndarray:
    """Return float channel in [0, 1] using Trilogy log scaling."""
    flat = data.ravel()
    valid = flat[np.isfinite(flat)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    data_sorted = np.sort(valid)
    stats = compute_robust_stats(data_sorted, presorted=True)

    cfg = config
    if cfg.auto_adjust:
        cfg = auto_adjust_parameters(cfg, stats, channel)

    unsatpercent = 1.0 - 0.01 * cfg.satpercent
    if cfg.noise is not None and cfg.saturate is not None:
        levels = (0.0, float(cfg.noise), float(cfg.saturate))
    else:
        levels = _levels_from_stats(
            data_sorted,
            stats,
            unsatpercent,
            noisesig=cfg.noisesig,
            correctbias=cfg.correctbias,
            noisesig0=cfg.noisesig0,
        )

    noiselum = cfg.noiselums.get(channel, cfg.noiselum)
    k = compute_scaling_k(levels, noiselum)
    scaled_u8 = apply_scaling(data, levels, k)
    return scaled_u8.astype(np.float32) / 255.0


def create_euclid_trilogy_rgb(
    images: Mapping[str, np.ndarray],
    *,
    noiselum: float = 0.15,
    satpercent: float = 0.001,
    colorsatfac: float = 1.4,
    noisesig: float = 1.0,
    noisesig0: float = 2.0,
    correctbias: bool = True,
    sky_subtract: bool = True,
) -> np.ndarray:
    """Build a Trilogy-style RGB float array in [0, 1] from Euclid bands."""
    if not _TRILOGY_OK:
        raise ImportError(
            "trilogy>=1.1 is required for create_euclid_trilogy_rgb "
            "(pip install 'trilogy>=1.1')"
        )

    I = _pick_band(images, ("EUCLID_VIS", "VIS", "I", "IE"))
    Y = _pick_band(images, ("EUCLID_Y", "Y", "YE"))
    J = _pick_band(images, ("EUCLID_J", "J", "JE"))
    H = _pick_band(images, ("EUCLID_H", "H", "HE"))

    if sky_subtract:
        for arr in (I, Y, J, H):
            arr -= float(np.median(arr[np.isfinite(arr)]))

    R = H
    G = 0.5 * (Y + J)
    B = I

    cfg = TrilogyConfig(
        satpercent=satpercent,
        noiselum=noiselum,
        noisesig=noisesig,
        noisesig0=noisesig0,
        correctbias=correctbias,
        colorsatfac=colorsatfac,
        auto_adjust=True,
        samplesize=min(I.shape),
    )

    r = _scale_channel(R, cfg, "R")
    g = _scale_channel(G, cfg, "G")
    b = _scale_channel(B, cfg, "B")

    rgb = np.stack([r, g, b], axis=0)  # (3, H, W) for adjust_saturation
    if abs(colorsatfac - 1.0) > 1e-6:
        rgb = adjust_saturation(rgb * 255.0, colorsatfac)
        rgb = np.clip(rgb, 0, 255) / 255.0

    return np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0).astype(np.float32)


def trilogy_params_from_config(cfg: Optional[Dict] = None) -> Dict:
    cfg = cfg or {}
    rgb = (cfg.get("output") or {}).get("rgb") or {}
    t = rgb.get("trilogy") or {}
    if not isinstance(t, dict):
        t = {}
    return dict(
        noiselum=float(t.get("noiselum", 0.15)),
        satpercent=float(t.get("satpercent", 0.001)),
        colorsatfac=float(t.get("colorsatfac", 1.4)),
        noisesig=float(t.get("noisesig", 1.0)),
        noisesig0=float(t.get("noisesig0", 2.0)),
        correctbias=bool(t.get("correctbias", True)),
        sky_subtract=bool(t.get("sky_subtract", True)),
    )
