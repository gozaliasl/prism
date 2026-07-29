"""Euclid RGB composites following Mischa Schirmer's ``eummy`` pipeline.

Reference: https://github.com/schirmermischa/eummy  (PyPI: eummy)

Pipeline (default MER mapping):
  1. Per-band flux scales  VIS/Y/J/H  (eummy --scales)
  2. B = VIS,  G = (Y+J)/2,  R = H
  3. L = VIS with adaptive H blend (--fr)
  4. Shared arcsinh stretch with black/white levels (--pivot, --blackwhite)
  5. Mild S-curve contrast on L
  6. CIELab: chrominance from BGR, replace L* with L, saturate a*/b*
  7. Optional unsharp mask

Simulated cutouts use different absolute flux units than MER stacks, so
``blackwhite`` is estimated from the scaled luminance unless overridden.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# eummy v1.3.2 defaults
EUMMY_SCALES = (0.002039, 0.5950, 1.0000, 1.0985)  # I Y J H  (divide by these)
EUMMY_PIVOT = 0.15
EUMMY_FR = 0.3
EUMMY_SATURATE = 2.0
EUMMY_CONTRAST = 1.0
EUMMY_UM = (1.6, 0.75, 0.09)  # FWHM, strength, threshold


def _as_f32(a: np.ndarray) -> np.ndarray:
    return np.asanyarray(a, dtype=np.float32).copy()


def _pick_band(images: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    lower = {k.lower(): k for k in images}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return _as_f32(images[key])
    raise KeyError(f"None of {list(names)} found in images keys={list(images)}")


def _estimate_blackwhite(ref: np.ndarray) -> Tuple[float, float]:
    """Map eummy's MER blackwhite=[-1.3, 7000] onto our flux units."""
    finite = ref[np.isfinite(ref)]
    if finite.size == 0:
        return -1.0, 1.0
    sky = float(np.median(finite))
    mad = float(np.median(np.abs(finite - sky)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(finite) + 1e-12)
    # After sky-subtract, sky≈0; black slightly negative like eummy's -1.3
    black = sky - 1.8 * sigma
    # White: high but not the absolute peak (avoids single-pixel blowout)
    white = float(np.percentile(finite, 99.7))
    if white <= black + 5 * sigma:
        white = black + max(15.0 * sigma, float(np.percentile(finite, 99.95)) - black, 1e-3)
    return black, white


def _arcsinh_norm(ch: np.ndarray, pivot: float, black: float, white: float) -> np.ndarray:
    b = np.arcsinh(pivot * black)
    w = np.arcsinh(pivot * white)
    scale = 1.0 / max(w - b, 1e-12)
    out = (np.arcsinh(pivot * ch) - b) * scale
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _contrast_scurve(L: np.ndarray, contrast: float) -> np.ndarray:
    if contrast == 0:
        return L
    # eummy: y = c*(0.5707 x^3 - 1.8298 x^2 + 2.2592 x - x) + x
    x = np.clip(L, 0.0, 1.0)
    return (contrast * (0.5707 * x**3 - 1.8298 * x**2 + 2.2592 * x - x) + x).astype(np.float32)


def _unsharp_mask(rgb: np.ndarray, radius: float, strength: float, threshold: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return rgb
    # eummy FWHM → sigma; mild sharpen of luminance
    sigma = max(radius / 2.355, 0.01)
    luma = np.mean(rgb, axis=2, keepdims=True)
    blur = gaussian_filter(luma, sigma=sigma)
    high = luma - blur
    mask = np.abs(high) > threshold
    luma2 = np.where(mask, luma + strength * high, luma)
    # Preserve chrominance ratios
    scale = luma2 / np.maximum(luma, 1e-6)
    return np.clip(rgb * scale, 0.0, 1.0).astype(np.float32)


def _lab_luminance_swap(rgb: np.ndarray, L: np.ndarray, saturate: float) -> np.ndarray:
    """CIELab: replace L* with eummy luminance L, boost a*/b*."""
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    L = np.clip(L, 0.0, 1.0).astype(np.float32)
    try:
        import cv2

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        lab[:, :, 1] = np.clip(lab[:, :, 1] * saturate, -128, 127)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * saturate, -128, 127)
        lab[:, :, 0] = L * 100.0
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    except Exception:
        # Fallback: luma swap in RGB space (no OpenCV)
        luma = np.mean(rgb, axis=2, keepdims=True)
        chroma = rgb - luma
        rgb2 = L[..., None] + chroma * saturate
        return np.clip(rgb2, 0.0, 1.0).astype(np.float32)


def _sky_subtract(im: np.ndarray) -> np.ndarray:
    sky = float(np.median(im[np.isfinite(im)])) if np.isfinite(im).any() else 0.0
    return im - sky


def create_euclid_eummy_rgb(
    images: Mapping[str, np.ndarray],
    *,
    scales: Sequence[float] | str = "auto",
    pivot: float = EUMMY_PIVOT,
    fr: float = EUMMY_FR,
    saturate: float = EUMMY_SATURATE,
    contrast: float = EUMMY_CONTRAST,
    blackwhite: Optional[Tuple[float, float]] = None,
    blend_iy: bool = False,
    fi: float = 1.6,
    unsharp: Optional[Tuple[float, float, float]] = EUMMY_UM,
    sky_subtract: bool = True,
) -> np.ndarray:
    """Build an eummy-style RGB float array in [0, 1] from Euclid band images.

    ``images`` keys may be ``EUCLID_VIS`` / ``EUCLID_Y`` / … or ``VIS`` / ``Y`` / …

    ``scales="auto"`` (default): use eummy MER scales only when band medians
    differ by >50× (real MER units). Simulated cutouts share e-/s units across
    bands, so auto falls back to (1,1,1,1) after sky subtraction — otherwise
    VIS÷0.002 amplifies noise into purple grain.
    """
    I = _pick_band(images, ("EUCLID_VIS", "VIS", "I", "IE"))
    Y = _pick_band(images, ("EUCLID_Y", "Y", "YE"))
    J = _pick_band(images, ("EUCLID_J", "J", "JE"))
    H = _pick_band(images, ("EUCLID_H", "H", "HE"))

    if sky_subtract:
        I, Y, J, H = map(_sky_subtract, (I, Y, J, H))

    if isinstance(scales, str) and scales.lower() == "auto":
        meds = [float(np.median(np.abs(x))) for x in (I, Y, J, H)]
        meds = [m if m > 0 else 1e-12 for m in meds]
        ratio = max(meds) / min(meds)
        scales = EUMMY_SCALES if ratio > 50.0 else (1.0, 1.0, 1.0, 1.0)

    si, sy, sj, sh = [float(s) for s in scales]
    I = I / si
    Y = Y / sy
    J = J / sj
    H = H / sh

    if blend_iy:
        B = (Y + fi * I) / (1.0 + fi)
        G = J
    else:
        B = I
        G = 0.5 * (Y + J)

    if fr > 0:
        w = fr * np.exp(-0.2 * np.abs(I))
        L = (I + w * H) / (1.0 + w)
    else:
        L = I.copy()
    R = H

    if blackwhite is None:
        black, white = _estimate_blackwhite(L)
    else:
        black, white = float(blackwhite[0]), float(blackwhite[1])

    B = _arcsinh_norm(B, pivot, black, white)
    G = _arcsinh_norm(G, pivot, black, white)
    R = _arcsinh_norm(R, pivot, black, white)
    L = _arcsinh_norm(L, pivot, black, white)
    L = _contrast_scurve(L, contrast)

    rgb = np.stack([R, G, B], axis=-1)
    rgb = _lab_luminance_swap(rgb, L, saturate)

    if unsharp is not None:
        rgb = _unsharp_mask(rgb, unsharp[0], unsharp[1], unsharp[2])

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def eummy_params_from_config(cfg: Optional[Dict] = None) -> Dict:
    """Read ``output.rgb.eummy`` overrides from simulator CONFIG."""
    cfg = cfg or {}
    rgb = (cfg.get("output") or {}).get("rgb") or {}
    e = rgb.get("eummy") or {}
    if not isinstance(e, dict):
        e = {}
    scales = e.get("scales", "auto")
    if isinstance(scales, str):
        pass
    else:
        scales = tuple(float(x) for x in scales)
    um = e.get("unsharp", list(EUMMY_UM))
    if um is False or um is None:
        unsharp = None
    else:
        unsharp = tuple(float(x) for x in um)
    bw = e.get("blackwhite")
    return dict(
        scales=scales,
        pivot=float(e.get("pivot", EUMMY_PIVOT)),
        fr=float(e.get("fr", EUMMY_FR)),
        saturate=float(e.get("saturate", EUMMY_SATURATE)),
        contrast=float(e.get("contrast", EUMMY_CONTRAST)),
        blackwhite=tuple(bw) if bw is not None else None,
        blend_iy=bool(e.get("blend_iy", False)),
        fi=float(e.get("fi", 1.6)),
        unsharp=unsharp,
        sky_subtract=bool(e.get("sky_subtract", True)),
    )
