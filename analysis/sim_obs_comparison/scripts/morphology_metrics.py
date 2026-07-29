#!/usr/bin/env python3
"""
Shared morphology/lensing metric extraction, used by both
compute_real_lens_morphology.py (real COSMOS-Web FITS cutouts) and
compute_sim_lens_morphology.py (simulated JADES/COSMOS-Web npz outputs).

Given a dict of band -> 2D ndarray (same pixel scale / cutout size for all
bands), measure_lens_arrays() returns the same metric set photutils
segmentation on an SNR-combined detection image (F150W+F277W, +F115W/F444W
if available), restricted to the central 6"x6" of the frame.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from prism.morphology.validation import compute_cas_gini_m20  # noqa: E402

PIXEL_SCALE = 0.03  # arcsec/pixel

CAS_GINI_M20_KEYS = ("concentration", "asymmetry", "smoothness", "clumpiness", "gini", "m20")


def background_stats(data):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    return median, std


def measure_lens_arrays(name, images, pixel_scale=PIXEL_SCALE, return_seg=False):
    """images: dict of band_name -> 2D ndarray (float).

    If return_seg=True, also returns (det, segm, rows) -- the SNR-combined
    detection image, the full-frame segmentation map, and the list of
    SourceCatalog rows that fall within the central 6"x6" box used for the
    metrics -- for diagnostic visualization (see visualize_segmentation.py).
    """
    bg = {}
    for band, d in images.items():
        bg[band] = background_stats(d)

    det_bands = [b for b in ["F150W", "F277W", "F115W", "F444W"] if b in images]
    det = None
    for b in det_bands:
        med, std = bg[b]
        snr = (images[b] - med) / max(std, 1e-12)
        det = snr if det is None else det + snr
    if det is None:
        return None

    threshold = 3.0 * np.sqrt(len(det_bands))
    segm = detect_sources(det, threshold, npixels=8)
    if segm is None:
        result = {"name": name, "n_components": 0}
        result["max_sep_arcsec"] = np.nan
        result["lens_reff_arcsec"] = np.nan
        result["lens_axis_ratio"] = np.nan
        result["arc_length_arcsec"] = np.nan
        result["arc_width_arcsec"] = np.nan
        result["arc_length_to_width"] = np.nan
        for key in CAS_GINI_M20_KEYS:
            result[key] = np.nan
        for band in images:
            med, std = bg[band]
            result[f"bg_rms_{band}"] = std
            result[f"sb_peak_{band}"] = float(np.max(images[band]))
            result[f"total_flux_{band}"] = np.nan
        if return_seg:
            return result, det, segm, []
        return result

    cat = SourceCatalog(det, segm)
    ny, nx = det.shape
    cy, cx = ny / 2.0, nx / 2.0

    half_box = 3.0 / pixel_scale  # pixels, central 6"x6"
    rows = []
    for src in cat:
        dx = src.xcentroid - cx
        dy = src.ycentroid - cy
        if abs(dx) > half_box or abs(dy) > half_box:
            continue
        rows.append(src)

    n_components = len(rows)

    max_sep = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dx = (rows[i].xcentroid - rows[j].xcentroid) * pixel_scale
            dy = (rows[i].ycentroid - rows[j].ycentroid) * pixel_scale
            sep = np.hypot(dx, dy)
            max_sep = max(max_sep, sep)

    central = min(rows, key=lambda s: np.hypot(s.xcentroid - cx, s.ycentroid - cy),
                   default=None)
    non_central = [s for s in rows if s is not central]
    arc = max(non_central, key=lambda s: s.area.value, default=None)

    result = {"name": name, "n_components": n_components, "max_sep_arcsec": max_sep}

    if central is not None:
        a = float(central.semimajor_sigma.value) * pixel_scale * 2.355
        b = float(central.semiminor_sigma.value) * pixel_scale * 2.355
        result["lens_reff_arcsec"] = (a + b) / 2.0
        result["lens_axis_ratio"] = b / a if a > 0 else np.nan
        central_mask = segm.data == central.label
        cas_gini_m20 = compute_cas_gini_m20(det, central_mask)
        result.update(cas_gini_m20)
    else:
        result["lens_reff_arcsec"] = np.nan
        result["lens_axis_ratio"] = np.nan
        for key in CAS_GINI_M20_KEYS:
            result[key] = np.nan

    if arc is not None:
        a = float(arc.semimajor_sigma.value) * pixel_scale * 2.355
        b = float(arc.semiminor_sigma.value) * pixel_scale * 2.355
        result["arc_length_arcsec"] = a
        result["arc_width_arcsec"] = b
        result["arc_length_to_width"] = a / b if b > 0 else np.nan
    else:
        result["arc_length_arcsec"] = np.nan
        result["arc_width_arcsec"] = np.nan
        result["arc_length_to_width"] = np.nan

    footprint = segm.data > 0
    for band, d in images.items():
        med, std = bg[band]
        result[f"bg_rms_{band}"] = std
        result[f"sb_peak_{band}"] = float(np.max(d))
        result[f"total_flux_{band}"] = float(np.sum((d - med)[footprint]))

    pairs = [("F115W", "F444W"), ("F115W", "F277W"), ("F150W", "F444W"), ("F277W", "F444W")]
    for b1, b2 in pairs:
        f1, f2 = result.get(f"total_flux_{b1}"), result.get(f"total_flux_{b2}")
        if f1 is not None and f2 is not None and not np.isnan(f1) and not np.isnan(f2) and f1 > 0 and f2 > 0:
            result[f"color_{b1}_{b2}"] = -2.5 * np.log10(f1 / f2)
        else:
            result[f"color_{b1}_{b2}"] = np.nan

    if return_seg:
        return result, det, segm, rows
    return result


def summarize(df, numeric_cols=None):
    import pandas as pd
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = {}
    for col in numeric_cols:
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        summary[col] = {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p16": float(np.percentile(vals, 16)),
            "p84": float(np.percentile(vals, 84)),
        }
    return summary
