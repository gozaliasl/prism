#!/usr/bin/env python3
"""
Compute quantitative morphology/lensing statistics for the 435 real COSMOS-Web
lenses in data/real_lenses/, to ground-truth the simulator (Phase 0 of the
"most realistic JWST strong lens simulation" effort).

For each lens (using all 4 NIRCam bands where available), this script
measures, via photutils source segmentation on an inverse-variance-weighted
detection image (F150W+F277W, the two highest-S/N bands):

  - n_components        : multiplicity proxy (number of detected segments
                           within the central 6" -- arcs/counter-images/
                           companions all count)
  - max_sep_arcsec       : maximum separation between any two component
                           centroids -- proxy for 2*theta_E (image separation)
  - arc_length_arcsec    : semi-major axis (Kron-like) of the largest
                           non-central segment -- "arc length"
  - arc_width_arcsec     : semi-minor axis of that segment -- "arc width"
  - arc_length_to_width  : ratio of the above
  - lens_reff_arcsec     : effective (half-light) radius of the central
                           (brightest, most central) segment -- lens galaxy size
  - lens_axis_ratio      : b/a of the central segment
  - color_XXX_YYY        : AB-like color (asinh flux ratio) between bands,
                           measured in a 0.3" aperture on the brightest pixel
  - bg_rms_<band>        : background RMS per band (electron-rate units, same
                           units as the FITS pixel values)
  - sb_peak_<band>       : peak surface brightness (pixel value) per band
  - total_flux_<band>    : segmentation-summed flux per band

Output: analysis/sim_obs_comparison/catalogs/real_lens_morphology.csv
Also writes summary distribution stats to:
  analysis/sim_obs_comparison/reports/real_lens_statistics_summary.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog
from photutils.utils import circular_footprint

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from prism.morphology.validation import compute_cas_gini_m20  # noqa: E402

CAS_GINI_M20_KEYS = ("concentration", "asymmetry", "smoothness", "clumpiness", "gini", "m20")

PIXEL_SCALE = 0.03  # arcsec/pixel (matches CDELT1=8.333e-6 deg)
BANDS = ["F115W", "F150W", "F277W", "F444W"]
REAL_LENS_DIR = Path("data/real_lenses")
OUT_DIR = Path("analysis/sim_obs_comparison")


def discover_lenses(real_lens_dir):
    lenses = defaultdict(dict)
    for band in BANDS:
        for f in real_lens_dir.glob(f"*_{band}.fits"):
            name = f.stem.replace(f"_{band}", "")
            lenses[name][band] = f
    return lenses


def load_band(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64)


def background_stats(data):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    return median, std


def measure_lens(name, band_paths):
    images = {}
    bg = {}
    for band, path in band_paths.items():
        d = load_band(path)
        images[band] = d
        bg_median, bg_std = background_stats(d)
        bg[band] = (bg_median, bg_std)

    # Detection image: sum of F150W + F277W (or whatever is available),
    # background-subtracted, in units of sigma
    det_bands = [b for b in ["F150W", "F277W", "F115W", "F444W"] if b in images]
    det = None
    for b in det_bands:
        med, std = bg[b]
        snr = (images[b] - med) / max(std, 1e-12)
        det = snr if det is None else det + snr
    if det is None:
        return None

    threshold = 3.0 * np.sqrt(len(det_bands))  # combined SNR threshold
    segm = detect_sources(det, threshold, npixels=8)
    if segm is None:
        result = {"name": name, "n_components": 0}
        for key in CAS_GINI_M20_KEYS:
            result[key] = np.nan
        return result

    cat = SourceCatalog(det, segm)
    ny, nx = det.shape
    cy, cx = ny / 2.0, nx / 2.0

    # Restrict to central 6" x 6" box (200x200 px for 0.03"/px, 300px frame)
    half_box = 3.0 / PIXEL_SCALE  # pixels
    rows = []
    for src in cat:
        dx = src.xcentroid - cx
        dy = src.ycentroid - cy
        if abs(dx) > half_box or abs(dy) > half_box:
            continue
        rows.append(src)

    n_components = len(rows)

    # Max pairwise separation (proxy for image separation / 2*theta_E)
    max_sep = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dx = (rows[i].xcentroid - rows[j].xcentroid) * PIXEL_SCALE
            dy = (rows[i].ycentroid - rows[j].ycentroid) * PIXEL_SCALE
            sep = np.hypot(dx, dy)
            max_sep = max(max_sep, sep)

    # Central (lens galaxy) segment: closest centroid to frame center
    central = min(rows, key=lambda s: np.hypot(s.xcentroid - cx, s.ycentroid - cy),
                   default=None)
    # Arc segment: largest non-central segment by area
    non_central = [s for s in rows if s is not central]
    arc = max(non_central, key=lambda s: s.area.value, default=None)

    result = {"name": name, "n_components": n_components,
              "max_sep_arcsec": max_sep}

    if central is not None:
        a = float(central.semimajor_sigma.value) * PIXEL_SCALE * 2.355  # FWHM-like
        b = float(central.semiminor_sigma.value) * PIXEL_SCALE * 2.355
        result["lens_reff_arcsec"] = (a + b) / 2.0
        result["lens_axis_ratio"] = b / a if a > 0 else np.nan
        central_mask = segm.data == central.label
        result.update(compute_cas_gini_m20(det, central_mask))
    else:
        result["lens_reff_arcsec"] = np.nan
        result["lens_axis_ratio"] = np.nan
        for key in CAS_GINI_M20_KEYS:
            result[key] = np.nan

    if arc is not None:
        a = float(arc.semimajor_sigma.value) * PIXEL_SCALE * 2.355
        b = float(arc.semiminor_sigma.value) * PIXEL_SCALE * 2.355
        result["arc_length_arcsec"] = a
        result["arc_width_arcsec"] = b
        result["arc_length_to_width"] = a / b if b > 0 else np.nan
    else:
        result["arc_length_arcsec"] = np.nan
        result["arc_width_arcsec"] = np.nan
        result["arc_length_to_width"] = np.nan

    # Per-band background, peak SB, total flux (within segmentation footprint)
    footprint = segm.data > 0
    for band in BANDS:
        if band in images:
            med, std = bg[band]
            result[f"bg_rms_{band}"] = std
            result[f"sb_peak_{band}"] = float(np.max(images[band]))
            result[f"total_flux_{band}"] = float(np.sum((images[band] - med)[footprint]))
        else:
            result[f"bg_rms_{band}"] = np.nan
            result[f"sb_peak_{band}"] = np.nan
            result[f"total_flux_{band}"] = np.nan

    # Colors: asinh ratio of total fluxes (avoids log of negative)
    pairs = [("F115W", "F444W"), ("F115W", "F277W"), ("F150W", "F444W"), ("F277W", "F444W")]
    for b1, b2 in pairs:
        f1, f2 = result.get(f"total_flux_{b1}"), result.get(f"total_flux_{b2}")
        if f1 is not None and f2 is not None and not np.isnan(f1) and not np.isnan(f2) and f1 > 0 and f2 > 0:
            result[f"color_{b1}_{b2}"] = -2.5 * np.log10(f1 / f2)
        else:
            result[f"color_{b1}_{b2}"] = np.nan

    return result


def main():
    lenses = discover_lenses(REAL_LENS_DIR)
    print(f"Found {len(lenses)} lenses")

    rows = []
    for i, (name, band_paths) in enumerate(sorted(lenses.items())):
        if len(band_paths) < 2:
            continue
        try:
            r = measure_lens(name, band_paths)
            if r is not None:
                rows.append(r)
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(lenses)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "catalogs" / "real_lens_morphology.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {out_csv}")

    # Summary statistics
    summary = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
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

    # Multiplicity classification counts
    mult = df["n_components"].fillna(0).astype(int)
    summary["multiplicity_histogram"] = {
        "single (1)": int((mult == 1).sum()),
        "double (2)": int((mult == 2).sum()),
        "triple (3)": int((mult == 3).sum()),
        "quad+ (>=4)": int((mult >= 4).sum()),
    }

    out_json = OUT_DIR / "reports" / "real_lens_statistics_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {out_json}")


if __name__ == "__main__":
    main()
