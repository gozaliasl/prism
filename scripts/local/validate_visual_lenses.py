"""Quantitative validation for the visual-HST-inspired strong-lens test batch.

For every rendered system, computes (independently of the pipeline's own
metadata where possible, for a genuine external check):
  - per-band peak/integrated arc S/N (from image_sources_only vs noise
    measured on image_final)
  - magnification (pipeline's own pre-noise, ray-traced value, from metadata)
  - number of distinct lensed-image connected regions above a documented
    S/N threshold
  - max tangential extent (deg) and radial width (arcsec) of the arc
  - Einstein radius and PSF FWHM in pixels, and their ratio
  - source offset / theta_E
  - intrinsic (catalog-magnitude) and lensed (measured-flux) source/lens
    flux ratios per band
  - fraction of arc-mask pixels blended with (dominated by) lens light

Applies the pass/fail thresholds from the task spec and writes a CSV.

Run: python scripts/local/validate_visual_lenses.py <output_dir> <catalog_csv> <out_csv>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

PIXEL_SCALE_ARCSEC = 0.031
BANDS = ["F115W", "F150W", "F277W", "F444W"]

# Documented thresholds (spec)
CONNECTED_PIXEL_MIN = 3       # min pixels for a "region" to count (not a hot pixel)
PEAK_SNR_THRESHOLD = 5.0      # per-pixel threshold defining an arc pixel
INTEGRATED_SNR_PASS = 20.0    # at least one band must clear this
PEAK_SNR_PASS = 5.0
THETA_E_OVER_FWHM_PASS = 1.5
MIN_IMAGE_REGIONS_PASS = 2
MIN_TANGENTIAL_SPAN_DEG_PASS = 45.0


def mad_sigma(im):
    med = np.median(im)
    return float(np.median(np.abs(im - med)) * 1.4826)


def corner_noise_sigma(im, corner_frac=0.13):
    """Sky-noise estimate from the 4 image corners only, far from the
    lens/arc light that dominates whole-frame statistics in a compact
    300x300 frame -- whole-frame MAD is contaminated by that extended
    signal (confirmed: ~25% higher than the corner-only estimate for this
    batch's typical system), so corners give a more defensible, less
    signal-biased noise floor for the S/N calculation.

    SUPERSEDED for batches with a realistic (COSMOS-density) field
    population: field galaxies land at random positions including the
    corners, so this estimate becomes unstable (contaminated whenever a
    field source happens to fall in a corner box) -- see
    robust_sky_sigma() below, which is now used instead. Kept only for
    reference/backward compatibility.
    """
    h, w = im.shape
    ch, cw = max(int(h * corner_frac), 10), max(int(w * corner_frac), 10)
    corners = np.concatenate([
        im[:ch, :cw].ravel(), im[:ch, -cw:].ravel(),
        im[-ch:, :cw].ravel(), im[-ch:, -cw:].ravel(),
    ])
    med = np.median(corners)
    sigma = float(np.median(np.abs(corners - med)) * 1.4826)
    if sigma <= 0:
        sigma = float(np.std(corners)) or 1e-12
    return sigma


def robust_sky_sigma(im, center, exclude_radius_px, n_clip_iter=4, clip_sigma=3.0):
    """Whole-frame sky-noise estimate, robust to BOTH the extended lens/arc
    light near the frame center AND scattered field galaxies anywhere else
    in the frame: excludes a disk around the lens (radius
    exclude_radius_px, sized off theta_E) up front, then iteratively
    sigma-clips remaining outlier (source) pixels via MAD before taking
    the final MAD as the noise floor. This replaces corner_noise_sigma(),
    which silently breaks once a field galaxy lands in a corner box.
    """
    h, w = im.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - center[0], xx - center[1])
    vals = im[r > exclude_radius_px].ravel()
    if vals.size < 20:
        vals = im.ravel()
    for _ in range(n_clip_iter):
        med = np.median(vals)
        sigma = float(np.median(np.abs(vals - med)) * 1.4826)
        if sigma <= 0:
            break
        keep = np.abs(vals - med) < clip_sigma * sigma
        if keep.sum() < 20 or keep.all():
            break
        vals = vals[keep]
    med = np.median(vals)
    sigma = float(np.median(np.abs(vals - med)) * 1.4826)
    if sigma <= 0:
        sigma = float(np.std(vals)) or 1e-12
    return sigma


def psf_fwhm_pixels(psf_kernel):
    """Estimate FWHM (pixels) from a normalized PSF kernel via its
    circularly-averaged radial profile."""
    ny, nx = psf_kernel.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    peak = psf_kernel.max()
    half = peak / 2.0
    r_flat = r.flatten()
    v_flat = psf_kernel.flatten()
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    v_sorted = v_flat[order]
    below = np.where(v_sorted < half)[0]
    if len(below) == 0:
        return float(2 * r_sorted[-1])
    r_half = r_sorted[below[0]]
    return float(2 * r_half)


def arc_geometry(arc_mask, center, pixel_scale=PIXEL_SCALE_ARCSEC):
    """Given a boolean arc-pixel mask and the (y, x) lens center in pixels,
    return (max_tangential_span_deg, radial_width_arcsec, n_regions,
    largest_region_area, is_elongated).

    is_elongated guards against a false positive found by inspection
    (2026-08-02, user-reported: BR_000012 is a compact blob/companion-like
    single image, not an arc, yet its scattered noisy pixels spanned a
    wide angular RANGE around the lens center purely by chance, passing
    the old tangential_span>=45deg clause with only 1 region). Uses the
    largest connected region's own principal-axis ratio (via the
    eigenvalues of its pixel-coordinate covariance matrix) -- a genuine
    tangential arc is elongated (ratio >> 1); a compact blob/point-like
    image is not (ratio ~ 1)."""
    labeled, n_regions = label(arc_mask)
    if n_regions == 0:
        return 0.0, 0.0, 0, 0, False
    sizes = np.bincount(labeled.flatten())[1:]
    valid_regions = int(np.sum(sizes >= CONNECTED_PIXEL_MIN))
    largest_label = int(np.argmax(sizes)) + 1
    largest_area = int(sizes.max()) if len(sizes) else 0

    ys, xs = np.nonzero(arc_mask)
    if len(ys) == 0:
        return 0.0, 0.0, valid_regions, largest_area, False
    dy = ys - center[0]
    dx = xs - center[1]
    r = np.hypot(dy, dx)
    theta = np.degrees(np.arctan2(dy, dx)) % 360.0

    # Tangential span: sort angles, find largest gap, span = 360 - largest_gap
    theta_sorted = np.sort(theta)
    gaps = np.diff(theta_sorted, append=theta_sorted[0] + 360.0)
    max_gap = gaps.max()
    tangential_span_deg = float(360.0 - max_gap)

    radial_width_arcsec = float((r.max() - r.min()) * pixel_scale)

    # POLAR elongation, not Cartesian PCA (fixed 2026-08-02): a
    # near-complete RING is isotropic in x-y space (Cartesian PCA gives
    # eigenvalue ratio ~1, wrongly flagging genuine rings as "not
    # elongated"), but is highly elongated along its own circumference --
    # arc-length-along-the-ring (mean_radius * angular_span) divided by
    # radial_width correctly stays large for both a thin partial arc AND
    # a thin full ring, while staying small for a compact/round blob.
    ys_l, xs_l = np.nonzero(labeled == largest_label)
    is_elongated = False
    if len(ys_l) >= CONNECTED_PIXEL_MIN:
        dy_l = ys_l - center[0]
        dx_l = xs_l - center[1]
        r_l = np.hypot(dy_l, dx_l)
        theta_l = np.degrees(np.arctan2(dy_l, dx_l)) % 360.0
        theta_l_sorted = np.sort(theta_l)
        gaps_l = np.diff(theta_l_sorted, append=theta_l_sorted[0] + 360.0)
        span_l_deg = float(360.0 - gaps_l.max())
        mean_r = float(np.mean(r_l))
        radial_width_l = float(r_l.max() - r_l.min())
        arc_length = mean_r * np.radians(span_l_deg)
        elongation_ratio = arc_length / max(radial_width_l, 1.0)
        is_elongated = elongation_ratio >= 2.0

    return tangential_span_deg, radial_width_arcsec, valid_regions, largest_area, is_elongated


def main():
    out_dir = Path(sys.argv[1])
    catalog_path = sys.argv[2]
    out_csv = Path(sys.argv[3])

    cat = pd.read_csv(catalog_path)
    npz_files = sorted((out_dir / "unified_npz").glob("*.npz"))
    psf_dir = out_dir / "psf_arrays"

    records = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["metadata"]))
        lens_id = meta.get("lens_id")
        if lens_id is None or lens_id >= len(cat):
            continue
        cat_row = cat.iloc[lens_id]

        psf_file = psf_dir / f"{f.stem}_psf.npz"
        psf_data = np.load(psf_file) if psf_file.exists() else None

        # FIX (2026-08-02, multi-telescope run): use THIS system's own
        # band names and pixel scale from its metadata instead of the
        # hardcoded JWST BANDS/PIXEL_SCALE_ARCSEC constants -- for any
        # non-JWST telescope the npz/PSF arrays are keyed by that
        # telescope's own band names (e.g. EUCLID_VIS), so the hardcoded
        # JWST names never matched, PSF FWHM silently came back NaN for
        # every band, and theta_E/FWHM failed for 100% of systems
        # regardless of actual visibility.
        bands = meta.get("bands") or BANDS
        pixel_scale = float(meta.get("delta_pix") or PIXEL_SCALE_ARCSEC)

        theta_E = meta.get("theta_E")
        theta_E_pix = theta_E / pixel_scale if theta_E else np.nan
        source_x = meta.get("source_x")
        source_y = meta.get("source_y")
        offset_over_theta_E = (
            float(np.hypot(source_x, source_y) / theta_E)
            if (source_x is not None and source_y is not None and theta_E) else np.nan
        )

        image_final = d["image_final"]  # (n_bands, H, W)
        image_sources_only = d["image_sources_only"]
        image_lens_only = d["image_lens_only"]
        H, W = image_final.shape[1], image_final.shape[2]
        center = ((H - 1) / 2.0, (W - 1) / 2.0)  # lens is at frame center by construction

        rec = dict(
            file=f.name, lens_id=lens_id,
            theta_E_arcsec=theta_E, theta_E_pix=theta_E_pix,
            lens_redshift=meta.get("lens_redshift"), source_redshift=meta.get("source_redshift"),
            source_x=source_x, source_y=source_y, offset_over_theta_E=offset_over_theta_E,
            magnification_model=meta.get("magnification"),
            lens_radius=meta.get("lens_radius"), source_radius=meta.get("source_radius"),
        )

        band_peak_snr, band_int_snr, band_fwhm_pix, band_theta_over_fwhm = {}, {}, {}, {}
        band_flux_ratio_lensed, band_flux_ratio_intrinsic, band_blend_frac = {}, {}, {}
        band_n_regions, band_tang_span, band_rad_width = {}, {}, {}
        band_elongated = {}

        for bi, band in enumerate(bands):
            arc = image_sources_only[bi]
            lens_only = image_lens_only[bi]
            final = image_final[bi]
            _excl_r = 3.0 * theta_E_pix if (theta_E_pix and not np.isnan(theta_E_pix)) else 0.15 * min(H, W)
            noise_sigma = robust_sky_sigma(final, center, _excl_r)
            if noise_sigma <= 0:
                noise_sigma = float(np.std(final)) or 1e-12

            peak_snr = float(arc.max() / noise_sigma)
            arc_mask = arc > (PEAK_SNR_THRESHOLD * noise_sigma)
            integrated_snr = float(
                arc[arc_mask].sum() / (noise_sigma * np.sqrt(max(int(arc_mask.sum()), 1)))
            ) if arc_mask.any() else 0.0

            band_peak_snr[band] = peak_snr
            band_int_snr[band] = integrated_snr

            tang_span, rad_width, n_regions, largest_area, is_elongated = arc_geometry(arc_mask, center, pixel_scale)
            band_n_regions[band] = n_regions
            band_elongated[band] = is_elongated
            band_tang_span[band] = tang_span
            band_rad_width[band] = rad_width

            if psf_data is not None and band in psf_data:
                fwhm = psf_fwhm_pixels(psf_data[band])
            else:
                fwhm = np.nan
            band_fwhm_pix[band] = fwhm
            band_theta_over_fwhm[band] = (theta_E_pix / fwhm) if fwhm and fwhm > 0 else np.nan

            flux_lens = float(lens_only.sum())
            flux_arc = float(arc.sum())
            band_flux_ratio_lensed[band] = (flux_arc / flux_lens) if flux_lens > 0 else np.nan

            src_mag_col = f"source_mag_{band.lower()}"
            lens_mag_col = f"lens_mag_{band.lower()}"
            if src_mag_col in cat_row and lens_mag_col in cat_row:
                dmag = float(cat_row[src_mag_col]) - float(cat_row[lens_mag_col])
                band_flux_ratio_intrinsic[band] = float(10 ** (-0.4 * dmag))
            else:
                band_flux_ratio_intrinsic[band] = np.nan

            if arc_mask.any():
                blended = (lens_only[arc_mask] > arc[arc_mask])
                band_blend_frac[band] = float(blended.mean())
            else:
                band_blend_frac[band] = np.nan

        best_band = max(bands, key=lambda b: band_int_snr[b])
        rec["best_band"] = best_band
        rec["max_integrated_snr"] = band_int_snr[best_band]
        rec["max_peak_snr"] = max(band_peak_snr.values())
        rec["max_n_regions"] = max(band_n_regions.values())
        rec["max_tangential_span_deg"] = max(band_tang_span.values())
        rec["max_radial_width_arcsec"] = max(band_rad_width.values())
        rec["any_band_elongated"] = bool(any(band_elongated.values()))
        rec["theta_E_over_fwhm_best_band"] = band_theta_over_fwhm.get(best_band, np.nan)
        rec["min_theta_E_over_fwhm"] = np.nanmin(list(band_theta_over_fwhm.values()))
        rec["mean_blend_fraction"] = float(np.nanmean(list(band_blend_frac.values())))

        for band in bands:
            rec[f"peak_snr_{band}"] = band_peak_snr[band]
            rec[f"integrated_snr_{band}"] = band_int_snr[band]
            rec[f"fwhm_pix_{band}"] = band_fwhm_pix[band]
            rec[f"flux_ratio_lensed_{band}"] = band_flux_ratio_lensed[band]
            rec[f"flux_ratio_intrinsic_{band}"] = band_flux_ratio_intrinsic[band]
            rec[f"blend_frac_{band}"] = band_blend_frac[band]

        # Pass/fail per documented thresholds
        reasons = []
        cond_snr_int = rec["max_integrated_snr"] >= INTEGRATED_SNR_PASS
        if not cond_snr_int:
            reasons.append(f"integrated_snr {rec['max_integrated_snr']:.1f} < {INTEGRATED_SNR_PASS}")
        cond_snr_peak = rec["max_peak_snr"] >= PEAK_SNR_PASS
        if not cond_snr_peak:
            reasons.append(f"peak_snr {rec['max_peak_snr']:.1f} < {PEAK_SNR_PASS}")
        cond_res = (not np.isnan(rec["min_theta_E_over_fwhm"])) and rec["min_theta_E_over_fwhm"] >= THETA_E_OVER_FWHM_PASS
        if not cond_res:
            reasons.append(f"theta_E/FWHM {rec['min_theta_E_over_fwhm']:.2f} < {THETA_E_OVER_FWHM_PASS}")
        # FIX (2026-08-02, user-reported false positive, BR_000012): a
        # tangential-span pass with only 1 region must also come from a
        # genuinely elongated feature, not a compact blob/point-like image
        # whose scattered pixels happened to span a wide angle by chance.
        cond_morph = (rec["max_n_regions"] >= MIN_IMAGE_REGIONS_PASS) or (
            rec["max_tangential_span_deg"] >= MIN_TANGENTIAL_SPAN_DEG_PASS and rec["any_band_elongated"]
        )
        if not cond_morph:
            reasons.append(
                f"regions={rec['max_n_regions']} span={rec['max_tangential_span_deg']:.0f}deg "
                f"elongated={rec['any_band_elongated']} < thresholds"
            )

        rec["pass"] = bool(cond_snr_int and cond_snr_peak and cond_res and cond_morph)
        rec["fail_reasons"] = "; ".join(reasons) if reasons else ""

        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    n_pass = int(df["pass"].sum())
    print(f"{n_pass}/{len(df)} systems PASS")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
