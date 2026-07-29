#!/usr/bin/env python3
"""Per-band (F115W/F150W/F277W/F444W) quality comparison: real COSMOS-Web
lens cutouts (data/real_lenses) vs. simulated TNG50/TNG100 outputs.

Targets band-specific artifacts flagged during visual QC -- especially in
F277W/F444W, where coarser TNG-particle smoothing and broader PSFs can leave
either residual "graininess" (individual star particles visible) or
over-smoothing relative to real JWST data. For each band this computes:

  - bg_rms, peak SB
  - CAS/Gini/M20 (concentration, asymmetry, smoothness, clumpiness, gini, m20)
  - "grain_frac": fraction of power spectrum (within the source footprint)
    at spatial frequencies above half the Nyquist -- a proxy for
    pixel-to-pixel graininess / particle-dot texture. Higher = grainier.

Simulated objects are additionally tagged with:
  - lens_redshift bucket (low/mid/high, from saved metadata)
  - TNG SF class (star_forming/quiescent, from tng_lens.ssfr_per_yr)
  - sim source (TNG50 / TNG100)

Output: per-object CSV + a markdown summary table (real vs. sim, per band,
with sim broken down by redshift bucket and SF class), written to
/Volumes/extHD/jwst_lens_outputs/qc_reports/<run_tag>/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from prism.morphology.validation import compute_cas_gini_m20  # noqa: E402

BANDS = ["F115W", "F150W", "F277W", "F444W"]
CAS_KEYS = ("concentration", "asymmetry", "smoothness", "clumpiness", "gini", "m20")

REPO = Path(__file__).resolve().parents[3]
REAL_DIR = REPO / "data" / "real_lenses"

SIM_DIRS = {
    "TNG50": Path("/Volumes/extHD/jwst_lens_outputs/tng50_particle_test100"),
    "TNG100": Path("/Volumes/extHD/jwst_lens_outputs/tng100_particle_test100"),
}


def grain_frac(image, mask):
    """Fraction of power spectrum (within mask's bounding box) at spatial
    frequencies above half-Nyquist. Computed on the masked, background
    subtracted cutout padded with zeros outside the mask."""
    if mask is None or not mask.any():
        sub = image
    else:
        sub = np.where(mask, image, 0.0)
    f = np.fft.fft2(sub)
    p = np.abs(f) ** 2
    p = np.fft.fftshift(p)
    ny, nx = p.shape
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - ny / 2.0, xx - nx / 2.0)
    rmax = min(ny, nx) / 2.0
    total = p.sum()
    if total <= 0:
        return np.nan
    high = p[r > 0.5 * rmax].sum()
    return float(high / total)


def band_metrics(image):
    image = np.nan_to_num(image.astype(np.float64))
    _, med, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    snr = (image - med) / max(std, 1e-12)
    segm = detect_sources(snr, 3.0, npixels=8)
    result = {"bg_rms": float(std), "peak": float(np.max(image))}
    if segm is None:
        for k in CAS_KEYS:
            result[k] = np.nan
        result["grain_frac"] = grain_frac(image - med, None)
        return result
    mask = segm.data > 0
    cas = compute_cas_gini_m20(image - med, mask)
    result.update(cas)
    result["grain_frac"] = grain_frac(image - med, mask)
    return result


def load_real():
    rows = []
    fits_files = sorted(REAL_DIR.glob("*_F115W.fits"))
    for f in fits_files:
        name = f.name.replace("_F115W.fits", "")
        for band in BANDS:
            bf = REAL_DIR / f"{name}_{band}.fits"
            if not bf.exists():
                continue
            data = fits.getdata(bf)
            m = band_metrics(data)
            m.update(source="REAL", name=name, band=band, z_bucket="n/a", sf_class="n/a")
            rows.append(m)
    return rows


def z_bucket(z):
    if z < 0.6:
        return "low_z(<0.6)"
    if z < 1.0:
        return "mid_z(0.6-1.0)"
    return "high_z(>=1.0)"


def sf_class(tng_lens):
    ssfr = tng_lens.get("ssfr_per_yr") if tng_lens else None
    if ssfr is None:
        return "unknown"
    return "star_forming" if ssfr > 1e-11 else "quiescent"


def load_sim(sim_name, sim_dir, limit=None):
    rows = []
    npz_dir = sim_dir / "unified_npz"
    files = sorted(npz_dir.glob("PRISM_lens_*.npz"))
    files = [f for f in files if "epoch" not in f.name]  # skip time-delay epoch frames
    if limit:
        files = files[:limit]
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["metadata"]))
        bands = meta.get("bands", BANDS)
        cube = d["image_final"]
        zb = z_bucket(float(meta.get("lens_redshift", np.nan)))
        sf = sf_class(meta.get("tng_lens"))
        for i, band in enumerate(bands):
            m = band_metrics(cube[i])
            m.update(source=sim_name, name=f.stem, band=band, z_bucket=zb, sf_class=sf)
            rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-sim", type=int, default=None,
                    help="limit number of sim lenses per dataset (for quick runs)")
    ap.add_argument("--out", type=str,
                    default="/Volumes/extHD/jwst_lens_outputs/qc_reports/per_band_v7")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real lens cutouts...")
    rows = load_real()
    print(f"  {len(rows)} (band x lens) rows")

    for sim_name, sim_dir in SIM_DIRS.items():
        print(f"Loading {sim_name} sim outputs from {sim_dir} ...")
        sim_rows = load_sim(sim_name, sim_dir, limit=args.limit_sim)
        print(f"  {len(sim_rows)} (band x lens) rows")
        rows.extend(sim_rows)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_band_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    # --- Summary: real vs sim, per band ---
    metric_cols = ["bg_rms", "peak", "grain_frac"] + list(CAS_KEYS)
    lines = []
    lines.append("# Per-band quality comparison: real vs. simulated\n")
    lines.append(f"Real lenses: {df[df.source=='REAL'].name.nunique()}")
    for sim_name in SIM_DIRS:
        sub = df[df.source == sim_name]
        lines.append(f"{sim_name} sim lenses: {sub.name.nunique()}")
    lines.append("")

    for band in BANDS:
        lines.append(f"## {band}\n")
        lines.append("| metric | REAL | TNG50 | TNG100 | TNG50-REAL | TNG100-REAL |")
        lines.append("|---|---|---|---|---|---|")
        bsub = df[df.band == band]
        for metric in metric_cols:
            real_vals = bsub[(bsub.source == "REAL")][metric].dropna()
            t50_vals = bsub[(bsub.source == "TNG50")][metric].dropna()
            t100_vals = bsub[(bsub.source == "TNG100")][metric].dropna()
            rmean = real_vals.mean() if len(real_vals) else np.nan
            t50mean = t50_vals.mean() if len(t50_vals) else np.nan
            t100mean = t100_vals.mean() if len(t100_vals) else np.nan
            lines.append(
                f"| {metric} | {rmean:.3f} | {t50mean:.3f} | {t100mean:.3f} "
                f"| {t50mean - rmean:+.3f} | {t100mean - rmean:+.3f} |"
            )
        lines.append("")

    # --- Sim breakdown by redshift bucket and SF class, F277W/F444W only ---
    for band in ["F277W", "F444W"]:
        lines.append(f"## {band} -- sim breakdown by lens redshift / TNG SF class\n")
        lines.append("| source | z_bucket | sf_class | n | grain_frac | smoothness | concentration |")
        lines.append("|---|---|---|---|---|---|---|")
        bsub = df[(df.band == band) & (df.source != "REAL")]
        for (src, zb, sf), g in bsub.groupby(["source", "z_bucket", "sf_class"]):
            lines.append(
                f"| {src} | {zb} | {sf} | {len(g)} | {g.grain_frac.mean():.3f} "
                f"| {g.smoothness.mean():.3f} | {g.concentration.mean():.3f} |"
            )
        lines.append("")

    report_path = out_dir / "per_band_quality_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
