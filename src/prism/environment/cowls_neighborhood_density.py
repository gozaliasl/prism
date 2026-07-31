"""Empirical field-galaxy neighborhood statistics around real COWLS lenses.

Measures the ACTUAL local population (counts, magnitude, redshift, stellar
mass distributions) within a fixed projected radius of each real COWLS
strong-lens position, using the full COSMOS-Web photometric catalog
(data/galaxy_catalog.fits, 784016 galaxies with real RA/DEC/z/mass/
per-band magnitudes). This replaces the earlier "COSMOS-Web average
surface density at a single mag cut" approach with the actual spatial
clustering/environment real lenses sit in -- lens sightlines are not a
random draw from the field (massive galaxies cluster), so this captures a
real overdensity/underdensity signal that a flat mag-cut density does not.

COWLS lens positions are recovered by joining data/cowls_processed_catalog.csv
(no RA/DEC) back to data/cosmos_web_lens_structural_properties.csv (has
RA/DEC) on the shared measured columns (LP_zfinal, LP_mass_med_PDF,
mag_f277w, nsersic_f277w) -- confirmed to recover all 356/356 COWLS rows.

Usage:
    from cowls_neighborhood_density import measure_cowls_neighborhoods, summarize_neighborhoods
    neigh = measure_cowls_neighborhoods(radius_arcmin=0.5)
    summary = summarize_neighborhoods(neigh, mag_limit=24.5)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_GALAXY_CATALOG_FITS = _DATA_DIR / "galaxy_catalog.fits"
_COWLS_CSV = _DATA_DIR / "cowls_processed_catalog.csv"
_STRUCT_CSV = _DATA_DIR / "cosmos_web_lens_structural_properties.csv"

_field_cache: Optional[pd.DataFrame] = None
_cowls_positions_cache: Optional[pd.DataFrame] = None


def load_field_catalog() -> pd.DataFrame:
    """Full COSMOS-Web photometric catalog (784016 galaxies) as a DataFrame
    with native-byte-order float columns (FITS big-endian arrays need an
    explicit cast for pandas/numpy reductions)."""
    global _field_cache
    if _field_cache is not None:
        return _field_cache
    from astropy.io import fits

    with fits.open(_GALAXY_CATALOG_FITS) as hdul:
        data = hdul[1].data
        cols = {
            "RA": data["RA_DETEC"], "DEC": data["DEC_DETEC"],
            "z": data["LP_zfinal"], "mass": data["LP_mass_med_PDF"],
            "mag_f115w": data["mag_f115w"], "mag_f150w": data["mag_f150w"],
            "mag_f277w": data["mag_f277w"], "mag_f444w": data["mag_f444w"],
        }
        df = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in cols.items()})
    _field_cache = df
    return df


def load_cowls_lens_positions() -> pd.DataFrame:
    """356 real COWLS lens positions, recovered by joining
    cowls_processed_catalog.csv (properties, no RA/DEC) back onto
    cosmos_web_lens_structural_properties.csv (has RA/DEC) via the shared
    measured columns. Confirmed 356/356 rows matched."""
    global _cowls_positions_cache
    if _cowls_positions_cache is not None:
        return _cowls_positions_cache
    cowls = pd.read_csv(_COWLS_CSV)
    struct = pd.read_csv(_STRUCT_CSV)
    merged = cowls.merge(
        struct[["RA", "DEC", "LP_zfinal", "LP_mass_med_PDF", "mag_f277w", "nsersic_f277w"]],
        on=["LP_zfinal", "LP_mass_med_PDF", "mag_f277w", "nsersic_f277w"],
        how="left",
    )
    n_matched = merged["RA"].notna().sum()
    if n_matched < len(merged):
        print(f"[WARNING] load_cowls_lens_positions: only {n_matched}/{len(merged)} "
              "COWLS rows matched back to RA/DEC")
    _cowls_positions_cache = merged.dropna(subset=["RA", "DEC"]).reset_index(drop=True)
    return _cowls_positions_cache


def measure_cowls_neighborhoods(radius_arcmin: float = 0.5) -> pd.DataFrame:
    """For each real COWLS lens, find all COSMOS-Web field galaxies within
    `radius_arcmin` (projected, flat-sky approx -- valid at this small
    scale) and return one row per (lens, neighbor) pair with the neighbor's
    z/mass/magnitudes and its separation in arcsec from the lens.

    Excludes the lens galaxy itself (matched by near-zero separation).
    """
    lenses = load_cowls_lens_positions()
    field = load_field_catalog()

    field_ra = field["RA"].to_numpy()
    field_dec = field["DEC"].to_numpy()
    radius_deg = radius_arcmin / 60.0

    rows = []
    for lens_idx, lens in lenses.iterrows():
        cosdec = np.cos(np.radians(lens["DEC"]))
        dra = (field_ra - lens["RA"]) * cosdec
        ddec = field_dec - lens["DEC"]
        sep_deg = np.sqrt(dra ** 2 + ddec ** 2)
        sep_arcsec = sep_deg * 3600.0
        mask = (sep_deg < radius_deg) & (sep_arcsec > 0.3)  # exclude the lens itself
        if not mask.any():
            continue
        sub = field.loc[mask].copy()
        sub["lens_idx"] = lens_idx
        sub["lens_RA"] = lens["RA"]
        sub["lens_DEC"] = lens["DEC"]
        sub["lens_z"] = lens["LP_zfinal"]
        sub["lens_mass"] = lens["LP_mass_med_PDF"]
        sub["sep_arcsec"] = sep_arcsec[mask]
        rows.append(sub)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_neighborhoods(neigh: pd.DataFrame, mag_limit: float = 27.0,
                             mag_band: str = "mag_f150w",
                             radius_arcmin: float = 0.5) -> dict:
    """Summary statistics of the real COWLS-lens neighborhood population:
    counts per lens (surface density), and marginal z/mass/magnitude
    distributions of the neighbors, cut at `mag_limit` in `mag_band`.

    IMPORTANT: `mag_limit` here is an ANALYSIS depth, not a rendering
    depth -- COSMOS-Web completeness extends to ~mag 28, so use something
    close to that (26-28) to characterize the true population.

    CORRECTED 2026-08-01 (adversarial audit finding C-1): the originally
    claimed 1.68-1.72x lens-sightline overdensity was a footprint-area
    measurement artifact in compare_to_field_average() (RA/Dec bounding
    box vs the true non-rectangular mosaic footprint, ~1.63x overstated
    area), not a real physical signal. After fixing the footprint to a
    grid-cell method and filtering sentinels consistently, the real
    overdensity is 1.03-1.07x (mean ~1.05x) across mag<24.5 through
    mag<28 -- still stable across depth, but a much smaller, more
    physically modest signal than originally reported.
    """
    area_arcmin2 = np.pi * radius_arcmin ** 2
    n_lenses = neigh["lens_idx"].nunique()

    # Reject catalog sentinels (-99/-999) BEFORE the magnitude cut; a
    # sentinel value trivially satisfies "< mag_limit" (it's very negative)
    # and silently inflates counts/densities if not excluded here too,
    # not just in the z/mass columns below. (Bug found by adversarial
    # audit 2026-08-01: this asymmetry -- z/mass sentinel-filtered but
    # magnitude not -- let null-photometry rows contaminate every count.)
    cut = neigh[(neigh[mag_band] > 15.0) & (neigh[mag_band] < mag_limit)]
    # z=-99 / mass=-99.9 are catalog "no measurement" flags, not real values
    clean = cut[(cut["z"] > -90) & (cut["mass"] > -90)]
    counts_per_lens = cut.groupby("lens_idx").size()
    # lenses with zero neighbors under the cut don't appear in counts_per_lens; reindex
    counts_per_lens = counts_per_lens.reindex(neigh["lens_idx"].unique(), fill_value=0)

    return {
        "n_lenses": n_lenses,
        "radius_arcmin": radius_arcmin,
        "mag_limit": mag_limit,
        "mag_band": mag_band,
        "mean_count_per_lens": float(counts_per_lens.mean()),
        "median_count_per_lens": float(counts_per_lens.median()),
        "std_count_per_lens": float(counts_per_lens.std()),
        "mean_density_per_arcmin2": float(counts_per_lens.mean() / area_arcmin2),
        "z_distribution": clean["z"].describe().to_dict(),
        "mass_distribution": clean["mass"].describe().to_dict(),
        "mag_distribution": cut[mag_band].describe().to_dict(),
        "sep_arcsec_distribution": cut["sep_arcsec"].describe().to_dict(),
        "frac_missing_z_or_mass": float(1.0 - len(clean) / max(len(cut), 1)),
    }


def compare_to_field_average(mag_limit: float = 24.5, mag_band: str = "mag_f150w") -> dict:
    """Compare the COWLS-lens-neighborhood density to the naive full-field
    average density at the same mag cut, to quantify real lens-sightline
    overdensity/underdensity (massive lens galaxies are not randomly
    positioned -- they trace large-scale structure).

    Uses a grid-cell footprint (15" cells, counting only occupied cells),
    NOT an RA/Dec bounding box. Bug found by adversarial audit 2026-08-01:
    the COSMOS-Web mosaic is not a filled rectangle, so a bounding box
    overstates the area by ~1.63x and understates the true density by the
    same factor -- this repo's own cosmos_field_density_per_arcmin2()
    already documents and fixes this exact problem for its own density
    measurement; compare_to_field_average previously did not, which meant
    the "1.70x lens-sightline overdensity" reported earlier was actually
    just this area bug (1.63x) plus a small (~6%) real signal -- confirmed
    by direct recomputation: bbox-footprint density 37.16/arcmin^2 (the
    exact "37" the old docstring quoted) vs grid-footprint 60.04/arcmin^2,
    against a COWLS-neighborhood density of 63.80/arcmin^2 -> real
    overdensity ~1.06x, not 1.70x.
    """
    field = load_field_catalog()
    ok = (field[mag_band] > 15.0) & np.isfinite(field["RA"]) & np.isfinite(field["DEC"])
    f = field.loc[ok]

    cell_deg = 15.0 / 3600.0  # 15" grid cells
    ix = ((f["RA"] - f["RA"].min()) / cell_deg).astype(int)
    iy = ((f["DEC"] - f["DEC"].min()) / cell_deg).astype(int)
    n_occupied_cells = len(set(zip(ix, iy)))
    footprint_arcmin2 = n_occupied_cells * (15.0 / 60.0) ** 2

    n_bright = int((f[mag_band] < mag_limit).sum())
    field_avg_density = n_bright / footprint_arcmin2
    return {
        "field_average_density_per_arcmin2": field_avg_density,
        "footprint_arcmin2": footprint_arcmin2,
        "n_galaxies_brighter_than_cut": n_bright,
        "footprint_method": "grid_15arcsec_occupied_cells",
    }
