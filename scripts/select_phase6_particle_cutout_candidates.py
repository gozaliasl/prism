#!/usr/bin/env python3
"""Select new TNG100-1 (snapshot, subhalo_id) candidates for local
particle-cutout download (Phase 6: grow the generative-morphology
training set, biased toward low-z star-forming disks so the conditional
VAE has a chance to learn spiral-arm / multi-component structure).

Reads the local TNG catalog (``build_tng_local_catalog.py`` output),
excludes subhalos that already have a local ``.h5`` cutout
(``src.tng_galaxy_selector.local_particle_path``), and selects:

1. Priority pool: snapshot redshift <= ``--z-max`` (default 1.0), logM in
   [``--logm-min``, ``--logm-max``] (default 9.5-11.5), and star-forming
   (``ssfr_per_yr > 1e-11``) -- these are the disk-galaxy candidates most
   likely to show spiral arms / clumpy multi-component structure once
   rendered.
2. Fill pool: same redshift/mass cuts but any sSFR (adds some
   passive/quiescent low-z galaxies for condition-class diversity), used
   to top up to ``--n-total`` if the priority pool is smaller.

Both pools are stratified across snapshots and mass bins. Result is written
as a CSV with one row per selected subhalo, ready to feed
``batch_fetch_galaxygenius_stamps.py --candidates <output>``.

Usage::

    python3 scripts/select_phase6_particle_cutout_candidates.py \\
        --catalog /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet \\
        --output /Volumes/extHD/tng_local_catalog/phase6_particle_cutout_candidates.csv \\
        --n-total 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tng_galaxy_selector import load_local_catalog, local_particle_path  # noqa: E402

MASS_BIN_EDGES = [9.5, 10.0, 10.5, 11.0, 11.5, 12.5]
MASS_BIN_LABELS = ["9.5-10.0", "10.0-10.5", "10.5-11.0", "11.0-11.5", "11.5-12.5"]
QUENCHED_SSFR_THRESHOLD = 1e-11


def _stratified_sample(cat: pd.DataFrame, n_total: int, rng: np.random.Generator) -> pd.DataFrame:
    cat = cat.copy()
    cat["mass_bin"] = pd.cut(cat["stellar_mass_logmsun"], bins=MASS_BIN_EDGES, labels=MASS_BIN_LABELS)

    snapshots = sorted(cat["snapshot"].unique())
    if not snapshots:
        return cat.iloc[0:0]
    n_per_snapshot = max(1, n_total // len(snapshots))

    selected_frames = []
    for snap in snapshots:
        snap_cat = cat[cat["snapshot"] == snap]
        n_bins = snap_cat["mass_bin"].nunique()
        n_per_bin = max(1, n_per_snapshot // max(n_bins, 1))
        for _, cell in snap_cat.groupby("mass_bin", observed=True):
            n = min(len(cell), n_per_bin)
            if n == 0:
                continue
            idx = rng.choice(cell.index, size=n, replace=False)
            selected_frames.append(cell.loc[idx])

    if not selected_frames:
        return cat.iloc[0:0]
    return pd.concat(selected_frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=str,
                   default="/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet")
    p.add_argument("--output", type=str,
                   default="/Volumes/extHD/tng_local_catalog/phase6_particle_cutout_candidates.csv")
    p.add_argument("--n-total", type=int, default=1500,
                   help="Target total number of new candidates")
    p.add_argument("--z-max", type=float, default=1.0,
                   help="Max snapshot redshift for the low-z disk-galaxy bias")
    p.add_argument("--logm-min", type=float, default=9.5)
    p.add_argument("--logm-max", type=float, default=12.5)
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    cat = load_local_catalog(args.catalog)
    rng = np.random.default_rng(args.seed)

    cat = cat[(cat["snapshot_redshift"] <= args.z_max) &
              (cat["stellar_mass_logmsun"] >= args.logm_min) &
              (cat["stellar_mass_logmsun"] <= args.logm_max)].copy()

    has_cutout = cat.apply(
        lambda r: local_particle_path(int(r["snapshot"]), int(r["subhalo_id"])) is not None, axis=1)
    cat = cat[~has_cutout].copy()
    print(f"{len(cat)} candidates without a local .h5 cutout "
          f"(z<={args.z_max}, logM {args.logm_min}-{args.logm_max})")

    star_forming = cat[cat["ssfr_per_yr"] > QUENCHED_SSFR_THRESHOLD]
    print(f"  of which star-forming (ssfr > {QUENCHED_SSFR_THRESHOLD:g}): {len(star_forming)}")

    priority = _stratified_sample(star_forming, args.n_total, rng)
    print(f"Priority (star-forming, low-z disk) sample: {len(priority)}")

    n_remaining = args.n_total - len(priority)
    if n_remaining > 0:
        remaining_pool = cat.drop(index=priority.index, errors="ignore")
        fill = _stratified_sample(remaining_pool, n_remaining, rng)
        print(f"Fill (any sSFR, low-z) sample: {len(fill)}")
        selected = pd.concat([priority, fill], ignore_index=True)
    else:
        selected = priority

    selected = selected.drop_duplicates(subset=["snapshot", "subhalo_id"])
    selected = selected.sort_values(["snapshot", "mass_bin"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    print(f"\nSelected {len(selected)} new candidates")
    print(f"Snapshots covered: {sorted(selected['snapshot'].unique())}")
    print(f"Redshift range: {selected['snapshot_redshift'].min():.2f} - {selected['snapshot_redshift'].max():.2f}")
    print(f"Star-forming fraction: {(selected['ssfr_per_yr'] > QUENCHED_SSFR_THRESHOLD).mean():.2f}")
    print(f"Mass-bin counts:\n{selected['mass_bin'].value_counts().sort_index()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
