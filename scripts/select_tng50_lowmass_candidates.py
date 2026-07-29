#!/usr/bin/env python3
"""Select TNG50-1 (snapshot, subhalo_id) candidates for local particle-cutout
download, spanning logM* ~ 8-12.5. Primarily targets low-mass (logM*<10.5)
field/source/companion galaxies where TNG100-1 cutouts are too sparse (~250
star particles at logM*=9.5) for smooth particle-rendered morphology, but
also includes the (rare, due to TNG50's smaller box) high-mass end so lens
galaxies can benefit from TNG50's resolution too.

TNG50-1 has ~16x better star-particle mass resolution than TNG100-1 (at the
cost of a ~9.7x smaller box volume), so the same stellar mass yields ~16x
more particles -- e.g. ~4000 particles at logM*=9.5 instead of ~250.

Reads the local TNG50-1 catalog (``build_tng_local_catalog.py --sim
TNG50-1`` output), excludes subhalos that already have a local ``.h5``
cutout (``src.tng_galaxy_selector.local_particle_path(..., sim="TNG50-1")``),
and selects a stratified sample across snapshots and mass bins. Result is
written as a CSV with a ``sim=TNG50-1`` column, ready to feed
``batch_fetch_galaxygenius_stamps.py --candidates <output>``.

Usage::

    python3 scripts/select_tng50_lowmass_candidates.py \\
        --catalog /Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet \\
        --output /Volumes/extHD/tng_local_catalog/tng50_lowmass_candidates.csv \\
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

MASS_BIN_EDGES = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.5]
MASS_BIN_LABELS = ["8.0-8.5", "8.5-9.0", "9.0-9.5", "9.5-10.0", "10.0-10.5",
                    "10.5-11.0", "11.0-11.5", "11.5-12.5"]


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
                   default="/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet")
    p.add_argument("--output", type=str,
                   default="/Volumes/extHD/tng_local_catalog/tng50_lowmass_candidates.csv")
    p.add_argument("--n-total", type=int, default=1500,
                   help="Target total number of new candidates")
    p.add_argument("--logm-min", type=float, default=8.0)
    p.add_argument("--logm-max", type=float, default=12.5)
    p.add_argument("--seed", type=int, default=44)
    args = p.parse_args()

    cat = load_local_catalog(args.catalog)
    if cat is None or len(cat) == 0:
        print(f"No catalog found at {args.catalog} (or it is empty) -- nothing to select")
        return
    rng = np.random.default_rng(args.seed)

    cat = cat[(cat["stellar_mass_logmsun"] >= args.logm_min) &
              (cat["stellar_mass_logmsun"] <= args.logm_max)].copy()
    print(f"{len(cat)} TNG50-1 subhalos in logM {args.logm_min}-{args.logm_max}")

    has_cutout = cat.apply(
        lambda r: local_particle_path(int(r["snapshot"]), int(r["subhalo_id"]), sim="TNG50-1") is not None, axis=1)
    cat = cat[~has_cutout].copy()
    print(f"{len(cat)} candidates without a local TNG50-1 .h5 cutout")

    selected = _stratified_sample(cat, args.n_total, rng)
    selected = selected.drop_duplicates(subset=["snapshot", "subhalo_id"])
    selected = selected.sort_values(["snapshot", "mass_bin"]).reset_index(drop=True)
    selected["sim"] = "TNG50-1"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    print(f"\nSelected {len(selected)} new candidates")
    if len(selected):
        print(f"Snapshots covered: {sorted(selected['snapshot'].unique())}")
        print(f"Redshift range: {selected['snapshot_redshift'].min():.2f} - {selected['snapshot_redshift'].max():.2f}")
        print(f"Mass-bin counts:\n{selected['mass_bin'].value_counts().sort_index()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
