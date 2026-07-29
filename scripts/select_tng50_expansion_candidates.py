#!/usr/bin/env python3
"""Select new TNG50-1 candidates for download to expand local h5 coverage to ~10,000 total.

Currently have ~3,897 TNG50-1 h5 files. Selects ~6,000 more to reach 10,000+,
stratified across logM bins and redshifts (snapshots).

Usage::
    python3 scripts/select_tng50_expansion_candidates.py \
        --catalog /Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet \
        --output /Volumes/extHD/tng_local_catalog/tng50_expansion_candidates.csv \
        --n-total 6000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tng_galaxy_selector import load_local_catalog, local_particle_path

MASS_BIN_EDGES = [8.0, 9.0, 10.0, 10.5, 11.0, 12.0]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet")
    p.add_argument("--output", default="/Volumes/extHD/tng_local_catalog/tng50_expansion_candidates.csv")
    p.add_argument("--n-total", type=int, default=6000)
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    cat = load_local_catalog(args.catalog)
    if cat is None:
        print(f"ERROR: Could not load catalog: {args.catalog}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    print(f"Catalog: {len(cat)} rows")
    print(f"Checking existing h5 files...")

    # Find rows without local particle files
    missing = []
    for _, row in cat.iterrows():
        sim = str(row.get("sim", "TNG50-1"))
        if local_particle_path(int(row["snapshot"]), int(row["subhalo_id"]), sim=sim) is None:
            missing.append(row)

    missing_df = pd.DataFrame(missing)
    print(f"Missing h5 files: {len(missing_df)} / {len(cat)}")

    if len(missing_df) == 0:
        print("All subhalos already have local particle files!")
        sys.exit(0)

    # Stratify by mass bin and snapshot
    missing_df = missing_df[
        (missing_df["stellar_mass_logmsun"] >= 8.0) &
        (missing_df["stellar_mass_logmsun"] <= 12.0)
    ].copy()

    missing_df["mass_bin"] = pd.cut(
        missing_df["stellar_mass_logmsun"], bins=MASS_BIN_EDGES,
        labels=[f"{MASS_BIN_EDGES[i]}-{MASS_BIN_EDGES[i+1]}" for i in range(len(MASS_BIN_EDGES)-1)]
    )

    n_bins = missing_df.groupby(["snapshot", "mass_bin"], observed=True).size()
    print(f"\nAvailable by (snapshot, mass_bin):\n{n_bins[n_bins > 0].head(20)}")

    n_per_bin = max(1, args.n_total // len(n_bins[n_bins > 0]))
    selected = []
    for (snap, mb), grp in missing_df.groupby(["snapshot", "mass_bin"], observed=True):
        n = min(n_per_bin, len(grp))
        if n > 0:
            idx = rng.choice(len(grp), size=n, replace=False)
            selected.append(grp.iloc[idx])

    result = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if len(result) > args.n_total:
        idx = rng.choice(len(result), size=args.n_total, replace=False)
        result = result.iloc[idx].copy()

    out_cols = ["sim", "snapshot", "subhalo_id", "stellar_mass_logmsun", "snapshot_redshift", "ssfr_per_yr"]
    out = result[[c for c in out_cols if c in result.columns]].copy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSelected {len(out)} candidates → {args.output}")
    print(f"Mass distribution:\n{out['stellar_mass_logmsun'].describe()}")

if __name__ == "__main__":
    main()
