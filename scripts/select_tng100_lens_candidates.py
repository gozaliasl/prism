#!/usr/bin/env python3
"""Select TNG100-1 lens-candidate subhalos to finalize TNG particle-cutout coverage.

Rationale (see project memory / conversation): TNG50's small box (35 Mpc/h)
under-samples massive ellipticals and group/cluster environments, which is
exactly what the lens population needs (logM* > 10.5, group/rich_group
environment, z spanning the lens redshift range). TNG50 cutouts already
dominate the local archive (11,501 vs. 4,382 for TNG100), so this script
targets TNG100-1 specifically, stratified across snapshot x mass-bin x
environment, and excludes subhalos that already have a local .h5 cutout.

Usage::

    python3 scripts/select_tng100_lens_candidates.py \\
        --catalog /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet \\
        --existing-dir /Volumes/extHD/galaxygenius_build/workspace/data \\
        --out /Volumes/extHD/tng_local_catalog/tng100_lens_candidates.csv \\
        --n-target 4000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MASS_BINS = [10.5, 10.8, 11.1, 11.4, 11.7, 12.0, 12.6]
ENV_PRIORITY = ["rich_group", "group", "pair", "isolated"]
ENV_WEIGHT = {"rich_group": 0.40, "group": 0.35, "pair": 0.15, "isolated": 0.10}


def existing_pairs(existing_dir: Path) -> set[tuple[int, int]]:
    pairs = set()
    for f in existing_dir.glob("TNG_100_snap_*_subhalo_*.h5"):
        m = re.match(r"TNG_100_snap_(\d+)_subhalo_(\d+)\.h5", f.name)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet")
    p.add_argument("--existing-dir", default="/Volumes/extHD/galaxygenius_build/workspace/data")
    p.add_argument("--out", default="/Volumes/extHD/tng_local_catalog/tng100_lens_candidates.csv")
    p.add_argument("--n-target", type=int, default=4000,
                    help="Total number of new candidates to select")
    p.add_argument("--mass-min", type=float, default=10.5,
                    help="Minimum log10(M*/Msun) — lens-like massive galaxies")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.catalog)
    df = df[df["stellar_mass_logmsun"] >= args.mass_min].copy()
    print(f"Catalog rows with logM*>={args.mass_min}: {len(df)}")

    existing = existing_pairs(Path(args.existing_dir))
    print(f"Existing local TNG100 cutouts: {len(existing)}")

    df["_have"] = list(zip(df["snapshot"].astype(int), df["subhalo_id"].astype(int)))
    df = df[~df["_have"].isin(existing)].drop(columns="_have")
    print(f"Remaining candidates after excluding existing: {len(df)}")

    df["mass_bin"] = pd.cut(df["stellar_mass_logmsun"], bins=MASS_BINS, right=False)
    df = df[df["environment"].isin(ENV_PRIORITY)]

    # Stratified sample: allocate the target count across environment groups
    # by priority weight, then spread evenly across snapshot x mass_bin cells
    # within each environment so coverage isn't dominated by one redshift.
    # Any shortfall (an environment with fewer candidates than its quota) is
    # rolled forward to the next environment in priority order so the overall
    # n_target is still reached from the larger, lower-priority pools.
    selected_frames = []
    used_ids = set()
    carry = 0
    for env, weight in ENV_WEIGHT.items():
        env_df = df[(df["environment"] == env) & (~df.index.isin(used_ids))]
        if env_df.empty:
            continue
        n_env = int(round(args.n_target * weight)) + carry
        cells = env_df.groupby(["snapshot", "mass_bin"], observed=True)
        n_cells = max(len(cells), 1)
        per_cell = max(1, -(-n_env // n_cells))  # ceil, so small cells don't starve the quota
        picks = []
        remaining = n_env
        for _, grp in cells:
            if remaining <= 0:
                break
            k = min(per_cell, len(grp), remaining)
            picks.append(grp.sample(n=k, random_state=args.seed))
            remaining -= k
        env_sel = pd.concat(picks) if picks else env_df.iloc[0:0]
        if len(env_sel) > n_env:
            env_sel = env_sel.sample(n=n_env, random_state=args.seed)
        selected_frames.append(env_sel)
        used_ids.update(env_sel.index)
        carry = max(0, n_env - len(env_sel))
        print(f"  env={env:10s} weight={weight:.2f} target={n_env:5d} selected={len(env_sel):5d} "
              f"carry_to_next={carry:5d}  (pool={len(env_df)} across {n_cells} snapshot x mass-bin cells)")

    selected = pd.concat(selected_frames).drop_duplicates(subset=["snapshot", "subhalo_id"])
    if carry > 0:
        print(f"  NOTE: {carry} candidates still short of n_target (mass-bin>=10.5 pool exhausted)")
    if len(selected) > args.n_target:
        selected = selected.sample(n=args.n_target, random_state=args.seed)

    selected = selected.copy()
    selected["sim"] = "TNG100-1"
    out_cols = ["sim", "snapshot", "subhalo_id", "halo_id", "stellar_mass_logmsun",
                "environment", "snapshot_redshift", "halfmassrad_stars_kpc"]
    selected[out_cols].sort_values(["snapshot", "subhalo_id"]).to_csv(args.out, index=False)
    print(f"\nWrote {len(selected)} candidates -> {args.out}")
    print(selected["environment"].value_counts())
    print(selected["stellar_mass_logmsun"].describe())


if __name__ == "__main__":
    main()
