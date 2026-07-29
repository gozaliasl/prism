#!/usr/bin/env python3
"""Build a disk-budget-capped, prioritized render queue from local TNG cutouts.

Rendering every locally downloaded cutout (~15,900 .h5 files) through
GalaxyGenius/SKIRT would need ~300GB of output (~19MB/galaxy final
mock_JWST products), against ~312GB free on /Volumes/extHD shared with an
active download. This script instead selects a disk-safe subset, prioritized:

  1. All TNG100 lens-candidate cutouts just downloaded (massive ellipticals /
     group+cluster environments) -- highest priority, smallest pool.
  2. A stratified sample of remaining cutouts (sim x snapshot x mass-bin x
     environment) for source/field diversity, filling the remaining budget.

Usage::

    python3 scripts/select_render_priority_list.py \\
        --data-dir /Volumes/extHD/galaxygenius_build/workspace/data \\
        --budget-gb 110 \\
        --mb-per-galaxy 19 \\
        --out /Volumes/extHD/tng_local_catalog/render_priority_queue.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MASS_BINS = [8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.6]


def parse_cutouts(data_dir: Path) -> pd.DataFrame:
    rows = []
    for f in data_dir.glob("TNG_*_snap_*_subhalo_*.h5"):
        m = re.match(r"TNG_(50|100)_snap_(\d+)_subhalo_(\d+)\.h5", f.name)
        if not m:
            continue
        sim_num, snap, subhalo = m.groups()
        rows.append({
            "sim_num": sim_num,
            "sim_prefix": f"TNG_{sim_num}",
            "sim": f"TNG{sim_num}-1",
            "snapshot": int(snap),
            "subhalo_id": int(subhalo),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="/Volumes/extHD/galaxygenius_build/workspace/data")
    p.add_argument("--catalog-100", default="/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet")
    p.add_argument("--catalog-50", default="/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet")
    p.add_argument("--lens-candidates", default="/Volumes/extHD/tng_local_catalog/tng100_lens_candidates.csv")
    p.add_argument("--budget-gb", type=float, default=110.0)
    p.add_argument("--mb-per-galaxy", type=float, default=19.0)
    p.add_argument("--out", default="/Volumes/extHD/tng_local_catalog/render_priority_queue.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    budget_n = int(args.budget_gb * 1024 / args.mb_per_galaxy)
    print(f"Disk budget: {args.budget_gb} GB / {args.mb_per_galaxy} MB per galaxy -> {budget_n} galaxies max")

    cutouts = parse_cutouts(Path(args.data_dir))
    print(f"Local cutouts found: {len(cutouts)}")

    cat100 = pd.read_parquet(args.catalog_100)
    cat100["sim"] = "TNG100-1"
    cat50 = pd.read_parquet(args.catalog_50)
    cat50["sim"] = "TNG50-1" if "sim" not in cat50.columns else cat50["sim"]
    cat = pd.concat([cat100, cat50], ignore_index=True)
    cat = cat[["sim", "snapshot", "subhalo_id", "stellar_mass_logmsun", "environment", "snapshot_redshift"]]

    merged = cutouts.merge(cat, on=["sim", "snapshot", "subhalo_id"], how="left")
    n_unmatched = merged["stellar_mass_logmsun"].isna().sum()
    if n_unmatched:
        print(f"  WARNING: {n_unmatched} local cutouts have no catalog match (will be deprioritized)")
    merged = merged.dropna(subset=["stellar_mass_logmsun"])

    # Priority 1: TNG100 lens candidates just downloaded.
    queue_frames = []
    try:
        lens_cand = pd.read_csv(args.lens_candidates)[["sim", "snapshot", "subhalo_id"]]
        lens_cand["sim"] = "TNG100-1"
        p1 = merged.merge(lens_cand, on=["sim", "snapshot", "subhalo_id"], how="inner")
        p1 = p1.drop_duplicates(subset=["sim", "snapshot", "subhalo_id"])
        print(f"Priority 1 (TNG100 lens candidates, downloaded+local): {len(p1)}")
        queue_frames.append(p1.assign(priority=1))
        used = set(zip(p1["sim"], p1["snapshot"], p1["subhalo_id"]))
    except FileNotFoundError:
        print("  lens-candidates CSV not found, skipping priority 1")
        used = set()

    remaining_budget = max(0, budget_n - sum(len(f) for f in queue_frames))
    pool = merged[~merged.apply(lambda r: (r["sim"], r["snapshot"], r["subhalo_id"]) in used, axis=1)].copy()
    pool["mass_bin"] = pd.cut(pool["stellar_mass_logmsun"], bins=MASS_BINS, right=False)

    cells = pool.groupby(["sim", "snapshot", "mass_bin", "environment"], observed=True)
    n_cells = max(len(cells), 1)
    per_cell = max(1, -(-remaining_budget // n_cells))
    picks = []
    remaining = remaining_budget
    for _, grp in cells:
        if remaining <= 0:
            break
        k = min(per_cell, len(grp), remaining)
        picks.append(grp.sample(n=k, random_state=args.seed))
        remaining -= k
    p2 = pd.concat(picks) if picks else pool.iloc[0:0]
    print(f"Priority 2 (stratified source/field sample): {len(p2)} "
          f"(target {remaining_budget} across {n_cells} sim x snapshot x mass-bin x env cells)")
    queue_frames.append(p2.assign(priority=2))

    queue = pd.concat(queue_frames).drop_duplicates(subset=["sim", "snapshot", "subhalo_id"])
    queue = queue.sort_values(["priority", "sim", "snapshot", "subhalo_id"])
    queue.to_csv(args.out, index=False)

    total_gb = len(queue) * args.mb_per_galaxy / 1024
    print(f"\nWrote {len(queue)} galaxies to render -> {args.out}")
    print(f"Estimated final output size: {total_gb:.1f} GB")
    print(queue["priority"].value_counts())
    print(queue.groupby("priority")["sim"].value_counts())


if __name__ == "__main__":
    main()
