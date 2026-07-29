#!/usr/bin/env python3
"""Select ~1000 new TNG100-1 (snapshot, subhalo_id) candidates for local
particle-cutout download (Phase 5 of the galaxy_morphology roadmap).

Reads the local TNG catalog (``build_tng_local_catalog.py`` output),
excludes subhalos that already have a local ``.h5`` cutout
(``src.tng_galaxy_selector.local_particle_path``), restricts to logM
8.0-11.5 (the lens/source/field/companion mass range used by ``tng_mode``),
and samples up to ``--n-per-snapshot`` new candidates per snapshot (21
snapshots total), stratified by mass bin within each snapshot.

The result is written as a CSV with one row per selected subhalo, ready to
feed ``batch_fetch_galaxygenius_stamps.py --candidates <output>``.

Usage::

    python3 scripts/select_phase5_particle_cutout_candidates.py \\
        --catalog /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet \\
        --output /Volumes/extHD/tng_local_catalog/phase5_particle_cutout_candidates.csv \\
        --n-total 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tng_galaxy_selector import load_local_catalog, local_particle_path  # noqa: E402

MASS_BIN_EDGES = [8.0, 9.0, 10.0, 10.5, 11.0, 11.5]
MASS_BIN_LABELS = ["8.0-9.0", "9.0-10.0", "10.0-10.5", "10.5-11.0", "11.0-11.5"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=str,
                    default="/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet")
    p.add_argument("--output", type=str,
                    default="/Volumes/extHD/tng_local_catalog/phase5_particle_cutout_candidates.csv")
    p.add_argument("--n-total", type=int, default=1000,
                    help="Target total number of new candidates")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-csv", type=str, action="append", default=[],
                    help="Path to a previously-selected candidates CSV "
                         "((snapshot, subhalo_id) pairs to exclude in "
                         "addition to subhalos with existing .h5 cutouts). "
                         "May be passed multiple times.")
    args = p.parse_args()

    cat = load_local_catalog(args.catalog)
    rng = np.random.default_rng(args.seed)

    cat = cat[(cat["stellar_mass_logmsun"] >= MASS_BIN_EDGES[0]) &
              (cat["stellar_mass_logmsun"] <= MASS_BIN_EDGES[-1])].copy()

    has_cutout = cat.apply(
        lambda r: local_particle_path(int(r["snapshot"]), int(r["subhalo_id"])) is not None, axis=1)
    cat = cat[~has_cutout].copy()
    print(f"{len(cat)} candidates without a local .h5 cutout (logM 8.0-11.5)")

    for exclude_path in args.exclude_csv:
        prev = pd.read_csv(exclude_path)
        prev_pairs = set(zip(prev["snapshot"].astype(int), prev["subhalo_id"].astype(int)))
        before = len(cat)
        cat = cat[~cat.apply(lambda r: (int(r["snapshot"]), int(r["subhalo_id"])) in prev_pairs, axis=1)].copy()
        print(f"Excluded {before - len(cat)} candidates already in {exclude_path}")

    cat["mass_bin"] = pd.cut(cat["stellar_mass_logmsun"], bins=MASS_BIN_EDGES, labels=MASS_BIN_LABELS)

    snapshots = sorted(cat["snapshot"].unique())
    n_per_snapshot = max(1, args.n_total // len(snapshots))

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

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(["snapshot", "mass_bin"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    print(f"Selected {len(selected)} new candidates")
    print(f"Snapshots covered: {sorted(selected['snapshot'].unique())}")
    print(f"Redshift range: {selected['snapshot_redshift'].min():.2f} - {selected['snapshot_redshift'].max():.2f}")
    print(f"Mass-bin counts:\n{selected['mass_bin'].value_counts().sort_index()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
