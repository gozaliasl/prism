#!/usr/bin/env python3
"""Select a diverse, stratified sample of TNG100-1 subhalos for GalaxyGenius stamps.

Reads the local TNG catalog built by ``build_tng_local_catalog.py`` and
selects candidates spanning redshift/snapshot, stellar-mass, environment
(isolated/pair/group/rich_group), and sSFR-derived galaxy type
(passive/star_forming/dusty_starburst -- same thresholds as
``tng_sed_galaxy_type`` in ``jwst_lens_simulator.py``).

Goal: a visually diverse set of lens/source/field/environmental galaxy
templates (different morphologies, colors, sizes) across the full
lens (z~0.2-1.5), source, and field redshift ranges used by the
strong-lens simulator -- for more realistic ML training data.

For each (snapshot, mass_bin, environment, sed_type) cell with at least one
subhalo, up to ``--n-per-cell`` are randomly sampled. The result is written
as a CSV with one row per selected subhalo, ready to feed
``fetch_galaxygenius_subhalo_particles.py --snap <snapshot> --subhalo <subhalo_id>``.

Usage::

    python3 scripts/select_galaxygenius_stamp_candidates.py \\
        --catalog /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet \\
        --output /Volumes/extHD/tng_local_catalog/galaxygenius_stamp_candidates.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tng_galaxy_selector import load_local_catalog  # noqa: E402

# Same thresholds as TNG_QUENCHED_SSFR_THRESHOLD / TNG_STARBURST_SSFR_THRESHOLD
# in src/jwst_lens_simulator.py.
QUENCHED_SSFR_THRESHOLD = 1e-11
STARBURST_SSFR_THRESHOLD = 1e-9

MASS_BIN_EDGES = [8.0, 9.0, 10.0, 10.5, 11.0, 11.5, 12.5]
MASS_BIN_LABELS = ["8.0-9.0", "9.0-10.0", "10.0-10.5", "10.5-11.0", "11.0-11.5", "11.5-12.5"]

ENVIRONMENTS = ["isolated", "pair", "group", "rich_group"]


def sed_type(ssfr: float) -> str:
    if ssfr < QUENCHED_SSFR_THRESHOLD:
        return "passive"
    if ssfr > STARBURST_SSFR_THRESHOLD:
        return "dusty_starburst"
    return "star_forming"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=str,
                    default="/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet")
    p.add_argument("--output", type=str,
                    default="/Volumes/extHD/tng_local_catalog/galaxygenius_stamp_candidates.csv")
    p.add_argument("--n-per-cell", type=int, default=2,
                    help="Max subhalos to sample per (snapshot, mass_bin, environment, sed_type) cell")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cat = load_local_catalog(args.catalog)
    rng = np.random.default_rng(args.seed)

    cat = cat.copy()
    cat["mass_bin"] = pd.cut(cat["stellar_mass_logmsun"], bins=MASS_BIN_EDGES, labels=MASS_BIN_LABELS)
    cat["sed_type"] = cat["ssfr_per_yr"].apply(sed_type)

    selected_frames = []
    group_cols = ["snapshot", "mass_bin", "environment", "sed_type"]
    for _, cell in cat.groupby(group_cols, observed=True):
        n = min(len(cell), args.n_per_cell)
        idx = rng.choice(cell.index, size=n, replace=False)
        selected_frames.append(cell.loc[idx])

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(["snapshot", "mass_bin", "environment", "sed_type"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    print(f"Selected {len(selected)} candidates from {len(cat)} catalog rows")
    print(f"Snapshots covered: {sorted(selected['snapshot'].unique())}")
    print(f"Redshift range: {selected['snapshot_redshift'].min():.2f} - {selected['snapshot_redshift'].max():.2f}")
    print(f"Mass-bin counts:\n{selected['mass_bin'].value_counts().sort_index()}")
    print(f"Environment counts:\n{selected['environment'].value_counts()}")
    print(f"SED-type counts:\n{selected['sed_type'].value_counts()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
