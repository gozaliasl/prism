#!/usr/bin/env python3
"""Select TNG50-1 (snapshot, subhalo_id) candidates for local particle-cutout
download, prioritizing the MOST MASSIVE galaxies first -- these are the most
relevant for lens-galaxy morphology (massive ellipticals/early-types).

Reads the local TNG50-1 catalog, excludes subhalos that already have a local
``.h5`` cutout (``src.tng_galaxy_selector.local_particle_path(...,
sim="TNG50-1")``), sorts the remainder by stellar mass descending, and takes
the top ``--n-total``. If fewer than ``--n-total`` remain above
``--logm-min``, the pool is widened downward in mass until ``--n-total`` is
reached (still highest-mass-first).

Usage::

    python3 scripts/select_tng50_massive_lens_candidates.py \\
        --catalog /Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet \\
        --output /Volumes/extHD/tng_local_catalog/tng50_massive_lens_candidates.csv \\
        --n-total 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tng_galaxy_selector import load_local_catalog, local_particle_path  # noqa: E402

MASS_BIN_EDGES = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.5]
MASS_BIN_LABELS = ["8.0-8.5", "8.5-9.0", "9.0-9.5", "9.5-10.0", "10.0-10.5",
                    "10.5-11.0", "11.0-11.5", "11.5-12.5"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=str,
                   default="/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet")
    p.add_argument("--output", type=str,
                   default="/Volumes/extHD/tng_local_catalog/tng50_massive_lens_candidates.csv")
    p.add_argument("--n-total", type=int, default=1500,
                   help="Target total number of new candidates")
    p.add_argument("--logm-min", type=float, default=11.0,
                   help="Preferred minimum logM* -- widened downward if not enough candidates")
    p.add_argument("--logm-max", type=float, default=12.5)
    args = p.parse_args()

    cat = load_local_catalog(args.catalog)
    if cat is None or len(cat) == 0:
        print(f"No catalog found at {args.catalog} (or it is empty) -- nothing to select")
        return

    cat = cat[cat["stellar_mass_logmsun"] <= args.logm_max].copy()
    has_cutout = cat.apply(
        lambda r: local_particle_path(int(r["snapshot"]), int(r["subhalo_id"]), sim="TNG50-1") is not None, axis=1)
    cat = cat[~has_cutout].copy()
    print(f"{len(cat)} TNG50-1 candidates without a local .h5 cutout (logM <= {args.logm_max})")

    cat = cat.sort_values("stellar_mass_logmsun", ascending=False)

    n_above = (cat["stellar_mass_logmsun"] >= args.logm_min).sum()
    print(f"  of which logM >= {args.logm_min}: {n_above}")

    selected = cat.head(args.n_total).copy()
    selected["mass_bin"] = pd.cut(selected["stellar_mass_logmsun"], bins=MASS_BIN_EDGES, labels=MASS_BIN_LABELS)
    selected["sim"] = "TNG50-1"
    selected = selected.sort_values(["snapshot", "stellar_mass_logmsun"], ascending=[True, False]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    print(f"\nSelected {len(selected)} new candidates (most massive first)")
    if len(selected):
        print(f"logM range: {selected['stellar_mass_logmsun'].min():.2f} - {selected['stellar_mass_logmsun'].max():.2f}")
        print(f"Snapshots covered: {sorted(selected['snapshot'].unique())}")
        print(f"Mass-bin counts:\n{selected['mass_bin'].value_counts().sort_index()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
