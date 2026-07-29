#!/usr/bin/env python3
"""Report COSMOS-Web 1′ field densities and validate a simulation run.

Usage:
  python scripts/local/report_cosmos_field_density.py
  python scripts/local/report_cosmos_field_density.py outputs/euclid_paper_physics_1arcmin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from prism.core.simulator import (  # noqa: E402
    cosmos_field_density_per_arcmin2,
    field_galaxy_count_target,
)


def print_cosmos_table() -> None:
    print("COSMOS-Web measured galaxy surface density")
    print("(data/galaxy_catalog.fits, F115W, footprint-corrected)")
    print(f"{'mag_lim':>8} {'Σ/arcmin²':>10} {'N(1′×1′)':>10} {'δ_group×1.8':>12}")
    for m in (21.0, 21.5, 22.0, 23.0, 23.5, 24.0, 24.5, 25.0, 26.0):
        dens = cosmos_field_density_per_arcmin2(m)
        print(f"{m:8.1f} {dens:10.2f} {dens:10.1f} {dens * 1.8:12.1f}")
    print()
    cfg = {
        "field": {"density_mag_limit": 23.5},
        "catalogs": {"galaxy_catalog_fits": "data/galaxy_catalog.fits"},
    }
    print("Expected counts in Euclid 1′ (600×600 @ 0.10″/px), mag < 23.5:")
    for name, mean in [("isolated_field", 2.5), ("galaxy_pair", 3.0), ("group", 4.5)]:
        m, s = field_galaxy_count_target(600, 0.10, {"galaxy_count_mean": mean}, cfg)
        print(f"  {name:16s}  N = {m:5.1f} ± {s:4.1f}")


def validate_run(sim_dir: Path) -> None:
    cat = sim_dir / "cosmos_training_catalog_lens_and_nonlens.csv"
    if not cat.exists():
        raise SystemExit(f"No catalog in {sim_dir}")
    df = pd.read_csv(cat)
    print(f"\nRun validation: {sim_dir}")
    print(f"  n_lenses = {len(df)}")

    # Field counts from NPZ metadata when available
    n_fields = []
    shears = []
    sigmas = []
    theta_es = []
    npz_dir = sim_dir / "unified_npz"
    for path in sorted(npz_dir.glob("PRISM_lens_*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = data["metadata"]
        if isinstance(meta, np.ndarray):
            meta = meta.item()
        if isinstance(meta, str):
            meta = json.loads(meta)
        if not isinstance(meta, dict):
            continue
        if "n_field_galaxies" in meta:
            n_fields.append(float(meta["n_field_galaxies"]))
        fi = meta.get("field_info") or {}
        if isinstance(fi, dict) and "n_field_galaxies" in fi:
            n_fields.append(float(fi["n_field_galaxies"]))
        for k in ("shear_gamma1", "gamma1", "external_shear_g"):
            if k in meta and meta[k] is not None:
                try:
                    shears.append(abs(float(meta[k])))
                except Exception:
                    pass
        if "sigma_kms" in meta and meta["sigma_kms"] is not None:
            try:
                sigmas.append(float(meta["sigma_kms"]))
            except Exception:
                pass
        if "theta_E" in meta:
            theta_es.append(float(meta["theta_E"]))

    if "n_field_galaxies" in df.columns:
        n_fields = df["n_field_galaxies"].astype(float).tolist()
    if n_fields:
        arr = np.asarray(n_fields, dtype=float)
        print(f"  n_field_galaxies: mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
              f"min={arr.min():.0f}  max={arr.max():.0f}")
        print(f"  (COSMOS mag<23.5 isolated ≈30; group ≈54)")
    else:
        print("  n_field_galaxies: not found in catalog/metadata")

    if "theta_E" in df.columns:
        te = df["theta_E"].astype(float)
        print(f"  theta_E: mean={te.mean():.2f}\"  median={te.median():.2f}\"  "
              f"range=[{te.min():.2f}, {te.max():.2f}]")
    elif theta_es:
        te = np.asarray(theta_es)
        print(f"  theta_E: mean={te.mean():.2f}\"  range=[{te.min():.2f}, {te.max():.2f}]")

    if sigmas:
        s = np.asarray(sigmas)
        print(f"  sigma_kms (FP): mean={s.mean():.0f}  range=[{s.min():.0f}, {s.max():.0f}]")
    if shears:
        g = np.asarray(shears)
        print(f"  |shear| samples: mean={g.mean():.3f}  range=[{g.min():.3f}, {g.max():.3f}]")

    # Class mix
    for col in ("lens_system_class", "environment"):
        if col in df.columns:
            print(f"  {col}:")
            for k, v in df[col].value_counts().items():
                print(f"    {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    print_cosmos_table()
    if args.sim_dir is not None:
        validate_run(args.sim_dir.resolve())


if __name__ == "__main__":
    main()
