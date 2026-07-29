#!/usr/bin/env python3
"""Pre-fetch a local TNG100-1 subhalo catalog for offline "TNG Mode".

Downloads subhalo physical properties (stellar mass, SFR, half-mass radius,
metallicity, group membership) for a set of snapshots/redshifts and a
stellar-mass range, and writes a single Parquet catalog that
``select_tng_galaxy_local`` (in ``src/tng_galaxy_selector.py``) can query
with zero network calls during simulation runs.

This replaces the live-API path used by ``query_tng_properties`` (which was
making 1-3 API calls *per lens/source/field galaxy*, each potentially slow
due to TNG API / MPCDF server latency).

Usage::

    TNG_API_KEY=... python3 scripts/build_tng_local_catalog.py \\
        --output /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet

Resumable: per-subhalo detail records and per-snapshot bulk listings are
cached as JSON under ``data/tng_catalogs/`` (existing cache from earlier TNG
Mode runs is reused), and per-snapshot Parquet files are written as soon as
each snapshot finishes, so an interrupted run can simply be restarted.

Scope/limitations:
- TNG100-1's stellar-mass resolution limit is ~10^8 Msun (a few hundred star
  particles), so this catalog cannot reach JADES-depth (~30.5 mag) dwarf
  galaxies at high redshift -- ``--logM-min`` below ~8.0 will mostly return
  poorly-resolved subhalos.
- For each snapshot, ALL subhalos with logM >= ``--logM-stratify-threshold``
  are kept (these are rare and important for lens/massive-companion
  matching); subhalos in [``--logM-min``, ``--logM-stratify-threshold``) are
  randomly subsampled down to fill the remaining ``--max-per-snapshot``
  budget. ``group_massive_subhalo_count`` (used for environment
  classification) is computed from the subhalos actually fetched for that
  snapshot, so it is a lower bound on the true companion count when the
  low-mass population is subsampled.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tng_galaxy_selector import (  # noqa: E402
    CACHE_DIR,
    ENV_MIN_COMPANION_LOGM,
    SNAPSHOT_REDSHIFTS,
    classify_environment,
    detail_to_record,
    fetch_all_subhalo_summaries,
    fetch_subhalo_detail,
)

# Default snapshots: nearest TNG100-1 snapshots to z = 0.0 .. 6.5, spanning
# the lens (z~0.2-1.5), lensed-source, and field-galaxy redshift ranges used
# by prism.core.simulator, extended up to z~6.5 (snap12) for high-z
# source/field coverage. Note: TNG100-1's stellar-mass resolution and box
# size make snap12-20 (z~4.2-6.5) increasingly sparse/small-N at the high-mass
# end -- see "Scope/limitations" above.
DEFAULT_SNAPSHOTS = [99, 91, 84, 78, 72, 67, 63, 59, 56, 53, 50, 46, 43, 40, 37, 33, 30, 28, 25, 23, 21,
                     20, 19, 18, 17, 16, 15, 14, 13, 12]


def build_snapshot_catalog(
    snap: int,
    logM_min: float,
    logM_max: float,
    logM_stratify_threshold: float,
    max_per_snapshot: int,
    sleep_s: float,
    rng: np.random.Generator,
    sim: str = "TNG100-1",
) -> pd.DataFrame:
    snap_z = SNAPSHOT_REDSHIFTS[snap]
    print(f"[snap {snap} z={snap_z:.3f}] fetching bulk subhalo listing "
          f"(logM {logM_min:.1f}-{logM_max:.1f}, sim={sim})...")
    summaries = fetch_all_subhalo_summaries(snap, logM_min, logM_max, sim=sim)
    print(f"[snap {snap}] {len(summaries)} subhalos in mass range")

    high_mass = [s for s in summaries if s["mass_log_msun"] >= logM_stratify_threshold]
    low_mass = [s for s in summaries if s["mass_log_msun"] < logM_stratify_threshold]

    # Split the per-snapshot detail-fetch budget between the high-mass and
    # low-mass populations (each capped at half), so a single snapshot can
    # never balloon to tens of thousands of detail requests even if the
    # "high-mass" population isn't actually rare for this snapshot/threshold.
    n_high_budget = min(len(high_mass), max_per_snapshot // 2)
    if len(high_mass) > n_high_budget:
        idx = rng.choice(len(high_mass), size=n_high_budget, replace=False)
        high_mass = [high_mass[i] for i in idx]

    n_low_budget = max(0, max_per_snapshot - len(high_mass))
    if len(low_mass) > n_low_budget:
        idx = rng.choice(len(low_mass), size=n_low_budget, replace=False)
        low_mass = [low_mass[i] for i in idx]

    selected = high_mass + low_mass
    print(f"[snap {snap}] selected {len(selected)} subhalos for detail fetch "
          f"({len(high_mass)} high-mass >= logM {logM_stratify_threshold:.1f}, "
          f"{len(low_mass)} subsampled low-mass)")

    details = []
    for i, s in enumerate(selected):
        cache_path = CACHE_DIR / f"{sim}_snap{snap}_subhalo{s['id']}.json"
        was_cached = cache_path.exists()
        detail = fetch_subhalo_detail(snap, s["id"], sim=sim)
        details.append(detail)
        if not was_cached:
            time.sleep(sleep_s)
        if (i + 1) % 100 == 0:
            print(f"[snap {snap}] fetched detail for {i + 1}/{len(selected)} subhalos")

    # group_massive_subhalo_count computed locally from the fetched set
    # (lower bound if the low-mass population was subsampled -- see module
    # docstring).
    grnr_counts: dict[int, int] = {}
    for d in details:
        if d["mass_log_msun"] >= ENV_MIN_COMPANION_LOGM:
            grnr_counts[d["grnr"]] = grnr_counts.get(d["grnr"], 0) + 1

    records = []
    for d in details:
        group_massive_count = grnr_counts.get(d["grnr"], 1)
        records.append(detail_to_record(snap, snap_z, d, group_massive_count, sim=sim))

    df = pd.DataFrame.from_records(records)
    print(f"[snap {snap}] done: {len(df)} rows, "
          f"environments: {df['environment'].value_counts().to_dict() if len(df) else {}}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshots", type=str, default=",".join(str(s) for s in DEFAULT_SNAPSHOTS),
                    help="Comma-separated TNG100-1 snapshot numbers")
    p.add_argument("--logM-min", type=float, default=8.0)
    p.add_argument("--logM-max", type=float, default=12.5)
    p.add_argument("--logM-stratify-threshold", type=float, default=10.0,
                    help="Keep ALL subhalos above this logM; subsample below it")
    p.add_argument("--max-per-snapshot", type=int, default=1500)
    p.add_argument("--sleep", type=float, default=0.3,
                    help="Seconds to sleep after each *new* (non-cached) detail request")
    p.add_argument("--sim", type=str, default="TNG100-1", choices=["TNG100-1", "TNG50-1"],
                    help="IllustrisTNG simulation to query")
    p.add_argument("--output", type=str,
                    default="/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    snapshots = [int(s) for s in args.snapshots.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    out_path = Path(args.output)
    out_dir = out_path.parent
    snap_dir = out_dir / "per_snapshot" / args.sim
    snap_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for snap in snapshots:
        snap_path = snap_dir / f"snap{snap}.parquet"
        if snap_path.exists():
            print(f"[snap {snap}] already built ({snap_path}), loading from disk")
            df = pd.read_parquet(snap_path)
        else:
            df = build_snapshot_catalog(
                snap, args.logM_min, args.logM_max,
                args.logM_stratify_threshold, args.max_per_snapshot,
                args.sleep, rng, sim=args.sim,
            )
            df.to_parquet(snap_path)
            print(f"[snap {snap}] saved {snap_path}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path)
    print(f"\nCombined catalog: {len(combined)} rows across {len(snapshots)} snapshots")
    print(f"Saved to {out_path}")
    print(combined.groupby("snapshot")["environment"].value_counts())


if __name__ == "__main__":
    main()
