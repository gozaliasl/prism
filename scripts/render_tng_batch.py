#!/usr/bin/env python3
"""Batch-render local TNG particle cutouts through GalaxyGenius/SKIRT.

Disk-aware, resumable, priority-aware:
  - Priority 1 (dynamic): TNG100 lens-candidate cutouts
    (tng100_lens_candidates.csv) that have finished downloading -- re-checked
    every cycle so newly arrived downloads jump the queue.
  - Priority 2 (static): the stratified source/field sample from
    render_priority_queue.csv.
  - Stops automatically once free disk space on the cutout volume drops
    below --min-free-gb, or once --budget-gb of new output has been written.
  - A manifest CSV (--manifest) records sim/snapshot/subhalo/status/runtime
    so a killed/restarted run skips already-attempted galaxies.
  - Each render's heavy dataCubes/Subhalo_<ID> intermediate (~1.5GB) is
    deleted immediately after a successful render; mock_JWST/Subhalo_<ID>
    output is renamed to mock_JWST/{SIM}_{ID} to avoid TNG50/TNG100 ID
    collisions before the next subhalo reuses the bare directory name.

Usage::

    nohup python3 scripts/render_tng_batch.py \\
        --budget-gb 110 --min-free-gb 20 \\
        > /tmp/tng_render_batch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

WORKSPACE = Path("/Volumes/extHD/galaxygenius_build/workspace")
RENDER_SCRIPT = WORKSPACE / "run_inclinations_generic.py"
PYTHON_BIN = sys.executable

MANIFEST_COLS = ["sim_prefix", "sim", "snapshot", "subhalo_id", "status", "seconds", "timestamp"]


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def load_manifest(manifest_path: Path) -> set[tuple[str, int, int]]:
    done = set()
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["sim_prefix"], int(row["snapshot"]), int(row["subhalo_id"])))
    return done


def append_manifest(manifest_path: Path, row: dict) -> None:
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def cutouts_for_pairs(data_dir: Path, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a (sim, snapshot, subhalo_id) candidate frame to those with a
    local .h5+.json pair actually present on disk."""
    keep = []
    for _, r in pairs_df.iterrows():
        sim_prefix = "TNG_100" if str(r["sim"]).startswith("TNG100") else "TNG_50"
        stem = f"{sim_prefix}_snap_{int(r['snapshot'])}_subhalo_{int(r['subhalo_id'])}"
        if (data_dir / f"{stem}.h5").exists() and (data_dir / f"{stem}.json").exists():
            keep.append(r)
    return pd.DataFrame(keep) if keep else pairs_df.iloc[0:0]


def render_one(sim_prefix: str, snap: int, subhalo: int, redshift: float, timeout: int) -> tuple[bool, float, str]:
    t0 = time.time()
    cmd = [PYTHON_BIN, str(RENDER_SCRIPT), sim_prefix.replace("TNG_", "TNG_"), str(snap), str(subhalo), str(redshift)]
    try:
        proc = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0 and f"RENDER_OK" in proc.stdout
        msg = "" if ok else (proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:])
        return ok, time.time() - t0, msg
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, time.time() - t0, str(exc)


def finalize_output(sim_prefix: str, subhalo: int) -> None:
    """Rename mock_JWST output to namespace it by sim, delete heavy dataCube."""
    bare = WORKSPACE / "mock_JWST" / f"Subhalo_{subhalo}"
    namespaced = WORKSPACE / "mock_JWST" / f"{sim_prefix}_{subhalo}"
    if bare.exists():
        if namespaced.exists():
            shutil.rmtree(namespaced)
        bare.rename(namespaced)
    cube = WORKSPACE / "dataCubes" / f"Subhalo_{subhalo}"
    if cube.exists():
        shutil.rmtree(cube, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(WORKSPACE / "data"))
    p.add_argument("--lens-candidates", default="/Volumes/extHD/tng_local_catalog/tng100_lens_candidates.csv")
    p.add_argument("--priority-queue", default="/Volumes/extHD/tng_local_catalog/render_priority_queue.csv")
    p.add_argument("--manifest", default="/Volumes/extHD/tng_local_catalog/render_manifest.csv")
    p.add_argument("--snap-redshift-map", default="/tmp/snap_redshift_map.csv",
                    help="CSV with columns snapshot,snapshot_redshift (shared by TNG50/TNG100)")
    p.add_argument("--budget-gb", type=float, default=110.0)
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--mb-per-galaxy", type=float, default=19.0)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--max-galaxies", type=int, default=10**9)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest)
    snap_z = pd.read_csv(args.snap_redshift_map).set_index("snapshot")["snapshot_redshift"].to_dict()

    static_queue = pd.read_csv(args.priority_queue)
    n_budget = int(args.budget_gb * 1024 / args.mb_per_galaxy)

    n_done = n_failed = n_skipped_existing = 0
    bytes_written_gb = 0.0
    t_start = time.time()

    while n_done < args.max_galaxies and n_done < n_budget:
        free_gb = disk_free_gb(data_dir)
        if free_gb < args.min_free_gb:
            print(f"STOP: free disk {free_gb:.1f}GB < min_free_gb={args.min_free_gb}GB", flush=True)
            break

        done_pairs = load_manifest(manifest_path)

        # Re-check dynamic priority 1: lens candidates that now have local cutouts.
        next_row = None
        try:
            lens_cand = pd.read_csv(args.lens_candidates)
            lens_cand["sim"] = "TNG100-1"
            avail = cutouts_for_pairs(data_dir, lens_cand)
            avail = avail[~avail.apply(
                lambda r: ("TNG_100", int(r["snapshot"]), int(r["subhalo_id"])) in done_pairs, axis=1)]
            if len(avail):
                next_row = avail.iloc[0]
                sim_prefix = "TNG_100"
        except FileNotFoundError:
            pass

        if next_row is None:
            pending = static_queue[~static_queue.apply(
                lambda r: (r["sim_prefix"], int(r["snapshot"]), int(r["subhalo_id"])) in done_pairs, axis=1)]
            if pending.empty:
                print("Priority queue exhausted.", flush=True)
                break
            next_row = pending.iloc[0]
            sim_prefix = next_row["sim_prefix"]

        snap = int(next_row["snapshot"])
        subhalo = int(next_row["subhalo_id"])
        redshift = snap_z.get(snap)
        if redshift is None:
            redshift = float(next_row.get("snapshot_redshift", 0.5))

        stem = f"{sim_prefix}_snap_{snap}_subhalo_{subhalo}"
        if not (data_dir / f"{stem}.h5").exists():
            append_manifest(manifest_path, dict(sim_prefix=sim_prefix, sim=next_row.get("sim", ""),
                                                 snapshot=snap, subhalo_id=subhalo, status="missing_cutout",
                                                 seconds=0, timestamp=time.time()))
            continue

        ok, secs, err = render_one(sim_prefix, snap, subhalo, redshift, args.timeout)
        status = "ok" if ok else "failed"
        if ok:
            finalize_output(sim_prefix, subhalo)
            n_done += 1
            bytes_written_gb += args.mb_per_galaxy / 1024
        else:
            n_failed += 1
            print(f"  FAILED {stem}: {err[:300]}", flush=True)

        append_manifest(manifest_path, dict(sim_prefix=sim_prefix, sim=next_row.get("sim", ""),
                                             snapshot=snap, subhalo_id=subhalo, status=status,
                                             seconds=round(secs, 1), timestamp=time.time()))

        total = n_done + n_failed
        if total % 10 == 0:
            elapsed_h = (time.time() - t_start) / 3600
            rate = n_done / max(elapsed_h, 1e-6)
            print(f"[{total}] done={n_done} failed={n_failed} free_disk={free_gb:.1f}GB "
                  f"written~{bytes_written_gb:.1f}GB elapsed={elapsed_h:.2f}h rate={rate:.1f}/h", flush=True)

    elapsed_h = (time.time() - t_start) / 3600
    print(f"\nFinished batch: done={n_done} failed={n_failed} elapsed={elapsed_h:.2f}h", flush=True)


if __name__ == "__main__":
    main()
