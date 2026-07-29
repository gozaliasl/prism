#!/usr/bin/env python3
"""Batch-download TNG100-1 particle cutouts + detail JSON for GalaxyGenius.

Iterates over the candidate list produced by
``select_galaxygenius_stamp_candidates.py`` and, for each
``(snapshot, subhalo_id)``, fetches the detail JSON and particle cutout
``.h5`` via the same logic as ``fetch_galaxygenius_subhalo_particles.py``
(producing ``TNG_100_snap_<snap>_subhalo_<id>.h5`` /
``TNG_100_snap_<snap>_subhalo_<id>.json`` pairs under ``--output-dir``).

Resumable: a candidate is skipped if both its ``.json`` and ``.h5`` files
already exist in ``--output-dir``.

Usage::

    TNG_API_KEY=... python3 scripts/batch_fetch_galaxygenius_stamps.py \\
        --candidates /Volumes/extHD/tng_local_catalog/galaxygenius_stamp_candidates.csv \\
        --output-dir /Volumes/extHD/galaxygenius_build/workspace/data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tng_galaxy_selector import _headers, fetch_subhalo_detail, SIM_FILE_PREFIX  # noqa: E402

DEFAULT_OUTPUT_DIR = "/Volumes/extHD/galaxygenius_build/workspace/data"


def download_cutout(cutout_url: str, h5_path: Path, max_retries: int = 5) -> None:
    urls_to_try = [cutout_url]
    # If URL points to EU node, also try main node as fallback
    if "data-eu.tng-project.org" in cutout_url:
        urls_to_try.append(cutout_url.replace("data-eu.tng-project.org", "data.tng-project.org"))

    last_exc = None
    for url in urls_to_try:
        for attempt in range(max_retries):
            try:
                r = requests.get(url, headers=_headers(), stream=True, timeout=600)
                r.raise_for_status()
                last_exc = None
                break
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                # Don't retry 403 on this URL — try next URL instead
                if hasattr(exc, 'response') and exc.response is not None and exc.response.status_code == 403:
                    print(f"    403 on {url.split('/')[2]}, trying alternative...")
                    break
                if attempt == max_retries - 1:
                    break
                wait = 2 ** attempt * 5
                print(f"    attempt {attempt + 1} failed ({exc}), retrying in {wait}s...")
                time.sleep(wait)
        if last_exc is None:
            break
    if last_exc is not None:
        raise last_exc
    tmp_path = h5_path.with_suffix(".h5.part")
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp_path.rename(h5_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=str,
                    default="/Volumes/extHD/tng_local_catalog/galaxygenius_stamp_candidates.csv")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds to sleep after each new (non-cached) detail/cutout request")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates)
    n_total = len(candidates)
    n_done = n_skipped = n_failed = 0

    for i, row in candidates.iterrows():
        snap = int(row["snapshot"])
        subhalo = int(row["subhalo_id"])
        sim = str(row["sim"]) if "sim" in row and pd.notna(row["sim"]) else "TNG100-1"
        stem = f"{SIM_FILE_PREFIX[sim]}_snap_{snap}_subhalo_{subhalo}"
        json_path = out_dir / f"{stem}.json"
        h5_path = out_dir / f"{stem}.h5"

        if json_path.exists() and h5_path.exists():
            n_skipped += 1
            continue

        try:
            detail = fetch_subhalo_detail(snap, subhalo, sim=sim)
            if not json_path.exists():
                json_path.write_text(json.dumps(detail, indent=4))
                time.sleep(args.sleep)

            if not h5_path.exists():
                cutout_url = detail["cutouts"]["subhalo"]
                download_cutout(cutout_url, h5_path)
                time.sleep(args.sleep)

            n_done += 1
        except Exception as exc:
            n_failed += 1
            print(f"  FAILED snap={snap} subhalo={subhalo}: {exc}")
            continue

        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            print(f"[{i + 1}/{n_total}] done={n_done} skipped={n_skipped} failed={n_failed}")

    print(f"\nFinished: {n_done} downloaded, {n_skipped} already present, {n_failed} failed "
          f"(out of {n_total} candidates)")


if __name__ == "__main__":
    main()
