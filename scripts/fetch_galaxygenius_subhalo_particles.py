#!/usr/bin/env python3
"""Download a TNG100-1 subhalo's particle cutout + detail JSON for GalaxyGenius.

This produces the ``TNG_100_snap_<snap>_subhalo_<id>.h5`` /
``TNG_100_snap_<snap>_subhalo_<id>.json`` pair that
``run_inclinations_*.py`` (in ``/Volumes/extHD/galaxygenius_build/workspace/``)
expects via ``preprocess.inputSubhaloParticleFile``.

Usage::

    TNG_API_KEY=... python3 scripts/fetch_galaxygenius_subhalo_particles.py \\
        --snap 84 --subhalo 12345
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tng_galaxy_selector import _headers, fetch_subhalo_detail, SIM_FILE_PREFIX  # noqa: E402

DEFAULT_OUTPUT_DIR = "/Volumes/extHD/galaxygenius_build/workspace/data"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--subhalo", type=int, required=True)
    p.add_argument("--sim", type=str, default="TNG100-1", choices=["TNG100-1", "TNG50-1"])
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{SIM_FILE_PREFIX[args.sim]}_snap_{args.snap}_subhalo_{args.subhalo}"

    detail = fetch_subhalo_detail(args.snap, args.subhalo, sim=args.sim)

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(detail, indent=4))
    print(f"Saved {json_path}")

    h5_path = out_dir / f"{stem}.h5"
    if h5_path.exists():
        print(f"{h5_path} already exists, skipping download")
        return

    cutout_url = detail["cutouts"]["subhalo"]
    print(f"Downloading particle cutout from {cutout_url} ...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = requests.get(cutout_url, headers=_headers(), stream=True, timeout=600)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  attempt {attempt + 1} failed ({exc}), retrying in {wait}s...")
            time.sleep(wait)
    with open(h5_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"Saved {h5_path} ({h5_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
