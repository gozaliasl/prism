#!/usr/bin/env python3
"""
Fetch bulge/disk kinematic decomposition (Thob et al. 2019) and structural
parameters from the EAGLE public database, to calibrate
src/galaxy_morphology's per-type bt_fractions / r_frac / axis-ratio recipes
against real simulated galaxies (Tier 1 of the EAGLE roadmap).

Requires an EAGLE public database account (free registration at
http://icc.dur.ac.uk/Eagle/database.php). Provide credentials via
environment variables -- do NOT hardcode them here or paste them into chat:

    export EAGLE_DB_USER=your_username
    export EAGLE_DB_PASS=your_password

Usage: conda run -n astro-clean python analysis/galaxy_morphology/fetch_eagle_morphology_catalog.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import eagleSqlTools as sql

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "eagle_morphology_catalog.csv"

# RefL0100N1504 = 100 Mpc reference box (largest public EAGLE volume).
# SnapNum 28 = z=0; include a few higher-z snapshots for redshift evolution
# of disc-to-total fractions and sizes.
SNAPSHOTS = {
    28: 0.0,
    19: 1.0,
    15: 2.0,
    12: 3.0,
}

QUERY_TEMPLATE = """
SELECT
    SH.GalaxyID,
    SH.SnapNum,
    SH.MassType_Star,
    SH.SubGroupNumber,
    SS.DiscToTotal,
    SS.KappaCo,
    SS.Ellipticity,
    SS.Triaxiality,
    SIZES.HalfMassRad_Star
FROM
    RefL0100N1504_Subhalo AS SH,
    RefL0100N1504_SizesShapes AS SS,
    RefL0100N1504_Sizes AS SIZES
WHERE
    SH.SnapNum = {snap}
    AND SH.MassType_Star > 1.0e9
    AND SH.GalaxyID = SS.GalaxyID
    AND SH.GalaxyID = SIZES.GalaxyID
"""


def main():
    user = os.environ.get("EAGLE_DB_USER")
    password = os.environ.get("EAGLE_DB_PASS")
    if not user or not password:
        print("Set EAGLE_DB_USER and EAGLE_DB_PASS environment variables "
              "(register at http://icc.dur.ac.uk/Eagle/database.php).")
        sys.exit(1)

    con = sql.connect(user, password=password)

    frames = []
    for snap, z in SNAPSHOTS.items():
        query = QUERY_TEMPLATE.format(snap=snap)
        print(f"Querying SnapNum={snap} (z~{z}) ...")
        result = sql.execute_query(con, query)
        df = pd.DataFrame(result)
        df['redshift'] = z
        frames.append(df)
        print(f"  -> {len(df)} galaxies")

    full = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(full)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
