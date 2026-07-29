#!/usr/bin/env python3
"""
Compute the same morphology/lensing statistics as
compute_real_lens_morphology.py, but for simulated lenses produced by the
JADES 10k run (unified_npz/*.npz, image_final stack ordered per metadata
'bands').

Usage:
  python compute_sim_lens_morphology.py <sim_output_dir> [--n_max N] [--seed S]

Output:
  analysis/sim_obs_comparison/catalogs/sim_lens_morphology.csv
  analysis/sim_obs_comparison/reports/sim_lens_statistics_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from morphology_metrics import measure_lens_arrays, summarize, PIXEL_SCALE

OUT_DIR = Path(__file__).resolve().parents[1]  # analysis/sim_obs_comparison
COMPARE_BANDS = ["F115W", "F150W", "F277W", "F444W"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir", help="Path to unified_npz directory or run output dir")
    ap.add_argument("--n_max", type=int, default=0, help="Max number of lenses to process (0 = all available)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sim_dir = Path(args.sim_dir)
    npz_dir = sim_dir / "unified_npz" if (sim_dir / "unified_npz").is_dir() else sim_dir
    files = sorted(npz_dir.glob("*.npz"))
    print(f"Found {len(files)} simulated lens npz files in {npz_dir}")

    if args.n_max and len(files) > args.n_max:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(files), size=args.n_max, replace=False)
        files = [files[i] for i in sorted(idx)]
        print(f"Subsampled to {len(files)} files")

    rows = []
    for i, f in enumerate(files):
        try:
            d = np.load(f, allow_pickle=True)
            meta = json.loads(str(d["metadata"].item())) if isinstance(d["metadata"].item(), str) else d["metadata"].item()
            bands = meta["bands"]
            stack = d["image_final"]  # (n_bands, ny, nx)
            band_idx = {b: bands.index(b) for b in COMPARE_BANDS if b in bands}
            images = {b: stack[idx].astype(np.float64) for b, idx in band_idx.items()}
            if len(images) < 2:
                continue
            r = measure_lens_arrays(f.stem, images)
            if r is None:
                continue
            r["theta_E"] = meta.get("theta_E", np.nan)
            r["lens_redshift"] = meta.get("lens_redshift", np.nan)
            r["source_redshift"] = meta.get("source_redshift", np.nan)
            rows.append(r)
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(files)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "catalogs" / "sim_lens_morphology.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {out_csv}")

    summary = summarize(df)
    mult = df["n_components"].fillna(0).astype(int)
    summary["multiplicity_histogram"] = {
        "single (1)": int((mult == 1).sum()),
        "double (2)": int((mult == 2).sum()),
        "triple (3)": int((mult == 3).sum()),
        "quad+ (>=4)": int((mult >= 4).sum()),
    }
    summary["source_npz_dir"] = str(npz_dir)
    summary["n_files_total_available"] = len(sorted(npz_dir.glob("*.npz")))
    summary["n_files_processed"] = len(df)

    out_json = OUT_DIR / "reports" / "sim_lens_statistics_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote summary -> {out_json}")


if __name__ == "__main__":
    main()
