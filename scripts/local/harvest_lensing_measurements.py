#!/usr/bin/env python3
"""Harvest κ/γ/μ/flexion scalar measurements from kappa_maps/*.npz into a CSV.

Merges with the main training catalog when available.

Usage:
  python scripts/local/harvest_lensing_measurements.py outputs/euclid_paper_select_50_q1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

SCALAR_KEYS = [
    "theta_E_eff", "kappa_max", "kappa_mean", "mu_max",
    "gamma_mag_max", "gamma_mag_mean", "critical_area",
    "F_mag_max", "F_mag_mean", "G_mag_max", "G_mag_mean",
    "ext_theta_E_eff", "ext_kappa_max", "ext_kappa_mean", "ext_mu_max",
    "ext_gamma_mag_max", "ext_gamma_mag_mean", "ext_critical_area",
    "ext_F_mag_max", "ext_F_mag_mean", "ext_G_mag_max", "ext_G_mag_mean",
    "ext_fov_arcmin", "ext_num_pix", "ext_delta_pix",
]


def _scalar(v):
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return float(v.item())
        return float(np.asarray(v).ravel()[0])
    if isinstance(v, (bytes, str)):
        return str(v)
    try:
        return float(v)
    except Exception:
        return v


def harvest_one(npz: Path) -> dict:
    d = np.load(npz, allow_pickle=True)
    row = {
        "kappa_file": npz.name,
        "lens_id": _scalar(d["lens_id"]) if "lens_id" in d.files else npz.stem.split("_")[0],
        "category": str(d["category"]) if "category" in d.files else "",
        "sub_type": str(d["sub_type"]) if "sub_type" in d.files else "",
    }
    for k in SCALAR_KEYS:
        if k in d.files:
            row[k] = _scalar(d[k])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    kappa_dir = run_dir / "kappa_maps"
    files = sorted(kappa_dir.glob("*_kappa_data.npz"))
    if not files:
        raise SystemExit(f"No kappa NPZ in {kappa_dir}")

    rows = [harvest_one(p) for p in files]
    meas = pd.DataFrame(rows)

    # Normalize lens_id to zero-padded string for joins
    def _pad_id(x):
        try:
            return f"{int(float(x)):06d}"
        except Exception:
            s = str(x)
            return s.zfill(6) if s.isdigit() else s

    meas["lens_id_str"] = meas["lens_id"].map(_pad_id)

    cat_path = run_dir / "cosmos_training_catalog_lens_and_nonlens.csv"
    if cat_path.exists():
        cat = pd.read_csv(cat_path)
        # catalog rows are ordered; also try lens_id column
        if "lens_id" in cat.columns:
            cat["lens_id_str"] = cat["lens_id"].astype(str).str.extract(r"(\d+)")[0].fillna("").map(
                lambda s: s.zfill(6) if s else ""
            )
        else:
            cat["lens_id_str"] = [f"{i:06d}" for i in range(len(cat))]
        merged = cat.merge(meas, on="lens_id_str", how="left", suffixes=("", "_kappa"))
    else:
        merged = meas

    out_csv = run_dir / "lensing_measurements.csv"
    out_json = run_dir / "lensing_measurements_summary.json"
    merged.to_csv(out_csv, index=False)

    summary = {
        "n_kappa_files": len(files),
        "n_merged_rows": len(merged),
        "theta_E_eff": merged["theta_E_eff"].describe().to_dict() if "theta_E_eff" in merged else {},
        "kappa_max": merged["kappa_max"].describe().to_dict() if "kappa_max" in merged else {},
        "mu_max": merged["mu_max"].describe().to_dict() if "mu_max" in merged else {},
        "columns": list(merged.columns),
    }
    out_json.write_text(json.dumps(summary, indent=2, default=float))

    print(f"[MEAS] {len(files)} kappa NPZ → {out_csv}")
    print(f"[MEAS] summary → {out_json}")
    if "theta_E_eff" in merged:
        print(merged[["lens_id_str", "theta_E_eff", "kappa_max", "mu_max", "gamma_mag_mean"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
