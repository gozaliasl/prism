#!/usr/bin/env python3
"""Group-aware train/val/test split for a PRISM training catalog.

FIX (adversarial audit finding C-14, 2026-08-01): the generator produces
`variations_per_base` (default 25) near-duplicate renders per real COSMOS-Web
catalog row -- same lens galaxy structural/photometric properties, only
theta_E/source position/noise realization perturbed. A naive random
train/val/test split leaks near-copies of validation/test lenses into
training. `base_lens_id` (already recorded in every catalog row) identifies
which base catalog row a given render came from -- this script splits on
THAT, not on individual rows, so no base lens appears in more than one split.

Usage:
    python make_grouped_split.py <training_catalog.csv> --out-dir <dir> \
        --val-frac 0.15 --test-frac 0.15 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def grouped_split(df: pd.DataFrame, group_col: str, val_frac: float,
                   test_frac: float, seed: int) -> dict:
    if group_col not in df.columns:
        raise ValueError(
            f"'{group_col}' column not found in catalog -- cannot do a "
            "group-aware split. Available columns: " + ", ".join(df.columns)
        )
    groups = df[group_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    train_groups = set(groups[n_test + n_val:])

    return {
        "train": df[df[group_col].isin(train_groups)].reset_index(drop=True),
        "val": df[df[group_col].isin(val_groups)].reset_index(drop=True),
        "test": df[df[group_col].isin(test_groups)].reset_index(drop=True),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("catalog", help="Path to a PRISM training catalog CSV")
    ap.add_argument("--group-col", default="base_lens_id")
    ap.add_argument("--out-dir", default=None,
                     help="Output directory (default: alongside input catalog)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    splits = grouped_split(df, args.group_col, args.val_frac, args.test_frac, args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.catalog).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.catalog).stem

    for name, split_df in splits.items():
        out_path = out_dir / f"{stem}_{name}.csv"
        split_df.to_csv(out_path, index=False)
        n_groups = split_df[args.group_col].nunique() if len(split_df) else 0
        print(f"{name}: {len(split_df)} rows, {n_groups} unique {args.group_col} -> {out_path}")

    # Sanity check: no group leakage across splits.
    train_g = set(splits["train"][args.group_col].dropna())
    val_g = set(splits["val"][args.group_col].dropna())
    test_g = set(splits["test"][args.group_col].dropna())
    overlap = (train_g & val_g) | (train_g & test_g) | (val_g & test_g)
    if overlap:
        raise AssertionError(f"Group leakage detected across splits: {overlap}")
    print("OK: no base-lens-id leakage across train/val/test.")


if __name__ == "__main__":
    main()
