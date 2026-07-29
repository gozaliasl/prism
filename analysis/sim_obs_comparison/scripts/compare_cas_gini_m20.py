#!/usr/bin/env python3
"""
Compare CAS (Concentration, Asymmetry, Smoothness), clumpiness, Gini and M20
non-parametric morphology statistics between the real COSMOS-Web sample
(analysis/sim_obs_comparison/catalogs/real_lens_morphology.csv) and the
simulated sample (analysis/sim_obs_comparison/catalogs/sim_lens_morphology.csv),
both produced by morphology_metrics.measure_lens_arrays() via
galaxy_morphology.validation.compute_cas_gini_m20.

Output:
  analysis/sim_obs_comparison/reports/phase3_cas_gini_m20_comparison.md
  analysis/sim_obs_comparison/reports/phase3_cas_gini_m20_comparison.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1]
METRICS = ["concentration", "asymmetry", "smoothness", "clumpiness", "gini", "m20"]
EXPECTED_RANGES = {
    "concentration": (0.0, None),
    "asymmetry": (0.0, None),
    "smoothness": (None, None),
    "clumpiness": (None, None),
    "gini": (0.0, 1.0),
    "m20": (None, 0.0),
}


def summary_stats(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def main():
    real_df = pd.read_csv(OUT_DIR / "catalogs" / "real_lens_morphology.csv")
    sim_df = pd.read_csv(OUT_DIR / "catalogs" / "sim_lens_morphology.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    lines = []
    lines.append("# Phase 7: CAS / Gini / M20 real vs. simulated comparison\n")
    lines.append(f"Real sample (COSMOS-Web): {len(real_df)} lenses, "
                  f"{real_df['gini'].notna().sum()} with valid metrics.")
    lines.append(f"Simulated sample: {len(sim_df)} lenses, "
                  f"{sim_df['gini'].notna().sum()} with valid metrics.\n")
    lines.append("| metric | real n | real mean | real median | sim n | sim mean | sim median | range check |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for ax, metric in zip(axes.flat, METRICS):
        real_vals = real_df[metric].dropna().values
        sim_vals = sim_df[metric].dropna().values
        real_stat = summary_stats(real_vals)
        sim_stat = summary_stats(sim_vals)

        lo, hi = EXPECTED_RANGES[metric]
        all_vals = np.concatenate([real_vals, sim_vals]) if len(real_vals) and len(sim_vals) else np.array([])
        in_range = True
        if len(all_vals):
            if lo is not None and np.any(all_vals < lo - 1e-6):
                in_range = False
            if hi is not None and np.any(all_vals > hi + 1e-6):
                in_range = False
        range_str = "OK" if in_range else "OUT OF RANGE"

        lines.append(
            f"| {metric} | {real_stat['n']} | {real_stat['mean']:.3f} | {real_stat['median']:.3f} "
            f"| {sim_stat['n']} | {sim_stat['mean']:.3f} | {sim_stat['median']:.3f} | {range_str} |"
        )

        bins = 25
        ax.hist(real_vals, bins=bins, alpha=0.5, density=True, label="real (COSMOS-Web)", color="tab:blue")
        ax.hist(sim_vals, bins=bins, alpha=0.5, density=True, label="simulated", color="tab:orange")
        ax.set_title(metric)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUT_DIR / "reports" / "phase3_cas_gini_m20_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")

    lines.append("\nRange checks: Gini in [0,1], concentration > 0, M20 < 0 (expected for "
                  "centrally-concentrated light profiles).\n")
    lines.append(f"![comparison]({fig_path.name})\n")

    out_md = OUT_DIR / "reports" / "phase3_cas_gini_m20_comparison.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
