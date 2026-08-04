"""Build the 4 required contact sheets for the visual-HST-inspired test batch,
using ONE fixed RGB stretch (computed once, globally, across the whole batch)
so no per-object display tuning can be mistaken for a real detection.

Run: python scripts/local/build_contact_sheets.py <output_dir> <validation_csv>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = ["F115W", "F150W", "F277W", "F444W"]


def compute_global_stretch(npz_files, black_pct=50.0, vmax_pct=99.9, softening_frac=0.05):
    """One black level + arcsinh softening scale per band, from ALL systems'
    image_final pooled together -- used identically for every system."""
    pooled = {b: [] for b in BANDS}
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        im = d["image_final"]
        for bi, b in enumerate(BANDS):
            pooled[b].append(im[bi].ravel())
    stretch = {}
    for b in BANDS:
        allpix = np.concatenate(pooled[b])
        black = np.percentile(allpix, black_pct)
        sub = np.clip(allpix - black, 0, None)
        vmax = np.percentile(sub, vmax_pct)
        if vmax <= 0:
            vmax = sub.max() or 1.0
        stretch[b] = dict(black=float(black), vmax=float(vmax), soft=float(vmax * softening_frac))
    return stretch


def apply_fixed_rgb(images_by_band, stretch):
    """images_by_band: {band: 2D array}. Returns (H,W,3) in [0,1]."""
    def ch(band):
        im = images_by_band[band].astype(np.float64)
        s = stretch[band]
        sub = np.clip(im - s["black"], 0, None)
        return np.clip(np.arcsinh(sub / s["soft"]) / np.arcsinh(s["vmax"] / s["soft"]), 0, 1)
    r = ch("F444W")
    g = 0.5 * (ch("F277W") + ch("F150W"))
    b = ch("F115W")
    return np.stack([r, g, b], axis=-1)


def main():
    out_dir = Path(sys.argv[1])
    val_csv = Path(sys.argv[2])
    df = pd.read_csv(val_csv)
    npz_files = sorted((out_dir / "unified_npz").glob("*.npz"))
    contact_dir = out_dir / "contact_sheets"
    contact_dir.mkdir(exist_ok=True)

    stretch = compute_global_stretch(npz_files)
    print("Global stretch (fixed across whole batch):")
    for b, s in stretch.items():
        print(f"  {b}: black={s['black']:.4f} vmax={s['vmax']:.4f} soft={s['soft']:.4f}")

    n = len(npz_files)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    # --- Sheet 1: full RGB, fixed stretch ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i, f in enumerate(npz_files):
        ax = axes[i // ncols, i % ncols]
        d = np.load(f, allow_pickle=True)
        im = d["image_final"]
        rgb = apply_fixed_rgb({b: im[bi] for bi, b in enumerate(BANDS)}, stretch)
        ax.imshow(rgb, origin="lower")
        row = df[df["file"] == f.name]
        status = "PASS" if len(row) and bool(row.iloc[0]["pass"]) else "FAIL"
        ax.set_title(f"{f.stem}\n{status}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Full RGB (combined, noisy) -- one fixed stretch for the whole batch", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(contact_dir / "sheet1_full_rgb.png", dpi=130)
    plt.close(fig)

    # --- Sheet 2: arc-only RGB, fixed stretch ---
    arc_stretch = compute_global_stretch(npz_files, black_pct=1.0, vmax_pct=99.9, softening_frac=0.15)
    for f in npz_files:
        pass
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i, f in enumerate(npz_files):
        ax = axes[i // ncols, i % ncols]
        d = np.load(f, allow_pickle=True)
        im = d["image_sources_only"]
        rgb = apply_fixed_rgb({b: im[bi] for bi, b in enumerate(BANDS)}, arc_stretch)
        ax.imshow(rgb, origin="lower")
        ax.set_title(f.stem, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Arc-only (noiseless, lens-light-subtracted) -- one fixed stretch", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(contact_dir / "sheet2_arc_only_rgb.png", dpi=130)
    plt.close(fig)

    # --- Sheet 3: lens-only / arc-only / combined triplets ---
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    axes = np.atleast_2d(axes)
    for i, f in enumerate(npz_files):
        d = np.load(f, allow_pickle=True)
        lens_only = apply_fixed_rgb({b: d["image_lens_only"][bi] for bi, b in enumerate(BANDS)}, stretch)
        arc_only = apply_fixed_rgb({b: d["image_sources_only"][bi] for bi, b in enumerate(BANDS)}, arc_stretch)
        combined = apply_fixed_rgb({b: d["image_final"][bi] for bi, b in enumerate(BANDS)}, stretch)
        for j, (img, label) in enumerate([(lens_only, "lens-only"), (arc_only, "arc-only"), (combined, "combined")]):
            axes[i, j].imshow(img, origin="lower")
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(label, fontsize=10)
        axes[i, 0].set_ylabel(f.stem, fontsize=7)
    fig.suptitle("Lens-only / Arc-only / Combined diagnostic triplets", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(contact_dir / "sheet3_triplets.png", dpi=130)
    plt.close(fig)

    # --- Sheet 4: failed systems with reasons ---
    failed = df[~df["pass"]]
    if len(failed) > 0:
        nf = len(failed)
        ncols_f = min(5, nf)
        nrows_f = int(np.ceil(nf / ncols_f))
        fig, axes = plt.subplots(nrows_f, ncols_f, figsize=(3.2 * ncols_f, 3.6 * nrows_f))
        axes = np.atleast_2d(axes)
        for i, (_, row) in enumerate(failed.iterrows()):
            ax = axes[i // ncols_f, i % ncols_f]
            f = out_dir / "unified_npz" / row["file"]
            d = np.load(f, allow_pickle=True)
            im = d["image_final"]
            rgb = apply_fixed_rgb({b: im[bi] for bi, b in enumerate(BANDS)}, stretch)
            ax.imshow(rgb, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            reason = str(row["fail_reasons"])[:60]
            ax.set_xlabel(reason, fontsize=6.5, wrap=True)
            ax.set_title(row["file"], fontsize=7)
        for j in range(nf, nrows_f * ncols_f):
            axes[j // ncols_f, j % ncols_f].axis("off")
        fig.suptitle(f"Failed systems ({nf}/{len(df)}) with failure reasons", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(contact_dir / "sheet4_failed_systems.png", dpi=130)
        plt.close(fig)
        print(f"Saved sheet4 with {nf} failed systems")
    else:
        print("No failed systems -- sheet4 skipped")

    print(f"Contact sheets saved to {contact_dir}")


if __name__ == "__main__":
    main()
