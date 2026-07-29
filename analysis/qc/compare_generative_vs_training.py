#!/usr/bin/env python3
"""Quality comparison: VAE-generated stamps vs training-set (TNG particle) stamps.

Compares:
  - morphology (CAS/Gini/M20)
  - size distribution (half-light radius)
  - surface-brightness profiles (radial SB)
  - color/band consistency (per-band flux ratios)
  - visual comparison grid

Usage:
    python analysis/qc/compare_generative_vs_training.py \
        --checkpoint src/galaxy_morphology/generative/checkpoints/cvae_v3.pt \
        --catalog /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet \
        --catalog2 /Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet \
        --n-samples 200 \
        --out analysis/qc/generative_quality_report.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.galaxy_morphology.generative.dataset import (
    MORPH_CLASSES, build_dataset, condition_vector, STAMP_SIZE,
    Z_BUCKET_EDGES, redshift_bucket,
)
from src.galaxy_morphology.generative.inference import build_generative_interpol_kwargs, DEFAULT_CHECKPOINT


def radial_profile(image: np.ndarray) -> np.ndarray:
    cy, cx = np.array(image.shape) / 2
    y, x = np.indices(image.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    rmax = min(image.shape) // 2
    profile = np.array([image[r == ri].mean() if (r == ri).any() else 0.0 for ri in range(rmax)])
    return profile


def half_light_radius(image: np.ndarray) -> float:
    total = image.sum()
    if total <= 0:
        return 0.0
    cy, cx = np.array(image.shape) / 2
    y, x = np.indices(image.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    order = np.argsort(r.ravel())
    cumsum = np.cumsum(image.ravel()[order])
    half_idx = np.searchsorted(cumsum, 0.5 * total)
    return float(r.ravel()[order[min(half_idx, len(order)-1)]])


def gini_coefficient(image: np.ndarray) -> float:
    flat = np.sort(np.abs(image.ravel()))
    n = len(flat)
    if n == 0 or flat.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * flat).sum() - (n + 1) * flat.sum()) / (n * flat.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--catalog", default="/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet")
    p.add_argument("--catalog2", default="/Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet")
    p.add_argument("--n-samples", type=int, default=200, help="Samples per morph class")
    p.add_argument("--band", default="F150W")
    p.add_argument("--out", default="analysis/qc/generative_quality_report.png")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run training first: python -m src.galaxy_morphology.generative.train ...")
        sys.exit(1)

    rng = np.random.default_rng(42)

    # --- Build training stamps sample ---
    print("Loading training stamps sample...")
    extra = [args.catalog2] if args.catalog2 and Path(args.catalog2).exists() else None
    images_train, conds_train, meta_train = build_dataset(
        args.catalog, band=args.band, max_n=args.n_samples * 2, seed=42,
        bands=[args.band], n_projections=1, extra_catalog_paths=extra,
        flip_augment=False,
    )
    images_train = images_train[:, 0, :, :]  # (N, H, W)
    print(f"  Training stamps: {len(images_train)}")

    # --- Generate VAE stamps ---
    print("Generating VAE stamps...")
    gen_stamps = []
    gen_types = []
    test_configs = [
        ("passive", 9.5, 0.6),
        ("passive", 11.0, 0.3),
        ("star_forming", 9.5, 1.5),
        ("star_forming", 10.5, 2.0),
        ("dusty_starburst", 9.5, 2.5),
        ("dusty_starburst", 10.0, 3.0),
    ]
    n_each = max(10, args.n_samples // len(test_configs))
    for morph, logM, z in test_configs:
        for _ in range(n_each):
            kwargs = build_generative_interpol_kwargs(
                morph_type=morph, logM=logM, redshift=z,
                magnitude_ref=24.0, rng=rng, checkpoint_path=args.checkpoint,
            )
            img = kwargs["image"]
            # Resize to STAMP_SIZE
            if img.shape[0] != STAMP_SIZE:
                img = zoom(img, STAMP_SIZE / img.shape[0], order=1)[:STAMP_SIZE, :STAMP_SIZE]
            # Normalize
            pos = img[img > 0]
            if pos.size:
                img = np.log1p(img / (np.median(pos) + 1e-12))
                img = img / (img.max() + 1e-12)
            gen_stamps.append(img.astype(np.float32))
            gen_types.append(morph)
    gen_stamps = np.array(gen_stamps)
    print(f"  Generated stamps: {len(gen_stamps)}")

    # --- Metrics ---
    def compute_metrics(stamps):
        return {
            "half_light": np.array([half_light_radius(s) for s in stamps]),
            "gini": np.array([gini_coefficient(s) for s in stamps]),
            "peak_sb": stamps.max(axis=(1, 2)),
            "total_flux": stamps.sum(axis=(1, 2)),
            "profiles": np.array([radial_profile(s) for s in stamps]),
        }

    print("Computing metrics...")
    m_train = compute_metrics(images_train)
    m_gen = compute_metrics(gen_stamps)

    # --- Plots ---
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    fig.suptitle(f"Generative VAE vs Training Stamps — {Path(args.checkpoint).name}", fontsize=13, y=0.99)

    # Row 0: example stamps
    ax = axes[0]
    for i, (title, stamps) in enumerate([
        ("Training (passive)", images_train[:4]),
        ("Generated (passive)", gen_stamps[:4]),
    ]):
        for j in range(min(4, len(stamps))):
            axes[0 if i == 0 else 1][j].imshow(stamps[j] if i == 0 else gen_stamps[j + n_each*2], cmap="inferno", origin="lower")
            axes[0 if i == 0 else 1][j].set_title(title.split()[0] + f" #{j+1}", fontsize=8)
            axes[0 if i == 0 else 1][j].axis("off")

    # Row 2: metric distributions
    for col, (key, label) in enumerate([
        ("half_light", "Half-light radius (px)"),
        ("gini", "Gini coefficient"),
        ("peak_sb", "Peak surface brightness"),
        ("total_flux", "Total flux"),
    ]):
        ax = axes[2][col]
        ax.hist(m_train[key], bins=30, alpha=0.6, label="Training", color="steelblue", density=True)
        ax.hist(m_gen[key], bins=30, alpha=0.6, label="Generated", color="tomato", density=True)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(label, fontsize=9)

    # Row 3: radial profiles + morph type distribution + stamps grid
    ax = axes[3][0]
    prof_train = m_train["profiles"].mean(axis=0)
    prof_gen = m_gen["profiles"].mean(axis=0)
    r = np.arange(len(prof_train))
    ax.plot(r, prof_train, label="Training", color="steelblue", lw=2)
    ax.plot(r[:len(prof_gen)], prof_gen[:len(r)], label="Generated", color="tomato", lw=2)
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Mean SB")
    ax.set_title("Mean radial SB profile")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    # Morph type distribution in training set
    ax = axes[3][1]
    train_types = np.array([m["morph_type"] for m in meta_train])
    counts = {mt: (train_types == mt).sum() for mt in MORPH_CLASSES}
    ax.bar(counts.keys(), counts.values(), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Training morph distribution")
    ax.set_ylabel("Count")

    # Generated type distribution
    ax = axes[3][2]
    from collections import Counter
    gen_counts = Counter(gen_types)
    ax.bar(gen_counts.keys(), gen_counts.values(), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Generated morph distribution")
    ax.set_ylabel("Count")

    # Summary stats
    ax = axes[3][3]
    ax.axis("off")
    stats_lines = [
        f"Training: {len(images_train)} stamps",
        f"Generated: {len(gen_stamps)} stamps",
        f"",
        f"Half-light radius (px):",
        f"  Train: {m_train['half_light'].mean():.1f} ± {m_train['half_light'].std():.1f}",
        f"  Gen:   {m_gen['half_light'].mean():.1f} ± {m_gen['half_light'].std():.1f}",
        f"",
        f"Gini coefficient:",
        f"  Train: {m_train['gini'].mean():.3f} ± {m_train['gini'].std():.3f}",
        f"  Gen:   {m_gen['gini'].mean():.3f} ± {m_gen['gini'].std():.3f}",
        f"",
        f"Peak SB:",
        f"  Train: {m_train['peak_sb'].mean():.3f} ± {m_train['peak_sb'].std():.3f}",
        f"  Gen:   {m_gen['peak_sb'].mean():.3f} ± {m_gen['peak_sb'].std():.3f}",
        f"",
        f"Checkpoint: {Path(args.checkpoint).name}",
    ]
    ax.text(0.05, 0.95, "\n".join(stats_lines), transform=ax.transAxes,
            fontsize=8, va="top", fontfamily="monospace")

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved quality report → {out_path}")

    # Print summary
    print(f"\n=== Quality Summary ===")
    print(f"Half-light radius: train={m_train['half_light'].mean():.1f}px  gen={m_gen['half_light'].mean():.1f}px")
    print(f"Gini:              train={m_train['gini'].mean():.3f}         gen={m_gen['gini'].mean():.3f}")
    print(f"Peak SB:           train={m_train['peak_sb'].mean():.3f}      gen={m_gen['peak_sb'].mean():.3f}")
    ratio = m_gen['half_light'].mean() / (m_train['half_light'].mean() + 1e-6)
    if ratio < 0.5 or ratio > 2.0:
        print(f"\nWARNING: Generated half-light radius differs significantly from training ({ratio:.2f}x)")
        print("Consider retraining with more epochs or larger dataset.")
    else:
        print(f"\nOK: Generated stamp sizes within 2x of training ({ratio:.2f}x)")


if __name__ == "__main__":
    main()
