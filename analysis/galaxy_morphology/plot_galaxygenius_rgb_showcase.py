#!/usr/bin/env python3
"""
3x4 RGB morphology showcase from GalaxyGenius JWST mock images (Zhou et al. 2025).

Each panel combines F070W (B), F150W (G), and F444W (R) from the same subhalo
row in the published combined figures (F444W stands in for F2100W in the paper RGB).

Usage:
  conda run -n astro-clean python analysis/galaxy_morphology/plot_galaxygenius_rgb_showcase.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.morphology.galaxygenius_assets import (
    load_catalog,
    load_morphology_rgb,
    morphology_order,
    panel_annotation,
    stretch_rgb,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "analysis" / "galaxy_morphology" / "reports" / "figures"
PANEL_SIZE = 256


def normalize_panel_rgb(rgb: np.ndarray, target_size: int = PANEL_SIZE) -> np.ndarray:
    """Center-crop to square and resize all channels identically."""
    from scipy.ndimage import zoom

    h, w, _ = rgb.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = rgb[y0:y0 + side, x0:x0 + side]
    if side != target_size:
        scale = target_size / side
        square = zoom(square, (scale, scale, 1), order=1)
    return square


def render_panel(ax, rgb, title, param_text=None):
    rgb = stretch_rgb(normalize_panel_rgb(rgb))
    ax.imshow(
        rgb, origin="lower", aspect="equal", interpolation="lanczos",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, PANEL_SIZE - 0.5)
    ax.set_ylim(-0.5, PANEL_SIZE - 0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if param_text:
        ax.text(
            0.03, 0.97, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
            verticalalignment="top", fontsize=7.5, family="monospace",
        )


def main():
    catalog = load_catalog()
    order = morphology_order(catalog)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3, 4, figsize=(18, 13.5),
        gridspec_kw={"wspace": 0.08, "hspace": 0.35},
    )
    axes = axes.ravel()

    for idx, morph_type in enumerate(order):
        rgb, meta = load_morphology_rgb(morph_type, figures_dir=FIGURES_DIR)
        title = f"{meta['label']}\n({morph_type})"
        param_text = panel_annotation(meta)
        render_panel(axes[idx], rgb, title, param_text)

    fig.suptitle(
        "GalaxyGenius morphology models — JWST RGB (F070W / F150W / F444W, "
        "Zhou et al. 2025)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.03, wspace=0.10, hspace=0.38)
    out_path = FIGURES_DIR / "galaxygenius_morphology_rgb_showcase.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
