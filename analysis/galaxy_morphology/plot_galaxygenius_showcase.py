#!/usr/bin/env python3
"""
3x4 morphology showcase using GalaxyGenius mock JWST images (Zhou et al. 2025).

Unlike the parametric GALFIT/lenstronomy showcase, each panel is cropped from
published SKIRT + mock-observation outputs (IllustrisTNG / EAGLE subhalos from
arXiv:2506.15060, Figure 6 and Figure 10b).

Usage:
  conda run -n astro-clean python analysis/galaxy_morphology/plot_galaxygenius_showcase.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.morphology.galaxygenius_assets import (
    load_catalog,
    load_morphology_panel,
    morphology_order,
    panel_annotation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "analysis" / "galaxy_morphology" / "reports" / "figures"
PANEL_SIZE = 256  # uniform square output (pixels) for every morphology panel


def normalize_panel(image: np.ndarray, target_size: int = PANEL_SIZE) -> np.ndarray:
    """Center-crop to square, then resize so every panel has identical shape."""
    from scipy.ndimage import zoom

    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = img[y0:y0 + side, x0:x0 + side]
    if side != target_size:
        square = zoom(square, target_size / side, order=1)
    return square


def _display_norm(image: np.ndarray) -> PowerNorm:
    """Percentile stretch on published log-display panels."""
    pos = np.clip(image, 0, None)
    vmax = max(float(np.percentile(pos, 99.5)), 1.0)
    return PowerNorm(gamma=0.45, vmin=0, vmax=vmax)


def render_panel(ax, image, title, param_text=None):
    image = normalize_panel(image)
    norm = _display_norm(image)
    ax.imshow(
        image, origin="lower", cmap="gray", norm=norm,
        aspect="equal", interpolation="lanczos",
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
        image, meta = load_morphology_panel(morph_type, figures_dir=FIGURES_DIR)
        title = f"{meta['label']}\n({morph_type})"
        param_text = panel_annotation(meta)
        render_panel(axes[idx], image, title, param_text)

    fig.suptitle(
        "GalaxyGenius morphology models (JWST NIRCam, SKIRT radiative transfer, "
        "Zhou et al. 2025)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.03, wspace=0.10, hspace=0.38)
    out_path = FIGURES_DIR / "galaxygenius_morphology_showcase.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
