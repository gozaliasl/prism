#!/usr/bin/env python3
"""
Strong-lens showcase: GalaxyGenius morphology stamps as deflector light.

Each panel shows the same SIE+shear mass model and lensed source, but the
central lens galaxy light comes from the GalaxyGenius SKIRT mock images used in
galaxygenius_morphology_showcase.png (12 morphological types).

Usage:
  conda run -n astro-clean python analysis/galaxy_morphology/plot_galaxygenius_lens_showcase.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prism.morphology.galaxygenius_assets import morphology_order, panel_annotation
from prism.morphology.lens_stamp_sim import LensStampSimConfig, simulate_galaxygenius_lens_system
from showcase_common import MORPH_PARAMS, PANEL_SIZE

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "analysis" / "galaxy_morphology" / "reports" / "figures"

SIM_CFG = LensStampSimConfig(
    numpix=300,
    pixel_scale=0.031,
    theta_E=1.15,
    lens_q=0.75,
    lens_pa_deg=30.0,
    lens_mag=21.0,
    lens_re_arcsec=0.85,
    source_mag=22.0,
)


def _crop_center(image: np.ndarray, size: int = PANEL_SIZE) -> np.ndarray:
    from scipy.ndimage import zoom

    h, w = image.shape
    side = min(h, w, size * 2)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    patch = image[y0:y0 + side, x0:x0 + side]
    if patch.shape[0] != size:
        patch = zoom(patch, size / patch.shape[0], order=1)
    return patch


def render_panel(ax, image, title, param_text=None):
    image = _crop_center(image)
    pos = np.clip(image, 0, None)
    vmax = max(float(np.percentile(pos, 99.7)), 1e-12)
    norm = PowerNorm(gamma=0.42, vmin=0, vmax=vmax)
    fov = SIM_CFG.numpix * SIM_CFG.pixel_scale / 2.0
    ax.imshow(
        image,
        origin="lower",
        cmap="magma",
        norm=norm,
        extent=[-fov, fov, -fov, fov],
        aspect="equal",
        interpolation="lanczos",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("arcsec", fontsize=8)
    ax.set_ylabel("arcsec", fontsize=8)
    ax.tick_params(labelsize=7)
    if param_text:
        ax.text(
            0.03, 0.97, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
            verticalalignment="top", fontsize=6.5, family="monospace",
        )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = morphology_order()

    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    axes = axes.ravel()

    for idx, morph_type in enumerate(order):
        params = MORPH_PARAMS[morph_type]
        # Edge-on lenses use stronger flattening; ETGs use catalog q
        lens_q = params["q"] if morph_type == "edge_on" else min(params["q"], SIM_CFG.lens_q)
        cfg = replace(SIM_CFG, lens_q=lens_q, lens_re_arcsec=params["R_eff"])
        image, meta = simulate_galaxygenius_lens_system(
            morph_type, cfg, figures_dir=FIGURES_DIR,
        )
        title = f"{params['label']}\n({morph_type})"
        lines = [
            panel_annotation(meta),
            "lenstronomy INTERPOL lens light",
            f"θ_E={cfg.theta_E:.2f}\"  lens mag={cfg.lens_mag:.1f}",
            "single ImageModel.image() pass",
        ]
        render_panel(axes[idx], image, title, "\n".join(lines))

    fig.suptitle(
        "GalaxyGenius lens galaxies via lenstronomy INTERPOL + lensed source (F150W)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.05, wspace=0.18, hspace=0.32)
    out_path = FIGURES_DIR / "galaxygenius_lens_morphology_showcase.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
