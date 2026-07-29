#!/usr/bin/env python3
"""
Morphology method showcases — compare rendering backends before pipeline integration.

Produces individual 3x4 figures for each approach and one 2x2 comparison mosaic:

  1. GalSim single Sersic
  2. GalSim native multi-component (bulge/disk/bar/ring/…)
  3. Lenstronomy single Sersic  (current parametric baseline)
  4. Lenstronomy native multi-component
  5. GalaxyGenius SKIRT stamps (simulation-based)
  6. GalaxyGenius RGB (F070W/F150W/F444W)

All panels are normalized to uniform 256x256 squares for fair visual comparison.

Usage:
  conda run -n astro-clean python analysis/galaxy_morphology/plot_morphology_methods_showcase.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prism.morphology.galaxygenius_assets import (
    load_morphology_panel,
    load_morphology_rgb,
    morphology_order,
    panel_annotation,
)
from galsim_showcase import render_galsim_multicomponent, render_galsim_single
from lenstronomy_showcase import render_lenstronomy_multicomponent, render_lenstronomy_single
from showcase_common import (
    BAND,
    MORPH_ORDER,
    MORPH_PARAMS,
    PANEL_SIZE,
    render_grayscale_panel,
    render_rgb_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "analysis" / "galaxy_morphology" / "reports" / "figures"


def _load_morph_cfg():
    with open(REPO_ROOT / "configs" / "default_config.yaml") as f:
        full_cfg = yaml.safe_load(f)
    morph_cfg = dict(full_cfg.get("morphology", {}))
    morph_cfg["multicomponent_enabled"] = True
    return morph_cfg


def _make_grid_figure():
    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    return fig, axes.ravel()


def _render_method_showcase(render_fn, suptitle, out_name, *, context=None):
    fig, axes = _make_grid_figure()
    order = MORPH_ORDER
    for idx, morph_type in enumerate(order):
        params = MORPH_PARAMS[morph_type]
        title = f"{params['label']}\n({morph_type})"
        if context is not None:
            image, param_text = render_fn(morph_type, params, context, idx)
        else:
            image, param_text = render_fn(morph_type, params, idx)
        render_grayscale_panel(axes[idx], image, title, param_text)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.03, wspace=0.10, hspace=0.38)
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _render_galaxygenius_grayscale():
    fig, axes = _make_grid_figure()
    order = morphology_order()
    for idx, morph_type in enumerate(order):
        image, meta = load_morphology_panel(morph_type, figures_dir=FIGURES_DIR)
        params = MORPH_PARAMS[morph_type]
        title = f"{params['label']}\n({morph_type})"
        render_grayscale_panel(axes[idx], image, title, panel_annotation(meta))

    fig.suptitle(
        "GalaxyGenius morphology stamps (JWST F150W, SKIRT radiative transfer)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.03, wspace=0.10, hspace=0.38)
    out_path = FIGURES_DIR / "galaxygenius_morphology_showcase.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _render_galaxygenius_rgb():
    fig, axes = _make_grid_figure()
    order = morphology_order()
    for idx, morph_type in enumerate(order):
        rgb, meta = load_morphology_rgb(morph_type, figures_dir=FIGURES_DIR)
        params = MORPH_PARAMS[morph_type]
        title = f"{params['label']}\n({morph_type})"
        render_rgb_panel(axes[idx], rgb, title, panel_annotation(meta))

    fig.suptitle(
        "GalaxyGenius morphology stamps — JWST RGB (F070W / F150W / F444W)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.03, wspace=0.10, hspace=0.38)
    out_path = FIGURES_DIR / "galaxygenius_morphology_rgb_showcase.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _build_method_cache(morph_cfg, rng):
    """Pre-render all methods so the comparison mosaic reuses identical arrays."""
    cache = {}
    for morph_type in MORPH_ORDER:
        params = MORPH_PARAMS[morph_type]
        gg_image, gg_meta = load_morphology_panel(morph_type, figures_dir=FIGURES_DIR)
        cache[morph_type] = {
            "galsim_single": render_galsim_single(morph_type, params, 0),
            "galsim_multi": render_galsim_multicomponent(morph_type, params, morph_cfg, rng, 0),
            "lenstronomy_single": render_lenstronomy_single(morph_type, params, 0),
            "lenstronomy_multi": render_lenstronomy_multicomponent(morph_type, params, morph_cfg, rng, 0),
            "galaxygenius": (gg_image, panel_annotation(gg_meta)),
        }
    return cache


def _render_comparison_mosaic(cache):
    """2x2 mosaic: top=GalSim, bottom=Lenstronomy; left=single, right=multi + GG inset."""
    fig = plt.figure(figsize=(22, 18))
    outer = fig.add_gridspec(2, 2, wspace=0.12, hspace=0.16)

    blocks = [
        ("GalSim — single Sersic", "galsim_single"),
        ("GalSim — multi-component", "galsim_multi"),
        ("Lenstronomy — single Sersic", "lenstronomy_single"),
        ("Lenstronomy — multi-component", "lenstronomy_multi"),
    ]

    for block_idx, (block_title, method_key) in enumerate(blocks):
        row, col = divmod(block_idx, 2)
        inner = outer[row, col].subgridspec(3, 4, wspace=0.08, hspace=0.35)
        for morph_idx, morph_type in enumerate(MORPH_ORDER):
            ax = fig.add_subplot(inner[morph_idx // 4, morph_idx % 4])
            image, param_text = cache[morph_type][method_key]
            params = MORPH_PARAMS[morph_type]
            title = f"{params['label']}"
            render_grayscale_panel(ax, image, title, param_text)
        fig.text(
            0.25 + 0.5 * col, 0.97 - 0.48 * row, block_title,
            ha="center", va="top", fontsize=15, fontweight="bold",
        )

    fig.suptitle(
        "Morphology rendering methods comparison (F150W, uniform 256 px panels)\n"
        "GalaxyGenius stamps shown separately — see galaxygenius_*_showcase.png",
        fontsize=16, fontweight="bold", y=1.0,
    )
    out_path = FIGURES_DIR / "morphology_methods_comparison_galsim_lenstronomy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _render_row_comparison(cache):
    """12 rows x 5 columns: one morphology per row, one method per column."""
    methods = [
        ("GalSim\nsingle", "galsim_single"),
        ("GalSim\nmulti", "galsim_multi"),
        ("Lenstronomy\nsingle", "lenstronomy_single"),
        ("Lenstronomy\nmulti", "lenstronomy_multi"),
        ("GalaxyGenius\nstamp", "galaxygenius"),
    ]

    fig, axes = plt.subplots(len(MORPH_ORDER), len(methods), figsize=(16, 28))
    for row, morph_type in enumerate(MORPH_ORDER):
        params = MORPH_PARAMS[morph_type]
        for col, (col_title, method_key) in enumerate(methods):
            ax = axes[row, col]
            entry = cache[morph_type][method_key]
            image, param_text = entry
            title = params["label"] if col == 0 else ""
            if col == 0:
                ax.set_ylabel(morph_type, fontsize=9, rotation=90, labelpad=12)
            if row == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold")
            render_grayscale_panel(ax, image, title, param_text if col == len(methods) - 1 else None)

    fig.suptitle(
        "All morphology methods side-by-side (F150W / GalaxyGenius band, 256 px panels)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.02, wspace=0.06, hspace=0.12)
    out_path = FIGURES_DIR / "morphology_methods_comparison_all.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    morph_cfg = _load_morph_cfg()
    rng = np.random.default_rng(42)

    _render_method_showcase(
        render_galsim_single,
        f"GalSim single Sersic morphology ({BAND}, PSF FWHM=0.064\")",
        "galsim_single_morphology_showcase.png",
    )
    _render_method_showcase(
        lambda mt, p, ctx, idx: render_galsim_multicomponent(mt, p, ctx, rng, idx),
        "GalSim native multi-component morphology (bulge/disk/bar/ring, default_config.yaml)",
        "galsim_multicomponent_morphology_showcase.png",
        context=morph_cfg,
    )
    _render_method_showcase(
        render_lenstronomy_single,
        "Lenstronomy single Sersic morphology (parametric baseline)",
        "lenstronomy_single_morphology_showcase.png",
    )
    _render_method_showcase(
        lambda mt, p, ctx, idx: render_lenstronomy_multicomponent(mt, p, ctx, rng, idx),
        "Lenstronomy native multi-component morphology (pipeline build_light_model path)",
        "lenstronomy_multicomponent_morphology_showcase.png",
        context=morph_cfg,
    )

    _render_galaxygenius_grayscale()
    _render_galaxygenius_rgb()

    cache = _build_method_cache(morph_cfg, rng)
    _render_comparison_mosaic(cache)
    _render_row_comparison(cache)


if __name__ == "__main__":
    main()
