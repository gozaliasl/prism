"""Shared morphology showcase constants and panel rendering helpers."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import PowerNorm

PANEL_SIZE = 256
PIXEL_SCALE = 0.031  # JWST NIRCam short-wave (arcsec/pixel)
BAND = "F150W"
TOTAL_MAG = 21.0
PSF_FWHM = 0.064

MORPH_PARAMS = {
    "elliptical":    dict(n_sersic=4.2, q=0.75, R_eff=0.8,  label="Elliptical"),
    "s0":            dict(n_sersic=2.8, q=0.85, R_eff=0.6,  label="S0 / Early-type"),
    "spiral":        dict(n_sersic=1.5, q=0.70, R_eff=1.0,  label="Spiral"),
    "late_spiral":   dict(n_sersic=1.0, q=0.45, R_eff=1.2,  label="Late Spiral (Sc/Sd)"),
    "edge_on":       dict(n_sersic=1.0, q=0.20, R_eff=1.1,  label="Edge-on Disk"),
    "barred_spiral": dict(n_sersic=1.5, q=0.75, R_eff=1.0,  label="Barred Spiral"),
    "ring":          dict(n_sersic=2.0, q=0.85, R_eff=1.3,  label="Ring Galaxy"),
    "post_merger":   dict(n_sersic=3.2, q=0.65, R_eff=0.9,  label="Post-Merger"),
    "irregular":     dict(n_sersic=0.6, q=0.55, R_eff=0.9,  label="Irregular"),
    "primordial":    dict(n_sersic=0.3, q=0.60, R_eff=0.7,  label="Primordial"),
    "clumpy":        dict(n_sersic=0.5, q=0.35, R_eff=1.1,  label="Clumpy"),
    "starburst":     dict(n_sersic=0.6, q=0.50, R_eff=0.8,  label="Starburst"),
}

MORPH_ORDER = list(MORPH_PARAMS.keys())


def q_to_e1e2(q: float) -> tuple[float, float]:
    e = (1.0 - q) / (1.0 + q)
    return e, 0.0


def mag_to_flux(total_mag: float, zp: float = 28.0) -> float:
    return 10.0 ** ((zp - total_mag) / 2.5)


def normalize_panel(image: np.ndarray, target_size: int = PANEL_SIZE) -> np.ndarray:
    """Center-crop to square and resize to a uniform panel size."""
    from scipy.ndimage import zoom

    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        h, w, c = img.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        square = img[y0:y0 + side, x0:x0 + side]
        if side != target_size:
            square = zoom(square, (target_size / side, target_size / side, 1), order=1)
        return square

    h, w = img.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = img[y0:y0 + side, x0:x0 + side]
    if side != target_size:
        square = zoom(square, target_size / side, order=1)
    return square


def display_norm(image: np.ndarray, gamma: float = 0.45) -> PowerNorm:
    pos = np.clip(image, 0, None)
    vmax = max(float(np.percentile(pos, 99.5)), 1e-12)
    return PowerNorm(gamma=gamma, vmin=0, vmax=vmax)


def stretch_rgb(rgb: np.ndarray, percentile: float = 99.5, gamma: float = 0.85) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float64)
    for i in range(3):
        plane = np.clip(rgb[:, :, i], 0, None)
        scale = float(np.percentile(plane, percentile))
        if scale <= 0:
            scale = 1.0
        out[:, :, i] = np.clip(plane / scale, 0, 1) ** gamma
    return out


def render_grayscale_panel(ax, image, title, param_text=None, *, cmap="gray"):
    image = normalize_panel(image)
    norm = display_norm(image)
    ax.imshow(
        image, origin="lower", cmap=cmap, norm=norm,
        aspect="equal", interpolation="lanczos",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, PANEL_SIZE - 0.5)
    ax.set_ylim(-0.5, PANEL_SIZE - 0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if param_text:
        ax.text(
            0.03, 0.97, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
            verticalalignment="top", fontsize=6.5, family="monospace",
        )


def render_rgb_panel(ax, rgb, title, param_text=None):
    rgb = stretch_rgb(normalize_panel(rgb))
    ax.imshow(rgb, origin="lower", aspect="equal", interpolation="lanczos")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, PANEL_SIZE - 0.5)
    ax.set_ylim(-0.5, PANEL_SIZE - 0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if param_text:
        ax.text(
            0.03, 0.97, param_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
            verticalalignment="top", fontsize=6.5, family="monospace",
        )


def save_showcase_grid(
    fig,
    axes,
    order,
    render_fn,
    suptitle,
    out_path,
    *,
    context=None,
):
    """Fill a 3x4 axis grid and save."""
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
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
