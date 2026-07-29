#!/usr/bin/env python3
"""
PRISM Kappa Map Output Module

Computes and saves convergence (κ), shear (γ), magnification (μ), and
flexion (F, G) maps for gravitational lens systems in the PRISM pipeline.

Functions:
  - compute_kappa_products()  : Compute all lensing quantities on a grid
  - save_kappa_outputs()      : Save NPY, NPZ, and publication-quality JPGs
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter

# Suppress matplotlib and lenstronomy warnings during computation
warnings.filterwarnings("ignore", category=UserWarning)


def _setup_logger(out_dir: str) -> logging.Logger:
    """Setup logger for kappa errors."""
    log_path = Path(out_dir) / "kappa_errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("prism_kappa")
    if logger.hasHandlers():
        logger.handlers.clear()
    
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    return logger


def _choose_extended_grid(native_delta_pix: float, fov_arcmin: float, max_grid: int = 1200) -> Tuple[float, int]:
    """Pick a pixel scale/grid size for an extended-FOV lensing map that covers
    ``fov_arcmin`` without exceeding ``max_grid`` pixels per side.

    Lensing quantities (kappa/gamma/flexion) for analytic profiles vary smoothly
    on scales far larger than a typical detector pixel, so there is no need to
    sample an extended map at the native (often sub-pixel-arcsec) image
    resolution out to a 1' field -- doing so would scale compute cost with the
    full pixel count of e.g. a JWST-resolution grid (millions of pixels) for no
    physical benefit. This coarsens delta_pix only as much as needed to keep
    the grid at or below ``max_grid`` pixels per side; native resolution is
    kept if it already fits.
    """
    target_num_pix = fov_arcmin * 60.0 / native_delta_pix
    if target_num_pix <= max_grid:
        return native_delta_pix, int(round(target_num_pix))
    delta_pix_ext = fov_arcmin * 60.0 / max_grid
    return delta_pix_ext, max_grid


def compute_kappa_products(
    lens_model,
    kwargs_lens: List[Dict],
    num_pix: int = 300,
    delta_pix: float = 0.031,
    compute_flexion: bool = True,
    extended_fov_arcmin: Optional[float] = None,
    extended_max_grid: int = 1200,
) -> Dict[str, np.ndarray | float | list]:
    """
    Compute convergence, shear, magnification, deflection, and flexion on a grid.

    Args:
        lens_model: lenstronomy LensModel object
        kwargs_lens: list of parameter dicts for lens profiles
        num_pix: image size in pixels
        delta_pix: pixel scale in arcsec/pixel
        compute_flexion: if True, compute first (F) and second (G) flexion maps
        extended_fov_arcmin: if set, also compute lensing maps on a second,
            larger grid covering this FOV in arcmin (e.g. 1.0 -> 1' x 1'),
            independent of the telescope's native image pixel scale. The
            extended grid's pixel scale is auto-coarsened (see
            ``_choose_extended_grid``) so compute cost stays bounded by
            ``extended_max_grid`` pixels per side regardless of how fine
            ``delta_pix`` is. Results stored with 'ext_' prefix in the
            returned dict, plus 'ext_delta_pix'/'ext_fov_arcmin'/'ext_num_pix'
            recording the grid actually used.
        extended_max_grid: maximum pixels per side for the extended grid.

    Returns:
        Dict with keys:
          - 'kappa': 2D convergence map [num_pix, num_pix]
          - 'gamma1', 'gamma2': shear components
          - 'gamma_mag': |gamma| magnitude
          - 'mu': magnification map (clipped ±1000)
          - 'alpha_x', 'alpha_y': deflection angles in arcsec
          - 'theta_E_eff': effective Einstein radius (arcsec)
          - 'kappa_max': peak convergence
          - 'mu_max': peak magnification (after clipping)
          - 'critical_area': fraction of pixels with kappa >= 1
          - 'extent': [xmin, xmax, ymin, ymax] in arcsec
          If compute_flexion=True:
          - 'F1', 'F2': first flexion components (arcsec^-1)
          - 'G1', 'G2': second flexion components (arcsec^-1)
          - 'F_mag', 'G_mag': flexion magnitudes
          - 'F_mag_max', 'F_mag_mean', 'G_mag_max', 'G_mag_mean': scalar summaries
          - 'gamma_mag_max', 'gamma_mag_mean', 'kappa_mean': scalar summaries
          If extended_fov_arcmin is set:
          - 'ext_*': all of the above on the extended grid
          - 'ext_delta_pix', 'ext_fov_arcmin', 'ext_num_pix': grid actually used
    """
    result = _compute_on_grid(lens_model, kwargs_lens, num_pix, delta_pix, compute_flexion)

    if extended_fov_arcmin is not None:
        ext_delta_pix, ext_num_pix = _choose_extended_grid(delta_pix, extended_fov_arcmin, extended_max_grid)
        ext_result = _compute_on_grid(
            lens_model, kwargs_lens, ext_num_pix, ext_delta_pix, compute_flexion
        )
        for k, v in ext_result.items():
            result[f"ext_{k}"] = v
        result["ext_delta_pix"] = ext_delta_pix
        result["ext_fov_arcmin"] = extended_fov_arcmin
        result["ext_num_pix"] = ext_num_pix

    return result


def _compute_on_grid(
    lens_model,
    kwargs_lens: List[Dict],
    num_pix: int,
    delta_pix: float,
    compute_flexion: bool,
) -> Dict:
    """Compute all lensing products on a single grid."""
    # Create coordinate grid centered on origin, in arcsec
    pixel_coords = np.arange(num_pix) - num_pix / 2.0
    x_arcsec = pixel_coords * delta_pix
    y_arcsec = pixel_coords * delta_pix
    xx, yy = np.meshgrid(x_arcsec, y_arcsec)

    # Compute lensing quantities
    kappa = lens_model.kappa(xx, yy, kwargs_lens).astype(np.float32)
    gamma1, gamma2 = lens_model.gamma(xx, yy, kwargs_lens)
    gamma1, gamma2 = gamma1.astype(np.float32), gamma2.astype(np.float32)

    # Clip extreme shear values (numerical instability at boundaries)
    # Typical shear is 0-1, clip outliers to prevent visualization artifacts
    gamma1_max = np.percentile(np.abs(gamma1[~np.isinf(gamma1)]), 99.9) if np.any(~np.isinf(gamma1)) else 1.0
    gamma2_max = np.percentile(np.abs(gamma2[~np.isinf(gamma2)]), 99.9) if np.any(~np.isinf(gamma2)) else 1.0
    clip_val = max(1.0, gamma1_max, gamma2_max)
    gamma1 = np.clip(gamma1, -clip_val, clip_val)
    gamma2 = np.clip(gamma2, -clip_val, clip_val)

    gamma_mag = np.sqrt(gamma1**2 + gamma2**2).astype(np.float32)

    # Deflection angles
    alpha_x, alpha_y = lens_model.alpha(xx, yy, kwargs_lens)
    alpha_x, alpha_y = alpha_x.astype(np.float32), alpha_y.astype(np.float32)

    # Magnification from Jacobian: μ = 1 / [(1 - κ - γ) × (1 - κ + γ)]
    #                             = 1 / [(1 - κ)^2 - γ^2]
    denominator = (1.0 - kappa)**2 - (gamma1**2 + gamma2**2)
    denominator = np.clip(denominator, 1e-6, None)  # Avoid division by zero
    mu = (1.0 / denominator).astype(np.float32)
    mu = np.clip(mu, -1000.0, 1000.0)  # Clip extreme values

    # Find effective Einstein radius: radius where mean(kappa) = 1 within aperture
    theta_E_eff = _compute_theta_E_effective(kappa, delta_pix)

    # Summary statistics
    kappa_max = float(np.max(kappa))
    mu_max = float(np.max(np.abs(mu)))
    critical_area = float(np.sum(kappa >= 1.0) / (num_pix**2))

    # Extent for imshow in arcsec
    half_fov = (num_pix / 2.0) * delta_pix
    extent = [-half_fov, half_fov, -half_fov, half_fov]

    result = {
        "kappa": kappa,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "gamma_mag": gamma_mag,
        "mu": mu,
        "alpha_x": alpha_x,
        "alpha_y": alpha_y,
        "theta_E_eff": theta_E_eff,
        "kappa_max": kappa_max,
        "kappa_mean": float(np.mean(kappa)),
        "mu_max": mu_max,
        "gamma_mag_max": float(np.max(gamma_mag)),
        "gamma_mag_mean": float(np.mean(gamma_mag)),
        "critical_area": critical_area,
        "extent": extent,
    }

    if compute_flexion:
        flexion = _compute_flexion_maps(alpha_x, alpha_y, delta_pix)
        flexion["F_mag_max"] = float(np.max(flexion["F_mag"]))
        flexion["F_mag_mean"] = float(np.mean(flexion["F_mag"]))
        flexion["G_mag_max"] = float(np.max(flexion["G_mag"]))
        flexion["G_mag_mean"] = float(np.mean(flexion["G_mag"]))
        result.update(flexion)

    return result


def _compute_flexion_maps(
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    delta_pix: float,
) -> Dict[str, np.ndarray]:
    """
    Compute first (F) and second (G) flexion from deflection angle maps.

    Flexion definitions (spin-1 F and spin-3 G):
      F1 = d(alpha_x)/dx,  F2 = d(alpha_y)/dx  (= d(alpha_x)/dy by symmetry)
      G1 = d(alpha_x)/dx - d(alpha_y)/dy
      G2 = d(alpha_x)/dy + d(alpha_y)/dx

    All quantities in arcsec^-1. np.gradient uses central differences
    internally, so boundary pixels use one-sided differences automatically.

    Note: the x-axis in the lensing grid corresponds to axis=1 (columns)
    and y-axis to axis=0 (rows), matching the meshgrid convention in
    _compute_on_grid where xx varies along columns.
    """
    # Gradient w.r.t. x (columns, axis=1) and y (rows, axis=0)
    dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x.astype(np.float64), delta_pix)
    dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y.astype(np.float64), delta_pix)

    # First flexion F (spin-1): gradient of convergence = divergence of alpha / 2
    F1 = (dalpha_x_dx + dalpha_y_dy).astype(np.float32) * 0.5  # = d(kappa)/dx
    F2 = (dalpha_x_dy + dalpha_y_dx).astype(np.float32) * 0.5  # = d(kappa)/dy
    F_mag = np.sqrt(F1**2 + F2**2).astype(np.float32)

    # Second flexion G (spin-3): curl-like combination
    G1 = (dalpha_x_dx - dalpha_y_dy).astype(np.float32)
    G2 = (dalpha_x_dy + dalpha_y_dx).astype(np.float32)
    G_mag = np.sqrt(G1**2 + G2**2).astype(np.float32)

    return {
        "F1": F1,
        "F2": F2,
        "F_mag": F_mag,
        "G1": G1,
        "G2": G2,
        "G_mag": G_mag,
    }


def _compute_theta_E_effective(kappa: np.ndarray, delta_pix: float) -> float:
    """
    Find effective Einstein radius as the radius within which the mean
    (enclosed) convergence equals 1 -- the standard definition, equivalent to
    the enclosed mass equalling the critical mass pi * theta_E^2 * Sigma_cr.

    Returns:
        theta_E_eff in arcsec, or -1.0 if not found within grid.
    """
    num_pix = kappa.shape[0]
    max_radius_pix = num_pix / 2.0

    center = (num_pix / 2.0, num_pix / 2.0)
    yy, xx = np.meshgrid(np.arange(num_pix), np.arange(num_pix))
    rr_pix = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)

    # mean_enclosed(kappa)(r) = (1/(pi r^2)) * integral_{r'<r} kappa dA is
    # monotonically decreasing from infinity (centrally cuspy profile) to 0
    # at large r, so there is exactly one crossing of mean_enclosed = 1 --
    # that crossing radius is theta_E by definition. (Note: the *local*
    # annulus-averaged kappa crosses 1 at r = theta_E/2 for an SIS/SIE, not
    # at theta_E -- using the enclosed mean is essential for correctness.)
    # Exclude the pixel(s) sitting exactly on the lens center: a centrally
    # cuspy profile sampled at r=0 gives a numerically near-infinite kappa
    # (a discretization artifact, not a physical value), which would swamp
    # the cumulative sum at every radius.
    nonzero = rr_pix.ravel() > 0
    rr_nz = rr_pix.ravel()[nonzero]
    kappa_nz = kappa.ravel()[nonzero]

    order = np.argsort(rr_nz)
    rr_sorted = rr_nz[order]
    kappa_sorted = kappa_nz[order]
    cum_sum = np.cumsum(kappa_sorted)
    n = np.arange(1, len(kappa_sorted) + 1)
    mean_enclosed = cum_sum / n  # mean kappa over all pixels with r' <= r

    valid = rr_sorted <= (max_radius_pix - 2.0)
    rr_valid = rr_sorted[valid]
    mean_valid = mean_enclosed[valid]

    below_one = np.where(mean_valid < 1.0)[0]
    if len(below_one) == 0:
        return -1.0  # mean(kappa) never drops below 1 within the grid

    idx = below_one[0]
    if idx == 0:
        return -1.0  # mean(kappa) < 1 even at the innermost pixel

    return float(rr_valid[idx] * delta_pix)


def save_kappa_outputs(
    kappa_dict: Dict,
    out_dir: str,
    lens_id: str,
    category: str,
    sub_type: str,
) -> bool:
    """
    Save kappa outputs: NPY, NPZ, single-panel JPG, and 4-panel diagnostic.
    
    Args:
        kappa_dict: output of compute_kappa_products()
        out_dir: output directory path
        lens_id: unique identifier (e.g., 'lens_00042')
        category: 'single', 'pair', or 'group'
        sub_type: e.g., 'EPL+SHEAR', 'SIE+SIE', 'NFW+SHEAR'
    
    Returns:
        True if successful, False if error (logged to kappa_errors.log)
    """
    logger = _setup_logger(out_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # A) Save raw kappa NPY
        npy_path = out_path / f"{lens_id}_kappa.npy"
        np.save(npy_path, kappa_dict["kappa"])
        
        # B) Save comprehensive NPZ
        npz_dict = {
            k: v for k, v in kappa_dict.items()
            if isinstance(v, (np.ndarray, float, int, list))
        }
        npz_dict.update({
            "lens_id": lens_id,
            "category": category,
            "sub_type": sub_type,
        })
        npz_path = out_path / f"{lens_id}_kappa_data.npz"
        np.savez_compressed(npz_path, **npz_dict)
        
        # C) Single-panel kappa JPG
        _save_kappa_jpg(kappa_dict, out_path, lens_id, category, sub_type)
        
        # D) 4-panel diagnostic JPG
        _save_kappa_panel_jpg(kappa_dict, out_path, lens_id, category, sub_type)

        # E) Flexion panel JPG (if computed)
        if "F1" in kappa_dict:
            _save_flexion_panel_jpg(kappa_dict, out_path, lens_id, category, sub_type)

        # F) Extended-FOV kappa + flexion panel JPGs (if computed)
        if "ext_kappa" in kappa_dict:
            fov_tag = f"{kappa_dict.get('ext_fov_arcmin', '?')}arcmin"
            ext_panel_dict = {k[4:]: v for k, v in kappa_dict.items()
                               if k.startswith("ext_") and isinstance(v, (np.ndarray, list, float))}
            _save_kappa_panel_jpg(ext_panel_dict, out_path, f"{lens_id}_ext{fov_tag}", category, sub_type)
            if "ext_F1" in kappa_dict:
                _save_flexion_panel_jpg(
                    ext_panel_dict, out_path, f"{lens_id}_ext{fov_tag}", category, sub_type,
                )

        return True
        
    except Exception as e:
        logger.error(f"{lens_id}: {category}/{sub_type} — {type(e).__name__}: {e}")
        return False


def _save_kappa_jpg(
    kappa_dict: Dict,
    out_path: Path,
    lens_id: str,
    category: str,
    sub_type: str,
) -> None:
    """Save single-panel publication-quality kappa map."""
    from PIL import Image
    import io
    
    kappa = kappa_dict["kappa"]
    gamma1 = kappa_dict["gamma1"]
    gamma2 = kappa_dict["gamma2"]
    extent = kappa_dict["extent"]
    theta_E_eff = kappa_dict["theta_E_eff"]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # Colormap: vmin=0, vmax=max(2.0, 95th percentile)
    vmax = max(2.0, np.percentile(kappa, 95))
    im = ax.imshow(kappa, extent=extent, origin="lower", cmap="inferno",
                   vmin=0, vmax=vmax)

    # Genuine critical curve: zero-Jacobian locus det(A) = (1-kappa)^2 - |gamma|^2 = 0
    # (equivalently 1/mu = 0). This is the physically correct critical curve and can
    # differ in topology from the kappa=1 contour (e.g. it also traces radial critical
    # curves for elliptical/multi-component lenses, which kappa=1 does not capture).
    det_A = (1.0 - kappa) ** 2 - (gamma1 ** 2 + gamma2 ** 2)
    ax.contour(det_A, levels=[0.0], extent=extent, origin="lower",
               colors="cyan", linewidths=1.5)
    # Kappa=1 contour shown for reference only (not the true critical curve)
    ax.contour(kappa, levels=[1.0], extent=extent, origin="lower",
               colors="white", linewidths=1.0, linestyles="dashed", alpha=0.6)
    
    # Labels and title
    ax.set_xlabel("RA offset (arcsec)", fontsize=11)
    ax.set_ylabel("Dec offset (arcsec)", fontsize=11)
    ax.set_title(f"{lens_id} | {category} | {sub_type}", fontsize=12, fontweight="bold")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="κ (convergence)")
    
    # Annotate theta_E_eff in corner
    if theta_E_eff > 0:
        ax.text(0.05, 0.05, f"θ_E = {theta_E_eff:.2f}\"",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                color="white")
    
    plt.tight_layout()
    
    # Save as PNG first, then convert to JPG with PIL quality control
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format="png", bbox_inches="tight", dpi=150)
    png_buffer.seek(0)
    plt.close(fig)
    
    # Convert PNG to JPG with quality control
    img_png = Image.open(png_buffer)
    if img_png.mode == "RGBA":
        # Convert RGBA to RGB
        rgb_img = Image.new("RGB", img_png.size, (255, 255, 255))
        rgb_img.paste(img_png, mask=img_png.split()[3] if len(img_png.split()) > 3 else None)
        img_png = rgb_img
    
    jpg_path = out_path / f"{lens_id}_kappa.jpg"
    img_png.save(jpg_path, format="JPEG", quality=92)


def _save_kappa_panel_jpg(
    kappa_dict: Dict,
    out_path: Path,
    lens_id: str,
    category: str,
    sub_type: str,
) -> None:
    """Save 4-panel diagnostic figure."""
    from PIL import Image
    import io
    
    kappa = kappa_dict["kappa"]
    gamma_mag = kappa_dict["gamma_mag"]
    mu = kappa_dict["mu"]
    gamma1 = kappa_dict["gamma1"]
    gamma2 = kappa_dict["gamma2"]
    extent = kappa_dict["extent"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    
    # [0, 0] Kappa map
    ax = axes[0, 0]
    vmax_k = max(2.0, np.percentile(kappa, 95))
    im0 = ax.imshow(kappa, extent=extent, origin="lower", cmap="inferno",
                    vmin=0, vmax=vmax_k)
    # Genuine critical curve: zero-Jacobian locus det(A) = (1-kappa)^2 - |gamma|^2 = 0
    # (i.e. 1/mu = 0), computed directly from the kappa/gamma maps rather than relying
    # on the kappa=1 proxy, which can have different topology for non-circular lenses.
    det_A = (1.0 - kappa) ** 2 - (gamma1 ** 2 + gamma2 ** 2)
    ax.contour(det_A, levels=[0.0], extent=extent, origin="lower",
               colors="cyan", linewidths=1.2)
    ax.contour(kappa, levels=[1.0], extent=extent, origin="lower",
               colors="white", linewidths=0.8, linestyles="dashed", alpha=0.6)
    ax.set_title("Convergence (κ) — cyan: critical curve (det A = 0)", fontsize=10, fontweight="bold")
    ax.set_xlabel("RA (arcsec)")
    ax.set_ylabel("Dec (arcsec)")
    plt.colorbar(im0, ax=ax, label="κ")
    
    # [0, 1] Shear magnitude
    ax = axes[0, 1]
    im1 = ax.imshow(gamma_mag, extent=extent, origin="lower", cmap="viridis",
                    vmin=0, vmax=0.5)
    ax.set_title("Shear Magnitude (|γ|)", fontsize=11, fontweight="bold")
    ax.set_xlabel("RA (arcsec)")
    ax.set_ylabel("Dec (arcsec)")
    plt.colorbar(im1, ax=ax, label="|γ|")
    
    # [1, 0] Log magnification
    ax = axes[1, 0]
    mu_safe = np.clip(mu, 0.01, 1000)
    log_mu = np.log10(mu_safe)
    im2 = ax.imshow(log_mu, extent=extent, origin="lower", cmap="RdYlBu_r",
                    vmin=-1, vmax=3)
    # Overlay the genuine critical curve (det A = 0, i.e. 1/mu = 0): the locus
    # where the magnification formally diverges, consistent with the kappa-panel
    # overlay above (rather than an arbitrary mu=2 magnification-threshold contour).
    det_A_mu = (1.0 - kappa) ** 2 - (gamma1 ** 2 + gamma2 ** 2)
    ax.contour(det_A_mu, levels=[0.0], extent=extent, origin="lower",
               colors="red", linewidths=1.2)
    ax.set_title("Log Magnification (log₁₀|μ|)", fontsize=11, fontweight="bold")
    ax.set_xlabel("RA (arcsec)")
    ax.set_ylabel("Dec (arcsec)")
    plt.colorbar(im2, ax=ax, label="log₁₀|μ|")
    
    # [1, 1] Kappa + gamma vector overlay
    ax = axes[1, 1]
    im3 = ax.imshow(kappa, extent=extent, origin="lower", cmap="gray",
                    vmin=0, vmax=2.0, alpha=0.7)
    
    # Quiver plot of shear (subsample every 10 pixels)
    # Normalize vectors for visualization (prevent single outliers from dominating)
    num_pix = kappa.shape[0]
    pixel_coords = np.arange(num_pix) - num_pix / 2.0
    delta_pix = extent[1] / (num_pix / 2.0)
    x_arcsec = pixel_coords * delta_pix
    y_arcsec = pixel_coords * delta_pix
    xx, yy = np.meshgrid(x_arcsec, y_arcsec)
    
    # Coarser subsampling and larger, higher-contrast arrows so the
    # characteristic rotation of the shear pattern around the deflector
    # remains legible at print resolution (the previous step=10/cool/thin
    # combination rendered as a dense grid of near-invisible dots).
    step = 18
    g1_sub = gamma1[::step, ::step]
    g2_sub = gamma2[::step, ::step]

    # Normalize shear vectors for uniform arrow size (color encode magnitude)
    gmag_sub = np.sqrt(g1_sub**2 + g2_sub**2)
    gmag_sub = np.clip(gmag_sub, 1e-6, None)  # Avoid division by zero
    g1_norm = g1_sub / gmag_sub
    g2_norm = g2_sub / gmag_sub

    # Color arrows by shear magnitude with a high-contrast colormap against
    # the grey kappa background; longer, thicker arrows for legibility
    ax.quiver(xx[::step, ::step], yy[::step, ::step],
              g1_norm, g2_norm,
              gmag_sub, cmap="plasma", scale=14, scale_units="inches",
              width=0.006, headwidth=3.5, headlength=4.5, alpha=0.95)
    
    ax.set_title("Kappa (grey) + Shear Vectors", fontsize=11, fontweight="bold")
    ax.set_xlabel("RA (arcsec)")
    ax.set_ylabel("Dec (arcsec)")
    plt.colorbar(im3, ax=ax, label="κ")
    
    fig.suptitle(f"PRISM Lensing Maps — {lens_id}", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    # Save as PNG first, then convert to JPG
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format="png", bbox_inches="tight", dpi=150)
    png_buffer.seek(0)
    plt.close(fig)
    
    # Convert PNG to JPG
    img_png = Image.open(png_buffer)
    if img_png.mode == "RGBA":
        rgb_img = Image.new("RGB", img_png.size, (255, 255, 255))
        rgb_img.paste(img_png, mask=img_png.split()[3] if len(img_png.split()) > 3 else None)
        img_png = rgb_img
    
    jpg_path = out_path / f"{lens_id}_kappa_panel.jpg"
    img_png.save(jpg_path, format="JPEG", quality=92)


def _save_flexion_panel_jpg(
    kappa_dict: Dict,
    out_path: Path,
    lens_id: str,
    category: str,
    sub_type: str,
) -> None:
    """Save 4-panel flexion diagnostic (F1, F2, G1, G2) as JPEG."""
    from PIL import Image
    import io

    extent = kappa_dict["extent"]
    F1 = kappa_dict["F1"]
    F2 = kappa_dict["F2"]
    F_mag = kappa_dict["F_mag"]
    G1 = kappa_dict["G1"]
    G2 = kappa_dict["G2"]
    G_mag = kappa_dict["G_mag"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)

    def _panel(ax, data, title, cmap):
        v = max(np.percentile(np.abs(data), 99), 1e-8)
        im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap,
                       vmin=-v, vmax=v)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("RA (arcsec)")
        ax.set_ylabel("Dec (arcsec)")
        plt.colorbar(im, ax=ax, label='arcsec$^{-1}$')

    def _panel_pos(ax, data, title, cmap):
        v = max(np.percentile(data, 99), 1e-8)
        im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap, vmin=0, vmax=v)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("RA (arcsec)")
        ax.set_ylabel("Dec (arcsec)")
        plt.colorbar(im, ax=ax, label='arcsec$^{-1}$')

    _panel(axes[0, 0], F1, "F1 (first flexion, x)", "RdBu_r")
    _panel(axes[0, 1], F2, "F2 (first flexion, y)", "RdBu_r")
    _panel_pos(axes[0, 2], F_mag, "|F| (first flexion magnitude)", "plasma")
    _panel(axes[1, 0], G1, "G1 (second flexion, spin-3 x)", "PRGn")
    _panel(axes[1, 1], G2, "G2 (second flexion, spin-3 y)", "PRGn")
    _panel_pos(axes[1, 2], G_mag, "|G| (second flexion magnitude)", "inferno")

    fig.suptitle(
        f"PRISM Flexion Maps — {lens_id}  [{category} | {sub_type}]",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout()

    png_buf = io.BytesIO()
    plt.savefig(png_buf, format="png", bbox_inches="tight", dpi=150)
    png_buf.seek(0)
    plt.close(fig)

    img_png = Image.open(png_buf)
    if img_png.mode == "RGBA":
        rgb = Image.new("RGB", img_png.size, (255, 255, 255))
        rgb.paste(img_png, mask=img_png.split()[3])
        img_png = rgb

    jpg_path = out_path / f"{lens_id}_flexion_panel.jpg"
    img_png.save(jpg_path, format="JPEG", quality=92)
