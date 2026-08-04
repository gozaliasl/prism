"""Publication-quality kappa/shear/magnification (and optional flexion)
figures, zoomed to a few arcsec around the deflector (not the full arcmin
extent), so the Einstein ring / critical curve structure is actually
resolved.

Run: python scripts/local/build_kappa_panel_paper.py <kappa_data_npz> <out_png> [zoom_arcsec]
     python scripts/local/build_kappa_panel_paper.py <kappa_data_npz> <out_png> [zoom_arcsec] --flexion
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    npz_path = Path(sys.argv[1])
    out_png = Path(sys.argv[2])
    zoom_arcsec = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    flexion_mode = "--flexion" in sys.argv

    d = np.load(npz_path, allow_pickle=True)
    extent = d["extent"]  # [xmin, xmax, ymin, ymax] arcsec, full frame
    theta_E_eff = float(d["theta_E_eff"])

    h, w = d["kappa"].shape
    full_w_arcsec = extent[1] - extent[0]
    pix_scale = full_w_arcsec / w
    half_px = int(round(zoom_arcsec / pix_scale))
    # Clamp to the array's actual half-size to avoid a negative slice start
    # (which Python silently wraps instead of erroring) when the requested
    # zoom exceeds the rendered FOV.
    half_px = min(half_px, h // 2, w // 2)
    zoom_arcsec = half_px * pix_scale
    cy, cx = h // 2, w // 2
    sl = (slice(cy - half_px, cy + half_px), slice(cx - half_px, cx + half_px))
    zext = [-zoom_arcsec, zoom_arcsec, -zoom_arcsec, zoom_arcsec]

    if flexion_mode:
        build_flexion_figure(d, sl, zext, out_png, theta_E_eff)
        return

    kappa_z = d["kappa"][sl]
    gamma_z = d["gamma_mag"][sl]
    mu_z = d["mu"][sl]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    im0 = axes[0].imshow(kappa_z, origin="lower", extent=zext, cmap="inferno", vmin=0, vmax=max(2.0, np.percentile(kappa_z, 99.5)))
    axes[0].contour(kappa_z, levels=[1.0], extent=zext, colors="cyan", linewidths=1.2)
    axes[0].set_title(r"Convergence $\kappa$ (cyan: critical curve, $\kappa$=1)")
    plt.colorbar(im0, ax=axes[0], label=r"$\kappa$", fraction=0.046)

    im1 = axes[1].imshow(gamma_z, origin="lower", extent=zext, cmap="viridis")
    axes[1].set_title(r"Shear magnitude $|\gamma|$")
    plt.colorbar(im1, ax=axes[1], label=r"$|\gamma|$", fraction=0.046)

    log_mu = np.log10(np.clip(np.abs(mu_z), 1e-3, None))
    im2 = axes[2].imshow(log_mu, origin="lower", extent=zext, cmap="RdYlBu_r", vmin=-1, vmax=2)
    axes[2].set_title(r"Log magnification $\log_{10}|\mu|$")
    plt.colorbar(im2, ax=axes[2], label=r"$\log_{10}|\mu|$", fraction=0.046)

    for ax in axes:
        ax.set_xlabel('RA offset (")')
        ax.set_ylabel('Dec offset (")')

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"theta_E_eff (from kappa map) = {theta_E_eff:.4f} arcsec")


def build_flexion_figure(d, sl, zext, out_png, theta_E_eff):
    # Flexion formally diverges toward the deflector centre, so a linear
    # scale is dominated by a single saturated pixel and hides the
    # surrounding radial structure -- use log10 as for the magnification
    # panel.
    F_mag = np.log10(np.clip(d["F_mag"][sl], 1e-3, None))
    G_mag = np.log10(np.clip(d["G_mag"][sl], 1e-3, None))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    im0 = axes[0].imshow(F_mag, origin="lower", extent=zext, cmap="magma", vmin=-2, vmax=1.6)
    axes[0].set_title(r"First flexion $\log_{10}|F|$")
    plt.colorbar(im0, ax=axes[0], label=r'$\log_{10}|F|$ (arcsec$^{-1}$)', fraction=0.046)

    im1 = axes[1].imshow(G_mag, origin="lower", extent=zext, cmap="magma", vmin=-2, vmax=1.6)
    axes[1].set_title(r"Second flexion $\log_{10}|G|$")
    plt.colorbar(im1, ax=axes[1], label=r'$\log_{10}|G|$ (arcsec$^{-1}$)', fraction=0.046)

    for ax in axes:
        ax.set_xlabel('RA offset (")')
        ax.set_ylabel('Dec offset (")')

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"F_mag max={F_mag.max():.3f}  G_mag max={G_mag.max():.3f}")


if __name__ == "__main__":
    main()
