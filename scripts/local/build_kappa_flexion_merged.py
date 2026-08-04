"""Merged 5-panel figure: kappa, shear, magnification, and both flexion
fields, all in a single row, zoomed to a few arcsec around the deflector.

Run: python scripts/local/build_kappa_flexion_merged.py <kappa_data_npz> <out_png> [zoom_arcsec]
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

    d = np.load(npz_path, allow_pickle=True)
    extent = d["extent"]
    theta_E_eff = float(d["theta_E_eff"])

    h, w = d["kappa"].shape
    pix_scale = (extent[1] - extent[0]) / w
    half_px = int(round(zoom_arcsec / pix_scale))
    # Clamp to the array's actual half-size -- requesting a zoom wider than
    # the rendered FOV would otherwise give a negative slice start, which
    # Python silently wraps to the array's far end instead of erroring,
    # producing a garbled/blank-looking crop.
    half_px = min(half_px, h // 2, w // 2)
    actual_zoom_arcsec = half_px * pix_scale
    cy, cx = h // 2, w // 2
    sl = (slice(cy - half_px, cy + half_px), slice(cx - half_px, cx + half_px))
    zext = [-actual_zoom_arcsec, actual_zoom_arcsec, -actual_zoom_arcsec, actual_zoom_arcsec]

    kappa_z = d["kappa"][sl]
    gamma_z = d["gamma_mag"][sl]
    mu_z = d["mu"][sl]
    F_z = np.log10(np.clip(d["F_mag"][sl], 1e-3, None))
    G_z = np.log10(np.clip(d["G_mag"][sl], 1e-3, None))
    log_mu_z = np.log10(np.clip(np.abs(mu_z), 1e-3, None))

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))

    im0 = axes[0].imshow(kappa_z, origin="lower", extent=zext, cmap="inferno", vmin=0, vmax=max(2.0, np.percentile(kappa_z, 99.5)))
    axes[0].contour(kappa_z, levels=[1.0], extent=zext, colors="cyan", linewidths=1.2)
    axes[0].set_title(r"Convergence $\kappa$")
    plt.colorbar(im0, ax=axes[0], label=r"$\kappa$", fraction=0.046)

    im1 = axes[1].imshow(gamma_z, origin="lower", extent=zext, cmap="viridis")
    axes[1].set_title(r"Shear $|\gamma|$")
    plt.colorbar(im1, ax=axes[1], label=r"$|\gamma|$", fraction=0.046)

    im2 = axes[2].imshow(log_mu_z, origin="lower", extent=zext, cmap="RdYlBu_r", vmin=-1, vmax=2)
    axes[2].set_title(r"$\log_{10}|\mu|$")
    plt.colorbar(im2, ax=axes[2], label=r"$\log_{10}|\mu|$", fraction=0.046)

    im3 = axes[3].imshow(F_z, origin="lower", extent=zext, cmap="magma", vmin=-2, vmax=1.6)
    axes[3].set_title(r"$\log_{10}|F|$")
    plt.colorbar(im3, ax=axes[3], label=r"$\log_{10}|F|$", fraction=0.046)

    im4 = axes[4].imshow(G_z, origin="lower", extent=zext, cmap="magma", vmin=-2, vmax=1.6)
    axes[4].set_title(r"$\log_{10}|G|$")
    plt.colorbar(im4, ax=axes[4], label=r"$\log_{10}|G|$", fraction=0.046)

    for ax in axes:
        ax.set_xlabel('RA offset (")')
        ax.set_ylabel('Dec offset (")')

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"theta_E_eff = {theta_E_eff:.4f} arcsec")


if __name__ == "__main__":
    main()
