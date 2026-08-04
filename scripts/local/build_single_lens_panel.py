"""Single-lens 5-panel figure: 4 bands (grayscale) + RGB, matching the
reference contact-sheet style (band labels on top, black background).

Run: python scripts/local/build_single_lens_panel.py <npz_path> <out_png>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def band_stretch(x, black_pct=50.0, vmax_pct=99.7, soft_frac=0.08):
    black = np.percentile(x, black_pct)
    sub = np.clip(x - black, 0, None)
    vmax = np.percentile(sub, vmax_pct)
    if vmax <= 0:
        vmax = sub.max() or 1.0
    soft = vmax * soft_frac
    return np.clip(np.arcsinh(sub / soft) / np.arcsinh(vmax / soft), 0, 1)


def main():
    npz_path = Path(sys.argv[1])
    out_png = Path(sys.argv[2])

    d = np.load(npz_path, allow_pickle=True)
    im = d["image_final"]
    meta = json.loads(str(d["metadata"]))
    bands = meta["bands"]

    stretched = [band_stretch(im[bi]) for bi in range(len(bands))]
    rgb = np.stack([stretched[-1], 0.5 * (stretched[-2] + stretched[1]), stretched[0]], axis=-1)

    ncols = len(bands) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.2), facecolor="black")
    for i, band in enumerate(bands):
        ax = axes[i]
        ax.imshow(stretched[i], cmap="gray", origin="lower")
        ax.set_title(band, color="white", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax = axes[-1]
    ax.imshow(rgb, origin="lower")
    ax.set_title("RGB", color="white", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor="black", bbox_inches="tight")
    print(f"Saved {out_png}")
    # Print the physical parameters for the caller to use in a paper caption.
    print(f"lens_id={meta.get('lens_id')}")
    print(f"theta_E_arcsec={meta.get('theta_E')}")
    print(f"theta_E_source={meta.get('theta_E_source', 'synthetic')}")
    print(f"lens_redshift={meta.get('lens_redshift')}")
    print(f"source_redshift={meta.get('source_redshift')}")
    print(f"magnification={meta.get('magnification')}")
    print(f"lens_sigma_kms={meta.get('lens_sigma_kms')}")
    print(f"lens_radius={meta.get('lens_radius')}")
    print(f"source_radius={meta.get('source_radius')}")
    print(f"n_field_galaxies={len(json.loads(str(d['metadata'])).get('field_magnitudes', []) or [])}"
          if 'field_magnitudes' in meta else "")


if __name__ == "__main__":
    main()
