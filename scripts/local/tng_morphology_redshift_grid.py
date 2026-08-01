"""Grid figure: rows = redshift bins, columns = TNG morphology/physical
classes, each cell = a single real TNG100-1 subhalo rendered from its own
downloaded star-particle cutout via prism.morphology.tng_particle_light
(the SAME procedural particle-projection renderer the main pipeline uses
for lens/source light -- not a fabricated Sersic stand-in).

Class definitions are physically-motivated proxies built from quantities
actually measured in the TNG local catalog (sSFR, gas fraction, half-mass
radius, velocity dispersion) -- there is no true "morphology" label in this
catalog, so we do not invent one; each column instead selects a real
physical regime that is expected to *look* different when rendered:

  - "Quiescent/Elliptical": low sSFR, low gas fraction, massive
    -> classically dispersion-dominated, smooth featureless light profile
  - "Star-forming/Disky":   moderate sSFR, moderate gas fraction
    -> expected to show a disk when the projection is inclined
  - "Starburst/Irregular":  very high sSFR, high gas fraction
    -> patchy/clumpy star-forming regions, more structure
  - "Compact/Low-mass":     lower stellar mass, small half-mass radius
    -> compact, dwarf/S0-like

Run: python scripts/local/tng_morphology_redshift_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.selection.tng_galaxy_selector import local_particle_path  # noqa: E402
from prism.morphology.tng_particle_light import _band_images  # noqa: E402

CATALOG_PATH = "/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog_enriched.parquet"
OUT_PATH = Path(__file__).resolve().parents[2] / "outputs_test" / "tng_morphology_redshift_grid.png"

TARGET_REDSHIFTS = [0.2, 0.5, 1.0, 2.0]
MIN_PARTICLES = 800  # relaxed from the pipeline's default 2000 to get columns to fill at high-z

CLASS_DEFS = [
    ("Quiescent / Elliptical",
     lambda df: (df["ssfr_per_yr"] < 1e-11) & (df["stellar_mass_logmsun"] > 10.3)
                & (df["gas_mass_msun"] / 10 ** df["stellar_mass_logmsun"] < 0.05)),
    ("Star-forming / Disky",
     lambda df: (df["ssfr_per_yr"] >= 1e-11) & (df["ssfr_per_yr"] < 5e-10)
                & (df["gas_mass_msun"] / 10 ** df["stellar_mass_logmsun"] >= 0.05)
                & (df["stellar_mass_logmsun"] > 9.5)),
    ("Starburst / Irregular",
     lambda df: (df["ssfr_per_yr"] >= 5e-10)
                & (df["gas_mass_msun"] / 10 ** df["stellar_mass_logmsun"] > 0.15)),
    ("Compact / Low-mass",
     lambda df: (df["stellar_mass_logmsun"] >= 8.5) & (df["stellar_mass_logmsun"] < 9.5)
                & (df["halfmassrad_stars_kpc"] < 2.0)),
]

DISPLAY_BAND = "F150W"


def nearest_snapshot(df, target_z):
    zs = df[["snapshot", "snapshot_redshift"]].drop_duplicates()
    idx = (zs["snapshot_redshift"] - target_z).abs().idxmin()
    return int(zs.loc[idx, "snapshot"]), float(zs.loc[idx, "snapshot_redshift"])


def pick_candidate(df, snapshot, class_mask_fn, rng):
    snap_df = df[df["snapshot"] == snapshot]
    mask = class_mask_fn(snap_df)
    candidates = snap_df[mask]
    if len(candidates) == 0:
        return None
    order = rng.permutation(len(candidates))
    for i in order:
        row = candidates.iloc[int(i)]
        p = local_particle_path(int(row["snapshot"]), int(row["subhalo_id"]),
                                 min_particles=MIN_PARTICLES, sim="TNG100-1")
        if p is not None:
            return row, p
    return None


def stretch(img):
    img = np.clip(img, 0, None)
    p99 = np.percentile(img, 99.5)
    if p99 <= 0:
        return np.zeros_like(img)
    return np.clip(np.arcsinh(img / (p99 / 5.0)) / np.arcsinh(5.0), 0, 1)


def main():
    df = pd.read_parquet(CATALOG_PATH)
    rng = np.random.default_rng(7)

    n_rows = len(TARGET_REDSHIFTS)
    n_cols = len(CLASS_DEFS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))

    for i, target_z in enumerate(TARGET_REDSHIFTS):
        snapshot, actual_z = nearest_snapshot(df, target_z)
        for j, (class_name, mask_fn) in enumerate(CLASS_DEFS):
            ax = axes[i, j]
            result = pick_candidate(df, snapshot, mask_fn, rng)
            if result is None:
                ax.text(0.5, 0.5, "no local\ncutout match", ha="center", va="center",
                        fontsize=9, color="gray", transform=ax.transAxes)
                ax.set_facecolor("black")
            else:
                row, particle_file = result
                band_imgs = _band_images(particle_file, float(row["halfmassrad_stars_kpc"]), rng)
                img = stretch(band_imgs[DISPLAY_BAND])
                ax.imshow(img, cmap="inferno", origin="lower")
                ax.text(0.03, 0.03, f"subh {int(row['subhalo_id'])}\n"
                        f"logM*={row['stellar_mass_logmsun']:.1f}",
                        transform=ax.transAxes, fontsize=6.5, color="white", va="bottom")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(class_name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"z = {actual_z:.2f}\n(snap {snapshot})", fontsize=10)

    fig.suptitle("Real TNG100-1 galaxies by redshift and physical class\n"
                  "(rendered from actual star-particle cutouts, F150W-equivalent luminosity)",
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
