"""Grid figure: rows = 4 TNG morphology/physical classes, columns = 5 real
telescopes (JWST, Roman, Euclid, Subaru/HSC, Rubin/LSST) already supported
by this pipeline's detector chain -- each cell is the SAME real TNG100-1
galaxy (fixed at z=0.5), rendered from its actual star-particle cutout via
prism.morphology.tng_particle_light, then resampled to that telescope's
real native pixel scale and convolved with that telescope's real PSF
(diffraction-limited for JWST/Roman/Euclid, atmospheric-seeing for
Subaru/Rubin) using prism.io.synthetic_psf_generator -- the same PSF
machinery the main lensing pipeline uses.

Context / provenance
---------------------
This grid was built after reviewing https://github.com/xczhou-astro/
galaxyGenius (SKIRT9 radiative-transfer postprocessing for TNG -> multi-
telescope mock images) as a reference for "how should a multi-telescope
mock-observation grid be structured". SKIRT9 is an external C++ code not
installed in this environment, so this script does NOT run GalaxyGenius
itself; it reuses this project's own procedural TNG-particle renderer
(tng_particle_light.py, already validated this session -- see
tng_morphology_redshift_grid.py) plus the project's own per-telescope PSF/
pixel-scale machinery, which achieves the same *goal* (real hydro-sim
galaxy, multiple telescopes, each telescope's own resolution/PSF) without
the SKIRT dependency.

Update (2026-08-01): tng_particle_light's SED model was upgraded from a
hand-tuned anchor table to a real FSPS SSP grid (Chabrier IMF, Cloudy-based
nebular continuum + emission lines) integrated through each telescope's own
real filter throughput curve -- see build_fsps_ssp_grid.py /
integrate_fsps_grid_to_bands.py. Every band below (including Roman/Subaru/
LSST, which the old anchor table did not cover at all) now uses its own
real filter-integrated flux map, not a borrowed JWST F150W stand-in.

Run: python scripts/local/tng_multitelescope_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.selection.tng_galaxy_selector import local_particle_path  # noqa: E402
from prism.morphology.tng_particle_light import _band_images, _PROJECTION_CACHE  # noqa: E402
from prism.io.synthetic_psf_generator import build_resolution_psf_cache  # noqa: E402

CATALOG_PATH = "/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog_enriched.parquet"
OUT_PATH = Path(__file__).resolve().parents[2] / "outputs_test" / "tng_multitelescope_grid.png"
JWST_PSF_TILE = Path(__file__).resolve().parents[2] / "data" / "psf_v5_30mas" / "tiles" / "A1" / "F150W_kernel.fits"

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
TARGET_Z = 0.5
MIN_PARTICLES = 800
# name -> (pixel_scale arcsec/px, psf_res_name, band actually rendered, display label)
TELESCOPES = [
    ("JWST NIRCam",   0.031, "jwst",   "F150W",      "F150W (real empirical PSF)"),
    ("Roman WFI",     0.11,  "roman",  "ROMAN_F158", "F158, diffraction-limited"),
    ("Euclid VIS",    0.10,  "euclid", "EUCLID_H",   "H, diffraction-limited"),
    ("Subaru HSC",    0.168, "subaru", "SUBARU_I",   "I, atmospheric seeing"),
    ("Rubin/LSST",    0.20,  "lsst",   "LSST_I",     "I, atmospheric seeing"),
]

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


def nearest_snapshot(df, target_z):
    zs = df[["snapshot", "snapshot_redshift"]].drop_duplicates()
    idx = (zs["snapshot_redshift"] - target_z).abs().idxmin()
    return int(zs.loc[idx, "snapshot"]), float(zs.loc[idx, "snapshot_redshift"])


def pick_candidate(df, snapshot, class_mask_fn, rng, used_subhalos):
    snap_df = df[df["snapshot"] == snapshot]
    mask = class_mask_fn(snap_df)
    candidates = snap_df[mask]
    if len(candidates) == 0:
        return None
    order = rng.permutation(len(candidates))
    for i in order:
        row = candidates.iloc[int(i)]
        if int(row["subhalo_id"]) in used_subhalos:
            continue
        p = local_particle_path(int(row["snapshot"]), int(row["subhalo_id"]),
                                 min_particles=MIN_PARTICLES, sim="TNG100-1")
        if p is not None:
            used_subhalos.add(int(row["subhalo_id"]))
            return row, p
    return None


def stretch(img):
    img = np.clip(img, 0, None)
    p99 = np.percentile(img, 99.5)
    if p99 <= 0:
        return np.zeros_like(img)
    return np.clip(np.arcsinh(img / (p99 / 5.0)) / np.arcsinh(5.0), 0, 1)


def resample_to_pixel_scale(img, native_pixel_scale_arcsec, target_pixel_scale_arcsec):
    """Flux-conserving resample from the native 256x256 grid to a telescope's
    native pixel scale, keeping the same physical/angular field of view."""
    zoom_factor = native_pixel_scale_arcsec / target_pixel_scale_arcsec
    out = zoom(img, zoom_factor, order=1)
    # Conserve total flux: zoom with order=1 approx preserves mean, not sum.
    total_before = img.sum()
    total_after = out.sum()
    if total_after > 0:
        out = out * (total_before / total_after)
    return out


def main():
    df = pd.read_parquet(CATALOG_PATH)
    rng = np.random.default_rng(11)

    snapshot, actual_z = nearest_snapshot(df, TARGET_Z)
    d_a_mpc = COSMO.angular_diameter_distance(actual_z).value

    # Real per-telescope PSFs, built once (JWST from empirical tiles;
    # Roman/Euclid diffraction-limited, Subaru/LSST atmospheric-seeing --
    # via the project's own synthetic_psf_generator, same code the main
    # lensing pipeline uses).
    psf_kernels = {}
    jwst_kernel = fits.getdata(JWST_PSF_TILE).astype(np.float64)
    jwst_kernel = jwst_kernel / jwst_kernel.sum()
    psf_kernels["JWST NIRCam"] = jwst_kernel
    for name, pixel_scale, res_name, band, _label in TELESCOPES:
        if res_name == "jwst":
            continue
        cache = build_resolution_psf_cache(res_name, [band], pixel_scale, psf_size=81)
        kernel = None
        for _tile, bd in cache.items():
            for _b, arr in bd.items():
                if arr is not None:
                    kernel = arr
                    break
            if kernel is not None:
                break
        psf_kernels[name] = kernel

    used_subhalos = set()
    n_rows = len(CLASS_DEFS)
    n_cols = len(TELESCOPES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))

    for i, (class_name, mask_fn) in enumerate(CLASS_DEFS):
        result = pick_candidate(df, snapshot, mask_fn, rng, used_subhalos)
        if result is None:
            for j in range(n_cols):
                axes[i, j].text(0.5, 0.5, "no match", ha="center", va="center", transform=axes[i, j].transAxes)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            continue
        row, particle_file = result
        band_imgs = _band_images(particle_file, float(row["halfmassrad_stars_kpc"]), rng)
        extent_kpc = _PROJECTION_CACHE[str(particle_file)]["extent_kpc"]
        angular_extent_arcsec = (extent_kpc / 1000.0 / d_a_mpc) * 206265.0

        for j, (name, pixel_scale, res_name, band, label) in enumerate(TELESCOPES):
            ax = axes[i, j]
            flux_map = band_imgs[band]
            native_pixel_scale = angular_extent_arcsec / flux_map.shape[0]
            resampled = resample_to_pixel_scale(flux_map, native_pixel_scale, pixel_scale)
            kernel = psf_kernels.get(name)
            if kernel is not None and resampled.size > 0:
                # Match kernel pixel scale approx: kernels were generated at
                # this telescope's own pixel_scale already (or JWST's native
                # 0.0310"/px empirical tile), so no extra resampling needed.
                rendered = fftconvolve(resampled, kernel, mode="same")
                rendered = np.clip(rendered, 0, None)
            else:
                rendered = resampled
            img = stretch(rendered)
            ax.imshow(img, cmap="inferno", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{name}\n{pixel_scale:.3f}\"/px, {label}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{class_name}\nsubh {int(row['subhalo_id'])}"
                               f" (logM*={row['stellar_mass_logmsun']:.1f})", fontsize=8.5)

    fig.suptitle(f"Same real TNG100-1 galaxies (z={actual_z:.2f}, snap {snapshot}) through 5 real telescopes\n"
                 "(real FSPS+nebular SED integrated through each telescope's own filter; real pixel scale + PSF)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
