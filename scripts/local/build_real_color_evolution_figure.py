"""Rebuild the color-evolution figure (Panels A/B) from PRISM's REAL
empirical SED code (src/prism/io/empirical_sed_templates.py) and real
filter throughput curves, replacing the previous standalone script's
fabricated placeholder formulas.

Panel A: the 4 real SED template shapes (generate_empirical_sed). The
         passive template is drawn with a small (2%) wavelength offset
         purely for visual separation from the near-overlapping
         dusty-starburst curve; this is a plotting aid only, applied
         identically to the whole curve, and carries no physical meaning.
Panel B: real synthetic-photometry color-redshift tracks for star-forming
         and passive templates, integrated against real JWST filter
         throughput curves (not a hand-picked formula).

Run: python scripts/local/build_real_color_evolution_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Volumes/exthd-prism/prism-lensing/src")
from prism.io.empirical_sed_templates import generate_empirical_sed

RNG = np.random.default_rng(42)
SED_TYPES = ["star_forming", "passive", "post_starburst", "dusty_starburst"]
COLORS = {"star_forming": "tab:blue", "passive": "tab:red",
          "post_starburst": "tab:purple", "dusty_starburst": "tab:brown"}
OUT_PATH = Path("/Volumes/exthd-prism/prism-lensing/outputs_test/fig07_real_color_evolution.png")

# Small (2%) visual-only wavelength offset applied to the passive template
# so its curve is distinguishable from dusty_starburst, which otherwise
# nearly overlaps it across most of the plotted range.
PANEL_A_WAVE_SHIFT = {"passive": 1.02}


def load_filter_curve(path: Path):
    """Load a real filter throughput .txt/.dat file: 2 columns, wavelength
    (Angstrom or micron, auto-detected) and throughput."""
    try:
        arr = np.loadtxt(path, comments="#")
    except ValueError:
        arr = np.loadtxt(path, comments="#", skiprows=1)
    wave, thr = arr[:, 0], arr[:, 1]
    if wave.max() > 100:  # Angstrom -> micron
        wave = wave / 1e4
    order = np.argsort(wave)
    return wave[order], thr[order]


NIRCAM_DIR = Path("/Volumes/exthd-prism/prism-data/filter_throughputs/nircam/mean_throughputs")
NIRCAM_CURVES = {
    "F115W": NIRCAM_DIR / "F115W_May2024_mean_system_throughput.txt",
    "F150W": NIRCAM_DIR / "F150W_May2024_mean_system_throughput.txt",
    "F277W": NIRCAM_DIR / "F277W_May2024_mean_system_throughput.txt",
    "F444W": NIRCAM_DIR / "F444W_May2024_mean_system_throughput.txt",
}


def synthetic_mag(sed_type: str, redshift: float, filter_curve) -> float:
    """Real synthetic photometry: redshift the empirical SED, integrate
    against the real filter throughput curve, return an AB-like relative
    magnitude (arbitrary zero point -- only COLORS from this function are
    physically meaningful, matching how Panel B is actually used)."""
    fwave, fthr = filter_curve
    # Rest-frame wavelengths needed to cover the observed filter after
    # redshifting: rest = obs / (1+z)
    rest_wave = fwave / (1.0 + redshift)
    sed_flux = generate_empirical_sed(sed_type, rest_wave, rng=RNG)
    # f_lambda -> f_nu-like weighting for AB mags: integrate f_lambda * lambda * T(lambda) d(lambda)
    num = np.trapz(sed_flux * fthr * fwave, fwave)
    denom = np.trapz(fthr * fwave, fwave)
    flux_density = num / max(denom, 1e-30)
    return -2.5 * np.log10(max(flux_density, 1e-30))


def main():
    nircam_curves = {b: load_filter_curve(p) for b, p in NIRCAM_CURVES.items()}

    fig = plt.figure(figsize=(10.5, 4.5))
    gs = fig.add_gridspec(1, 2, wspace=0.32)

    # --- Panel A: real SED template shapes ---
    axA = fig.add_subplot(gs[0])
    wave_rest = np.logspace(np.log10(0.1), np.log10(30), 800)
    for sed_type in SED_TYPES:
        flux = generate_empirical_sed(sed_type, wave_rest, rng=RNG)
        wave_plot = wave_rest * PANEL_A_WAVE_SHIFT.get(sed_type, 1.0)
        axA.plot(wave_plot, flux / np.nanmax(flux), label=sed_type.replace("_", " "),
                  color=COLORS[sed_type], lw=1.5)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel(r"Rest-frame wavelength ($\mu$m)")
    axA.set_ylabel("Normalized flux")
    axA.set_title("(A) Real empirical SED templates")
    axA.legend(fontsize=7, loc="lower right")
    axA.set_ylim(1e-4, 2)

    # --- Panel B: real synthetic-photometry color-redshift tracks ---
    # Restricted to z=0.2-1.8: beyond this the rest-frame wavelength window
    # sampled by F115W crosses below the empirical templates' hard
    # blue-side flux cutoff (~0.35-0.4um, visible in Panel A), causing the
    # integrated flux to numerically collapse toward zero and the color to
    # diverge -- a real spectral-break-crossing effect, but the templates'
    # simplified hard cutoff (rather than a gradual physical decline) makes
    # the divergence numerically extreme rather than physically realistic,
    # so we do not display it as a quantitative result.
    axB = fig.add_subplot(gs[1])
    z_grid = np.linspace(0.2, 1.8, 25)
    for sed_type in ["star_forming", "passive"]:
        colors_115_150 = []
        for z in z_grid:
            m115 = synthetic_mag(sed_type, z, nircam_curves["F115W"])
            m150 = synthetic_mag(sed_type, z, nircam_curves["F150W"])
            colors_115_150.append(m115 - m150)
        axB.plot(z_grid, colors_115_150, color=COLORS[sed_type], lw=2,
                  label=sed_type.replace("_", " "))
    axB.set_xlabel("Redshift")
    axB.set_ylabel("F115W $-$ F150W (synthetic photometry)")
    axB.set_title("(B) Real synthetic-photometry color tracks")
    axB.legend(fontsize=8, loc="lower right")
    axB.axhline(0, color="gray", lw=0.5, ls=":")

    fig.suptitle("Rebuilt from PRISM's real empirical SED code "
                  "(replaces prior fabricated placeholder formulas)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
