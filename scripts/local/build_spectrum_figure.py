"""Real FSPS-based lens/source spectra for a single system, as saved by
PRISM in every output npz (lens_light_spectrum_*, source_spectrum_*).

Run: python scripts/local/build_spectrum_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = Path("/Volumes/exthd-prism/prism-lensing/outputs_test/refix_jwst_date_20260803_114958/unified_npz/PRISM_lens_SF_000002.npz")
OUT = Path("/Volumes/exthd-prism/prism-lensing/outputs_test/fig_real_spectra.png")


def main():
    d = np.load(NPZ, allow_pickle=True)
    meta = json.loads(str(d["metadata"]))

    lens_w = d["lens_light_spectrum_wave_obs_aa"] / 1e4
    lens_f = d["lens_light_spectrum_flux_fnu_cgs"]
    src_w = d["source_spectrum_wave_obs_aa"] / 1e4
    src_f = d["source_spectrum_flux_fnu_cgs"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(lens_w, lens_f, color="tab:red", lw=1.2,
            label=fr"Lens ($z_l={meta['lens_redshift']:.2f}$)")
    ax.plot(src_w, src_f, color="tab:blue", lw=1.2,
            label=fr"Source ($z_s={meta['source_redshift']:.2f}$)")

    # Mark JWST/NIRCam band positions used in this paper for context.
    band_centers = {"F115W": 1.154, "F150W": 1.501, "F277W": 2.786, "F444W": 4.421}
    for name, wc in band_centers.items():
        ax.axvline(wc, color="gray", lw=0.6, ls=":")

    ax.set_xlabel(r"Observed wavelength ($\mu$m)")
    ax.set_ylabel(r"$F_\nu$ (cgs)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.set_title(f"Real FSPS-based spectra, lens_id={meta.get('lens_id')} "
                 f"($\\theta_E={meta['theta_E']:.3f}''$)", fontsize=10)

    ymin = ax.get_ylim()[0]
    for i, (name, wc) in enumerate(band_centers.items()):
        ax.annotate(name, xy=(wc, ymin), xytext=(wc, ymin * (2.5 ** (1 + i % 2 * 2))),
                    fontsize=7, ha="center", color="dimgray", rotation=90)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
