"""One-time (expensive) step: precompute a (age x metallicity x wavelength)
single-stellar-population (SSP) spectral grid from FSPS, WITH nebular
emission (Cloudy-based, built into FSPS via add_neb_emission=True) --
replacing tng_particle_light.py's hand-tuned 5-point age-anchor /
per-band-metallicity-power-law approximation with a real stellar population
synthesis model (the actual technique iMaNGA uses MaStar/MappingsIII for,
just via FSPS's bundled Cloudy nebular grids instead of a separate
MappingsIII step).

Requires SPS_HOME to point at a full FSPS data checkout (isochrones +
spectral libraries), NOT just the pip-installed python-fsps bindings --
see scripts/local/build_fsps_ssp_grid.sh which fetches that data once.

Output: data/fsps_ssp_grid.npz containing
  wave_aa       : (n_wave,) rest-frame wavelength, Angstrom
  age_gyr       : (n_age,)  SSP ages, log-spaced
  logzsol       : (n_z,)    log10(Z/Zsun)
  l_nu_grid     : (n_age, n_z, n_wave)  L_nu [Lsun/Hz] per Msun formed,
                  nebular emission included

Run: SPS_HOME=/path/to/fsps python scripts/local/build_fsps_ssp_grid.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "fsps_ssp_grid.npz"

# Age grid: 1 Myr -> 13.5 Gyr, log-spaced (matches the physical range
# tng_particle_light.py's AGE_GRID_GYR already covers).
AGE_GRID_GYR = np.logspace(np.log10(0.001), np.log10(13.5), 24)

# Metallicity grid: logzsol = log10(Z/Zsun), spans TNG's star particle
# metallicity range (roughly 0.02-2.5 Zsun in the existing METAL_GRID_ZREL).
LOGZSOL_GRID = np.array([-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.4])


def main():
    if "SPS_HOME" not in os.environ:
        print("ERROR: SPS_HOME not set. Run scripts/local/build_fsps_ssp_grid.sh first "
              "to fetch the FSPS data checkout, then:\n"
              "  SPS_HOME=data/fsps_home/fsps-master python scripts/local/build_fsps_ssp_grid.py")
        sys.exit(1)

    import fsps

    sp = fsps.StellarPopulation(
        zcontinuous=1,       # continuous metallicity interpolation via logzsol
        sfh=0,                # SSP: single coeval burst
        imf_type=1,           # Chabrier (2003) IMF
        add_neb_emission=True,  # Cloudy-based nebular continuum + emission lines
        add_dust_emission=False,  # we apply our own TNG-gas-based dust screen downstream
        dust_type=0, dust1=0.0, dust2=0.0,  # no FSPS dust attenuation -- tng_particle_light.py applies its own gas-based screen
    )

    wave_aa = None
    l_nu_grid = np.zeros((len(AGE_GRID_GYR), len(LOGZSOL_GRID), 1), dtype=np.float64)

    for j, logzsol in enumerate(LOGZSOL_GRID):
        sp.params["logzsol"] = float(logzsol)
        for i, age_gyr in enumerate(AGE_GRID_GYR):
            wave, spec = sp.get_spectrum(tage=float(age_gyr), peraa=False)
            if wave_aa is None:
                wave_aa = wave
                l_nu_grid = np.zeros((len(AGE_GRID_GYR), len(LOGZSOL_GRID), len(wave_aa)), dtype=np.float64)
            l_nu_grid[i, j, :] = spec
        print(f"[FSPS] logzsol={logzsol:+.1f} done ({len(AGE_GRID_GYR)} ages)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        wave_aa=wave_aa,
        age_gyr=AGE_GRID_GYR,
        logzsol=LOGZSOL_GRID,
        l_nu_grid=l_nu_grid.astype(np.float32),
    )
    print(f"Saved {OUT_PATH}  (grid shape {l_nu_grid.shape}, "
          f"{OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
