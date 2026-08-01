"""Second stage: integrate the precomputed FSPS SSP spectral grid
(data/fsps_ssp_grid.npz, built by build_fsps_ssp_grid.py) against this
project's real per-telescope filter throughput curves
(prism.telescopes.multi_telescope_filters), producing a per-band
(age x metallicity) luminosity-per-unit-mass table -- the SAME shape/role
as tng_particle_light.py's old BAND_AGE_METAL_LUMINOSITY, but now real SPS
output (with nebular emission) integrated through real filter curves,
covering JWST + Roman + Euclid + Subaru + LSST instead of just the 8
JWST/Euclid bands the old anchor-table approximation supported.

This is a fast step (spectral integration only, no SKIRT/FSPS runtime
cost) so it's kept separate from the expensive FSPS grid generation.

Output: data/fsps_band_age_metal_grid.npz containing
  age_gyr    : (n_age,)
  logzsol    : (n_z,)
  bands      : (n_band,) string array of band names
  band_lum   : (n_band, n_age, n_z)  L_band [Lsun-equivalent per Msun formed]

Run: python scripts/local/integrate_fsps_grid_to_bands.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.telescopes.multi_telescope_filters import MultiTelescopeFilterSystem  # noqa: E402
from prism.telescopes.jwst_real_filter_transmission import RealJWSTFilterSystem  # noqa: E402

# The real per-telescope filter throughput files (JWST mean-system
# throughputs, Roman, Euclid, Subaru, LSST) live in the sibling repo's data/
# directory, not inside prism-lensing itself -- MultiTelescopeFilterSystem's
# and RealJWSTFilterSystem's own auto-detected workspace roots
# (walking up from src/prism/telescopes/) stop at src/prism/data (which
# exists but is empty of these files) before ever reaching a directory that
# actually has them, so both need an explicit path override here.
_FILTER_DATA_ROOT = Path("/Volumes/exthd-prism/jwst-mock-lens-simulator")

SSP_GRID_PATH = Path(__file__).resolve().parents[2] / "data" / "fsps_ssp_grid.npz"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "fsps_band_age_metal_grid.npz"

# All bands this pipeline's detector chain actually uses (5 telescopes).
BANDS = [
    "F070W", "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W",
    "EUCLID_VIS", "EUCLID_Y", "EUCLID_J", "EUCLID_H",
    "ROMAN_F062", "ROMAN_F087", "ROMAN_F106", "ROMAN_F129",
    "ROMAN_F146", "ROMAN_F158", "ROMAN_F184", "ROMAN_F213",
    "SUBARU_B", "SUBARU_V", "SUBARU_G", "SUBARU_R", "SUBARU_I", "SUBARU_Z", "SUBARU_Y",
    "LSST_U", "LSST_G", "LSST_R", "LSST_I", "LSST_Z", "LSST_Y",
]

C_AA_PER_S = 2.99792458e18  # speed of light, Angstrom/s


def integrate_band_luminosity(wave_aa, l_nu, filt_wave_um, filt_trans):
    """Photon-counting bandpass-averaged L_nu [Lsun/Hz] -> a single
    representative "band luminosity" per Msun formed, using the standard
    filter-integral convention (int f_nu T dnu/nu) / (int T dnu/nu)."""
    filt_wave_aa = filt_wave_um * 1e4
    order = np.argsort(filt_wave_aa)
    filt_wave_aa = filt_wave_aa[order]
    filt_trans = np.clip(filt_trans[order], 0.0, None)

    lo, hi = filt_wave_aa[0], filt_wave_aa[-1]
    mask = (wave_aa >= lo) & (wave_aa <= hi)
    if mask.sum() < 4:
        return 0.0
    w = wave_aa[mask]
    lnu = l_nu[mask]

    trans_interp = interp1d(filt_wave_aa, filt_trans, bounds_error=False, fill_value=0.0)
    t = trans_interp(w)

    # nu is descending (frequency decreases with wavelength), so trapezoid
    # over it returns a consistently negative-signed integral for both num
    # and den -- the ratio is still correct (signs cancel); do not treat
    # a negative `den` as "no throughput" (that was a real bug: it forced
    # every band to silently return 0.0).
    nu = C_AA_PER_S / w
    num = trapezoid(lnu * t / nu, nu)
    den = trapezoid(t / nu, nu)
    if den == 0:
        return 0.0
    return float(num / den)


def main():
    if not SSP_GRID_PATH.exists():
        print(f"ERROR: {SSP_GRID_PATH} not found -- run build_fsps_ssp_grid.py first.")
        sys.exit(1)

    grid = np.load(SSP_GRID_PATH)
    wave_aa = grid["wave_aa"]
    age_gyr = grid["age_gyr"]
    logzsol = grid["logzsol"]
    l_nu_grid = grid["l_nu_grid"]  # (n_age, n_z, n_wave)

    filters = MultiTelescopeFilterSystem(workspace_root=_FILTER_DATA_ROOT)
    jwst_filters = RealJWSTFilterSystem(
        data_path=_FILTER_DATA_ROOT / "data" / "nircam_throughputs" / "mean_throughputs")

    n_age, n_z = len(age_gyr), len(logzsol)
    band_lum = np.zeros((len(BANDS), n_age, n_z), dtype=np.float64)

    for b_idx, band in enumerate(BANDS):
        try:
            if band in jwst_filters.transmission_curves:
                filt_wave_um, filt_trans = jwst_filters.transmission_curves[band]
            else:
                filt_wave_um, filt_trans = filters.load_filter_transmission(band)
        except Exception as e:
            print(f"[SKIP] {band}: {e}")
            continue
        for i in range(n_age):
            for j in range(n_z):
                band_lum[b_idx, i, j] = integrate_band_luminosity(
                    wave_aa, l_nu_grid[i, j, :], filt_wave_um, filt_trans)
        print(f"[BAND] {band}: range {band_lum[b_idx].min():.3e} - {band_lum[b_idx].max():.3e}")

    np.savez_compressed(
        OUT_PATH,
        age_gyr=age_gyr,
        logzsol=logzsol,
        bands=np.array(BANDS),
        band_lum=band_lum,
    )
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
