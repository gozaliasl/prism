"""Unit-level check of the TNG particle-driven morphology builder
(``src/galaxy_morphology/tng_particle_light.py``).

For a handful of locally-downloaded TNG particle cutouts, builds INTERPOL
kwargs for all 4 JWST bands, checks the images are finite/non-negative and
that the per-band colors are physically sensible (younger/star-forming
subhalos bluer than old/quiescent ones), and finally renders one of them
through lenstronomy (``magnitude2amplitude`` + ``ImageModel.image()``) to
confirm the kwargs are usable end-to-end.
"""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prism.morphology.tng_particle_light import build_tng_particle_interpol_kwargs  # noqa: E402

PARTICLE_DIR = "/Volumes/extHD/galaxygenius_build/workspace/data"
LOCAL_CATALOG = "/Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet"

BANDS = ["F115W", "F150W", "F277W", "F444W"]


def find_test_subhalos(n: int = 5):
    files = sorted(glob.glob(os.path.join(PARTICLE_DIR, "TNG_100_snap_*_subhalo_*.h5")))
    cat = pd.read_parquet(LOCAL_CATALOG)
    rows = []
    for f in files:
        m = re.match(r"TNG_100_snap_(\d+)_subhalo_(\d+)\.h5", os.path.basename(f))
        snap, sid = int(m.group(1)), int(m.group(2))
        match = cat[(cat.snapshot == snap) & (cat.subhalo_id == sid)]
        if len(match) == 0:
            continue
        rows.append(dict(
            path=f, snapshot=snap, subhalo_id=sid,
            halfmassrad_stars_kpc=float(match.iloc[0]["halfmassrad_stars_kpc"]),
            ssfr_per_yr=float(match.iloc[0]["ssfr_per_yr"]),
        ))
        if len(rows) >= n:
            break
    return rows


def main():
    rng = np.random.default_rng(42)
    subhalos = find_test_subhalos(5)
    assert subhalos, "no local particle files matched the local TNG catalog"

    print(f"Testing {len(subhalos)} subhalos:")
    results = []
    for sh in subhalos:
        kw_by_band = {}
        for band in BANDS:
            kw = build_tng_particle_interpol_kwargs(
                band=band,
                particle_file=sh["path"],
                halfmassrad_stars_kpc=sh["halfmassrad_stars_kpc"],
                magnitude_ref=24.0,
                ref_band="F150W",
                center_x=0.0, center_y=0.0, phi_G=0.3,
                target_size_arcsec=1.2,
                rng=rng,
            )
            image = kw["image"]
            assert np.all(np.isfinite(image)), f"{sh['path']} {band}: non-finite pixels"
            assert np.all(image >= 0), f"{sh['path']} {band}: negative pixels"
            assert np.isfinite(kw["magnitude"]), f"{sh['path']} {band}: non-finite magnitude"
            kw_by_band[band] = kw

        color_115_444 = kw_by_band["F115W"]["magnitude"] - kw_by_band["F444W"]["magnitude"]
        results.append((sh, color_115_444))
        print(f"  snap{sh['snapshot']}_subhalo{sh['subhalo_id']}: "
              f"rhalf={sh['halfmassrad_stars_kpc']:.2f} kpc, "
              f"sSFR={sh['ssfr_per_yr']:.2e}/yr, "
              f"F115W-F444W={color_115_444:+.2f} "
              f"(F150W mag={kw_by_band['F150W']['magnitude']:.2f})")

    # Star-forming (high sSFR) subhalos should be bluer (smaller F115W-F444W)
    # than quiescent ones, on average.
    sf = [c for sh, c in results if sh["ssfr_per_yr"] > 1e-11]
    q = [c for sh, c in results if sh["ssfr_per_yr"] <= 1e-11]
    if sf and q:
        print(f"\nMean F115W-F444W: star-forming={np.mean(sf):+.2f}, quiescent={np.mean(q):+.2f}")
        assert np.mean(sf) < np.mean(q), "star-forming subhalos should be bluer than quiescent ones"

    # End-to-end lenstronomy render for the first subhalo.
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    sh = subhalos[0]
    band = "F150W"
    kw = build_tng_particle_interpol_kwargs(
        band=band, particle_file=sh["path"],
        halfmassrad_stars_kpc=sh["halfmassrad_stars_kpc"],
        magnitude_ref=22.0, ref_band="F150W",
        center_x=0.0, center_y=0.0, phi_G=0.0,
        target_size_arcsec=1.2, rng=rng,
    )
    kwargs_band = dict(
        pixel_scale=0.031, exposure_time=1028.0, magnitude_zero_point=28.09,
        read_noise=12.0, sky_brightness=27.0, ccd_gain=1.0,
        seeing=0.05, psf_type="GAUSSIAN",
    )
    sim = SimAPI(numpix=100, kwargs_single_band=kwargs_band,
                  kwargs_model=dict(lens_light_model_list=["INTERPOL"]))
    kw_amp, _, _ = sim.magnitude2amplitude(kwargs_lens_light_mag=[kw])
    im_model = sim.image_model_class(dict(supersampling_factor=1, supersampling_convolution=False))
    image = im_model.image(kwargs_lens=[], kwargs_source=[], kwargs_lens_light=kw_amp)
    assert np.all(np.isfinite(image)), "rendered image has non-finite pixels"
    assert image.sum() > 0, "rendered image is all zero"
    print(f"\nEnd-to-end render OK: shape={image.shape}, total flux={image.sum():.4g}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
