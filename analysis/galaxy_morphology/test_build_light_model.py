#!/usr/bin/env python3
"""
Unit-level check for galaxy_morphology.build_light_model: for every
morphology type, build a native multi-component fragment from a
representative base_params dict, verify flux fractions sum to 1 per band,
and render the resulting light_model_list + kwargs through a standalone
lenstronomy SimAPI to confirm finite, non-negative images.

Usage: conda run -n astro-clean python analysis/galaxy_morphology/test_build_light_model.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism.morphology import build_light_model, MORPH_COMPONENTS
from lenstronomy.SimulationAPI.sim_api import SimAPI


BANDS = ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']

KWARGS_SINGLE_BAND = dict(
    read_noise=4, ccd_gain=1.0, sky_brightness=22.0, exposure_time=1000,
    magnitude_zero_point=28.0, num_exposures=1, seeing=0.1,
    pixel_scale=0.03, psf_type='NONE',
)


def main():
    config = {'morphology': {'multicomponent_enabled': True}}
    rng = np.random.default_rng(42)

    base = dict(R_sersic=0.8, n_sersic=3.0, e1=0.15, e2=-0.08,
                 center_x=0.0, center_y=0.0)
    total_mag = {b: 21.0 for b in BANDS}

    n_failed = 0
    for morph in MORPH_COMPONENTS:
        fragment, kw_by_band, mt = build_light_model(
            'lens', base, total_mag, BANDS, rng, config, morph_type=morph)

        expected_n = len(MORPH_COMPONENTS[morph])
        assert len(fragment) == expected_n, f"{morph}: fragment len {len(fragment)} != {expected_n}"

        for b in BANDS:
            kw_list = kw_by_band[b]
            assert len(kw_list) == expected_n
            fracs = [10 ** (-0.4 * (kw['magnitude'] - 21.0)) for kw in kw_list]
            assert all(np.isfinite(f) for f in fracs), f"{morph}/{b}: non-finite flux fraction"
            total = sum(fracs)
            assert abs(total - 1.0) < 1e-6, f"{morph}/{b}: flux fractions sum to {total}, not 1"

        # Render through SimAPI for one band
        model_lists = dict(lens_model_list=[], lens_light_model_list=fragment,
                            source_light_model_list=[])
        sim = SimAPI(numpix=64, kwargs_single_band=KWARGS_SINGLE_BAND, kwargs_model=model_lists)
        kw_amp, _, _ = sim.magnitude2amplitude(kw_by_band['F150W'], [])
        im_model = sim.image_model_class(dict(supersampling_factor=1, supersampling_convolution=False))
        image = im_model.image(kwargs_lens=[], kwargs_source=[], kwargs_lens_light=kw_amp)

        ok = np.all(np.isfinite(image)) and np.all(image >= 0) and image.sum() > 0
        status = "OK" if ok else "FAIL"
        if not ok:
            n_failed += 1
        print(f"{morph:<16} components={fragment} sum={image.sum():.4g} {status}")

    if n_failed:
        print(f"\n{n_failed} morph type(s) FAILED")
        sys.exit(1)
    print("\nAll morphology types OK")


if __name__ == "__main__":
    main()
