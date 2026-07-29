"""Tests for Euclid Q1 catalogue and empirical PSF integration."""

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from prism.telescopes.euclid_q1_catalog import (
    EuclidQ1Catalog,
    apply_euclid_q1_physics,
    apply_euclid_q1_photometry,
    euclid_q1_enabled,
    get_euclid_q1_catalog,
    is_euclid_q1_psf_data,
)
from prism.io.synthetic_psf_generator import build_resolution_psf_cache, load_euclid_q1_psf_data


DATA_DIR = ROOT / 'data' / 'euclid_q1_psf'


@pytest.mark.skipif(pytest is None or not DATA_DIR.exists(), reason='pytest or Euclid Q1 data missing')
class TestEuclidQ1Catalog:
    def test_catalog_loads(self):
        cat = EuclidQ1Catalog(DATA_DIR)
        assert cat.n_systems > 300
        assert len(cat.psf_tiles) > 300

    def test_sample_physics(self):
        cat = EuclidQ1Catalog(DATA_DIR)
        rng = np.random.default_rng(0)
        phys = cat.sample_physics_batch(10, rng)
        assert len(phys['theta_E']) == 10
        assert np.all(phys['theta_E'] > 0)
        assert np.median(phys['theta_E']) > 0.5

    def test_sample_magnitudes(self):
        cat = EuclidQ1Catalog(DATA_DIR)
        rng = np.random.default_rng(1)
        lens, src = cat.sample_magnitudes_batch(5, rng, ['euclid_vis', 'euclid_y'])
        assert 'euclid_vis' in lens
        assert np.median(lens['euclid_vis']) > 18


@pytest.mark.skipif(not (DATA_DIR / 'tiles').exists(), reason='PSF tiles missing')
class TestEuclidQ1PSF:
    def test_load_psf_tiles(self):
        psf = load_euclid_q1_psf_data(DATA_DIR / 'tiles')
        assert is_euclid_q1_psf_data(psf)
        assert len(psf) > 300
        tile = next(iter(psf))
        assert psf[tile]['EUCLID_VIS'].shape == (101, 101)
        assert abs(psf[tile]['EUCLID_VIS'].sum() - 1.0) < 1e-5

    def test_build_cache_uses_empirical(self):
        cache = build_resolution_psf_cache(
            'euclid',
            ['EUCLID_VIS', 'EUCLID_Y', 'EUCLID_J', 'EUCLID_H'],
            0.10,
            psf_size=101,
        )
        assert is_euclid_q1_psf_data(cache)
        assert 'synthetic' not in cache


def test_euclid_q1_enabled_flag():
    assert euclid_q1_enabled({'telescope': 'euclid', 'euclid_q1': {'enabled': True}})
    assert not euclid_q1_enabled({'telescope': 'jwst', 'euclid_q1': {'enabled': True}})
    assert not euclid_q1_enabled({'telescope': 'euclid', 'euclid_q1': {'enabled': False}})


@pytest.mark.skipif(pytest is None or not DATA_DIR.exists(), reason='pytest or Euclid Q1 data missing')
def test_apply_priors_to_dataframe():
    import pandas as pd
    config = {
        'telescope': 'euclid',
        'euclid_q1': {
            'enabled': True,
            'data_dir': str(DATA_DIR),
            'use_population_priors': True,
            'use_q1_photometry': True,
            'use_q1_shear': True,
            'resample_theta_E': True,
        },
        'photometry': {
            'lens_mag_min': 18.0, 'lens_mag_max': 24.0,
            'source_mag_min': 19.0, 'source_mag_max': 27.0,
            'min_source_fainter_than_lens_mag': 1.0,
        },
        'telescope_configs': {
            'euclid': {'bands': ['EUCLID_VIS', 'EUCLID_Y', 'EUCLID_J', 'EUCLID_H']},
        },
    }
    df = pd.DataFrame({'theta_E': [1.0, 1.0], 'lens_id': ['a', 'b']})
    rng = np.random.default_rng(42)
    out = apply_euclid_q1_physics(df, config, rng)
    out = apply_euclid_q1_photometry(out, config, rng)
    assert 'lens_mag_euclid_vis' in out.columns
    assert 'shear_gamma1' in out.columns
    assert 'euclid_psf_tile' in out.columns
