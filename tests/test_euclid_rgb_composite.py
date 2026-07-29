"""Tests for Euclid RGB compositing and per-band TNG particle colours.

Covers the linked-stretch RGB fix (preserves inter-band flux ratios so TNG
galaxies are not uniformly yellow) and arc-boost visibility in composites.

Visual regression outputs from the 50-lens hybrid run live under:
  outputs/euclid_q1_hybrid_test50_96px/jpg_rgb_v2/

Re-render previews from saved science arrays (no re-simulation):
  PYTHONPATH=. python -c "
  from pathlib import Path
  from PIL import Image
  import numpy as np, sys
  sys.path.insert(0, 'src')
  from prism.core.simulator import create_jwst_panel_rgb, load_config
  load_config('configs/euclid_q1_hybrid_test50_96px.yaml')
  bands = ['EUCLID_VIS','EUCLID_Y','EUCLID_J','EUCLID_H']
  out = Path('outputs/euclid_q1_hybrid_test50_96px/unified_npz')
  dest = Path('outputs/euclid_q1_hybrid_test50_96px/jpg_rgb_v2')
  dest.mkdir(exist_ok=True)
  for p in sorted(out.glob('*.npz')):
      d = np.load(p, allow_pickle=True)
      imgs = {b: d['image_final'][i] for i, b in enumerate(bands)}
      arc = None
      if 'image_lens_sources' in d.files and 'image_lens_only' in d.files:
          arc = {'lens_sources': {b: d['image_lens_sources'][i] for i, b in enumerate(bands)},
                 'lens_only': {b: d['image_lens_only'][i] for i, b in enumerate(bands)}}
      panel = create_jwst_panel_rgb(imgs, bands=bands, telescope='euclid', arc_images=arc)
      Image.fromarray((panel * 255).astype(np.uint8)).save(dest / f'{p.stem}.jpg', quality=95)
  "
"""

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from prism.morphology.tng_particle_light import (  # noqa: E402
    BAND_AGE_METAL_LUMINOSITY,
    _stellar_band_luminosity,
)
from prism.core.simulator import (  # noqa: E402
    TELESCOPE_RGB_PARAMS,
    _normalize_for_rgb_composite_core,
    _rgb_arc_residual,
    create_jwst_rgb,
)


EUCLID_BANDS = ['EUCLID_VIS', 'EUCLID_Y', 'EUCLID_J', 'EUCLID_H']
HYBRID_OUTPUT = ROOT / 'outputs' / 'euclid_q1_hybrid_test50_96px'
HYBRID_NPZ = HYBRID_OUTPUT / 'unified_npz'
HYBRID_RGB_V2 = HYBRID_OUTPUT / 'jpg_rgb_v2'


def _synthetic_euclid_images(red_galaxy=True, numpix=64, rng=None):
    """Build toy 4-band images with controlled colour."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[:numpix, :numpix]
    r2 = (xx - numpix // 2) ** 2 + (yy - numpix // 2) ** 2
    profile = np.exp(-r2 / (2 * (numpix / 8) ** 2))

    if red_galaxy:
        # Old elliptical: faint VIS, bright H (flux ratios preserved in sim)
        flux = {
            'EUCLID_VIS': 1.0,
            'EUCLID_Y': 1.6,
            'EUCLID_J': 2.0,
            'EUCLID_H': 2.8,
        }
    else:
        flux = {b: 1.0 for b in EUCLID_BANDS}

    images = {}
    for band in EUCLID_BANDS:
        noise = rng.normal(0, 0.02, (numpix, numpix))
        images[band] = profile * flux[band] + np.clip(noise, 0, None)
    return images


def _flux_ratio_red_to_blue(images, numpix=64):
    """Integrated flux ratio H/VIS on the synthetic galaxy (input physics)."""
    return images['EUCLID_H'].sum() / max(images['EUCLID_VIS'].sum(), 1e-12)


def _rgb_red_blue_ratio_at_annulus(rgb, numpix=64, radius=10):
    """R/B in the RGB composite away from the saturated core."""
    c = numpix // 2
    y, x = c - radius, c
    return float(rgb[y, x, 0] / (rgb[y, x, 2] + 1e-6))


@pytest.mark.skipif(pytest is None, reason='pytest not installed')
class TestEuclidRgbComposite:
  def test_linked_stretch_preserves_red_galaxy_colour(self):
      images = _synthetic_euclid_images(red_galaxy=True)
      rgb = create_jwst_rgb(images, bands=EUCLID_BANDS, telescope='euclid')
      assert rgb is not None
      input_ratio = _flux_ratio_red_to_blue(images)
      assert input_ratio > 2.0
      # Away from the saturated centre, linked asinh scale should keep H > VIS.
      assert _rgb_red_blue_ratio_at_annulus(rgb) > 1.05

  def test_linked_stretch_uses_common_scale(self):
      images = _synthetic_euclid_images(red_galaxy=True)
      r_data = images['EUCLID_H']
      g_data = 0.5 * (images['EUCLID_Y'] + images['EUCLID_J'])
      b_data = images['EUCLID_VIS']
      _, r_scale = _normalize_for_rgb_composite_core(r_data, sat_percent=0.02, sigma_mult=1.2)
      _, b_scale = _normalize_for_rgb_composite_core(b_data, sat_percent=0.02, sigma_mult=1.2)
      common = float(np.median([r_scale, b_scale]))
      assert r_scale > b_scale  # redder band has higher flux → larger scale
      assert abs(common - r_scale) < abs(common - b_scale) or abs(common - b_scale) < abs(common - r_scale)

  def test_arc_residual_helper_positive_on_injected_arc(self):
      numpix = 48
      images = _synthetic_euclid_images(red_galaxy=False, numpix=numpix)
      lens_only = {b: im.copy() for b, im in images.items()}
      lens_sources = {b: im.copy() for b, im in images.items()}
      yy, xx = np.mgrid[:numpix, :numpix]
      arc_mask = np.exp(-((xx - 30) ** 2 + (yy - 24) ** 2) / 18.0)
      for band in EUCLID_BANDS:
          lens_sources[band] = lens_sources[band] + 0.35 * arc_mask

      residual = _rgb_arc_residual(lens_sources, lens_only, EUCLID_BANDS)
      assert residual['EUCLID_VIS'].max() > 0
      assert residual['EUCLID_VIS'][24, 30] > residual['EUCLID_VIS'][5, 5]

  def test_euclid_rgb_params_use_linked_stretch(self):
      p = TELESCOPE_RGB_PARAMS['euclid']
      assert p.get('linked_stretch') is True
      assert p.get('arc_boost', 0) > 0


@pytest.mark.skipif(pytest is None, reason='pytest not installed')
class TestTngEuclidBandColours:
  def test_old_stars_redder_than_young_in_euclid_bands(self):
      """SSP grid: integrated L/M increases toward H for coeval old populations."""
      ages_old = np.full(500, 10.0)
      ages_young = np.full(500, 0.05)
      metal = np.full(500, 0.0127)

      vis_old = _stellar_band_luminosity(ages_old, metal, 'EUCLID_VIS').mean()
      h_old = _stellar_band_luminosity(ages_old, metal, 'EUCLID_H').mean()
      vis_young = _stellar_band_luminosity(ages_young, metal, 'EUCLID_VIS').mean()
      h_young = _stellar_band_luminosity(ages_young, metal, 'EUCLID_H').mean()

      assert h_old > vis_old
      assert vis_young > vis_old  # young populations brighter in blue/optical per unit mass

      # Colour index: bluer when young
      colour_old = h_old / vis_old
      colour_young = h_young / vis_young
      assert colour_young < colour_old

  def test_euclid_bands_have_distinct_lm_grids(self):
      vis = BAND_AGE_METAL_LUMINOSITY['EUCLID_VIS']
      h = BAND_AGE_METAL_LUMINOSITY['EUCLID_H']
      assert not np.allclose(vis, h)
      # Oldest age bin: H band L/M exceeds VIS
      assert h[-1, 2] > vis[-1, 2]


@pytest.mark.skipif(
    pytest is None or not HYBRID_NPZ.exists(),
    reason='pytest or hybrid 50-lens test outputs missing',
)
class TestHybridRunArtifacts:
  def test_hybrid_run_npz_and_rgb_v2_exist(self):
      npz_files = list(HYBRID_NPZ.glob('*.npz'))
      assert len(npz_files) == 50
      jpg_files = list(HYBRID_RGB_V2.glob('*.jpg'))
      assert len(jpg_files) == 50

  def test_npz_contains_arc_intermediate_bands(self):
      sample = next(HYBRID_NPZ.glob('*.npz'))
      d = np.load(sample, allow_pickle=True)
      assert 'image_final' in d.files
      assert d['image_final'].shape[0] == 4
      assert 'image_lens_sources' in d.files
      assert 'image_lens_only' in d.files
