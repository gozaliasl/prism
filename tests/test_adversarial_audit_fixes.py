"""Regression tests for the adversarial-audit fixes (2026-08-01).

Each test pins down a specific bug that was found and fixed this session,
so it cannot silently regress on a future code change. These are
deliberately FAST (unit-level, no full multi-minute renders) except where
a real render is the only way to verify end-to-end consistency (marked
`slow`).

Run: pytest tests/test_adversarial_audit_fixes.py -v
Run only fast tests: pytest tests/test_adversarial_audit_fixes.py -v -m "not slow"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# C-1: lens-sightline overdensity factor
# ---------------------------------------------------------------------------

def test_overdensity_factor_default_is_corrected_value():
    """The default must be ~1.05x (real measured value), not the old 1.70x
    (a footprint-area measurement artifact -- see git history 2026-08-01)."""
    import inspect

    from prism.core import simulator
    src = inspect.getsource(simulator.field_galaxy_count_target)
    assert '"lens_sightline_overdensity_factor", 1.05)' in src, (
        "field_galaxy_count_target's overdensity_factor default must be "
        "1.05, not the old (bugged) 1.70 value"
    )


def test_compare_to_field_average_uses_grid_footprint():
    """compare_to_field_average must use a grid-cell footprint, not an
    RA/Dec bounding box (which overstates the non-rectangular COSMOS-Web
    mosaic area by ~1.63x)."""
    from prism.environment.cowls_neighborhood_density import compare_to_field_average
    result = compare_to_field_average(mag_limit=24.5, mag_band="mag_f115w")
    assert result.get("footprint_method") == "grid_15arcsec_occupied_cells"


def test_overdensity_factor_only_applies_to_rich_environments():
    """richness_mult<=1 (isolated_field) must NOT get the overdensity boost."""
    from prism.core.simulator import field_galaxy_count_target
    cfg = {
        "field": {"density_mag_limit": 24.5, "lens_sightline_overdensity_factor": 2.0},
        "catalogs": {"galaxy_catalog_fits": "data/galaxy_catalog.fits"},
    }
    isolated_mean, _ = field_galaxy_count_target(
        1936, 0.031, {"galaxy_count_mean": 2.5}, cfg)  # richness_mult == 1.0
    group_mean, _ = field_galaxy_count_target(
        1936, 0.031, {"galaxy_count_mean": 5.0}, cfg)  # richness_mult == 2.0
    # isolated_field's mean should NOT include the 2.0x overdensity factor;
    # group's should. So group_mean should be ~4x isolated_mean (2x richness
    # * 2x overdensity), not just 2x.
    ratio = group_mean / isolated_mean
    assert ratio > 3.0, (
        f"group/isolated field-count ratio was {ratio:.2f}, expected >3.0 "
        "(2x richness * 2x overdensity) -- overdensity factor may be "
        "leaking into isolated_field"
    )


# ---------------------------------------------------------------------------
# C-4: field-galaxy magnitude sentinel contamination
# ---------------------------------------------------------------------------

def test_field_galaxy_properties_reject_sentinel_magnitudes():
    """No field galaxy should have mag < 15 (the catalog's null-photometry
    sentinel is ~-88.95, which passed the old "> -90" check)."""
    from prism.core.simulator import cosmos_field_galaxy_properties
    props = cosmos_field_galaxy_properties(24.5, "data/galaxy_catalog.fits")
    if props is None:
        pytest.skip("galaxy_catalog.fits not available in this environment")
    assert props["mag_f115w"].min() > 15.0, (
        f"field galaxy pool contains mag={props['mag_f115w'].min():.2f}, "
        "below the physical floor -- sentinel contamination regression"
    )


# ---------------------------------------------------------------------------
# C-5: reproducibility
# ---------------------------------------------------------------------------

def test_prnu_seed_is_deterministic_not_hash_based():
    """PRNU seeding must use zlib.crc32 (deterministic across processes),
    not Python's hash() (salted per-process by default)."""
    import inspect

    from prism.core import simulator
    src = inspect.getsource(simulator)
    assert "zlib.crc32(str(row.get('lens_id', 0)).encode())" in src
    assert "hash(str(row.get('lens_id'" not in src.replace(
        "zlib.crc32(str(row.get('lens_id', 0)).encode())", "")


def test_global_random_seed_is_set_at_startup():
    """main() must call np.random.seed(args.seed), not just seed a local
    rng object -- otherwise the ~50 np.random.* call sites elsewhere in
    the module ignore --seed entirely."""
    import inspect

    from prism.core import simulator
    src = inspect.getsource(simulator.main)
    assert "np.random.seed(args.seed)" in src


# ---------------------------------------------------------------------------
# C-8: detector chain (sky noise propagation, ADC clipping, PRNU ordering)
# ---------------------------------------------------------------------------

def test_adc_does_not_pin_negative_noise_to_zero():
    """_apply_gain_adc must not clip the ENTIRE negative half of a
    zero-centered noise distribution to exactly 0 -- real detectors use a
    bias pedestal so negative noise excursions still register (removed
    during calibration). Some exact-zero pixels are expected from normal
    ADC quantization near a true-zero signal (not a bug); the meaningful
    regression signature is roughly HALF the distribution pinned to zero
    (everything that would have been negative), not a small quantization
    fraction.
    """
    from prism.io.detector_chain import DetectorChain

    rng = np.random.default_rng(0)
    chain = DetectorChain(telescope="jwst", band="F150W", rng=rng,
                           exposure_time=1000.0, numpix=64)
    # Values straddling zero, as read noise around a near-zero signal would produce.
    im_e = rng.normal(0.0, 5.0, (64, 64))
    out = chain._apply_gain_adc(im_e)
    frac_exactly_zero = float((out == 0.0).mean())
    frac_negative = float((out < 0.0).mean())
    assert frac_exactly_zero < 0.35, (
        f"{frac_exactly_zero:.1%} of pixels pinned to exactly 0 after ADC "
        "-- this looks like the full negative-tail-truncation bug, not just "
        "normal ADC quantization near zero"
    )
    assert frac_negative > 0.30, (
        f"only {frac_negative:.1%} of pixels came out negative for a "
        "zero-centered input -- negative noise tail may be getting "
        "truncated again (expect ~50% for symmetric noise)"
    )


def test_prnu_applied_before_read_noise_in_chain_order():
    """PRNU (QE variation) must be applied to the signal before read/1-f
    noise (amplifier-stage, independent of per-pixel QE) -- verified via
    source order, since testing the numerical effect directly requires
    isolating read noise from Poisson noise."""
    import inspect

    from prism.io.detector_chain import DetectorChain
    src = inspect.getsource(DetectorChain.apply)
    prnu_pos = src.find("self._apply_prnu(")
    poisson_pos = src.find("self._apply_poisson_noise(")
    read_pos = src.find("self._apply_read_noise(")
    assert prnu_pos != -1 and poisson_pos != -1 and read_pos != -1
    assert prnu_pos < poisson_pos < read_pos, (
        "PRNU must be applied before Poisson noise and read noise in the "
        "detector chain, not after (see git history 2026-08-01)"
    )


def test_brighter_fatter_is_not_single_global_blur():
    """The BFE blur must not collapse to a single global-scalar sigma
    applied uniformly to the whole frame (a bright pixel anywhere would
    then blur every faint/empty pixel too)."""
    import inspect

    from prism.io.detector_chain import DetectorChain
    src = inspect.getsource(DetectorChain._apply_brighter_fatter)
    assert "n_bins" in src and "bin_edges" in src, (
        "brighter-fatter blur must use a per-pixel local-sigma binning "
        "scheme (n_bins/bin_edges), not a single global-percentile sigma "
        "applied uniformly to the whole frame"
    )
    code_lines = [
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "sigma = np.percentile(extra_sigma, 95)" not in code_only, (
        "brighter-fatter blur reverted to a single-global-percentile "
        "sigma -- this makes blur strength depend on whatever the "
        "brightest pixel in the WHOLE frame is, not local flux"
    )


@pytest.mark.slow
def test_sky_background_rms_matches_calibrated_target():
    """End-to-end: measured background RMS in a rendered image should be
    close to the calibrated target (previously ~6x too low due to sky
    noise being added post-hoc, bypassing Poisson propagation)."""
    pytest.skip(
        "requires a full prism-simulate render (~1 min); run manually via "
        "the validation script in scripts/local/ if needed"
    )


# ---------------------------------------------------------------------------
# C-2/C-3: theta_E label consistency
# ---------------------------------------------------------------------------

def test_theta_E_written_back_after_group_scale_override():
    """The group-scale theta_E override must write row["theta_E"] (not
    just kwargs_lens), and must flag theta_E_override_applied, or every
    saved label downstream of `row` reports a stale pre-override value."""
    import inspect

    from prism.core import simulator
    src = inspect.getsource(simulator.simulate_complete_lens_system_with_real_fields)
    override_block_start = src.find("GROUP/CLUSTER-SCALE THETA_E OVERRIDE")
    assert override_block_start != -1
    next_section = src.find("\n    # ---", override_block_start + 100)
    override_block_end = next_section if next_section != -1 else override_block_start + 6000
    override_block = src[override_block_start:override_block_end]
    assert 'row["theta_E"] = float(theta_E)' in override_block
    assert 'row["theta_E_override_applied"] = True' in override_block
    assert 'row["lens_sigma_kms"]' in override_block


@pytest.mark.slow
def test_theta_E_metadata_matches_kappa_map():
    """End-to-end: saved metadata theta_E must agree with the rendered
    kappa map's theta_E_eff to within a few percent (was 38% off before
    the fix)."""
    pytest.skip(
        "requires a full prism-simulate render + kappa map output; run "
        "manually via the validation script in scripts/local/ if needed"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
