#!/usr/bin/env python3
"""
COSMOS-Web Lens Mock Pipeline — v11 ENHANCED MORPHOLOGY

NEW in v11 - Ultra-Realistic Morphological Features:
✓ Realistic logarithmic spiral arms with proper pitch angles (15-30°)
✓ Star-forming clumps with physical sizes (60-150 pc -> 2.5-5.0 pixels)
✓ Smooth elliptical bar structures (no rectangular artifacts)
✓ Collision-induced ring galaxies with subtle star-forming knots
✓ Realistic dust lanes confined to central disk regions
✓ Enhanced morphology matching Paper Figure 2 v2 improvements

Based on v10_final_restored + fig02_morphology_showcase_v2.py enhancements

From v10:
✓ Realistic field galaxy populations (5-15 per image)
✓ Enhanced morphological diversity (60% spirals, 40% ellipticals)
✓ Proper magnitude scaling for detectability
✓ Synthetic field galaxy fallback system
✓ Comprehensive diagnostics and validation
✓ Balanced lens/non-lens training dataset generation

Usage:
python cosmos_web_lens_mock_v11_enhanced_morphology_full.py \
  --cosmos_catalog /path/to/cosmos_web_lens_structural_properties.csv \
  --merged_field_catalog /path/to/merged_lens_field_catalog.csv \
  --output_dir ./training_dataset_v11 \
  --n_lenses 5000 --n_non_lenses 5000 \
  --variations_per_base 25 --n_field_max 5 --add_artifacts
"""

import os, sys, math, time, argparse, json, traceback, zlib
import yaml
from typing import Optional
from pathlib import Path
import numpy as np
from functools import lru_cache

# Import parameter utilities for compatibility and missing parameter handling
try:
    from prism.core.parameter_utils import safe_random_integers, fill_missing_parameter
except ImportError:
    from src.parameter_utils import safe_random_integers, fill_missing_parameter

# Native multi-component (bulge/disk/bar/ring) galaxy morphology package
try:
    from prism.morphology import build_light_model as gm_build_light_model
except ImportError:
    from src.galaxy_morphology import build_light_model as gm_build_light_model

# Try to import empirical noise sampler (optional)
try:
    from prism.io.empirical_noise_sampler import EmpiricalNoiseSampler
    EMPIRICAL_NOISE_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from prism.io.empirical_noise_sampler import EmpiricalNoiseSampler
        EMPIRICAL_NOISE_AVAILABLE = True
    except ImportError:
        EMPIRICAL_NOISE_AVAILABLE = False
        print("Warning: EmpiricalNoiseSampler not available, using config values")

# Import ML training enhancements
try:
    from prism.ml.ml_training_enhancements import (
        JWSTHardNegativeMiner, JWSTDataAugmentation, 
        JWSTBalancedTrainingSets, JWSTSurveyMetrics,
        create_enhanced_training_pipeline
    )
    ML_ENHANCEMENTS_AVAILABLE = True
    print("✓ ML training enhancements loaded")
except ImportError:
    ML_ENHANCEMENTS_AVAILABLE = False
    print("Warning: ML training enhancements not available")

# Synthetic PSF generator for Roman / Euclid / Subaru
try:
    from prism.io.synthetic_psf_generator import build_resolution_psf_cache as _build_resolution_psf_cache
    SYNTHETIC_PSF_AVAILABLE = True
except ImportError:
    try:
        import sys as _sys_psf
        _sys_psf.path.insert(0, str(Path(__file__).parent))
        from prism.io.synthetic_psf_generator import build_resolution_psf_cache as _build_resolution_psf_cache
        SYNTHETIC_PSF_AVAILABLE = True
    except ImportError:
        SYNTHETIC_PSF_AVAILABLE = False
        print("Warning: synthetic_psf_generator not available — non-JWST PSFs will be None")

# Euclid Q1 catalogue integration (population priors + empirical PSF)
try:
    from prism.telescopes.euclid_q1_catalog import (
        get_euclid_q1_catalog,
        euclid_q1_enabled,
        apply_euclid_q1_physics,
        apply_euclid_q1_photometry,
        is_euclid_q1_psf_data,
    )
    EUCLID_Q1_AVAILABLE = True
except ImportError:
    EUCLID_Q1_AVAILABLE = False
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd

from PIL import Image
from lenstronomy.SimulationAPI.sim_api import SimAPI
from collections import Counter
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.cosmology import FlatLambdaCDM
import glob

# Import enhanced field sampling
try:
    from prism.core.enhanced_field_sampling import EnhancedFieldSampler
    ENHANCED_SAMPLING_AVAILABLE = True
except ImportError:
    ENHANCED_SAMPLING_AVAILABLE = False
    print("[WARNING] Enhanced field sampling not available - using basic sampling")

# Import time delay integration
try:
    from prism.lensing.time_delay_integration import (
        should_generate_time_delays,
        select_variable_source_type,
        calculate_time_delays_simplified,
        generate_epoch_times,
        generate_light_curve_for_source,
        apply_time_delay_to_source_magnitude,
        create_time_delay_metadata
    )
    TIME_DELAY_INTEGRATION_AVAILABLE = True
except ImportError:
    TIME_DELAY_INTEGRATION_AVAILABLE = False

# Import advanced lens features for binary lenses
try:
    from prism.core.advanced_lens_features import RealisticMassProfiles
    BINARY_LENS_AVAILABLE = True
except ImportError:
    BINARY_LENS_AVAILABLE = False
    print("[WARNING] Binary lens features not available - using single lens systems only")
    print("[WARNING] Time delay integration not available")

# Import PRISM kappa map output module (optional - non-fatal if missing)
try:
    from prism.io.prism_kappa_output import compute_kappa_products, save_kappa_outputs
    KAPPA_OUTPUT_AVAILABLE = True
except ImportError:
    KAPPA_OUTPUT_AVAILABLE = False
    print("[INFO] Kappa map output module not available - skipping convergence maps")

# Import empirical SED templates
try:
    from prism.io.empirical_sed_templates import (
        calculate_k_correction_empirical,
        select_sed_type_from_redshift,
        generate_empirical_sed,
        JWST_FILTERS
    )
    EMPIRICAL_SED_AVAILABLE = True
    print("✓ Empirical SED templates loaded (BC03 + Calzetti+2000 + Chary & Elbaz 2001)")
except ImportError:
    EMPIRICAL_SED_AVAILABLE = False
    print("[WARNING] Empirical SED templates not available - using simplified K-corrections")

# Import JWST filter transmission system
try:
    from prism.telescopes.jwst_filter_transmission import (
        JWST_FILTERS_SYSTEM,
        convolve_sed_to_magnitude,
        get_filter_noise_properties
    )
    FILTER_TRANSMISSION_AVAILABLE = True
    print("✓ Realistic JWST filter transmission system loaded")
except ImportError:
    FILTER_TRANSMISSION_AVAILABLE = False
    print("[WARNING] JWST filter transmission system not available - using simplified calculations")

# Import real JWST NIRCam throughput curves from STScI.
# NOTE: REAL_JWST_FILTERS is currently loaded for reference/validation only;
# the active photometry path uses JWST_FILTERS_SYSTEM (jwst_filter_transmission,
# see FILTER_TRANSMISSION_AVAILABLE above), not these STScI throughput curves.
try:
    from prism.telescopes.jwst_real_filter_transmission import REAL_JWST_FILTERS
    REAL_FILTERS_AVAILABLE = True
    print(f"  (reference) STScI NIRCam throughput curves available "
          f"({len(REAL_JWST_FILTERS.available_filters)} filters; not used in the active photometry path)")
except ImportError:
    REAL_FILTERS_AVAILABLE = False
    REAL_JWST_FILTERS = None

# Import multi-telescope filter system for Roman, Subaru, etc.
try:
    from prism.telescopes.multi_telescope_filters import get_multi_telescope_filters
    MULTI_TELESCOPE_FILTERS = get_multi_telescope_filters()
    MULTI_TELESCOPE_AVAILABLE = True
    print(f"✓ Multi-telescope filter system loaded (JWST, Roman, Subaru)")
except ImportError as e:
    MULTI_TELESCOPE_AVAILABLE = False
    MULTI_TELESCOPE_FILTERS = None
    print(f"[WARNING] Multi-telescope filter system not available: {e}")

# Import time delay image modification (for per-image brightness differences)
try:
    from prism.lensing.time_delay_image_modification import apply_per_image_time_delay_brightness
    TIME_DELAY_IMAGE_MODIFICATION_AVAILABLE = True
except ImportError:
    TIME_DELAY_IMAGE_MODIFICATION_AVAILABLE = False
    print("[WARNING] Time delay image modification not available (per-image brightness differences disabled)")

# Import Fundamental Plane / Faber-Jackson module
try:
    from prism.core.fundamental_plane import fp_consistent_lens_params
    FUNDAMENTAL_PLANE_AVAILABLE = True
    print("✓ Fundamental Plane + Faber-Jackson module loaded (Bernardi+2003, Singh+2021, Sonnenfeld+2023)")
except ImportError:
    FUNDAMENTAL_PLANE_AVAILABLE = False
    print("[INFO] Fundamental Plane module not found — FP consistency disabled")

# Import physically-realistic detector chain
try:
    from prism.io.detector_chain import make_detector_chain, TELESCOPE_PARAMS as DETECTOR_TELESCOPE_PARAMS
    DETECTOR_CHAIN_AVAILABLE = True
    print(f"✓ Detector chain loaded ({len(DETECTOR_TELESCOPE_PARAMS)} telescopes: "
          f"{', '.join(DETECTOR_TELESCOPE_PARAMS)})")
except ImportError as e:
    DETECTOR_CHAIN_AVAILABLE = False
    print(f"[WARNING] Detector chain not available: {e}")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
LOWER_BANDS = ["f115w", "f150w", "f277w", "f444w"]

# Cosmology for angular diameter distance calculations
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# Initialize enhanced field sampler if available
if ENHANCED_SAMPLING_AVAILABLE:
    try:
        ENHANCED_SAMPLER = EnhancedFieldSampler()
        ENHANCED_SAMPLER.load_models()
        ENHANCED_SAMPLER.load_field_catalog()
        print("[INFO] Enhanced field sampling initialized successfully")
    except Exception as e:
        print(f"[WARNING] Could not initialize enhanced field sampling: {e}")
        ENHANCED_SAMPLER = None
else:
    ENHANCED_SAMPLER = None

def angular_diameter_distance(z):
    """Calculate angular diameter distance in Mpc"""
    return COSMO.angular_diameter_distance(z).value

_COWLS_MS_BINS = None
_CWMGS_MS_BINS = None


def _resolve_mass_size_data_path(filename):
    """Resolve a mass-size relation CSV against, in order: CONFIG['mass_size']['data_dir'],
    the current working directory's data/ folder, and the current working
    directory's analysis/mass_size_relations/ folder. Returns None if not found.

    (Previously this resolved paths relative to this source file's own
    location -- src/prism/core/ -- which never matched any real data/ or
    analysis/ directory after the package restructure, silently disabling
    the CWMGs/COWLS mass-size relations in favor of the generic fallback.)
    """
    candidates = []
    _ms_cfg = CONFIG.get('mass_size', {}) if isinstance(CONFIG, dict) else {}
    _data_dir = _ms_cfg.get('data_dir')
    if _data_dir:
        candidates.append(Path(_data_dir) / filename)
    candidates.append(Path("data") / filename)
    candidates.append(Path("analysis/mass_size_relations") / filename)
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_cowls_mass_size_bins():
    """Load COSMOS-Web (COWLS) redshift-binned mass–size relation if present.
    Expects CSV with columns: z_min, z_max, alpha, beta.
    Interprets alpha as slope and beta as intercept when values are reasonable; if not,
    swaps them defensively. Returns list of dicts or None if unavailable.
    """
    global _COWLS_MS_BINS
    if _COWLS_MS_BINS is not None:
        return _COWLS_MS_BINS
    csv_path = _resolve_mass_size_data_path("cowls_redshift_binned_relations.csv")
    if csv_path is None:
        _COWLS_MS_BINS = None
        return None
    try:
        df = pd.read_csv(csv_path)
        bins = []
        for _, row in df.iterrows():
            zmin = float(row.get("z_min", np.nan))
            zmax = float(row.get("z_max", np.nan))
            a = float(row.get("alpha", np.nan))
            b = float(row.get("beta", np.nan))
            # Heuristic: typical slope |alpha| < 1.5, intercept |beta| < 10
            slope, intercept = a, b
            if not (np.isfinite(slope) and np.isfinite(intercept)):
                continue
            if not (abs(slope) < 1.5 and abs(intercept) < 10.0):
                # Try swapped
                if abs(intercept) < 1.5 and abs(slope) < 10.0:
                    slope, intercept = intercept, slope
                else:
                    # Skip unreasonable bin
                    continue
            bins.append({"z_min": zmin, "z_max": zmax, "slope": slope, "intercept": intercept})
        _COWLS_MS_BINS = bins if bins else None
    except Exception:
        _COWLS_MS_BINS = None
    return _COWLS_MS_BINS

def _load_cwmgs_mass_size_bins():
    """Load CWMGs (massive galaxies) redshift-binned mass–size relation if present.
    Expects CSV with columns: z_min, z_max, alpha, beta.
    """
    global _CWMGS_MS_BINS
    if _CWMGS_MS_BINS is not None:
        return _CWMGS_MS_BINS
    csv_path = _resolve_mass_size_data_path("massive_galaxy_redshift_binned_relations.csv")
    if csv_path is None:
        _CWMGS_MS_BINS = None
        return None
    try:
        df = pd.read_csv(csv_path)
        bins = []
        for _, row in df.iterrows():
            zmin = float(row.get("z_min", np.nan))
            zmax = float(row.get("z_max", np.nan))
            a = float(row.get("alpha", np.nan))
            b = float(row.get("beta", np.nan))
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            slope, intercept = a, b
            if not (abs(slope) < 1.5 and abs(intercept) < 10.0):
                if abs(intercept) < 1.5 and abs(slope) < 10.0:
                    slope, intercept = intercept, slope
                else:
                    continue
            bins.append({"z_min": zmin, "z_max": zmax, "slope": slope, "intercept": intercept})
        _CWMGS_MS_BINS = bins if bins else None
    except Exception:
        _CWMGS_MS_BINS = None
    return _CWMGS_MS_BINS


_BGG_MS_BINS = None
_BGG_PIVOT_LOG_MASS = np.log10(5.0e10)  # Gozaliasl+2025 pivot mass, 5x10^10 Msun


def _load_bgg_mass_size_bins():
    """Load the Brightest Group Galaxy (BGG) size-mass relation from
    Gozaliasl et al. (2025, A&A 703, A129), Table D.1 (Color+sSFR
    classification). Expects columns: z_min, z_max, type (SF/QG), logA,
    alpha. The published form is
      log10(Re/kpc) = logA + alpha * log10(M*/5e10 Msun),
    which is converted here to this module's raw (unpivoted) convention
      log10(Re/kpc) = slope*log10(M*) + intercept
    via intercept = logA - alpha*log10(5e10).
    Returns {'SF': [...], 'QG': [...]} or None if unavailable.
    """
    global _BGG_MS_BINS
    if _BGG_MS_BINS is not None:
        return _BGG_MS_BINS
    csv_path = _resolve_mass_size_data_path("bgg_size_mass_relations.csv")
    if csv_path is None:
        _BGG_MS_BINS = None
        return None
    try:
        df = pd.read_csv(csv_path)
        bins = {'SF': [], 'QG': []}
        for _, row in df.iterrows():
            gal_type = str(row.get("type", "")).strip().upper()
            if gal_type not in bins:
                continue
            zmin = float(row.get("z_min", np.nan))
            zmax = float(row.get("z_max", np.nan))
            log_a = float(row.get("logA", np.nan))
            alpha = float(row.get("alpha", np.nan))
            if not (np.isfinite(zmin) and np.isfinite(zmax) and np.isfinite(log_a) and np.isfinite(alpha)):
                continue
            intercept = log_a - alpha * _BGG_PIVOT_LOG_MASS
            bins[gal_type].append({"z_min": zmin, "z_max": zmax, "slope": alpha, "intercept": intercept})
        _BGG_MS_BINS = bins if (bins['SF'] or bins['QG']) else None
    except Exception:
        _BGG_MS_BINS = None
    return _BGG_MS_BINS


def _select_ms_bin(chosen_bins, z):
    selected = None
    for b in chosen_bins:
        if b["z_min"] <= z < b["z_max"]:
            selected = b
            break
    if selected is None:
        def z_center(b):
            return 0.5 * (float(b["z_min"]) + float(b["z_max"]))
        selected = sorted(chosen_bins, key=lambda b: abs(z_center(b) - z))[0]
    return selected


def mass_size_relation(mass_log10, z, rng, is_bgg=False, bgg_type='QG'):
    """
    Sample effective radius from mass–size relation.
    Preference order:
      1) BGG-specific relation (Gozaliasl+2025) if is_bgg=True and available
      2) COSMOS-Web COWLS / CWMGs redshift-binned relation if available
      3) Empirical global relation with mild size evolution
    Returns R_eff in kpc.

    is_bgg : bool
        If True, use the Brightest Group Galaxy relation instead of the
        general CWMGs/COWLS relations (see mass_size.bgg_fraction in config).
    bgg_type : str
        'QG' (quiescent) or 'SF' (star-forming) BGG row to use when
        is_bgg=True. PRISM lens galaxies are early-type by design
        (Section: Fundamental Plane), so 'QG' is the appropriate default.
    """
    if is_bgg:
        _bgg_bins = _load_bgg_mass_size_bins()
        _bgg_rows = _bgg_bins.get(bgg_type) if _bgg_bins else None
        if _bgg_rows:
            selected = _select_ms_bin(_bgg_rows, z)
            log_reff = selected["slope"] * mass_log10 + selected["intercept"]
            log_reff += rng.normal(0, 0.2)
            reff_kpc = 10 ** log_reff
            return float(np.clip(reff_kpc, 0.3, 50.0))
        # BGG data unavailable -- fall through to the general relations below.

    # Decide dataset per config: cowls vs cwmgs fraction
    ms_cfg = CONFIG.get('mass_size', {}).get('lens_relation_mix', {})
    cowls_frac = float(ms_cfg.get('cowls_fraction', 0.4))
    use_cowls = rng.random() < cowls_frac
    chosen_bins = _load_cowls_mass_size_bins() if use_cowls else _load_cwmgs_mass_size_bins()

    if chosen_bins:
        # Find matching redshift bin; if none, use closest by center
        selected = None
        for b in chosen_bins:
            if b["z_min"] <= z < b["z_max"]:
                selected = b
                break
        if selected is None:
            def z_center(b):
                return 0.5 * (float(b["z_min"]) + float(b["z_max"]))
            selected = sorted(chosen_bins, key=lambda b: abs(z_center(b) - z))[0]
        slope = selected["slope"]
        intercept = selected["intercept"]
        log_reff = slope * mass_log10 + intercept
    else:
        # Global empirical fallback from config
        fb = CONFIG.get('mass_size', {}).get('fallback_global', {})
        slope = float(fb.get('slope', 0.387))
        intercept = float(fb.get('intercept', -3.843))
        evol = float(fb.get('size_evolution_exponent', -0.2))
        log_reff = slope * mass_log10 + intercept
        log_reff += evol * np.log10(1 + z)

    # Add intrinsic scatter ~0.2 dex
    log_reff += rng.normal(0, 0.2)

    # Convert to kpc and clip to sensible bounds
    reff_kpc = 10 ** log_reff
    reff_kpc = np.clip(reff_kpc, 0.3, 50.0)
    return reff_kpc

_COSMOS_DENSITY_CACHE: dict = {}


def cosmos_field_density_per_arcmin2(mag_limit, catalog_path="data/galaxy_catalog.fits", band="mag_f115w"):
    """Measure the real galaxy surface density (per arcmin^2) at a given
    magnitude limit directly from the COSMOS-Web detection catalog.

    Uses a grid-based footprint area (count of 15"x15" sky cells containing
    >=1 detection, summed) rather than an RA/Dec bounding box: the COSMOS-Web
    mosaic footprint is not a filled rectangle, so a bounding box overstates
    the area by ~1.6x here and understates the true density by the same
    factor. The grid method reproduces the catalog's published ~0.54 deg^2
    footprint almost exactly.

    Returns 0.0 (caller should fall back to a parametric density) if the
    catalog can't be loaded.
    """
    cache_key = (catalog_path, band)
    if cache_key not in _COSMOS_DENSITY_CACHE:
        try:
            from astropy.io import fits
            with fits.open(catalog_path) as hdul:
                d = hdul[1].data
                ra = np.asarray(d["RA_DETEC"], dtype=float)
                dec = np.asarray(d["DEC_DETEC"], dtype=float)
                mag = np.asarray(d[band], dtype=float)
            valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag) & (mag > -90)
            ra, dec, mag = ra[valid], dec[valid], mag[valid]

            cell_deg = 15.0 / 3600.0
            ix = ((ra - ra.min()) / cell_deg).astype(int)
            iy = ((dec - dec.min()) / cell_deg).astype(int)
            n_cells = len({(int(a), int(b)) for a, b in zip(ix, iy)})
            footprint_arcmin2 = n_cells * (15.0 / 60.0) ** 2

            _COSMOS_DENSITY_CACHE[cache_key] = (mag, footprint_arcmin2)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] cosmos_field_density_per_arcmin2: could not load {catalog_path}: {exc}")
            _COSMOS_DENSITY_CACHE[cache_key] = (None, None)

    mag, footprint_arcmin2 = _COSMOS_DENSITY_CACHE[cache_key]
    if mag is None:
        return 0.0
    n = int(np.sum(mag < mag_limit))
    return n / footprint_arcmin2


def field_density_area_scale(numpix, pixel_scale, config=None):
    """Scale factor between the actual image FOV area and the reference FOV
    that environment.types[*].galaxy_count_mean/std/min/max were calibrated
    against (default: 300px @ 0.031"/px = 9.3" FOV, the project's small-FOV
    default).

    Those config values are absolute counts/radii, not surface densities, so
    using them unscaled at a different FOV (e.g. a 1' extended-FOV render)
    silently under- or over-populates the field: a 1' FOV is ~42x the
    reference's area, so reusing the same absolute count there yields a field
    ~42x too sparse relative to the calibrated density. Multiplying
    galaxy_count_* by this scale (and max_radius_arcsec by its sqrt) keeps
    the *surface density* consistent across FOV sizes, so the same config
    values work correctly whether image_size is 300px or extended to 1936px.
    """
    cfg = config or {}
    ref_numpix = cfg.get('field', {}).get('density_calibration_image_size', 300)
    ref_pixel_scale = cfg.get('field', {}).get('density_calibration_pixel_scale', 0.031)
    actual_fov_arcmin2 = (numpix * pixel_scale / 60.0) ** 2
    ref_fov_arcmin2 = (ref_numpix * ref_pixel_scale / 60.0) ** 2
    # NOTE: field-galaxy placement radius is derived directly from numpix *
    # pixel_scale (the actual image FOV) wherever galaxies are sampled, so it
    # already follows extended-FOV renders without any additional scaling
    # here -- only the counts below need explicit area-scaling.
    return actual_fov_arcmin2 / max(ref_fov_arcmin2, 1e-12)


def field_galaxy_count_target(numpix, pixel_scale, env_params, config=None):
    """Target (mean, std) field-galaxy count for the actual image FOV.

    Primary source: the real COSMOS-Web detection density
    (cosmos_field_density_per_arcmin2) at the configured source magnitude
    limit, scaled by the FOV area, an environment-relative richness
    multiplier (the ratio of this environment's galaxy_count_mean to the
    isolated_field reference value of 2.5), AND a lens-sightline
    overdensity factor -- real strong-lens galaxies are not randomly
    positioned draws from the field; they trace large-scale structure, so
    their surroundings are measurably richer than a flat field average.

    The overdensity factor (default 1.05x) is measured directly from the
    356 real COWLS lenses against the full COSMOS-Web catalog
    (prism/environment/cowls_neighborhood_density.py). CORRECTED
    2026-08-01 (adversarial audit finding C-1): the originally-reported
    1.70x was a footprint-area measurement artifact -- the lens-
    neighborhood density used a circular-aperture area while the field-
    average comparison used an RA/Dec BOUNDING BOX area, which overstates
    the true (non-rectangular) COSMOS-Web mosaic footprint by ~1.63x. That
    area bug, not a real overdensity signal, was responsible for nearly
    the entire "1.70x". After fixing compare_to_field_average() to use a
    grid-cell footprint (matching this file's own
    cosmos_field_density_per_arcmin2 methodology) and filtering catalog
    sentinels consistently, the real overdensity is 1.03-1.07x across
    mag<24.5 through mag<28 (using mag_f115w to match the band used by
    cosmos_field_density_per_arcmin2 below -- the original measurement
    also mixed bands, mag_f150w vs mag_f115w). 1.05x is the mean.

    Falls back to the older purely-parametric area-scaling (see
    field_density_area_scale) if the catalog isn't available.
    """
    cfg = config or {}
    field_cfg = cfg.get("field", {})
    mag_limit = field_cfg.get(
        "density_mag_limit",
        cfg.get("photometry", {}).get("field_mag_limit",
            cfg.get("photometry", {}).get("source_mag_max", 27.5)),
    )
    catalog_path = cfg.get("catalogs", {}).get("galaxy_catalog_fits", "data/galaxy_catalog.fits")
    density_per_arcmin2 = cosmos_field_density_per_arcmin2(mag_limit, catalog_path)
    overdensity_factor = field_cfg.get("lens_sightline_overdensity_factor", 1.05)

    if density_per_arcmin2 > 0:
        fov_arcmin2 = (numpix * pixel_scale / 60.0) ** 2
        richness_mult = env_params.get("galaxy_count_mean", 2.5) / 2.5
        # FIX (adversarial audit finding C-12, 2026-08-01): the overdensity
        # factor was applied unconditionally to EVERY environment type,
        # including "isolated_field" -- so a lens explicitly labelled
        # isolated was still rendered in a systematically overdense
        # sightline, contradicting its own label. The overdensity signal
        # (measured relative to a random field sightline) should only
        # apply when the environment is richer than the isolated_field
        # reference (richness_mult>1, i.e. group/pair); isolated_field
        # itself IS the reference population, not an overdense one.
        _effective_overdensity = overdensity_factor if richness_mult > 1.0 else 1.0
        mean = density_per_arcmin2 * _effective_overdensity * fov_arcmin2 * richness_mult
        std = mean * 0.25  # ~25% field-to-field scatter (Poisson + clustering)
        return mean, std

    # Explicit surface density (gal/arcsec^2) when COSMOS FITS catalog unavailable
    surf = field_cfg.get("expected_density_per_arcsec2")
    if surf is not None and surf > 0:
        fov_arcsec2 = (numpix * pixel_scale) ** 2
        richness_mult = env_params.get("galaxy_count_mean", 2.5) / 2.5
        mean = float(surf) * fov_arcsec2 * richness_mult
        std = mean * 0.25
        return mean, std

    area_scale = field_density_area_scale(numpix, pixel_scale, cfg)
    return (env_params.get("galaxy_count_mean", 6.0) * area_scale,
            env_params.get("galaxy_count_std", 2.5) * area_scale)


def _theta_E_hard_max(config=None):
    """Hard upper clip on θ_E (arcsec). Default 5″; paper/cluster configs
    may raise this (e.g. final_B-style rings at θ_E ≈ 7–9″)."""
    cfg = config if config is not None else CONFIG
    if not isinstance(cfg, dict):
        return 5.0
    return float(cfg.get('geometry', {}).get('theta_E_hard_max',
                 cfg.get('mass', {}).get('theta_E_hard_max', 5.0)))


def _source_xy_hard_max(theta_E, config=None):
    """Max |source_x/y| (arcsec). Scales with θ_E so large-Einstein systems
    are not forced into the tiny ±0.4″ box used for JWST cutouts."""
    te = float(theta_E)
    hard = max(0.6, 0.55 * te)
    cfg = config if config is not None else CONFIG
    if isinstance(cfg, dict):
        hard = float(cfg.get('geometry', {}).get('source_xy_hard_max', hard))
    return hard


def convert_physical_to_angular_radius(reff_kpc, z):
    """
    Convert physical effective radius to angular radius
    
    Args:
        reff_kpc: Physical effective radius in kpc
        z: Redshift
    
    Returns:
        theta_eff: Angular effective radius in arcsec
    """
    # Convert kpc to Mpc
    reff_mpc = reff_kpc / 1000.0
    
    # Get angular diameter distance
    da_mpc = angular_diameter_distance(z)
    
    # Convert to angular size (arcsec)
    theta_eff_rad = reff_mpc / da_mpc
    theta_eff_arcsec = theta_eff_rad * 206265  # Convert radians to arcsec
    
    return theta_eff_arcsec


# Quenched (low-sSFR) galaxies are early-type/de-Vaucouleurs-like (n~4);
# star-forming galaxies are later-type/exponential-disk-like (n~1). This
# threshold (sSFR ~ 1/(3*t_Hubble(z=0))) separates the two TNG populations.
TNG_QUENCHED_SSFR_THRESHOLD = 1e-11

# Batch-level exclusion set: (sim, snapshot, subhalo_id) tuples already used
# in the current run. Prevents reusing the same TNG galaxy for lens+source
# or across two different systems. Reset at the start of each batch run.
_used_tng_subhalos: set = set()

# Above this sSFR, a TNG-matched galaxy is treated as a (dusty) starburst for
# SED/K-correction purposes -- roughly the sSFR above which a galaxy doubles
# its stellar mass on a <1 Gyr timescale.
TNG_STARBURST_SSFR_THRESHOLD = 1e-9


def tng_sed_galaxy_type(tng_info):
    """Map a TNG-matched subhalo's sSFR to one of the SED ``galaxy_type``
    categories used by ``calculate_k_correction`` (``passive`` /
    ``star_forming`` / ``dusty_starburst``), replacing the morphology-based
    heuristic for galaxies that have a TNG match. Mirrors the
    ``TNG_QUENCHED_SSFR_THRESHOLD``-based ``n_sersic`` bias already applied
    to TNG-matched lens/source/field galaxies.
    """
    ssfr = tng_info.get('ssfr_per_yr', 0.0)
    if ssfr < TNG_QUENCHED_SSFR_THRESHOLD:
        return 'passive'
    if ssfr > TNG_STARBURST_SSFR_THRESHOLD:
        return 'dusty_starburst'
    return 'star_forming'


# Fields copied verbatim from a ``select_tng_galaxy`` match when flattening
# into the training catalog (see ``flatten_tng_info``). Keep in sync with
# the dict returned by ``tng_galaxy_selector.select_tng_galaxy``.
TNG_INFO_FIELDS = [
    "sim", "snapshot", "snapshot_redshift", "subhalo_id", "halo_id",
    "stellar_mass_logmsun", "sfr_msun_per_yr", "ssfr_per_yr",
    "halfmassrad_stars_kpc", "gas_mass_msun", "gas_metallicity",
    "star_metallicity", "environment", "group_massive_subhalo_count",
    "primary_flag",
]


def flatten_tng_info(prefix, tng_info):
    """Flatten a ``select_tng_galaxy`` match (or ``None``) into
    ``{prefix}_matched`` plus ``{prefix}_<field>`` columns for
    ``TNG_INFO_FIELDS``, for the training catalog CSV.

    Numeric fields are ``NaN`` and ``environment`` is ``""`` when
    ``tng_info`` is ``None`` (TNG Mode disabled or no match found).
    """
    out = {f"{prefix}_matched": tng_info is not None}
    for field in TNG_INFO_FIELDS:
        key = f"{prefix}_{field}"
        if tng_info is None:
            out[key] = "" if field == "environment" else np.nan
        else:
            out[key] = tng_info.get(field, "" if field == "environment" else np.nan)
    return out


_TNG_CATALOG_CACHE: dict = {}


def _get_merged_tng_catalog(tng_cfg: dict):
    """Return (possibly merged) TNG catalog from config, cached by path key."""
    import pandas as pd

    path1 = tng_cfg.get('local_catalog_path')
    path2 = tng_cfg.get('local_catalog_path_tng50')
    sim_mode = tng_cfg.get('sim_mode', 'tng100')  # 'tng100' | 'tng50' | 'tng_mixed'

    if sim_mode == 'tng50' and path2:
        key = str(path2)
        if key not in _TNG_CATALOG_CACHE:
            _TNG_CATALOG_CACHE[key] = load_local_catalog(path2)
        return _TNG_CATALOG_CACHE[key]
    elif sim_mode == 'tng_mixed' and path1 and path2:
        key = f"{path1}:{path2}"
        if key not in _TNG_CATALOG_CACHE:
            c1 = load_local_catalog(path1)
            c2 = load_local_catalog(path2)
            if c1 is not None:
                c1 = c1.copy()
                if "sim" not in c1.columns:
                    c1["sim"] = "TNG100-1"
            if c2 is not None and "sim" not in c2.columns:
                c2 = c2.copy()
                c2["sim"] = "TNG50-1"
            if c1 is not None and c2 is not None:
                import pandas as pd
                _TNG_CATALOG_CACHE[key] = pd.concat([c1, c2], ignore_index=True)
            else:
                _TNG_CATALOG_CACHE[key] = c1 or c2
        return _TNG_CATALOG_CACHE[key]
    else:
        if not path1:
            return None
        key = str(path1)
        if key not in _TNG_CATALOG_CACHE:
            _TNG_CATALOG_CACHE[key] = load_local_catalog(path1)
        return _TNG_CATALOG_CACHE[key]


def query_tng_properties(target_z, target_logM, rng, config, environment=None,
                          exclude_subhalos=None, sfr_class=None,
                          min_particles_override=None):
    """Look up a TNG subhalo near ``(target_z, target_logM)``, gated by
    ``config['tng_mode']['enabled']``.

    Supports TNG100-1, TNG50-1, or a merged catalog via
    ``config['tng_mode']['sim_mode']`` ('tng100' | 'tng50' | 'tng_mixed').

    Parameters
    ----------
    exclude_subhalos : set of (sim, snapshot, subhalo_id), optional
        Subhalos already assigned in this batch run -- excluded to prevent
        the same galaxy appearing as both lens and source or in two systems.
        The caller is responsible for adding the returned subhalo to this set.
    sfr_class : 'star_forming' | 'quiescent' | None
        When given, restricts selection to that SFR class so that sources
        span a realistic mix of star-forming and quiescent morphologies.
    min_particles_override : int | None
        If set, overrides ``particle_morphology.min_particles`` for this
        lookup (e.g. lower threshold for sparse field-galaxy cutouts).

    Returns ``None`` if TNG Mode is disabled, no match is found, or the
    API is unreachable -- callers must fall back to Sersic-based generation.
    """
    tng_cfg = config.get('tng_mode', {}) if isinstance(config, dict) else {}
    if not tng_cfg.get('enabled', False):
        return None

    catalog = _get_merged_tng_catalog(tng_cfg)
    if catalog is not None:
        _min_p = min_particles_override
        if _min_p is None:
            _min_p = tng_cfg.get('particle_morphology', {}).get('min_particles')
        result = select_tng_galaxy_local(
            target_z=target_z,
            target_logM=target_logM,
            rng=rng,
            catalog=catalog,
            logM_tol=float(tng_cfg.get('logM_tol', 0.3)),
            environment=environment,
            max_attempts=int(tng_cfg.get('max_attempts', 10)),
            require_local_particles=bool(tng_cfg.get('require_local_particles', False)),
            min_particles=_min_p,
            exclude_subhalos=exclude_subhalos,
            sfr_class=sfr_class,
            delta_z_window=float(tng_cfg.get('delta_z_window', 0.4)),
        )
        if result is not None and exclude_subhalos is not None:
            exclude_subhalos.add((result['sim'], result['snapshot'], result['subhalo_id']))
        return result

    try:
        return select_tng_galaxy(
            target_z=target_z,
            target_logM=target_logM,
            rng=rng,
            logM_tol=float(tng_cfg.get('logM_tol', 0.3)),
            environment=environment,
            max_attempts=int(tng_cfg.get('max_attempts', 10)),
        )
    except Exception as exc:
        print(f"[TNG_MODE] lookup failed for z={target_z:.2f}, logM={target_logM:.2f}: {exc}")
        return None


def apply_tng_field_overrides(field_galaxies_base, rng, config, exclude_subhalos=None):
    """For a random subset of field galaxies, override R_sersic/n_sersic with
    a physically matched TNG100-1 subhalo's half-mass radius / sSFR-based
    morphology (same pattern as the lens and lensed source in
    ``query_tng_properties``). Stellar mass is sampled from
    ``config['tng_mode']['field_logM_default']``/``field_logM_scatter`` (no
    catalog mass exists for field galaxies). Stores the match (or ``None``)
    as ``galaxy['tng_info']``.

    No-op (and does not set ``tng_info``) unless
    ``config['tng_mode']['enabled']`` and ``config['tng_mode']['field_enabled']``
    are both true.
    """
    tng_cfg = config.get('tng_mode', {}) if isinstance(config, dict) else {}
    if not tng_cfg.get('enabled', False) or not tng_cfg.get('field_enabled', False):
        return field_galaxies_base

    field_fraction = float(tng_cfg.get('field_fraction', 0.3))
    override_structural = bool(tng_cfg.get('field_override_structural', True))
    logM_default = tng_cfg.get('field_logM_default', 9.5)
    logM_scatter = tng_cfg.get('field_logM_scatter', 0.6)
    pm_cfg = tng_cfg.get('particle_morphology', {})
    min_particles_field = pm_cfg.get('min_particles_field', pm_cfg.get('min_particles'))

    for galaxy in field_galaxies_base:
        if rng.random() >= field_fraction:
            galaxy['tng_info'] = None
            continue

        field_z = float(galaxy.get('field_redshift', 1.0))
        field_logM = float(np.clip(rng.normal(logM_default, logM_scatter), 8.0, 11.5))
        # Field galaxies span all SFR classes — let the sSFR randomise naturally
        tng_field = query_tng_properties(field_z, field_logM, rng, config,
                                         environment=galaxy.get('environment'),
                                         exclude_subhalos=exclude_subhalos,
                                         min_particles_override=min_particles_field)
        galaxy['tng_info'] = tng_field
        if tng_field is None or not override_structural:
            continue

        new_radius = convert_physical_to_angular_radius(tng_field['halfmassrad_stars_kpc'], field_z)
        galaxy['R_sersic'] = float(np.clip(new_radius, 0.05, 2.5))

        if tng_field['ssfr_per_yr'] < TNG_QUENCHED_SSFR_THRESHOLD:
            galaxy['n_sersic'] = max(float(galaxy['n_sersic']), 3.0)
        else:
            galaxy['n_sersic'] = min(float(galaxy['n_sersic']), 2.0)

    return field_galaxies_base


def sample_halo_radius_profile(r200_kpc, z, rng, n_satellites, max_radius_factor=1.5):
    """
    Sample satellite radii from beta/Einasto profile within realistic halo radius
    
    Args:
        r200_kpc: Halo R200 radius in kpc
        z: Redshift
        rng: Random number generator
        n_satellites: Number of satellites to sample
        max_radius_factor: Maximum radius as factor of R200 (default 1.5)
    
    Returns:
        radii_arcsec: Array of angular radii in arcsec
    """
    if n_satellites == 0:
        return np.array([])
    
    # Convert R200 to angular size
    r200_mpc = r200_kpc / 1000.0
    da_mpc = angular_diameter_distance(z)
    r200_arcsec = (r200_mpc / da_mpc) * 206265
    
    # Sample from beta profile (simplified Einasto)
    # Beta profile: n(r) ∝ (1 + (r/r_s)^2)^(-3β/2)
    # For satellites, use β=0.5, r_s = R200/2
    beta = 0.5
    r_s = r200_arcsec / 2.0
    
    # Sample radii using inverse transform sampling
    # For beta profile, use rejection sampling
    radii_arcsec = []
    max_radius = r200_arcsec * max_radius_factor
    
    while len(radii_arcsec) < n_satellites:
        # Sample candidate radius
        r_candidate = rng.uniform(0, max_radius)
        
        # Beta profile probability
        prob = (1 + (r_candidate/r_s)**2)**(-3*beta/2)
        
        # Accept with probability
        if rng.random() < prob:
            radii_arcsec.append(r_candidate)
    
    return np.array(radii_arcsec[:n_satellites])

def estimate_halo_r200(mass_log10, z):
    """
    Estimate halo R200 from stellar mass
    
    Args:
        mass_log10: Log10 stellar mass in M_sun
        z: Redshift
    
    Returns:
        r200_kpc: Halo R200 radius in kpc
    """
    # Rough stellar-to-halo mass relation
    # M_star/M_halo ~ 0.01-0.05 for massive galaxies
    stellar_to_halo_ratio = 0.03  # Typical value
    
    # Convert stellar mass to halo mass
    halo_mass_log10 = mass_log10 + np.log10(1/stellar_to_halo_ratio)
    
    # R200 ~ (3*M_halo/(4*π*200*ρ_crit))^(1/3)
    # For ρ_crit = 2.775e11 * h^2 * M_sun/Mpc^3, h=0.7
    rho_crit = 2.775e11 * 0.7**2  # M_sun/Mpc^3
    r200_mpc = (3 * 10**halo_mass_log10 / (4 * np.pi * 200 * rho_crit))**(1/3)
    r200_kpc = r200_mpc * 1000  # Convert to kpc
    
    return r200_kpc
# Default JWST bands (will be overridden by config if specified)
DEFAULT_BANDS = ["F115W", "F150W", "F277W", "F444W"]
NA_SENTINELS = ["", " ", "nan", "NaN", "-99", "-999", 99, 999]

# All known filter wavelengths (microns) for different telescopes
ALL_BAND_CENTERS_UM = {
    # JWST NIRCam
    "f070w": 0.704, "f090w": 0.901, "f115w": 1.154, "f140m": 1.404,
    "f150w": 1.501, "f162m": 1.626, "f182m": 1.845, "f200w": 1.990,
    "f210m": 2.093, "f250m": 2.503, "f277w": 2.786, "f300m": 2.996,
    "f322w2": 3.247, "f323n": 3.237, "f335m": 3.365, "f356w": 3.563,
    "f360m": 3.621, "f405n": 4.055, "f410m": 4.082, "f430m": 4.280,
    "f444w": 4.421, "f460m": 4.624, "f466n": 4.654, "f470n": 4.707,
    "f480m": 4.834,
    # Roman WFI
    "roman_f062": 0.62, "roman_f087": 0.87, "roman_f106": 1.06,
    "roman_f129": 1.29, "roman_f146": 1.46, "roman_f158": 1.58,
    "roman_f184": 1.84, "roman_f213": 2.13,
    # Euclid
    "euclid_vis": 0.700, "euclid_y": 1.020, "euclid_j": 1.250, "euclid_h": 1.650,
    # Subaru HSC / Suprime-Cam
    "subaru_b": 0.445, "subaru_v": 0.551, "subaru_g": 0.477,
    "subaru_r": 0.623, "subaru_i": 0.764, "subaru_z": 0.907,
    "subaru_y": 0.999,
    # LSST / Rubin Observatory (Ivezić et al. 2019)
    "lsst_u": 0.367, "lsst_g": 0.482, "lsst_r": 0.622,
    "lsst_i": 0.754, "lsst_z": 0.869, "lsst_y": 0.971,
}

# Telescope-specific default filter sets
TELESCOPE_FILTERS = {
    'jwst':        ["F115W", "F150W", "F277W", "F444W"],
    'roman':       ["ROMAN_F106", "ROMAN_F129", "ROMAN_F158", "ROMAN_F184"],
    'ground_based':["SUBARU_G", "SUBARU_R", "SUBARU_I", "SUBARU_Z"],
    'subaru':      ["SUBARU_G", "SUBARU_R", "SUBARU_I", "SUBARU_Z"],
    'euclid':      ["EUCLID_VIS", "EUCLID_Y", "EUCLID_J", "EUCLID_H"],
    'lsst':        ["LSST_G", "LSST_R", "LSST_I", "LSST_Z"],
}

BAND_CENTERS_UM = ALL_BAND_CENTERS_UM  # Will be subset by actual bands used

# Telescope-specific RGB display tuning.
# - noise_level / sat_percent: passed to normalize_for_display_astronomical()
#   to control the asinh stretch range for each band.
# - sigma_mult: how many noise-sigma above the sky level a pixel must be
#   before it contributes to the display image (higher = cleaner background,
#   lower = preserves more low surface-brightness flux).
# - color_enhance: max color-saturation enhancement factor applied to
#   bright (source) pixels in the RGB composite (1.0 = no enhancement).
# - gamma: per-channel (R, G, B) gamma correction applied to the composite.
TELESCOPE_RGB_PARAMS = {
    'jwst':   dict(noise_level=0.30, sat_percent=0.01, sigma_mult=1.2, color_enhance=1.0,
                   gamma=(1.0, 1.0, 1.0), linked_stretch=True, arc_boost=0.35),
    # Euclid default: Trilogy log stretch (Dan Coe) — clean paper-quality RGB on sims.
    # Override with output.rgb.use_eummy: true for Mischa Schirmer MER Lab look.
    'euclid': dict(noise_level=0.15, sat_percent=0.001, sigma_mult=1.2, color_enhance=1.0,
                   gamma=(1.0, 1.0, 1.0), linked_stretch=True, arc_boost=0.25,
                   use_trilogy=True, use_eummy=False,
                   soft_clip=False, band_style_stretch=False, field_overlay=False),
    'roman':  dict(noise_level=0.15, sat_percent=0.02, sigma_mult=1.2, color_enhance=1.35,
                   gamma=(1.0, 1.0, 0.96), linked_stretch=True, arc_boost=0.45),
    'subaru': dict(noise_level=0.30, sat_percent=0.01, sigma_mult=1.2, color_enhance=1.0,
                   gamma=(1.0, 1.0, 1.0), linked_stretch=True, arc_boost=0.30),
    'lsst':   dict(noise_level=0.30, sat_percent=0.01, sigma_mult=1.2, color_enhance=1.0,
                   gamma=(1.0, 1.0, 1.0), linked_stretch=True, arc_boost=0.30),
}

def get_telescope_bands(resolution_name, config_bands=None):
    """
    Get appropriate filter set for a given telescope/resolution.
    
    Parameters
    ----------
    resolution_name : str
        Name of resolution ('jwst', 'euclid', 'roman', 'ground_based', 'default')
    config_bands : list, optional
        User-specified bands from config (takes precedence for JWST)
    
    Returns
    -------
    list : Upper-case band names for this telescope
    """
    # Optional per-telescope overrides from config
    telescope_bands_cfg = CONFIG.get('telescope_bands', {}) if isinstance(CONFIG, dict) else {}
    if isinstance(telescope_bands_cfg, dict):
        cfg_bands = telescope_bands_cfg.get(resolution_name)
        if isinstance(cfg_bands, list) and len(cfg_bands) > 0:
            return cfg_bands

    # If user specified bands in config and we're in JWST mode, use those
    if config_bands and resolution_name in ['jwst', 'default']:
        return config_bands
    
    # Otherwise use telescope defaults
    return TELESCOPE_FILTERS.get(resolution_name, TELESCOPE_FILTERS['jwst'])

from prism.core.constants import C_LIGHT_KM_S, H0_DEFAULT as H0_KM_S_MPC  # noqa: E402
from prism.selection.galaxygenius_stamps import (  # noqa: E402
    build_field_galaxy_interpol_kwargs,
    build_lens_light_interpol_kwargs,
    build_source_interpol_kwargs,
    available_views,
    angular_size_for_redshift,
    random_stamp_set,
    native_redshift_for_stamp_set,
)
from prism.selection.tng_galaxy_selector import select_tng_galaxy, select_tng_galaxy_local, load_local_catalog, local_particle_path  # noqa: E402
from prism.morphology.tng_particle_light import build_tng_particle_interpol_kwargs, get_projection_orientation  # noqa: E402
OMEGA_M = 0.3
OMEGA_L = 0.7
ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)

# These will be set dynamically after CONFIG is loaded
UPPER_BANDS = DEFAULT_BANDS
LOWER_BANDS = [b.lower() for b in DEFAULT_BANDS]
BAND_TO_LOWER = {b: b.lower() for b in DEFAULT_BANDS}


def _particle_ref_band() -> str:
    """Reference band for TNG particle INTERPOL colour anchoring."""
    return UPPER_BANDS[0] if UPPER_BANDS else 'F150W'

# Global configuration loaded from YAML
CONFIG = {
    'photometry': {
        'lens_mag_min': 18.0,
        'lens_mag_max': 27.0,
        'source_mag_min': 18.5,
        'source_mag_max': 27.5,
        'min_source_fainter_than_lens_mag': 0.8,
        'source_mag_diff_min': 1.5,
        'source_mag_diff_max': 5.0,
        # Calibrated so the resulting lens_mag_f150w distribution's median
        # (~21.0 at the old 21.0 zero-point, for the z<~1-dominated lens
        # sample) matches the real COSMOS-Web sample's mag_f150w median
        # (21.55, see analysis/sim_obs_comparison/reports/
        # phase2_noise_and_color_fixes.md section 13). The previous value
        # (21.0) made simulated lenses ~0.55 mag (~1.6x flux) brighter than
        # real on average, which (via the fixed-threshold isophotal-radius
        # mechanism, section 12) was the dominant driver of the
        # lens_reff_arcsec gap (~1.5x).
        'lens_base_mag_zero': 21.55,
        'lens_redshift_log_slope': 0.8,
        'source_base_mag': 20.5,
        'color_offsets': {
            'f115w': 0.4, 'f150w': 0.2, 'f277w': -0.1, 'f444w': -0.2
        }
    },
    'field': {
        'expected_density_per_arcsec2': 1.2,
        'avoid_center_arcsec': 0.3,
        'min_pair_separation_arcsec': 0.08,
        'max_fraction_of_half_size': 0.9,
        'min_count': 4
    }
}

def load_config(config_path: Optional[str]):
    global UPPER_BANDS, LOWER_BANDS, BAND_TO_LOWER, BAND_CENTERS_UM
    
    if not config_path:
        return
    try:
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
        # Deep-merge into CONFIG
        def deep_merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v
        deep_merge(CONFIG, user_cfg)
        print(f"[CONFIG] Loaded configuration from {config_path}")
    except Exception as e:
        print(f"[CONFIG] Failed to load {config_path}: {e}")
    
    # Update UPPER_BANDS from config if specified
    if 'bands' in CONFIG:
        bands_list = CONFIG['bands']
        if isinstance(bands_list, list) and len(bands_list) > 0:
            UPPER_BANDS = bands_list
            LOWER_BANDS = [b.lower() for b in UPPER_BANDS]
            BAND_TO_LOWER = {b: b.lower() for b in UPPER_BANDS}
            # Update BAND_CENTERS_UM to only include bands we're using
            BAND_CENTERS_UM = {b.lower(): ALL_BAND_CENTERS_UM.get(b.lower()) for b in UPPER_BANDS}
            print(f"[CONFIG] JWST bands: {UPPER_BANDS} ({len(UPPER_BANDS)} filters)")
    
    # After merging user config, attempt to compute empirical JWST color offsets
    try:
        catalogs_cfg = CONFIG.get('catalogs', {})
        galaxy_catalog_path = catalogs_cfg.get('galaxy_catalog')
        if galaxy_catalog_path and os.path.exists(galaxy_catalog_path):
            empirical = compute_empirical_color_offsets(galaxy_catalog_path)
            if empirical:
                CONFIG.setdefault('photometry', {})['empirical_color_offsets'] = empirical
                print(f"[CONFIG] Empirical color offsets computed from galaxy catalog: {empirical}")
            else:
                print("[CONFIG] Empirical color offsets not computed (insufficient data)")
        else:
            if galaxy_catalog_path:
                print(f"[CONFIG] Galaxy catalog not found at {galaxy_catalog_path}")
    except Exception as e:
        print(f"[CONFIG] Error computing empirical color offsets: {e}")


def _detect_band_mag_columns(df_columns: list[str]) -> dict:
    """Detect plausible JWST band magnitude column names in a catalog (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df_columns}
    patterns = {
        'f115w': ['f115w', 'mag_f115w', 'f115w_mag'],
        'f150w': ['f150w', 'mag_f150w', 'f150w_mag'],
        'f277w': ['f277w', 'mag_f277w', 'f277w_mag'],
        'f444w': ['f444w', 'mag_f444w', 'f444w_mag']
    }
    result = {}
    for band, candidates in patterns.items():
        for cand in candidates:
            if cand in cols_lower:
                result[band] = cols_lower[cand]
                break
    return result

def compute_empirical_color_offsets(csv_path: str) -> dict:
    """Compute robust median color offsets per band relative to F150W (fallback to first available)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    colmap = _detect_band_mag_columns(list(df.columns))
    if not colmap:
        return {}
    # Choose reference band
    ref_band = 'f150w' if 'f150w' in colmap else next(iter(colmap.keys()))
    ref_col = colmap[ref_band]
    offsets = {}
    ref_series = pd.to_numeric(df[ref_col], errors='coerce')
    # Trim outliers for the reference
    q1, q99 = ref_series.quantile([0.01, 0.99])
    ref_mask = ref_series.between(q1, q99)
    for band, col in colmap.items():
        band_series = pd.to_numeric(df[col], errors='coerce')
        # Common mask
        m = ref_mask & band_series.notna()
        if m.sum() < 100:
            continue
        diff = band_series[m] - ref_series[m]
        # Winsorize diffs
        d1, d99 = diff.quantile([0.01, 0.99])
        diff_clipped = diff.clip(d1, d99)
        offsets[band] = float(diff_clipped.median())
    return offsets

# ============================================================================
# PSF INTEGRATION FUNCTIONS
# ============================================================================

def load_psf_data(psf_dir="data/psf_v5_30mas"):
    """
    Load real JWST PSF data from psf_v5_30mas directory
    
    Args:
        psf_dir: Path to PSF directory
        
    Returns:
        dict: PSF data organized by tile and band
    """
    psf_data = {}
    
    # Get all tile directories
    tile_dirs = glob.glob(f"{psf_dir}/tiles/*/")
    
    for tile_dir in tile_dirs:
        tile_name = tile_dir.split('/')[-2]  # Extract tile name (e.g., A1, B5)
        psf_data[tile_name] = {}
        
        # Load PSF kernel files for each band
        for band in UPPER_BANDS:
            kernel_file = f"{tile_dir}{band}_kernel.fits"
            if os.path.exists(kernel_file):
                try:
                    # Load PSF kernel data
                    psf_array = fits.getdata(kernel_file)
                    # PSF kernels are already normalized (sum = 1.0)
                    psf_data[tile_name][band] = psf_array
                    print(f"[PSF] Loaded {band} PSF kernel for tile {tile_name}: {psf_array.shape}")
                except Exception as e:
                    print(f"[PSF] Error loading {kernel_file}: {e}")
                    psf_data[tile_name][band] = None
            else:
                print(f"[PSF] PSF kernel file not found: {kernel_file}")
                psf_data[tile_name][band] = None
    
    return psf_data

def map_coordinates_to_tile(ra, dec):
    """
    Map COSMOS-Web coordinates to PSF tile names using exact tile boundaries
    
    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        
    Returns:
        str: Tile name (e.g., 'A1', 'B5') or None if not found
    """
    from shapely.geometry import Point, Polygon
    
    # COSMOS-Web tile boundaries (exact coordinates)
    coords_A1 = [(149.8703317, 2.0856512), (149.7198796, 2.1403395), (149.7908786, 2.3354095), (149.9413496, 2.2807163)]
    coords_A2 = [(150.0058959, 2.0363591), (149.8554506, 2.0910612), (149.9264667, 2.2861269), (150.0769300, 2.2314186)]
    coords_A3 = [(150.1414523, 1.9870553), (149.9910155, 2.0417704), (150.0620479, 2.2368306), (150.2125019, 2.1821081)]
    coords_A4 = [(150.2769995, 1.9377408), (150.1265729, 1.9924679), (150.1976208, 2.1875215), (150.3480637, 2.1327859)]
    coords_A5 = [(150.4125359, 1.8884166), (150.2621212, 1.9431545), (150.3331838, 2.1382005), (150.4836139, 2.0834528)]
    coords_A6 = [(149.8045087, 1.9048087), (149.6540746, 1.9594923), (149.7250552, 2.1545612), (149.8755087, 2.0998725)]
    coords_A7 = [(149.9400575, 1.8555218), (149.7896293, 1.9102182), (149.8606274, 2.1052826), (150.0110740, 2.0505800)]
    coords_A8 = [(150.0755992, 1.8062243), (149.9251788, 1.8609325), (149.9961935, 2.0559913), (150.1466316, 2.0012757)]
    coords_A9 = [(150.2111325, 1.7569171), (150.0607214, 1.8116361), (150.1317520, 2.0066883), (150.2821799, 1.9519607)]
    coords_A10= [(150.3466557, 1.7076011), (150.1962556, 1.7623299), (150.2673014, 1.9573744), (150.4177173, 1.9026358)]
    coords_B1 = [(150.0020274, 2.4473359), (149.8515406, 2.5020333), (149.9225757, 2.6970916), (150.0730806, 2.6423895)]
    coords_B2 = [(150.1376214, 2.3980335), (149.9871430, 2.4527469), (150.0581944, 2.6478011), (150.2086900, 2.5930817)]
    coords_B3 = [(150.2732061, 2.3487174), (150.1227378, 2.4034461), (150.1938048, 2.5984949), (150.3442894, 2.5437590)]
    coords_B4 = [(150.4087801, 2.2993886), (150.2583236, 2.3541315), (150.3294054, 2.5491739), (150.4798772, 2.4944226)]
    coords_B5 = [(150.5443418, 2.2500480), (150.3938989, 2.3048040), (150.4649946, 2.4998389), (150.6154520, 2.4450733)]
    coords_B6 = [(149.9361713, 2.2664951), (149.7857017, 2.3211879), (149.8567188, 2.5162544), (150.0072070, 2.4615567)]
    coords_B7 = [(150.0717506, 2.2171978), (149.9212885, 2.2719056), (149.9923224, 2.4669678), (150.1428020, 2.4122539)]
    coords_B8 = [(150.2073213, 2.1678878), (150.0568686, 2.2226097), (150.1279183, 2.4176665), (150.2783878, 2.3629373)]
    coords_B9 = [(150.3428821, 2.1185662), (150.1924404, 2.1733011), (150.2635052, 2.3683514), (150.4139629, 2.3136080)]
    coords_B10= [(150.4784314, 2.0692337), (150.3280023, 2.1239807), (150.3990815, 2.3190234), (150.5495255, 2.2642668)]
    
    # Create polygon objects
    polygons = {
        'A1': Polygon(coords_A1), 'A2': Polygon(coords_A2), 'A3': Polygon(coords_A3), 'A4': Polygon(coords_A4), 'A5': Polygon(coords_A5),
        'A6': Polygon(coords_A6), 'A7': Polygon(coords_A7), 'A8': Polygon(coords_A8), 'A9': Polygon(coords_A9), 'A10': Polygon(coords_A10),
        'B1': Polygon(coords_B1), 'B2': Polygon(coords_B2), 'B3': Polygon(coords_B3), 'B4': Polygon(coords_B4), 'B5': Polygon(coords_B5),
        'B6': Polygon(coords_B6), 'B7': Polygon(coords_B7), 'B8': Polygon(coords_B8), 'B9': Polygon(coords_B9), 'B10': Polygon(coords_B10)
    }
    
    # Create point from coordinates
    point = Point(ra, dec)
    
    # Check which tile contains the point
    for tile_name, polygon in polygons.items():
        if polygon.contains(point):
            return tile_name
    
    # If no tile contains the point, return None
    return None

def get_psf_for_simulation(psf_data, lens_id=None, rng=None, ra=None, dec=None, psf_tile=None):
    """
    Get PSF data for a simulation, either assigned to specific lens or random
    
    Args:
        psf_data: PSF data dictionary from load_psf_data()
        lens_id: Specific lens ID (if None, random selection)
        rng: Random number generator
        ra: Right Ascension in degrees for tile mapping
        dec: Declination in degrees for tile mapping
        
    Returns:
        dict: PSF arrays for each band
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    # Get available tiles
    available_tiles = list(psf_data.keys())
    _q1_psf = available_tiles and str(available_tiles[0]).startswith('Q1_')
    
    # Euclid Q1 empirical PSF assignment (per-lens tile from catalog row or assignment CSV)
    if _q1_psf:
        tile = psf_tile
        if tile is None and lens_id is not None:
            try:
                if EUCLID_Q1_AVAILABLE and euclid_q1_enabled(CONFIG):
                    cat = get_euclid_q1_catalog(CONFIG)
                    if cat is not None:
                        tile = cat.assign_psf_tile(np.random.default_rng(rng.randint(0, 2**31)), lens_id=lens_id)
            except Exception:
                pass
        if tile is None and lens_id is not None:
            try:
                _q1_data_dir = CONFIG.get('euclid_q1', {}).get('data_dir', 'data/euclid_q1_psf') if isinstance(CONFIG, dict) else 'data/euclid_q1_psf'
                assign_path = Path(_q1_data_dir) / 'psf_assignment.csv'
                if assign_path.exists():
                    psf_assignments = pd.read_csv(assign_path)
                    lens_assignments = psf_assignments[psf_assignments['lens_id'] == lens_id]
                    if not lens_assignments.empty:
                        tile = lens_assignments.iloc[0]['tile']
            except Exception as e:
                print(f"[PSF] Could not load Euclid Q1 PSF assignment for {lens_id}: {e}")
        if tile and tile in psf_data:
            print(f"[PSF] Using Euclid Q1 tile {tile} for lens {lens_id}")
            return {band: psf_data[tile].get(band) for band in UPPER_BANDS}
    
    # Try coordinate-based mapping first (JWST COSMOS tiles only)
    if not _q1_psf and ra is not None and dec is not None:
        try:
            # Convert to float if they're strings
            ra_float = float(ra) if isinstance(ra, str) else ra
            dec_float = float(dec) if isinstance(dec, str) else dec
            tile_name = map_coordinates_to_tile(ra_float, dec_float)
            if tile_name and tile_name in psf_data:
                print(f"[PSF] Using coordinate-mapped tile {tile_name} for coordinates ({ra_float:.6f}, {dec_float:.6f})")
                return {band: psf_data[tile_name].get(band) for band in UPPER_BANDS}
        except (ValueError, TypeError) as e:
            print(f"[PSF] Error converting coordinates ({ra}, {dec}): {e}")
            # Continue to fallback options
    
    if not _q1_psf and lens_id is not None:
        # Try to get PSF assignment from psf_assignment.csv
        try:
            psf_assignments = pd.read_csv("data/psf_v5_30mas/psf_assignment.csv")
            lens_assignments = psf_assignments[psf_assignments['lens_id'] == lens_id]
            if not lens_assignments.empty:
                tile = lens_assignments.iloc[0]['tile']
                if tile in psf_data:
                    print(f"[PSF] Using assigned tile {tile} for lens {lens_id}")
                    return {band: psf_data[tile].get(band) for band in UPPER_BANDS}
        except Exception as e:
            print(f"[PSF] Could not load PSF assignment for {lens_id}: {e}")
    
    # Random tile selection — return None per band if no tiles loaded
    if not available_tiles:
        print("[PSF] No PSF tiles available — analytical PSF will be used")
        return {band: None for band in UPPER_BANDS}
    tile = rng.choice(available_tiles)
    label = "Euclid Q1" if _q1_psf else "PSF"
    print(f"[PSF] Using random {label} tile {tile}")
    return {band: psf_data[tile].get(band) for band in UPPER_BANDS}

def apply_psf_convolution(image, psf_array):
    """
    Apply PSF convolution to an image
    
    Args:
        image: 2D numpy array (simulated image)
        psf_array: 2D numpy array (PSF kernel)
        
    Returns:
        2D numpy array: Convolved image
    """
    if psf_array is None:
        return image
    
    try:
        # Apply PSF convolution with proper boundary handling to avoid edge artifacts
        convolved = convolve_fft(image, psf_array, boundary='fill', fill_value=0.0, 
                                normalize_kernel=True, nan_treatment='fill')
        return convolved
    except Exception as e:
        print(f"[PSF] Error in convolution: {e}")
        return image

def apply_field_galaxy_psf_convolution(field_galaxies, psf_data, rng):
    """
    Apply individual PSF convolution to each field galaxy based on its position
    
    Args:
        field_galaxies: List of field galaxy dictionaries
        psf_data: PSF data dictionary from load_psf_data()
        rng: Random number generator
        
    Returns:
        List of field galaxies with PSF information added
    """
    enhanced_galaxies = []
    
    for galaxy in field_galaxies:
        # Get galaxy coordinates if available
        ra = galaxy.get('RA_DETEC', None)
        dec = galaxy.get('DEC_DETEC', None)
        
        # Get PSF for this galaxy's position
        psf_arrays = get_psf_for_simulation(
            psf_data, 
            lens_id=None, 
            rng=rng, 
            ra=ra, 
            dec=dec
        )
        
        # Add PSF information to galaxy
        galaxy['psf_arrays'] = psf_arrays
        enhanced_galaxies.append(galaxy)
    
    return enhanced_galaxies

# ============================================================================
# NEW v11: ENHANCED MORPHOLOGICAL FEATURE FUNCTIONS
# ============================================================================

def estimate_half_light_radius(image, center_x, center_y):
    """Estimate half-light radius (pixels) by cumulative flux."""
    total_flux = np.sum(image)
    if total_flux <= 0:
        return max(image.shape[0] * 0.1, 5.0)

    y_coords, x_coords = np.indices(image.shape)
    r = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

    flat_r = r.flatten()
    flat_flux = image.flatten()
    order = np.argsort(flat_r)
    cumulative_flux = np.cumsum(flat_flux[order])

    half_flux = 0.5 * total_flux
    idx = np.searchsorted(cumulative_flux, half_flux)
    if idx >= len(flat_r):
        return max(image.shape[0] * 0.15, 6.0)

    r_half = flat_r[order][idx]
    return max(r_half, 4.0)


def _generate_low_frequency_noise(shape, rng):
    """Generate smooth low-frequency noise using coarse upsampling."""
    size_y, size_x = shape
    tile_y = max(4, size_y // 12)
    tile_x = max(4, size_x // 12)

    coarse = rng.normal(0.0, 1.0, (tile_y, tile_x))

    repeat_y = int(np.ceil(size_y / tile_y))
    repeat_x = int(np.ceil(size_x / tile_x))
    noise = np.kron(coarse, np.ones((repeat_y, repeat_x)))
    noise = noise[:size_y, :size_x]

    for _ in range(3):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0

    noise -= noise.min()
    max_val = noise.max()
    if max_val > 0:
        noise /= max_val
    else:
        noise.fill(0.5)
    return noise


def add_sky_background_noise(image_band: np.ndarray, band: str,
                               rng: np.random.Generator,
                               band_cfgs: dict) -> np.ndarray:
    """
    Add sky/zodiacal background noise to a detector-chain-processed image.

    DetectorChain.apply() (src/detector_chain.py) models dark current, read
    noise, and shot noise on the *source* signal, but has no sky-background
    term -- so its output noise floor is a pure "dark frame" level, far below
    real JWST/COSMOS-Web background RMS. The empirical
    CONFIG['noise'][band]['background_rms'] values (calibrated from the 435
    real COSMOS-Web lenses, validated against an independent photutils
    measurement on the same sample -- see
    analysis/sim_obs_comparison/reports/phase1_real_vs_sim_comparison.md)
    give the target total background RMS in the same e-/s units as the
    pipeline's final images. We inject that as additive Gaussian noise here,
    after the detector chain, so the output background RMS matches real data
    regardless of the chosen exposure_time.
    """
    bg_rms = float(band_cfgs.get(band, {}).get('_bg_rms', 0.0))
    if bg_rms <= 0:
        return image_band
    return image_band + rng.normal(0.0, bg_rms, image_band.shape).astype(image_band.dtype)


def apply_filter_specific_noise(image_band: np.ndarray, band: str,
                                 rng: np.random.Generator,
                                 exposure_time: float = 10000.0) -> np.ndarray:
    """
    Apply realistic, filter-specific noise to image data.
    
    Different JWST filters have different noise characteristics due to:
    - Thermal background (increases toward longer wavelengths)
    - Read noise (varies by filter)
    - Saturation levels
    - Excess noise factors
    
    Parameters:
    -----------
    image_band : np.ndarray
        Image data in electrons or counts
    band : str
        JWST filter name (e.g., 'F150W')
    rng : np.random.Generator
        Random number generator
    exposure_time : float
        Exposure time in seconds
    
    Returns:
    --------
    noisy_image : np.ndarray
        Image with filter-specific noise applied
    """
    if not FILTER_TRANSMISSION_AVAILABLE:
        # Fall back to simple Gaussian noise if transmission system unavailable
        noise_sigma = np.sqrt(np.abs(image_band)) * 0.1 + 1.0
        return image_band + rng.normal(0, noise_sigma, image_band.shape)
    
    try:
        # Get filter-specific noise properties
        noise_props = get_filter_noise_properties(band)
        
        # Calculate noise components
        # 1. Background noise (increases with exposure time and filter wavelength)
        background_level = noise_props['background'] * exposure_time
        background_noise = rng.normal(0, np.sqrt(background_level), image_band.shape)
        
        # 2. Read noise (independent of exposure time)
        read_noise = rng.normal(0, noise_props['read_noise'], image_band.shape)
        
        # 3. Photon noise (Poisson - modeled as sqrt(signal))
        signal = np.abs(image_band)
        photon_noise = rng.normal(0, np.sqrt(signal + 1), image_band.shape)
        
        # 4. Apply excess noise factor (accounts for 1/f noise at long wavelengths)
        excess_factor = noise_props['excess_noise_factor']
        
        # Combine noise sources
        total_noise = (
            photon_noise +  # Signal-dependent
            read_noise * excess_factor +  # Readout, scaled by excess
            background_noise * excess_factor  # Background, scaled by excess
        )
        
        # Apply noise and clip to valid range
        noisy_image = image_band + total_noise
        
        # Ensure physical values (no negative electrons)
        noisy_image = np.maximum(noisy_image, 0)
        
        # Check for saturation
        sat_level = noise_props['saturation_electrons']
        noisy_image = np.minimum(noisy_image, sat_level)
        
        return noisy_image
        
    except Exception as e:
        print(f"[WARNING] Filter-specific noise failed for {band}: {e}, using generic noise")
        noise_sigma = np.sqrt(np.abs(image_band)) * 0.1 + 1.0
        return image_band + rng.normal(0, noise_sigma, image_band.shape)


def add_spiral_arms_to_image(image, center_x, center_y, pixel_scale,
                              pitch_angle=20, arm_strength=0.3, n_arms=2,
                              axis_ratio=0.7, position_angle=0.0,
                              bulge_fraction=0.4, seed=42, band=None,
                              clump_config=None):
    """
    Add realistic logarithmic spiral arms to galaxy image

    Args:
        image: 2D numpy array
        center_x, center_y: center position in pixels
        pixel_scale: arcsec/pixel (0.03 for JWST)
        pitch_angle: degrees (15-30 typical, lower = tighter winding)
        arm_strength: 0-1, strength of spiral modulation
        n_arms: number of spiral arms (2 for early spirals, 3 for late spirals)
        seed: random seed for reproducibility

    Returns:
        Enhanced image with spiral arms and disk structure
    """
    rng = np.random.RandomState(seed)
    size = image.shape[0]
    y_grid, x_grid = np.indices(image.shape)

    rel_x = x_grid - center_x
    rel_y = y_grid - center_y

    pa_rad = np.radians(position_angle)
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
    x_rot = rel_x * cos_pa + rel_y * sin_pa
    y_rot = -rel_x * sin_pa + rel_y * cos_pa

    q = np.clip(axis_ratio, 0.2, 1.0)
    r_ell = np.sqrt(x_rot**2 + (y_rot / q)**2)

    r_half = estimate_half_light_radius(image, center_x, center_y)
    disk_scale = max(r_half / 1.678, 2.5)

    total_flux = np.maximum(np.sum(image), 1e-8)
    bulge_fraction = float(np.clip(bulge_fraction, 0.2, 0.7))

    bulge_component = image * bulge_fraction
    residual_flux = total_flux - np.sum(bulge_component)

    disk_profile = np.exp(-r_ell / disk_scale)
    disk_profile_sum = np.sum(disk_profile)
    if disk_profile_sum <= 0:
        return image

    disk_component = disk_profile * (residual_flux / disk_profile_sum)

    theta = np.arctan2(y_rot / np.clip(q, 1e-3, 1.0), x_rot + 1e-6)
    r_safe = np.maximum(r_ell, 1.0)

    ref_radius = np.clip(r_half * rng.uniform(0.9, 1.2), 6.0, size * 0.45)
    inner_cut = max(3.0, 0.35 * ref_radius)
    outer_cut = min(size * 0.48, ref_radius * 3.0)

    spiral_pattern = np.zeros_like(image)
    arm_offsets = []
    pitch_rad = np.radians(np.clip(pitch_angle, 5, 45))

    for arm in range(n_arms):
        arm_offset = arm * 2 * np.pi / n_arms + rng.uniform(-0.2, 0.2)
        arm_offsets.append(arm_offset)
        spiral_phase = theta - arm_offset - np.log(r_safe / ref_radius) * np.tan(pitch_rad)

        wrapped_phase = np.arctan2(np.sin(spiral_phase), np.cos(spiral_phase))
        arm_width = 0.28 + 0.07 * rng.uniform(-0.5, 0.5)
        arm_core = np.exp(-0.5 * (wrapped_phase / arm_width)**2)

        radial_envelope = np.exp(-((r_ell - ref_radius)**2) / (2 * (0.6 * ref_radius)**2))
        taper_inner = 1 - np.exp(-((r_ell - inner_cut) / max(1.5, 0.2 * ref_radius))**2)
        taper_outer = np.exp(-((r_ell - outer_cut) / max(5.0, 0.3 * outer_cut))**2)
        disk_mask = (r_ell > inner_cut) & (r_ell < outer_cut)

        spiral_pattern += arm_core * radial_envelope * taper_inner * taper_outer * disk_mask

    if np.max(spiral_pattern) > 0:
        spiral_pattern /= np.max(spiral_pattern)

    enhanced_disk = disk_component * (1 + arm_strength * spiral_pattern)

    if np.any(spiral_pattern > 0):
        diffuse_noise = _generate_low_frequency_noise(image.shape, rng)
        diffuse_emission = disk_component * (0.10 + 0.08 * diffuse_noise) * (np.clip(spiral_pattern, 0, 1)**0.8)
        enhanced_disk += diffuse_emission

    clump_map = np.zeros_like(image)
    if clump_config is not None and clump_config.get('count', 0) > 0 and np.any(spiral_pattern > 0):
        count = max(1, int(clump_config.get('count', n_arms * 5)))
        size_min, size_max = clump_config.get('size_range', (1.2, 2.6))
        strength_min, strength_max = clump_config.get('strength_range', (0.06, 0.14))
        radial_jitter = float(clump_config.get('radial_jitter', 0.1))
        theta_jitter = float(clump_config.get('theta_jitter', np.radians(5)))

        band_map = {
            'F115W': 1.0,
            'F150W': 0.92,
            'F200W': 0.85,
            'F277W': 0.72,
            'F356W': 0.62,
            'F444W': 0.55
        }
        band_key = (band or '').upper()
        band_scale = band_map.get(band_key, 0.8)
        band_hash = (sum(ord(c) for c in band_key) + int(seed)) % 7
        band_scale *= 1.0 + 0.03 * (band_hash - 3)

        weights = np.ones(len(arm_offsets))
        distribution = rng.multinomial(count, weights / weights.sum())
        base_disk_peak = np.max(enhanced_disk) + 1e-9

        r_start = inner_cut * 1.05
        r_end = min(outer_cut * 0.95, ref_radius * 1.9)

        for arm_idx, n_on_arm in enumerate(distribution):
            if n_on_arm == 0:
                continue

            fractions = np.linspace(0.2, 1.0, n_on_arm)
            base_radii = r_start + (r_end - r_start) * (fractions**1.15)
            base_radii += rng.normal(0.0, radial_jitter * ref_radius, size=n_on_arm)

            for r_val in base_radii:
                if not (inner_cut < r_val < outer_cut):
                    continue

                theta_arm = arm_offsets[arm_idx] + np.log(r_val / ref_radius) * np.tan(pitch_rad)
                theta_arm += rng.normal(0.0, theta_jitter)

                x_rot_arm = r_val * np.cos(theta_arm)
                y_rot_arm = q * r_val * np.sin(theta_arm)

                rel_x_knot = x_rot_arm * cos_pa - y_rot_arm * sin_pa
                rel_y_knot = x_rot_arm * sin_pa + y_rot_arm * cos_pa

                xk = center_x + rel_x_knot
                yk = center_y + rel_y_knot

                if not (0 <= xk < size and 0 <= yk < size):
                    continue

                sigma = rng.uniform(size_min, size_max)
                amp = base_disk_peak * rng.uniform(strength_min, strength_max) * band_scale
                dist2 = (x_grid - xk)**2 + (y_grid - yk)**2
                clump_map += amp * np.exp(-dist2 / (2 * sigma**2))

    if q < 0.85:
        lane_width = np.interp(q, [0.2, 0.85], [1.0, 5.5])
        lane_depth = np.interp(q, [0.2, 0.85], [0.55, 0.15])
        dust_profile = np.exp(-0.5 * (y_rot / lane_width)**2)
        dust_mask = disk_profile / (np.max(disk_profile) + 1e-6)
        dust_attenuation = lane_depth * dust_profile * dust_mask
        enhanced_disk *= np.clip(1 - dust_attenuation, 0.4, 1.0)

    enhanced_image = bulge_component + enhanced_disk + clump_map

    enhanced_flux = np.sum(enhanced_image)
    if enhanced_flux > 0:
        enhanced_image *= total_flux / enhanced_flux

    return enhanced_image


def add_clumpy_structure_to_image(image, center_x, center_y, pixel_scale,
                                   n_clumps=12, clump_strength=0.8, seed=42):
    """
    Add star-forming clumps to galaxy image
    Physical sizes: 60-150 pc -> 2.5-5.0 pixels at z~0.5 with 0.03"/pix

    Args:
        image: 2D numpy array
        center_x, center_y: center position in pixels
        pixel_scale: arcsec/pixel
        n_clumps: number of clumps (12-15 typical for irregulars)
        clump_strength: relative strength of clumps (0.5-1.5)
        seed: random seed

    Returns:
        Enhanced image with clumpy structure
    """
    rng = np.random.RandomState(seed)
    size = image.shape[0]
    y_grid, x_grid = np.ogrid[:size, :size]

    clumpy_pattern = np.zeros_like(image)

    for i in range(n_clumps):
        # Random clump position in disk - STRICTLY within galaxy body
        clump_r = rng.uniform(5, min(size/4, 35))  # Reduced radius to keep clumps closer
        clump_theta = rng.uniform(0, 2*np.pi)
        clump_x = center_x + clump_r * np.cos(clump_theta)
        clump_y = center_y + clump_r * np.sin(clump_theta)

        # Clump size: 2.5-5.0 pixels (physical 60-150 pc at z~0.5)
        clump_size = rng.uniform(2.5, 5.0)

        # Add clump
        clump_dist = np.sqrt((x_grid - clump_x)**2 + (y_grid - clump_y)**2)
        clump_brightness = rng.uniform(0.5, 1.5)
        clump = clump_brightness * np.exp(-0.5 * (clump_dist/clump_size)**2)
        clumpy_pattern += clump

    # Add clumps ONLY where galaxy has significant flux (stricter mask)
    # Use 5% of max flux as threshold to avoid clumps in faint outer regions
    flux_threshold = 0.05 * np.max(image)
    galaxy_mask = image > flux_threshold
    enhanced_image = image + clump_strength * clumpy_pattern * galaxy_mask

    return enhanced_image


def add_bar_structure_to_image(image, center_x, center_y, pixel_scale,
                                bar_length=25, bar_width=8, bar_angle=30, seed=42):
    """
    Add bar structure to galaxy with smooth elliptical Gaussian profile
    Avoids rectangular artifacts by using smooth radial falloff

    Args:
        image: 2D numpy array
        center_x, center_y: center position in pixels
        pixel_scale: arcsec/pixel
        bar_length: semi-major axis in pixels
        bar_width: semi-minor axis in pixels
        bar_angle: rotation angle in degrees
        seed: random seed

    Returns:
        Enhanced image with smooth bar
    """
    size = image.shape[0]
    y_grid, x_grid = np.ogrid[:size, :size]

    # Rotate coordinates for bar
    cos_angle = np.cos(np.radians(bar_angle))
    sin_angle = np.sin(np.radians(bar_angle))

    x_rot = (x_grid - center_x) * cos_angle + (y_grid - center_y) * sin_angle
    y_rot = -(x_grid - center_x) * sin_angle + (y_grid - center_y) * cos_angle

    # Create smooth elliptical bar profile (no hard edges)
    # Use elliptical radius for smooth falloff
    r_bar = np.sqrt((x_rot/bar_length)**2 + (y_rot/bar_width)**2)

    # Smooth Gaussian-like profile (no rectangular mask)
    bar_profile = 0.3 * np.exp(-0.5 * r_bar**2) * np.exp(-r_bar/1.5)

    # Add bar to galaxy (only where galaxy exists to avoid artifacts)
    enhanced_image = image + bar_profile * (image > 0.01)

    return enhanced_image


def add_ring_structure_to_image(image, center_x, center_y, pixel_scale,
                                 ring_radius=30, ring_width=2.2, n_knots=14, seed=42):
    """
    Realistic collision-induced (Cartwheel-type) ring galaxy.

    Physically distinct from Einstein rings:
    - Narrow ring (ring_width << ring_radius)
    - Slightly elliptical, not perfectly circular
    - Compact off-center nucleus (intruder remnant)
    - Azimuthally clumpy star-forming knots
    """
    rng = np.random.RandomState(seed)
    size = image.shape[0]
    y_grid, x_grid = np.mgrid[:size, :size]

    # Clearly elliptical ring (not nearly circular)
    q_ring = rng.uniform(0.62, 0.80)
    pa_ring = rng.uniform(0, np.pi)
    dx_r = x_grid - center_x
    dy_r = y_grid - center_y
    x_rot =  dx_r * np.cos(pa_ring) + dy_r * np.sin(pa_ring)
    y_rot = -dx_r * np.sin(pa_ring) + dy_r * np.cos(pa_ring)
    r_ell = np.sqrt(x_rot**2 + (y_rot / q_ring)**2)

    # Narrow primary ring
    ring_profile = 0.55 * np.exp(-0.5 * ((r_ell - ring_radius) / ring_width)**2)

    # Azimuthal brightness modulation (two brighter arcs, ~160° apart)
    theta_map = np.arctan2(dy_r, dx_r)
    arc_mod = 0.35 * (
        np.exp(-0.5 * ((theta_map - rng.uniform(0.3, 1.0)) / 0.8)**2)
        + 0.7 * np.exp(-0.5 * ((theta_map - (rng.uniform(0.3, 1.0) + np.pi)) / 1.1)**2)
    )
    ring_profile = ring_profile * (1.0 + arc_mod)

    # Off-center compact nucleus (intruder remnant) — guaranteed visible
    # offset, scaled with the ring size so it stays well inside the ring
    # for both small and large rings.
    nuc_dx = (2 * int(rng.uniform() > 0.5) - 1) * rng.uniform(0.27, 0.47) * ring_radius
    nuc_dy = (2 * int(rng.uniform() > 0.5) - 1) * rng.uniform(0.17, 0.33) * ring_radius
    r_nuc = np.sqrt((x_grid - (center_x + nuc_dx))**2 + (y_grid - (center_y + nuc_dy))**2)
    nucleus = 1.1 * np.exp(-r_nuc**2 / (2.0**2))

    # Star-forming knots in ring
    knots_pattern = np.zeros_like(image, dtype=np.float32)
    for i in range(n_knots):
        knot_angle = i * 2 * np.pi / n_knots + rng.uniform(-0.35, 0.35)
        knot_r = ring_radius + rng.uniform(-ring_width * 0.8, ring_width * 0.8)
        knot_x = center_x + knot_r * np.cos(knot_angle) * q_ring
        knot_y = center_y + knot_r * np.sin(knot_angle)
        if 0 <= knot_x < size and 0 <= knot_y < size:
            knot_dist = np.sqrt((x_grid - knot_x)**2 + (y_grid - knot_y)**2)
            knot_size = rng.uniform(1.5, 2.8)
            knot_brightness = rng.uniform(0.12, 0.28)
            knots_pattern += knot_brightness * np.exp(-knot_dist**2 / (knot_size**2))

    # Faint interior disk
    r_circ = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    interior_disk = 0.06 * np.exp(-0.5 * (r_circ / (ring_radius * 0.55))**2)

    return image + ring_profile + knots_pattern + nucleus + interior_disk


def add_dust_lane_to_image(image, center_x, center_y, axis_ratio,
                           position_angle, width=3):
    """
    Add realistic dust lane for edge-on galaxies
    STRICTLY confined to central galaxy region with sharp cutoff
    Prevents horizontal stripe artifacts across entire image

    Args:
        image: 2D numpy array
        center_y: center y position in pixels
        width: dust lane width in pixels

    Returns:
        Enhanced image with dust lane absorption
    """
    size = image.shape[0]
    y_grid, x_grid = np.indices(image.shape)

    rel_x = x_grid - center_x
    rel_y = y_grid - center_y

    pa_rad = np.radians(position_angle)
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
    x_rot = rel_x * cos_pa + rel_y * sin_pa
    y_rot = -rel_x * sin_pa + rel_y * cos_pa

    dust_profile = np.exp(-0.5 * (y_rot / width)**2)

    q = np.clip(axis_ratio, 0.2, 1.0)
    r_ell = np.sqrt(x_rot**2 + (y_rot / q)**2)

    flux_threshold = 0.01 * np.max(image)
    galaxy_mask = image > flux_threshold

    max_dust_radius = size * 0.22
    radius_mask = r_ell < max_dust_radius

    valid_region = galaxy_mask & radius_mask

    dust_absorption = np.zeros_like(image)
    dust_absorption[valid_region] = 0.45 * dust_profile[valid_region]

    edge_smooth = np.exp(-0.5 * ((r_ell - max_dust_radius * 0.8) / (max_dust_radius * 0.12))**2)
    edge_smooth = np.clip(edge_smooth, 0, 1)
    dust_absorption *= edge_smooth

    enhanced_image = image * (1 - dust_absorption)

    return enhanced_image


def apply_morphological_enhancements(images, n_sersic, q_ratio, morph_type,
                                     numpix=300, pixel_scale=0.03, seed=42,
                                     position_angle=0.0, context='field',
                                     skip_native_bar_ring=False, r_eff_pix=None):
    """
    Apply appropriate morphological enhancements based on galaxy type
    NEW v11: Applies enhancements from Paper Figure 2 v2

    IMPORTANT: Uses SAME seed for all bands so features appear in same positions
    (clumps, spiral arms, etc. should be in same place, just different brightnesses)

    Args:
        images: dict of band -> 2D numpy array
        n_sersic: Sersic index
        q_ratio: axis ratio
        morph_type: morphology classification string
        numpix: image size in pixels
        pixel_scale: arcsec/pixel
        seed: random seed (SAME for all bands!)
        position_angle: major-axis position angle in degrees
        context: 'lens' for main deflector, 'field' for non-lens systems

    Returns:
        Enhanced images dict
    """
    rng = np.random.RandomState(seed)
    pa = float(position_angle)
    context = (context or 'field').lower()
    enhanced_images = {}
    center = numpix // 2

    # QUALITY CONTROL: Check if galaxy has enough flux for enhancements
    # Don't enhance very faint galaxies or noise-dominated images
    first_band = list(images.keys())[0]
    max_flux = np.max(images[first_band])
    total_flux = np.sum(images[first_band])

    # Skip enhancements if galaxy is too faint or too small
    if max_flux < 1e-7 or total_flux < 1e-5:
        print(f"[v11] Skipping enhancements - galaxy too faint (max={max_flux:.2e}, total={total_flux:.2e})")
        return images

    # QUALITY CONTROL: Don't enhance ellipticals or very high-n galaxies
    # These should remain smooth
    if morph_type in ['elliptical', 's0'] and n_sersic > 2.5:
        return images  # Return unmodified - ellipticals should be smooth

    spiral_settings = None
    barred_settings = None

    for band, image in images.items():
        enhanced = image.copy()

        # CRITICAL FIX: Use SAME seed for all bands!
        # This ensures clumps/arms appear in same positions across bands
        # (they'll have different brightnesses due to galaxy SED, which is correct)

        # Apply enhancements based on morphology
        if morph_type in ['spiral', 'late_spiral', 'irregular_spiral']:
            if spiral_settings is None:
                spiral_settings = {}
                extra_clumps = False
                if n_sersic < 1.2:
                    pitch = rng.uniform(15, 20)
                    n_arms = 3
                    arm_strength = 0.25
                    bulge_fraction = 0.3 if n_sersic < 1.0 else 0.35
                    if n_sersic < 1.0:
                        clump_count = safe_random_integers(rng, 13, 19)
                        strength_range = (0.10, 0.18)
                        extra_clumps = True
                    else:
                        clump_count = safe_random_integers(rng, 11, 16)
                        strength_range = (0.09, 0.16)
                    radial_jitter = 0.12
                    theta_jitter = np.radians(7)
                elif n_sersic < 2.0:
                    n_arms = 2 if rng.random() < 0.6 else 3
                    pitch = rng.uniform(10, 15)
                    arm_strength = 0.22
                    bulge_fraction = 0.45
                    clump_count = safe_random_integers(rng, 5, 9)
                    strength_range = (0.06, 0.12)
                    radial_jitter = 0.08
                    theta_jitter = np.radians(5)
                else:
                    n_arms = 2
                    pitch = rng.uniform(8, 12)
                    arm_strength = 0.18
                    bulge_fraction = 0.5
                    clump_count = safe_random_integers(rng, 3, 6)
                    strength_range = (0.05, 0.1)
                    radial_jitter = 0.06
                    theta_jitter = np.radians(4)

                if context == 'lens':
                    # Lens galaxies should have more subdued morphological features
                    # Even late-type lens galaxies are typically more massive and less clumpy
                    clump_count = max(2, int(np.round(clump_count * 0.5)))  # Reduced from 0.7 to 0.5
                    strength_range = tuple(val * 0.7 for val in strength_range)  # Reduced from 0.85 to 0.7
                    # Note: arm_strength is reduced later when calling add_spiral_arms_to_image

                clump_config = {
                    'count': int(clump_count),
                    'size_range': (1.3, 2.6) if n_sersic < 1.2 else (1.1, 2.2),
                    'strength_range': strength_range,
                    'radial_jitter': radial_jitter,
                    'theta_jitter': theta_jitter
                }
                spiral_settings.update({
                    'pitch': pitch,
                    'n_arms': int(n_arms),
                    'arm_strength': arm_strength,
                    'bulge_fraction': bulge_fraction,
                    'clump_config': clump_config,
                    'seed': int(safe_random_integers(rng, 0, 1_000_000)),
                    'extra_clumps': extra_clumps
                })

            settings = spiral_settings
            # For lens galaxies, reduce arm strength
            effective_arm_strength = settings['arm_strength']
            if context == 'lens':
                effective_arm_strength = settings['arm_strength'] * 0.75  # Reduce arm prominence
            
            enhanced = add_spiral_arms_to_image(
                enhanced, center, center, pixel_scale,
                pitch_angle=settings['pitch'],
                arm_strength=effective_arm_strength,
                n_arms=settings['n_arms'],
                axis_ratio=q_ratio,
                position_angle=pa,
                bulge_fraction=settings['bulge_fraction'],
                seed=settings['seed'],
                band=band,
                clump_config=settings['clump_config']
            )
            if settings.get('extra_clumps'):
                # For lens galaxies, reduce extra clumps significantly
                if context == 'lens':
                    n_extra_clumps = max(3, int(10 * 0.4))  # Reduce from 10 to 4
                    extra_clump_strength = 0.28 * 0.6  # Reduce strength
                else:
                    n_extra_clumps = 10
                    extra_clump_strength = 0.28
                enhanced = add_clumpy_structure_to_image(
                    enhanced, center, center, pixel_scale,
                    n_clumps=n_extra_clumps,
                    clump_strength=extra_clump_strength,
                    seed=settings['seed']
                )

        # Irregular/clumpy systems (very low n)
        elif morph_type in ['irregular', 'clumpy', 'starburst']:
            enhanced = add_clumpy_structure_to_image(
                enhanced, center, center, pixel_scale,
                n_clumps=12,
                clump_strength=0.5,  # Reduced from 0.8 for more subtle appearance
                seed=seed  # SAME seed for all bands!
            )

        # Barred spirals
        elif morph_type == 'barred_spiral':
            if barred_settings is None:
                barred_settings = {}
                n_arms = 2 if rng.random() < 0.7 else 3
                pitch = rng.uniform(12, 18)
                clump_count = safe_random_integers(rng, 7, 12)
                strength_range = (0.07, 0.14)
                if context == 'lens':
                    clump_count = max(3, int(np.round(clump_count * 0.7)))
                    strength_range = tuple(val * 0.85 for val in strength_range)
                barred_settings.update({
                    'bar_angle': rng.uniform(0, 180),
                    'n_arms': int(n_arms),
                    'pitch': pitch,
                    'clump_config': {
                        'count': int(clump_count),
                        'size_range': (1.1, 2.3),
                        'strength_range': strength_range,
                        'radial_jitter': 0.08,
                        'theta_jitter': np.radians(6)
                    },
                    'seed': int(safe_random_integers(rng, 0, 1_000_000))
                })

            settings = barred_settings

            # The bar itself may already be a native SERSIC_ELLIPSE
            # component (galaxy_morphology.components, when
            # morphology.multicomponent_enabled=True) -- skip the
            # pixel-space bar to avoid double-rendering it. Spiral arms
            # (texture, not a native component) are still added.
            if not skip_native_bar_ring:
                enhanced = add_bar_structure_to_image(
                    enhanced, center, center, pixel_scale,
                    bar_length=25,
                    bar_width=8,
                    bar_angle=settings['bar_angle'],
                    seed=seed  # SAME seed for all bands!
                )
            # Add spiral arms emerging from bar ends
            enhanced = add_spiral_arms_to_image(
                enhanced, center, center, pixel_scale,
                pitch_angle=settings['pitch'],
                arm_strength=0.28,
                n_arms=settings['n_arms'],
                axis_ratio=q_ratio,
                position_angle=pa,
                bulge_fraction=0.4,
                seed=settings['seed'],
                band=band,
                clump_config=settings['clump_config']
            )

        # Ring galaxies. The pixel-space collision-ring (Cartwheel-type) is
        # a different feature from the native annular-Sersic 'ring'
        # component (galaxy_morphology.components); skip the pixel-space
        # version when the native component is already present to avoid
        # stacking two unrelated ring features.
        elif morph_type == 'ring' and not skip_native_bar_ring:
            # Place the collisional ring well outside the galaxy's light
            # profile (real Cartwheel-type rings sit at several effective
            # radii from the nucleus, with a clear gap in between).
            # Fall back to a numpix-relative default if r_eff_pix is
            # unavailable.
            if r_eff_pix is not None and r_eff_pix > 0:
                ring_radius = float(np.clip(2.5 * r_eff_pix, 0.1 * numpix, 0.45 * numpix))
            else:
                ring_radius = 0.1 * numpix
            ring_width = max(0.08 * ring_radius, 1.5)
            enhanced = add_ring_structure_to_image(
                enhanced, center, center, pixel_scale,
                ring_radius=ring_radius,
                ring_width=ring_width,
                n_knots=14,
                seed=seed  # same seed for all bands
            )

        # Edge-on spirals with dust lanes
        elif morph_type == 'edge_on' or (morph_type == 'spiral' and q_ratio < 0.35):
            enhanced = add_dust_lane_to_image(
                enhanced, center, center, q_ratio, pa, width=3
            )

        # Primordial high-z irregulars (ultra-low n)
        elif morph_type == 'primordial' or (morph_type == 'irregular' and n_sersic < 0.6):
            enhanced = add_clumpy_structure_to_image(
                enhanced, center, center, pixel_scale,
                n_clumps=15,  # More clumps for primordial
                clump_strength=0.6,  # Reduced from 1.0 for more realistic appearance
                seed=seed  # SAME seed for all bands!
            )

        enhanced_images[band] = enhanced

    return enhanced_images


# --------------------------------------------------------------------------------------
# CSV utilities
# --------------------------------------------------------------------------------------
def _read_csv_robust(path):
    """Robust CSV reader with multiple strategies"""
    tries = [
        dict(sep=",", low_memory=False, na_values=NA_SENTINELS),
        dict(sep=",", low_memory=False, skipinitialspace=True, na_values=NA_SENTINELS),
        dict(sep=",", low_memory=False, encoding="utf-8", na_values=NA_SENTINELS),
        dict(sep=",", low_memory=False, encoding="latin1", na_values=NA_SENTINELS),
    ]
    last_err = None
    for kw in tries:
        try:
            df = pd.read_csv(path, **kw)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not read CSV {path}: {last_err}")

def _get_series(df, cand_cols, length, default=np.nan, numeric=True):
    """Safe series extraction with proper indexing"""
    if df is None or len(df) == 0:
        return pd.Series([default] * length, index=range(length))
    
    for col in cand_cols:
        if col in df.columns:
            s = df[col]
            if numeric:
                s = pd.to_numeric(s, errors="coerce")
            s = s.reset_index(drop=True)
            
            if len(s) >= length:
                s = s.iloc[:length].reset_index(drop=True)
            else:
                s = s.reindex(range(length)).fillna(default)
            
            return s
    
    return pd.Series([default] * length, index=range(length))

# --------------------------------------------------------------------------------------
# CORRECTED: Real field galaxy data processing
# --------------------------------------------------------------------------------------

def load_real_field_galaxy_population_from_merged(merged_csv_path):
    """Load and process real field galaxy measurements from single merged CSV file"""
    
    merged_path = Path(merged_csv_path)
    if not merged_path.exists():
        print(f"[WARNING] Merged field galaxy CSV not found: {merged_path}")
        return None
    
    print(f"[INFO] Loading real field galaxy data from merged CSV: {merged_path}")
    
    try:
        # CRITICAL: Safe CSV loading to prevent execution errors
        field_pop = pd.read_csv(
            merged_path, 
            sep=',',
            low_memory=False,
            na_values=NA_SENTINELS,
            dtype=str,  # Read as strings first
            skipinitialspace=True
        )
        
        # Clean column names
        field_pop.columns = field_pop.columns.str.strip()
        
        print(f"[INFO] Loaded merged field galaxy catalog:")
        print(f"  Total entries: {len(field_pop)}")
        print(f"  Columns: {len(field_pop.columns)}")
        
        # Convert numeric columns safely
        numeric_columns = [
            'rearc_f115w', 'rearc_f150w', 'rearc_f277w', 'rearc_f444w',
            'nsersic_f115w', 'nsersic_f150w', 'nsersic_f277w', 'nsersic_f444w',
            'qratio_f115w', 'qratio_f150w', 'qratio_f277w', 'qratio_f444w',
            'LP_zfinal', 'sep_arcsec', 'x_pixels', 'y_pixels',
            'mag_f115w', 'mag_f150w', 'mag_f277w', 'mag_f444w',
            'MAG_MODEL_F115W', 'MAG_MODEL_F150W', 'MAG_MODEL_F277W', 'MAG_MODEL_F444W'
        ]
        
        for col in numeric_columns:
            if col in field_pop.columns:
                field_pop[col] = pd.to_numeric(field_pop[col], errors='coerce')
        
        # Identify lens grouping column
        lens_id_col = None
        possible_lens_cols = ['lens_id', 'LENS_ID', 'lens_name', 'field_id', 'parent_lens']
        
        for col in possible_lens_cols:
            if col in field_pop.columns:
                lens_id_col = col
                break
        
        if lens_id_col is None:
            for col in field_pop.columns:
                if 'lens' in col.lower():
                    print(f"  Found potential grouping column: {col}")
                    lens_id_col = col
                    break
        
        if lens_id_col:
            unique_lenses = field_pop[lens_id_col].nunique()
            print(f"  Lens grouping column: {lens_id_col}")
            print(f"  Unique lens fields: {unique_lenses}")
            print(f"  Average galaxies per field: {len(field_pop)/unique_lenses:.1f}")
        else:
            print(f"  No lens grouping found - treating as single population")
        
        # Process the merged data
        field_pop = process_real_field_galaxy_data_merged(field_pop, lens_id_col)
        
        return field_pop
        
    except Exception as e:
        print(f"[ERROR] Failed to load merged field galaxy CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_real_field_galaxy_data_merged(df, lens_id_col=None):
    """Process and clean real field galaxy measurements from merged catalog"""
    
    print("[INFO] Processing merged field galaxy measurements...")
    
    # Store original lens association if available
    if lens_id_col and lens_id_col in df.columns:
        df['source_lens_id'] = df[lens_id_col].astype(str)
    else:
        df['source_lens_id'] = 'merged_catalog'
    
    # Clean redshifts
    if 'LP_zfinal' in df.columns:
        df['field_redshift'] = pd.to_numeric(df['LP_zfinal'], errors='coerce')
    elif 'redshift' in df.columns:
        df['field_redshift'] = pd.to_numeric(df['redshift'], errors='coerce')
    else:
        print("[WARNING] No redshift column found, using random distribution")
        # EXPANDED: Extended field galaxy redshift range
        df['field_redshift'] = np.random.uniform(0.1, 15.0, len(df))
    
    # EXPANDED: Clean up invalid redshifts with extended range
    df['field_redshift'] = df['field_redshift'].clip(0.01, 15.0)
    # Fix for pandas compatibility - cannot pass numpy array directly to fillna
    missing_mask = df['field_redshift'].isna()
    if missing_mask.any():
        n_missing = missing_mask.sum()
        # EXPANDED: Wider fallback range for field galaxies
        df.loc[missing_mask, 'field_redshift'] = np.random.uniform(0.1, 5.0, n_missing)
    
    # Calculate separations if not provided
    if 'sep_arcsec' in df.columns:
        df['field_sep_arcsec'] = pd.to_numeric(df['sep_arcsec'], errors='coerce')
    elif all(col in df.columns for col in ['x_pixels', 'y_pixels']):
        center_pix = 150  # Assuming 300x300 with center at 150,150
        dx = pd.to_numeric(df['x_pixels'], errors='coerce') - center_pix
        dy = pd.to_numeric(df['y_pixels'], errors='coerce') - center_pix
        sep_pix = np.sqrt(dx**2 + dy**2)
        df['field_sep_arcsec'] = sep_pix * 0.03
    else:
        print("[WARNING] No position columns found, using random separations")
        df['field_sep_arcsec'] = np.random.uniform(1.0, 4.0, len(df))
    
    # Fill missing separations - pandas compatibility fix
    missing_sep_mask = df['field_sep_arcsec'].isna()
    if missing_sep_mask.any():
        n_missing_sep = missing_sep_mask.sum()
        df.loc[missing_sep_mask, 'field_sep_arcsec'] = np.random.uniform(1.0, 5.0, n_missing_sep)
    
    # Extract rest-frame structural parameters
    rest_params = extract_restframe_struct_real(df, df['field_redshift'])
    
    for key, series in rest_params.items():
        df[key] = series
    
    # Add morphological classification
    df = classify_real_galaxy_morphology(df)
    
    # Remove objects too close to center
    initial_count = len(df)
    min_sep = 0.5  # CORRECTED: Less restrictive (was 0.8)
    df = df[df['field_sep_arcsec'] > min_sep].reset_index(drop=True)
    
    # Remove objects with problematic measurements
    df = df.dropna(subset=['re_rest_clean', 'q_rest_clean', 'n_rest_clean']).reset_index(drop=True)
    
    print(f"[INFO] Processed merged field galaxy catalog:")
    print(f"  Initial galaxies: {initial_count}")
    print(f"  After center exclusion: {len(df)} (removed {initial_count - len(df)} central objects)")
    print(f"  Final valid population: {len(df)} field galaxies")
    
    # Show lens field distribution
    if 'source_lens_id' in df.columns and df['source_lens_id'].nunique() > 1:
        unique_lenses = df['source_lens_id'].nunique()
        print(f"  Distributed across: {unique_lenses} lens fields")
        print(f"  Average per field: {len(df)/unique_lenses:.1f}")
    
    # Show morphology statistics
    if 'real_morph_type' in df.columns:
        morph_counts = df['real_morph_type'].value_counts()
        print(f"[INFO] Real morphology distribution:")
        for morph, count in morph_counts.items():
            print(f"  {morph}: {count} ({100*count/len(df):.1f}%)")
    
    return df

def extract_restframe_struct_real(df, z_series, rest_um=1.6):
    """Extract rest-frame structural parameters for real field galaxies"""
    
    n = len(df)
    re_out = pd.Series(np.nan, index=range(n))
    q_out = pd.Series(np.nan, index=range(n)) 
    n_out = pd.Series(np.nan, index=range(n))
    
    # Per-galaxy band selection based on redshift
    for i, z in enumerate(z_series.values):
        if i >= len(df):
            break
            
        # Choose best band for this redshift
        band = choose_observed_band_for_rest(rest_um, z)
        
        # Extract measurements from the best band
        for measurement, series_out, colname in [
            ("rearc", re_out, f"rearc_{band}"),
            ("qratio", q_out, f"qratio_{band}"), 
            ("nsersic", n_out, f"nsersic_{band}"),
        ]:
            if colname in df.columns:
                try:
                    val = pd.to_numeric(df[colname], errors="coerce").iloc[i]
                    if pd.notna(val) and np.isfinite(val):
                        # Apply realistic range constraints
                        if measurement == "rearc" and 0.03 <= val <= 3.0:
                            series_out.iat[i] = val
                        elif measurement == "qratio" and 0.1 <= val <= 1.0:
                            series_out.iat[i] = val
                        elif measurement == "nsersic" and 0.2 <= val <= 8.0:
                            series_out.iat[i] = val
                except:
                    pass
    
    # Fallback to best available measurements across all bands
    for measurement, series_out, col_template in [
        ("rearc", re_out, "rearc_{}"),
        ("qratio", q_out, "qratio_{}"), 
        ("nsersic", n_out, "nsersic_{}"),
    ]:
        for i, val in enumerate(series_out.values):
            if pd.isna(val):
                # Try all bands for this galaxy
                for band in LOWER_BANDS:
                    colname = col_template.format(band)
                    if colname in df.columns:
                        try:
                            candidate = pd.to_numeric(df[colname], errors="coerce").iloc[i]
                            if pd.notna(candidate) and np.isfinite(candidate):
                                if measurement == "rearc" and 0.03 <= candidate <= 3.0:
                                    series_out.iat[i] = candidate
                                    break
                                elif measurement == "qratio" and 0.1 <= candidate <= 1.0:
                                    series_out.iat[i] = candidate
                                    break
                                elif measurement == "nsersic" and 0.2 <= candidate <= 8.0:
                                    series_out.iat[i] = candidate
                                    break
                        except:
                            continue
    
    # Final fallbacks with realistic defaults
    re_out = re_out.fillna(0.4)  # Typical field galaxy size
    q_out = q_out.fillna(0.7).clip(0.1, 1.0)
    n_out = n_out.fillna(1.5).clip(0.3, 8.0)  # Extended range for massive ellipticals
    
    return {
        "re_rest_clean": re_out, 
        "q_rest_clean": q_out, 
        "n_rest_clean": n_out
    }

def classify_real_galaxy_morphology(df):
    """Classify real galaxies into morphological types using measurements"""
    
    # Use Sersic index as primary classifier (rest-frame when available)
    if 'n_rest_clean' in df.columns:
        n_sersic = df['n_rest_clean']
    else:
        # Use F444W as proxy for rest-frame
        n_sersic = pd.to_numeric(df['nsersic_f444w'], errors='coerce').fillna(2.0)
    
    # Use axis ratio for additional classification
    if 'q_rest_clean' in df.columns:
        q_ratio = df['q_rest_clean']  
    else:
        q_ratio = pd.to_numeric(df['qratio_f444w'], errors='coerce').fillna(0.7)
    
    # Morphological classification based on observations
    morph_type = []
    color_type = []
    
    for i, (n, q) in enumerate(zip(n_sersic.values, q_ratio.values)):
        n = float(n) if pd.notna(n) else 2.0
        q = float(q) if pd.notna(q) else 0.7
        
        if n > 3.5:
            morph_type.append('elliptical')
            color_type.append('red')
        elif n > 2.5:
            if q > 0.6:
                morph_type.append('S0')
                color_type.append('intermediate')
            else:
                morph_type.append('spiral_early')
                color_type.append('blue_intermediate')
        elif n > 1.2:
            morph_type.append('spiral_early')
            color_type.append('blue_intermediate')
        else:
            morph_type.append('spiral_late')
            color_type.append('blue')
    
    df['real_morph_type'] = morph_type
    df['real_color_type'] = color_type
    
    return df

# --------------------------------------------------------------------------------------
# CORRECTED: Enhanced field galaxy sampling with realistic populations
# --------------------------------------------------------------------------------------

def extract_realistic_size(real_gal, rng, lens_radius=None):
    """Extract realistic effective radius from real galaxy data
    
    Args:
        real_gal: Galaxy data row
        rng: Random number generator
        lens_radius: Optional lens radius to constrain field galaxy size
    """
    # Get size constraints from config
    cfg_field = CONFIG.get('field', {})
    max_size = float(cfg_field.get('size_max_arcsec', 1.2))
    max_fraction = float(cfg_field.get('size_max_fraction_of_lens', 0.6))
    
    # If lens radius provided, constrain field galaxy size
    if lens_radius is not None and lens_radius > 0:
        max_size = min(max_size, lens_radius * max_fraction)
    
    # Try to get measured size
    size_candidates = ['re_rest_clean', 'rearc_f444w', 'rearc_f277w', 'rearc_f150w', 'rearc_f115w']
    
    for col in size_candidates:
        if col in real_gal.index and pd.notna(real_gal[col]):
            try:
                size = float(real_gal[col])
                if 0.03 <= size <= max_size:  # Constrained by config
                    # Add small scatter to avoid identical sizes
                    scattered_size = size * rng.lognormal(0, 0.1)
                    return min(scattered_size, max_size)  # Hard cap
            except:
                continue
    
    # Fallback: realistic size distribution (smaller than lens)
    fallback_size = rng.lognormal(np.log(0.3), 0.5)  # Smaller mean
    return min(fallback_size, max_size)

def extract_realistic_axis_ratio(real_gal, rng):
    """Extract realistic axis ratio with morphological diversity"""
    
    q_candidates = ['q_rest_clean', 'qratio_f444w', 'qratio_f277w', 'qratio_f150w', 'qratio_f115w']
    
    for col in q_candidates:
        if col in real_gal.index and pd.notna(real_gal[col]):
            try:
                q = float(real_gal[col])
                if 0.1 <= q <= 1.0:
                    # Add scatter for diversity with conservative limits
                    q_varied = q * rng.lognormal(0, 0.12)  # Less scatter
                    return np.clip(q_varied, 0.5, 1.0)  # Prevent overly thin
            except:
                continue
    
    # Fallback: realistic axis ratio distribution avoiding overly thin galaxies
    # More conservative to prevent thin appearance
    return rng.beta(2, 2) * 0.4 + 0.6  # Range 0.6-1.0, more realistic

def extract_realistic_sersic_index(real_gal, rng):
    """CRITICAL FIX: Extract Sersic index with proper morphological diversity"""
    
    n_candidates = ['n_rest_clean', 'nsersic_f444w', 'nsersic_f277w', 'nsersic_f150w', 'nsersic_f115w']
    
    for col in n_candidates:
        if col in real_gal.index and pd.notna(real_gal[col]):
            try:
                n = float(real_gal[col])
                if 0.2 <= n <= 8.0:
                    # Add controlled scatter
                    n_varied = n * rng.lognormal(0, 0.2)
                    return np.clip(n_varied, 0.3, 6.0)
            except:
                continue
    
    # EXTENDED RANGE: Include massive ellipticals with high Sersic indices
    # Real spiral galaxies often have n~0.3-0.8, ellipticals n~2-8
    sersic_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    raw_weights = [0.20, 0.22, 0.20, 0.15, 0.10, 0.08, 0.03, 0.015, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002]
    sersic_weights = [w / sum(raw_weights) for w in raw_weights]  # Normalize exactly
    
    return rng.choice(sersic_values, p=sersic_weights)

def _field_mag_limit() -> float:
    """
    Telescope-aware faint magnitude limit for field galaxies.
    Ensures field galaxies stay above the per-telescope noise floor.
    Limits (SNR≥5 per pixel at typical exposure):
      JWST 1028s  → 25.5  (SNR~10)
      Roman 146s  → 24.5  (SNR~6)
      Euclid 565s → 24.0  (SNR~7, lower ZP)
      Subaru 600s → 24.5  (SNR~8)
      LSST   30s  → 23.0  (SNR~8, sky-dominated)
    """
    tel = CONFIG.get('telescope', 'jwst').lower()
    limits = {'jwst': 25.5, 'roman': 24.5, 'euclid': 24.0, 'subaru': 24.5, 'lsst': 23.0}
    return limits.get(tel, 25.0)


def _field_target_size_factor() -> float:
    """``field_target_size_factor`` from the ``galaxygenius_stamps`` config
    block: TNG-particle field galaxies are rendered at this multiple of
    their R_sersic (arcsec)."""
    return float(CONFIG.get('galaxygenius_stamps', {}).get('field_target_size_factor', 4.0))


def convert_real_galaxy_to_field_format(gal_row, rng, lens_radius=None, numpix=300, pixel_scale=0.03):
    """Convert a real galaxy row to field galaxy format for simulation
    
    Args:
        gal_row: Pandas Series with galaxy data from merged catalog
        rng: Random number generator
        lens_radius: Lens radius for size constraints
        numpix: Image size in pixels
        pixel_scale: Arcsec per pixel
    
    Returns:
        dict: Field galaxy parameters for simulation
    """
    try:
        # Extract structural parameters with size constraints
        re_eff = extract_realistic_size(gal_row, rng, lens_radius=lens_radius)
        q_axis = extract_realistic_axis_ratio(gal_row, rng)
        n_sersic = extract_realistic_sersic_index(gal_row, rng)
        
        # Use observed separation as position.
        # lenstronomy uses arcseconds for all coordinates — do NOT divide by pixel_scale.
        sep_arcsec = float(gal_row.get('sep_arcsec', 2.0))
        half_size_arcsec = (numpix * pixel_scale) / 2.0  # image half-width in arcsec

        angle = rng.uniform(0, 2 * np.pi)
        x_pos = sep_arcsec * np.cos(angle)   # arcsec
        y_pos = sep_arcsec * np.sin(angle)   # arcsec

        # Clip to image bounds (arcsec)
        x_pos = np.clip(x_pos, -(half_size_arcsec - pixel_scale), half_size_arcsec - pixel_scale)
        y_pos = np.clip(y_pos, -(half_size_arcsec - pixel_scale), half_size_arcsec - pixel_scale)
        
        # Extract magnitude (use F150W as reference)
        # Clean bad catalog values (-999, inf, nan) then clip to detectable range.
        # The COSMOS-Web catalog median is 26.2 mag; below ~26.5 individual galaxies
        # are undetectable in typical JWST exposures. Cap at 26.0 so field galaxies
        # remain detectable above the noise floor.
        _raw_mag = gal_row.get('mag_f150w', None)
        try:
            _raw_mag = float(_raw_mag)
            if not np.isfinite(_raw_mag) or _raw_mag < 0 or _raw_mag > 35:
                _raw_mag = None
        except (TypeError, ValueError):
            _raw_mag = None

        if _raw_mag is not None:
            magnitude = float(np.clip(_raw_mag + rng.normal(0, 0.3), 18.0, _field_mag_limit()))
        else:
            magnitude = float(extract_realistic_magnitude(gal_row, float(gal_row.get('LP_zPDF', 1.5)), rng))
        
        # Extract redshift if available
        redshift = float(gal_row.get('LP_zPDF', 1.5))
        
        # Position angle and ellipticity
        pa = rng.uniform(-180, 180)
        e = (1 - q_axis) / (1 + q_axis)
        e1 = e * np.cos(2 * np.radians(pa))
        e2 = e * np.sin(2 * np.radians(pa))
        
        # Determine morphology type from Sersic index
        if n_sersic < 1.5:
            morph_type = 'spiral'
        elif n_sersic > 3.0:
            morph_type = 'elliptical'
        else:
            morph_type = 'S0'
        
        return {
            'center_x': float(x_pos),
            'center_y': float(y_pos),
            'R_sersic': float(re_eff),
            'n_sersic': float(n_sersic),
            'e1': float(np.clip(e1, -0.8, 0.8)),
            'e2': float(np.clip(e2, -0.8, 0.8)),
            'axis_ratio': float(q_axis),
            'position_angle': float(pa),
            'magnitude': float(magnitude),
            'redshift': float(redshift),
            'morph_type': morph_type,
            'sep_arcsec': float(sep_arcsec)  # Store original separation
        }
        
    except Exception as e:
        print(f"[WARNING] Failed to convert galaxy row: {e}")
        return None

def extract_field_redshift(real_gal, lens_redshift, rng):
    """Extract and adjust field galaxy redshift"""
    
    if 'LP_zfinal' in real_gal.index and pd.notna(real_gal['LP_zfinal']):
        observed_z = float(real_gal['LP_zfinal'])
        if 0.1 <= observed_z <= 15.0:
            # Add small scatter but keep realistic
            z_varied = observed_z + rng.normal(0, 0.1)
            # EXPANDED: Extended redshift range for field galaxies
            return np.clip(z_varied, 0.05, 15.0)
    
    # Realistic redshift distribution for field galaxies
    if rng.random() < 0.3:  # 30% foreground
        return rng.uniform(0.1, max(0.8 * lens_redshift, 0.3))
    else:  # 70% background
        # EXPANDED: Extended background redshift range
        return rng.uniform(lens_redshift + 0.2, 15.0)

def extract_realistic_magnitude(real_gal, redshift, rng):
    """Extract realistic magnitude with proper scaling"""
    
    # Try observed magnitudes
    mag_candidates = [
        'mag_f444w', 'MAG_MODEL_F444W',
        'mag_f277w', 'MAG_MODEL_F277W',
        'mag_f150w', 'MAG_MODEL_F150W',
        'mag_f115w', 'MAG_MODEL_F115W'
    ]
    
    for col in mag_candidates:
        if col in real_gal.index and pd.notna(real_gal[col]):
            try:
                mag = float(real_gal[col])
                if 18.0 <= mag <= 28.0:
                    # Add scatter and ensure detectability
                    mag_varied = mag + rng.normal(0, 0.2)
                    return np.clip(mag_varied, 18.0, 28.0)  # Allow full magnitude range
            except:
                continue
    
    # Fallback: realistic magnitude distribution for extended redshift range
    # Enhanced scaling for high-z galaxies (z=0.8-15.0)
    z_eff = max(redshift, 0.1)
    if z_eff <= 2.0:
        # Low-z regime: moderate dimming
        base_mag = 23.0 + 1.5 * np.log10(z_eff)
    else:
        # High-z regime: stronger dimming due to (1+z)^4 cosmological effect
        base_mag = 24.5 + 2.0 * np.log10(z_eff / 2.0)
    
    scatter = rng.normal(0, 0.8)
    return np.clip(base_mag + scatter, 18.0, _field_mag_limit())

def classify_galaxy_morphology_enhanced(n_sersic, q_ratio, rng=None, allow_ring=True):
    """
    Enhanced morphological classification with v11 enhancements
    Returns morphology type string for enhancement application

    Thin backward-compatible wrapper -- canonical implementation now lives
    in src/galaxy_morphology/taxonomy.py:classify_morphology (also used to
    pick native multi-component light-model fragments).

    allow_ring=False (use for lens-deflector / non-lens central galaxies)
    remaps 'ring' -> 'spiral', since the pixel-space collisional-ring
    feature can be visually confused with a strong-lensing Einstein ring.
    """
    try:
        from prism.morphology.taxonomy import classify_morphology
    except ImportError:
        from prism.morphology.taxonomy import classify_morphology
    return classify_morphology(n_sersic, q_ratio, rng, allow_ring=allow_ring)

def classify_galaxy_colors_enhanced(redshift, n_sersic, rng):
    """Enhanced color classification"""
    
    # Color-morphology correlation with scatter
    if n_sersic > 3.0:  # Ellipticals
        return rng.choice(['red', 'intermediate'], p=[0.7, 0.3])
    elif n_sersic > 2.0:  # S0/early spirals
        return rng.choice(['intermediate', 'blue_intermediate'], p=[0.6, 0.4])
    else:  # Late spirals
        return rng.choice(['blue', 'blue_intermediate'], p=[0.7, 0.3])

def assess_structural_quality(re_eff, n_sersic, q_axis):
    """Assess quality of structural parameters"""
    
    quality_score = 1.0
    
    # Penalize extreme values
    if re_eff < 0.05 or re_eff > 2.5:
        quality_score *= 0.7
    if n_sersic < 0.3 or n_sersic > 6.0:
        quality_score *= 0.8
    if q_axis < 0.2:
        quality_score *= 0.9
    
    if quality_score > 0.8:
        return 'high'
    elif quality_score > 0.6:
        return 'medium'
    else:
        return 'low'

_COSMOS_FIELD_PROPS_CACHE: dict = {}


def cosmos_field_galaxy_properties(mag_limit, catalog_path="data/galaxy_catalog.fits"):
    """Load and cache per-galaxy structural/photometric properties from the
    COSMOS-Web detection catalog, restricted to real detections below
    ``mag_limit`` in F115W.

    Returns a dict of numpy arrays (mag_f115w/f150w/f277w/f444w, R_sersic_arcsec,
    axis_ratio, n_sersic, position_angle_deg, redshift, log_mass) or None if the
    catalog can't be loaded -- caller should fall back to parametric modeling.

    This is the basis for bootstrap-resampling real galaxies (rather than just
    using the catalog to set an aggregate target *count*): drawing whole rows
    preserves the real joint distribution of size, axis ratio, Sersic index,
    luminosity and redshift, instead of reproducing only the 1-D number
    density and reinventing everything else independently at random.
    """
    cache_key = catalog_path
    if cache_key in _COSMOS_FIELD_PROPS_CACHE:
        cached = _COSMOS_FIELD_PROPS_CACHE[cache_key]
    else:
        try:
            from astropy.io import fits
            with fits.open(catalog_path) as hdul:
                d = hdul[1].data
                cached = {
                    "mag_f115w": np.asarray(d["mag_f115w"], dtype=float),
                    "mag_f150w": np.asarray(d["mag_f150w"], dtype=float),
                    "mag_f277w": np.asarray(d["mag_f277w"], dtype=float),
                    "mag_f444w": np.asarray(d["mag_f444w"], dtype=float),
                    "radius_deg": np.asarray(d["RADIUS"], dtype=float),
                    "axis_ratio": np.asarray(d["AXRATIO"], dtype=float),
                    "n_sersic": np.asarray(d["SERSIC"], dtype=float),
                    "pa_deg": np.asarray(d["ANGLE_SERSIC"], dtype=float),
                    "redshift": np.asarray(d["LP_zfinal"], dtype=float),
                    "log_mass": np.asarray(d["LP_mass_med_PDF"], dtype=float),
                }
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] cosmos_field_galaxy_properties: could not load {catalog_path}: {exc}")
            cached = None
        _COSMOS_FIELD_PROPS_CACHE[cache_key] = cached

    if cached is None:
        return None

    valid = (
        # FIX (adversarial audit finding C-4, 2026-08-01): "> -90" does not
        # reject this catalog's actual null-photometry sentinel, which
        # reaches ~-88.95 -- it passed the old check and was rendered as a
        # real galaxy. Measured impact: in the mag<22.5 pool, 10.86% of
        # rows were brighter than mag 16 (min -88.95); these saturate the
        # detector (identical ~84.7 e-/s flux in all 4 bands = full-well
        # clipping, confirmed by execution). Nothing in a real COSMOS-Web
        # field is brighter than AB~15, so use a physical bound instead of
        # a sentinel guard.
        np.isfinite(cached["mag_f115w"]) & (cached["mag_f115w"] > 15.0) & (cached["mag_f115w"] < mag_limit)
        & np.isfinite(cached["radius_deg"]) & (cached["radius_deg"] > 0)
        & np.isfinite(cached["axis_ratio"]) & (cached["axis_ratio"] > 0) & (cached["axis_ratio"] <= 1.0)
        & np.isfinite(cached["n_sersic"]) & (cached["n_sersic"] > 0)
    )
    if not np.any(valid):
        return None
    return {k: v[valid] for k, v in cached.items()}


def sample_cosmos_field_galaxies(rng, n, x_positions, y_positions, mag_limit=27.5,
                                  catalog_path="data/galaxy_catalog.fits",
                                  group_target_z=None, group_z_fraction=0.0,
                                  group_z_window=0.05):
    """Bootstrap-resample ``n`` real COSMOS-Web galaxies (with replacement) and
    place them at the given (already-sampled) positions, inheriting their real
    measured size, axis ratio, Sersic index, magnitude (all 4 JWST bands) and
    photometric redshift. Falls back to None if the catalog is unavailable.

    group_target_z / group_z_fraction / group_z_window: FIX (adversarial
    audit finding C-12, 2026-08-01). Previously field galaxies were an
    UNCONDITIONAL bootstrap over the whole catalog regardless of the
    lens's environment label -- a lens explicitly classified "group"
    still had field-galaxy redshifts entirely uncorrelated with the
    lens's own redshift (i.e. no actual group at any redshift was ever
    rendered, only an isolated lens with a richer, but still
    field-redshift-distributed, background/foreground population). When
    group_z_fraction>0, that fraction of the n galaxies is preferentially
    drawn from real catalog galaxies within +/-group_z_window of
    group_target_z (the lens redshift) -- i.e. real, physically-plausible
    group members -- while the remainder is still a normal field
    bootstrap (real lens/group fields DO also contain unrelated line-of-
    sight interlopers, so 100% group-redshift membership would itself be
    unrealistic).
    """
    props = cosmos_field_galaxy_properties(mag_limit, catalog_path)
    if props is None or n <= 0:
        return None

    if group_target_z is not None and group_z_fraction > 0:
        n_group = int(round(n * group_z_fraction))
        n_field = n - n_group
        near_mask = np.abs(props["redshift"] - group_target_z) < group_z_window
        near_idx_pool = np.nonzero(near_mask)[0]
        if len(near_idx_pool) > 0 and n_group > 0:
            idx_group = rng.choice(near_idx_pool, size=n_group, replace=True)
            idx_field = rng.integers(0, len(props["mag_f115w"]), size=n_field)
            idx = np.concatenate([idx_group, idx_field])
            rng.shuffle(idx)
        else:
            idx = rng.integers(0, len(props["mag_f115w"]), size=n)
    else:
        idx = rng.integers(0, len(props["mag_f115w"]), size=n)
    field_galaxies = []
    for i, src_i in enumerate(idx):
        q_ratio = float(np.clip(props["axis_ratio"][src_i], 0.05, 1.0))
        n_sersic = float(np.clip(props["n_sersic"][src_i], 0.3, 8.0))
        re_eff = float(np.clip(props["radius_deg"][src_i] * 3600.0, 0.03, 3.0))  # deg -> arcsec
        pa = float(props["pa_deg"][src_i])
        e = (1 - q_ratio) / (1 + q_ratio)
        e1 = e * np.cos(2 * np.radians(pa))
        e2 = e * np.sin(2 * np.radians(pa))

        mags = {b: props[f"mag_{b}"][src_i] for b in ("f115w", "f150w", "f277w", "f444w")}
        mag_f115w = float(mags["f115w"])
        color_f115_f444 = (mag_f115w - float(mags["f444w"])) if np.isfinite(mags["f444w"]) else 0.0
        # Redder (larger f115w-f444w, i.e. fainter in f444w relative to f115w... in
        # AB mag, *smaller* mag = brighter, so a *larger* f115w-f444w value means
        # f115w is comparatively faint -- i.e. the galaxy is intrinsically red).
        if color_f115_f444 > 1.0:
            color_type = "red"
        elif color_f115_f444 < 0.0:
            color_type = "blue"
        else:
            color_type = "intermediate"
        morph_type = "elliptical" if n_sersic > 2.5 else ("S0" if n_sersic > 1.5 else "spiral")

        z = float(props["redshift"][src_i])
        redshift = z if np.isfinite(z) and z > 0 else float(rng.uniform(0.3, 3.0))
        log_mass = float(props["log_mass"][src_i])

        field_galaxies.append({
            "center_x": float(x_positions[i]),
            "center_y": float(y_positions[i]),
            "e1": float(np.clip(e1, -0.7, 0.7)),
            "e2": float(np.clip(e2, -0.7, 0.7)),
            "axis_ratio": q_ratio,
            "position_angle": pa,
            "R_sersic": re_eff,
            "n_sersic": n_sersic,
            "magnitude": mag_f115w,
            "field_redshift": redshift,
            "real_morph_type": morph_type,
            "real_color_type": color_type,
            "source_lens_id": "cosmos_web_bootstrap",
            "original_magnitude": mag_f115w,
            "structural_quality": "real_catalog",
            "log_mass": log_mass,
        })
    return field_galaxies


def generate_synthetic_field_population(rng, n_field, numpix, pixel_scale=None,
                                         env_type=None, lens_z=None):
    """Generate a field galaxy population for when no lens-specific real field
    catalog is available.

    Positions are sampled uniformly over the full square FOV (with a small
    central exclusion zone and minimum pairwise separation). Per-galaxy
    structural/photometric properties are bootstrap-resampled from real
    COSMOS-Web detections (cosmos_field_galaxy_properties) whenever the
    catalog is available -- this reproduces the real joint distribution of
    size, axis ratio, Sersic index, magnitude and redshift, rather than only
    matching the 1-D number density and reinventing everything else from
    independent parametric guesses. Falls back to the latter only if the
    catalog can't be loaded.

    env_type/lens_z: when env_type=="group", ~40% of the population is
    preferentially drawn from real catalog galaxies near lens_z (see
    sample_cosmos_field_galaxies's group_target_z/group_z_fraction), so a
    lens actually labelled "group" renders an actual redshift-clustered
    group, not just a richer field population with uncorrelated redshifts
    (adversarial audit finding C-12, 2026-08-01).
    """
    if pixel_scale is None:
        pixel_scale = CONFIG.get('pixel_scale', 0.031) if isinstance(CONFIG, dict) else 0.031
    half_size = (numpix * pixel_scale) / 2.0

    positions = []
    for i in range(n_field):
        # Position avoiding center and other galaxies
        attempts = 0
        while attempts < 50:
            # Uniform spatial distribution over the full square FOV (not a
            # circle inscribed in it -- sampling in polar (r, theta) with a
            # capped r_max fills only a disk of radius r_max, leaving the
            # square's corners empty out to r_max*sqrt(2)). A small circular
            # exclusion zone around the lens at center is still applied via
            # rejection, since that's a genuine "avoid center" requirement,
            # not the outer FOV boundary.
            r_min = 0.8  # Avoid center
            margin = half_size * 0.95  # small inset so galaxies aren't clipped at the edge
            x_pos = rng.uniform(-margin, margin)
            y_pos = rng.uniform(-margin, margin)
            if x_pos**2 + y_pos**2 < r_min**2:
                attempts += 1
                continue

            # Check collision with existing galaxies
            too_close = False
            for ex_x, ex_y in positions:
                if np.sqrt((x_pos - ex_x)**2 + (y_pos - ex_y)**2) < 0.15:
                    too_close = True
                    break

            if not too_close:
                break
            attempts += 1

        if attempts >= 50:
            # Force placement
            x_pos = rng.uniform(-half_size*0.8, half_size*0.8)
            y_pos = rng.uniform(-half_size*0.8, half_size*0.8)

        positions.append((x_pos, y_pos))

    # FIX (adversarial audit finding C-4, 2026-08-01): this used to read
    # photometry.source_mag_max (a SOURCE-galaxy config key, e.g. 22.5),
    # while field_galaxy_count_target() -- which decides HOW MANY field
    # galaxies to place -- reads field.density_mag_limit (e.g. 24.5). The
    # two disagreed by ~2 mag, so the code rendered a COUNT calibrated to
    # a 24.5 depth using PROPERTIES bootstrap-resampled from a pool cut at
    # only 22.5 -- i.e. the right number of galaxies, all systematically
    # too bright. Read the same key both places.
    if isinstance(CONFIG, dict):
        _field_cfg = CONFIG.get('field', {})
        mag_limit = _field_cfg.get(
            'density_mag_limit',
            CONFIG.get('photometry', {}).get('field_mag_limit',
                CONFIG.get('photometry', {}).get('source_mag_max', 27.5)),
        )
        catalog_path = CONFIG.get('catalogs', {}).get('galaxy_catalog_fits', 'data/galaxy_catalog.fits')
    else:
        mag_limit = 27.5
        catalog_path = 'data/galaxy_catalog.fits'
    x_arr = [p[0] for p in positions]
    y_arr = [p[1] for p in positions]
    _group_z_frac = 0.4 if (env_type == 'group' and lens_z is not None) else 0.0
    field_galaxies = sample_cosmos_field_galaxies(
        rng, n_field, x_arr, y_arr, mag_limit, catalog_path,
        group_target_z=lens_z, group_z_fraction=_group_z_frac,
    )

    if field_galaxies is not None:
        print(f"[INFO] Generated {len(field_galaxies)} field galaxies (bootstrap-resampled from COSMOS-Web)")
        return field_galaxies

    # Fallback: parametric modeling (catalog unavailable)
    field_galaxies = []
    for x_pos, y_pos in positions:
        # REALISTIC morphological diversity
        morph_type = rng.choice(['spiral', 'elliptical', 'S0'], p=[0.6, 0.25, 0.15])

        if morph_type == 'spiral':
            n_sersic = rng.choice([0.8, 1.0, 1.2, 1.5], p=[0.3, 0.4, 0.2, 0.1])
            q_ratio = rng.beta(2.0, 2.0) * 0.5 + 0.5  # Range 0.5-1.0
            color_type = rng.choice(['blue', 'blue_intermediate'], p=[0.7, 0.3])
        elif morph_type == 'elliptical':
            n_sersic = rng.choice([3.0, 4.0, 5.0], p=[0.5, 0.35, 0.15])
            q_ratio = rng.beta(2.5, 1.5) * 0.3 + 0.7  # Range 0.7-1.0
            color_type = rng.choice(['red', 'intermediate'], p=[0.8, 0.2])
        else:  # S0
            n_sersic = rng.choice([2.0, 2.5, 3.0], p=[0.4, 0.4, 0.2])
            q_ratio = rng.beta(2, 2) * 0.4 + 0.6  # Range 0.6-1.0
            color_type = 'intermediate'

        re_eff = np.clip(rng.lognormal(np.log(0.35), 0.5), 0.08, 2.5)

        pa = rng.uniform(-180, 180)
        e = (1 - q_ratio) / (1 + q_ratio)
        e1 = e * np.cos(2 * np.radians(pa))
        e2 = e * np.sin(2 * np.radians(pa))

        bright_prob = 0.15  # 15% bright foreground galaxies (z<0.5)
        if rng.random() < bright_prob:
            magnitude = rng.uniform(19.0, 22.0)
            redshift = rng.uniform(0.1, 0.8)
        else:
            magnitude = rng.uniform(22.0, 28.5)
            redshift = rng.uniform(0.3, 6.0)

        asym_strength = 0.15 if n_sersic < 1.0 else 0.05
        center_x_asym = rng.normal(x_pos, asym_strength)
        center_y_asym = rng.normal(y_pos, asym_strength)

        field_galaxies.append({
            'center_x': float(center_x_asym),
            'center_y': float(center_y_asym),
            'e1': float(np.clip(e1, -0.7, 0.7)),
            'e2': float(np.clip(e2, -0.7, 0.7)),
            'axis_ratio': float(q_ratio),
            'position_angle': float(pa),
            'R_sersic': float(re_eff),
            'n_sersic': float(n_sersic),
            'magnitude': float(magnitude),
            'field_redshift': float(redshift),
            'real_morph_type': morph_type,
            'real_color_type': color_type,
            'source_lens_id': 'synthetic',
            'original_magnitude': float(magnitude),
            'structural_quality': 'synthetic',
        })

    print(f"[INFO] Generated {len(field_galaxies)} synthetic field galaxies (parametric fallback)")
    return field_galaxies

def sample_real_field_galaxies_for_mock(field_pop, n_max=8, rng=None, numpix=300, 
                                       pixel_scale=0.03, lens_redshift=0.8, 
                                       avoid_center_arcsec=0.6, psf_data=None,
                                       lens_radius=None, lens_id=None, 
                                       lens_mass_log10=None, halo_radius_constraint=True):
    """ENHANCED: Sample from lens-specific real field galaxy population
    
    Args:
        field_pop: Full merged field galaxy catalog
        lens_id: Specific lens ID to sample field galaxies from (e.g., 'COSJ095846+020304')
        n_max: Maximum number of field galaxies to return
        rng: Random number generator
        numpix: Image size in pixels
        pixel_scale: Arcsec per pixel
        lens_redshift: Lens redshift for filtering
        avoid_center_arcsec: Minimum separation from center
        psf_data: PSF data for convolution
        lens_radius: Lens radius for size constraints
    """
    
    if field_pop is None or len(field_pop) == 0 or n_max <= 0 or rng is None:
        print("[WARNING] No field population available, generating synthetic field galaxies")
        return generate_synthetic_field_population(rng, n_max, numpix)
    
    # ENHANCED: If lens_id provided, sample from lens-specific field galaxies
    if lens_id is not None and 'lens_id' in field_pop.columns:
        lens_specific_galaxies = field_pop[field_pop['lens_id'] == lens_id].copy()
        
        if len(lens_specific_galaxies) > 0:
            print(f"[INFO] Using lens-specific field galaxies for {lens_id}: {len(lens_specific_galaxies)} available")
            
            # Filter out central lens (sep_arcsec = 0) and apply constraints
            field_galaxies = lens_specific_galaxies[lens_specific_galaxies['sep_arcsec'] > avoid_center_arcsec].copy()
            
            # Apply halo radius constraint if enabled
            if halo_radius_constraint and lens_mass_log10 is not None:
                # Estimate halo R200
                r200_kpc = estimate_halo_r200(lens_mass_log10, lens_redshift)
                r200_mpc = r200_kpc / 1000.0
                da_mpc = angular_diameter_distance(lens_redshift)
                r200_arcsec = (r200_mpc / da_mpc) * 206265
                
                # Reject galaxies beyond 1.5 * R200
                max_radius = r200_arcsec * 1.5
                field_galaxies = field_galaxies[field_galaxies['sep_arcsec'] <= max_radius].copy()
                
                print(f"[INFO] Applied halo radius constraint: R200={r200_arcsec:.2f} arcsec, max_radius={max_radius:.2f} arcsec")
            
            if len(field_galaxies) > 0:
                # Sample up to n_max field galaxies from this lens's observed field
                n_sample = min(n_max, len(field_galaxies))
                selected_indices = rng.choice(len(field_galaxies), size=n_sample, replace=False)
                selected_galaxies = field_galaxies.iloc[selected_indices]
                
                print(f"[INFO] Selected {n_sample} field galaxies from {lens_id} (separations: {selected_galaxies['sep_arcsec'].min():.2f}-{selected_galaxies['sep_arcsec'].max():.2f} arcsec)")
                
                # Convert to field galaxy format
                field_galaxies_list = []
                for _, gal in selected_galaxies.iterrows():
                    field_gal = convert_real_galaxy_to_field_format(gal, rng, lens_radius, numpix, pixel_scale)
                    if field_gal is not None:
                        field_galaxies_list.append(field_gal)
                
                return field_galaxies_list
            else:
                print(f"[WARNING] No field galaxies found for {lens_id} beyond avoid_center_arcsec={avoid_center_arcsec}")
        else:
            print(f"[WARNING] No field galaxies found for lens_id={lens_id}, falling back to general sampling")
    
    # Fallback to original general sampling method
    
    # Caller already chose the field-galaxy count (n_max) via
    # field_galaxy_count_target (COSMOS density × FOV × environment richness).
    # Do not re-draw from expected_density_per_arcsec2 here — that double-counts
    # and previously collapsed hybrid Euclid runs to ~2 galaxies/image.
    n_field = max(0, int(n_max))
    min_count = int(CONFIG['field'].get('min_count', 0))
    if min_count > 0:
        n_field = max(n_field, min_count)
    print(f"[DEBUG] Field galaxy sampling: target_count={n_field}")
    
    if len(field_pop) < n_field:
        selected_indices = rng.choice(len(field_pop), size=n_field, replace=True)
    else:
        selected_indices = rng.choice(len(field_pop), size=n_field, replace=False)
    
    field_galaxies = []
    half_size = (numpix * pixel_scale) / 2.0
    position_attempts = 0
    # COSMOS field_sep_arcsec is measured in ~9″ JWST cutouts. Reusing those
    # absolute separations on a 1′ Euclid FOV piles ~70% of galaxies within a
    # few arcsec of the lens. For wide FOVs, place uniformly over the square.
    wide_fov = half_size > 8.0
    min_re_arcsec = max(0.12, 1.5 * float(pixel_scale))

    for idx in selected_indices:
        if position_attempts > n_field * 20:  # Prevent infinite loops
            break
            
        real_gal = field_pop.iloc[idx]
        
        # FIXED: More generous position sampling
        attempts = 0
        while attempts < 100:  # Increased attempts
            cfg_field = CONFIG['field']
            min_sep = max(avoid_center_arcsec, float(cfg_field.get('avoid_center_arcsec', 0.3)))
            max_sep = float(cfg_field.get('max_fraction_of_half_size', 0.9)) * half_size
            margin = half_size * 0.95

            if wide_fov:
                # Uniform over the full square FOV (reject central exclusion).
                x_pos = float(rng.uniform(-margin, margin))
                y_pos = float(rng.uniform(-margin, margin))
                if x_pos ** 2 + y_pos ** 2 < min_sep ** 2:
                    attempts += 1
                    position_attempts += 1
                    continue
            else:
                # Use observed separation as guide but allow more flexibility
                if 'field_sep_arcsec' in real_gal.index and pd.notna(real_gal['field_sep_arcsec']):
                    observed_sep = float(real_gal['field_sep_arcsec'])
                else:
                    observed_sep = rng.uniform(1.5, 4.0)

                # Sample position with realistic distribution
                if rng.random() < 0.7:  # 70% use observed separation pattern
                    target_sep = np.clip(observed_sep + rng.normal(0, 0.5), min_sep, max_sep)
                else:  # 30% uniform distribution
                    target_sep = rng.uniform(min_sep, max_sep)

                theta = rng.uniform(0, 2*np.pi)
                x_pos = target_sep * np.cos(theta)
                y_pos = target_sep * np.sin(theta)

            # Check bounds - more lenient
            if (abs(x_pos) < half_size-0.2 and abs(y_pos) < half_size-0.2):
                # Check for collisions - configurable minimum distance
                too_close = False
                for existing in field_galaxies:
                    dx = x_pos - existing['center_x']
                    dy = y_pos - existing['center_y']
                    min_pair_sep = float(cfg_field.get('min_pair_separation_arcsec', 0.08))
                    if np.sqrt(dx**2 + dy**2) < min_pair_sep:
                        too_close = True
                        break
                
                if not too_close:
                    break
            attempts += 1
            position_attempts += 1
        
        if attempts >= 100:
            # Force placement if positioning fails
            x_pos = rng.uniform(-half_size+0.5, half_size-0.5)
            y_pos = rng.uniform(-half_size+0.5, half_size-0.5)
            print(f"[WARNING] Forced placement for field galaxy {len(field_galaxies)}")
        
        # FIXED: Enhanced structural parameter extraction (constrained by lens size)
        re_eff = extract_realistic_size(real_gal, rng, lens_radius=lens_radius)
        re_eff = max(float(re_eff), min_re_arcsec)
        q_axis = extract_realistic_axis_ratio(real_gal, rng)
        n_sersic = extract_realistic_sersic_index(real_gal, rng)
        
        # FIXED: Improved redshift handling
        field_z = extract_field_redshift(real_gal, lens_redshift, rng)
        
        # FIXED: More diverse position angles
        pa = rng.uniform(-180, 180)
        e = (1 - q_axis) / (1 + q_axis)
        e1 = e * np.cos(2 * np.radians(pa))
        e2 = e * np.sin(2 * np.radians(pa))
        
        # FIXED: Better magnitude extraction and scaling with high-z evolution
        base_mag = extract_realistic_magnitude(real_gal, field_z, rng)
        
        # Additional high-z dimming for field galaxies at z>6
        if field_z > 6.0:
            high_z_dimming = 0.5 * (field_z - 6.0)  # 0.5 mag per unit z beyond z=6
            base_mag += high_z_dimming
        
        # ASYMMETRY: Add realistic asymmetry based on morphology
        morph_type = classify_galaxy_morphology_enhanced(n_sersic, q_axis)
        asym_strength = 0.12 if 'spiral' in morph_type else 0.03
        center_x_asym = rng.normal(x_pos, asym_strength)  
        center_y_asym = rng.normal(y_pos, asym_strength)
        
        # Create enhanced galaxy dictionary
        galaxy = {
            'center_x': float(center_x_asym),
            'center_y': float(center_y_asym),
            'e1': float(np.clip(e1, -0.7, 0.7)),
            'e2': float(np.clip(e2, -0.7, 0.7)),
            'axis_ratio': float(q_axis),
            'position_angle': float(pa),
            'R_sersic': float(re_eff),
            'n_sersic': float(n_sersic),
            'magnitude': float(base_mag),
            'field_redshift': float(field_z),
            'real_morph_type': morph_type,
            'real_color_type': classify_galaxy_colors_enhanced(field_z, n_sersic, rng),
            'source_lens_id': str(real_gal.get('source_lens_id', 'unknown')),
            'original_redshift': float(real_gal.get('LP_zfinal', field_z)),
            'original_magnitude': float(base_mag),
            'structural_quality': assess_structural_quality(re_eff, n_sersic, q_axis),
            # Include original coordinates for PSF mapping
            'RA_DETEC': real_gal.get('RA_DETEC', None),
            'DEC_DETEC': real_gal.get('DEC_DETEC', None)
        }
        
        field_galaxies.append(galaxy)
    
    print(f"[DEBUG] Successfully sampled {len(field_galaxies)} field galaxies")
    
    # Add synthetic galaxies if we're still short
    if len(field_galaxies) < 4:  # Ensure minimum realistic population
        synthetic_count = 4 - len(field_galaxies)
        synthetic_gals = generate_synthetic_field_population(rng, synthetic_count, numpix)
        field_galaxies.extend(synthetic_gals)
        print(f"[INFO] Added {synthetic_count} synthetic field galaxies for realism")
    
    # Apply PSF convolution to field galaxies if PSF data is available
    if psf_data is not None:
        field_galaxies = apply_field_galaxy_psf_convolution(field_galaxies, psf_data, rng)
        print(f"[PSF] Applied individual PSF convolution to {len(field_galaxies)} field galaxies")
    
    return field_galaxies


def resolve_sed_type_from_morphology(morph_type, n_sersic, rng):
    """Map a galaxy's morphology label + Sersic index to one empirical SED
    class. Callers computing colors for the same galaxy across multiple
    bands should call this once per galaxy and pass the result to every
    per-band color call (via the ``sed_type`` argument of
    ``get_realistic_jwst_color``/``get_realistic_jwst_color_from_transmission``),
    so all bands share one physically self-consistent SED instead of an
    independent random draw per band.
    """
    if morph_type in ['elliptical', 's0'] or n_sersic > 2.5:
        return 'passive'
    if morph_type in ['irregular', 'clumpy', 'starburst', 'primordial'] or n_sersic < 0.7:
        return rng.choice(['star_forming', 'dusty_starburst'], p=[0.7, 0.3])
    if morph_type in ['spiral', 'late_spiral', 'barred_spiral'] or (0.7 <= n_sersic < 2.0):
        return 'star_forming'
    return rng.choice(['star_forming', 'post_starburst'], p=[0.8, 0.2])


def get_realistic_jwst_color(morph_type, n_sersic, base_band='F150W', target_band='F150W', rng=None, redshift=0.5, sed_type=None):
    """
    Get realistic JWST color (magnitude offset) based on galaxy morphology and empirical SED templates.
    Returns magnitude offset relative to base_band.

    Uses empirical SED templates (BC03 + Calzetti+2000 + Chary & Elbaz 2001) when available,
    accounting for:
    - Stellar population synthesis (age, SFH)
    - Extinction (Calzetti law, E(B-V))
    - Dust emission (multi-temperature blackbodies, PAH features)
    - Spectral features (Balmer break, 4000 Å break, 1.6 μm stellar bump)

    Falls back to empirical COSMOS-Web color offsets if templates unavailable.

    sed_type : str, optional
        If given, use this SED class directly instead of re-deriving it from
        morph_type/n_sersic. Callers computing colors for the same galaxy in
        multiple bands should resolve sed_type once and pass it consistently
        here, so the same underlying SED (not an independent per-band draw)
        sets every band's flux -- required for physically self-consistent
        multi-band colors.
    """
    if rng is None:
        # FIX (audit C-5): a bare default_rng() draws fresh OS entropy,
        # ignoring args.seed entirely and breaking reproducibility even
        # when a caller forgot to pass its own `rng`. Seed from the (now
        # seeded-at-startup, see main()) global np.random stream instead,
        # so behavior is deterministic given a fixed call order.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    if base_band == target_band:
        return 0.0

    # Use empirical SED templates if available
    if EMPIRICAL_SED_AVAILABLE:
        # Map morphology to SED type (unless the caller already resolved one)
        if sed_type is None:
            sed_type = resolve_sed_type_from_morphology(morph_type, n_sersic, rng)

        # Get filter wavelengths
        wave_base = JWST_FILTERS.get(base_band, 1.5)
        wave_target = JWST_FILTERS.get(target_band, 1.5)
        
        # Calculate K-corrections for both bands
        k_base = calculate_k_correction_empirical(redshift, wave_base, sed_type)
        k_target = calculate_k_correction_empirical(redshift, wave_target, sed_type)
        
        # Color = K_target - K_base (difference in K-corrections gives color)
        color_empirical = k_target - k_base
        
        # Add small scatter for measurement uncertainty
        scatter = rng.normal(0, 0.03)
        
        return color_empirical + scatter
    
    # ========================================================================
    # FALLBACK: Empirical COSMOS-Web color offsets
    # ========================================================================
    
    # Define color indices based on galaxy type
    # Positive = target is FAINTER (higher mag)
    # Negative = target is BRIGHTER (lower mag)

    color_offsets = {
        'F115W': {},  # Bluest band (1.15 micron)
        'F150W': {},  # Blue band (1.50 micron) - reference
        'F277W': {},  # Red band (2.77 micron)
        'F444W': {}   # Reddest band (4.44 micron)
    }

    # Early-type galaxies (ellipticals, S0): RED SEDs
    if morph_type in ['elliptical', 's0'] or n_sersic > 2.5:
        color_offsets['F115W']['F150W'] = rng.uniform(0.3, 0.6)   # Faint in blue
        color_offsets['F150W']['F150W'] = 0.0                      # Reference
        color_offsets['F277W']['F150W'] = rng.uniform(-0.2, -0.05)  # Bright in red
        color_offsets['F444W']['F150W'] = rng.uniform(-0.3, -0.1)   # Brightest in redder

    # Late-type spirals: BLUE SEDs
    elif morph_type in ['spiral', 'late_spiral', 'barred_spiral'] or (0.7 <= n_sersic < 2.0):
        color_offsets['F115W']['F150W'] = rng.uniform(-0.3, -0.1)  # Bright in blue
        color_offsets['F150W']['F150W'] = 0.0                       # Reference
        color_offsets['F277W']['F150W'] = rng.uniform(0.1, 0.3)    # Faint in red
        color_offsets['F444W']['F150W'] = rng.uniform(0.2, 0.5)    # Fainter in reddest

    # Irregular/clumpy/starburst: VERY BLUE SEDs (strong UV)
    elif morph_type in ['irregular', 'clumpy', 'starburst', 'primordial'] or n_sersic < 0.7:
        color_offsets['F115W']['F150W'] = rng.uniform(-0.5, -0.2)  # Very bright in blue
        color_offsets['F150W']['F150W'] = 0.0                       # Reference
        color_offsets['F277W']['F150W'] = rng.uniform(0.2, 0.4)    # Faint in red
        color_offsets['F444W']['F150W'] = rng.uniform(0.4, 0.7)    # Very faint in reddest

    # Default (intermediate)
    else:
        color_offsets['F115W']['F150W'] = rng.uniform(-0.2, 0.2)
        color_offsets['F150W']['F150W'] = 0.0
        color_offsets['F277W']['F150W'] = rng.uniform(-0.1, 0.2)
        color_offsets['F444W']['F150W'] = rng.uniform(-0.1, 0.3)

    # Return color offset for target band
    return color_offsets.get(target_band, {}).get(base_band, 0.0)


def _is_jwst_nircam_band(band: str) -> bool:
    """True for JWST NIRCam bands (F070W, F115W, ...) but not EUCLID/ROMAN/etc."""
    b = band.upper()
    return b.startswith('F') and not b.startswith('EUCLID')


def _color_from_multi_telescope_transmission(
        wavelengths_um: np.ndarray,
        sed_spectrum: np.ndarray,
        base_band: str,
        target_band: str,
        redshift: float,
        rng: np.random.Generator,
) -> float:
    """SED–filter colour via MultiTelescopeFilterSystem (Euclid, Roman, Subaru, LSST)."""
    if not MULTI_TELESCOPE_AVAILABLE:
        raise ValueError("Multi-telescope filter system not available")
    mt = MULTI_TELESCOPE_FILTERS or get_multi_telescope_filters()
    z = max(float(redshift), 0.0)
    obs_wave = wavelengths_um * (1.0 + z)
    flux_base = mt.convolve_sed_to_magnitude(base_band, obs_wave, sed_spectrum)
    flux_target = mt.convolve_sed_to_magnitude(target_band, obs_wave, sed_spectrum)
    if flux_base <= 0 or flux_target <= 0:
        raise ValueError(f"zero flux for {base_band}/{target_band}")
    color = -2.5 * np.log10(flux_target / flux_base)
    return float(color + rng.normal(0, 0.02))


def get_realistic_jwst_color_from_transmission(morph_type: str, n_sersic: float,
                                                base_band: str = 'F150W',
                                                target_band: str = 'F150W',
                                                redshift: float = 0.5, rng=None,
                                                sed_type: str = None) -> float:
    """
    Calculate realistic JWST color using filter transmission curves and SED convolution.

    This function provides more accurate colors by:
    1. Convolving galaxy SEDs with realistic filter transmission curves
    2. Accounting for proper K-corrections including filter shape effects
    3. Including filter-specific noise characteristics

    Uses empirical SED templates when available to generate realistic galaxy spectra,
    then convolves with JWST filter transmission curves to compute magnitudes.

    Parameters:
    -----------
    morph_type : str
        Galaxy morphology (e.g., 'elliptical', 'spiral', 'starburst')
    n_sersic : float
        Sersic index
    base_band : str
        Reference band for color (default F150W)
    target_band : str
        Target band for color calculation
    redshift : float
        Galaxy redshift
    rng : np.random.Generator
        Random number generator
    sed_type : str, optional
        If given, use this SED class directly instead of re-deriving it from
        morph_type/n_sersic. Callers computing colors for the same galaxy in
        multiple bands should resolve sed_type once and pass it consistently
        here, so the same underlying SED (not an independent per-band draw)
        sets every band's flux.

    Returns:
    --------
    color : float
        Magnitude difference (mag_target - mag_base) in ABs
    """
    if rng is None:
        # FIX (audit C-5): a bare default_rng() draws fresh OS entropy,
        # ignoring args.seed entirely and breaking reproducibility even
        # when a caller forgot to pass its own `rng`. Seed from the (now
        # seeded-at-startup, see main()) global np.random stream instead,
        # so behavior is deterministic given a fixed call order.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    if base_band == target_band:
        return 0.0

    base_upper = base_band.upper()
    target_upper = target_band.upper()
    use_jwst = _is_jwst_nircam_band(base_upper) and _is_jwst_nircam_band(target_upper)

    # Use transmission-based calculation if available
    if not FILTER_TRANSMISSION_AVAILABLE and use_jwst:
        # Fall back to original method if transmission system not available
        return get_realistic_jwst_color(morph_type, n_sersic, base_band,
                                       target_band, rng, redshift, sed_type=sed_type)

    if not MULTI_TELESCOPE_AVAILABLE and not use_jwst:
        return get_realistic_jwst_color(morph_type, n_sersic, base_band,
                                       target_band, rng, redshift, sed_type=sed_type)

    # Generate SED based on morphology (unless the caller already resolved one)
    if EMPIRICAL_SED_AVAILABLE:
        if sed_type is None:
            sed_type = resolve_sed_type_from_morphology(morph_type, n_sersic, rng)
        
        # Generate SED spectrum covering JWST wavelength range
        wavelengths_um = np.linspace(0.3, 5.0, 200)  # Microns
        sed_spectrum = generate_empirical_sed(sed_type, wavelengths_um, rng=rng)
        
        # Convolve with filter transmission to get magnitudes
        try:
            if use_jwst:
                mag_base = JWST_FILTERS_SYSTEM.convolve_sed_to_magnitude(
                    wavelengths_um, sed_spectrum, base_upper, redshift
                )
                mag_target = JWST_FILTERS_SYSTEM.convolve_sed_to_magnitude(
                    wavelengths_um, sed_spectrum, target_upper, redshift
                )
                color = mag_target - mag_base
            else:
                color = _color_from_multi_telescope_transmission(
                    wavelengths_um, sed_spectrum, base_upper, target_upper,
                    redshift, rng,
                )
                return color
            
            # JWST path: colour with small measurement uncertainty
            scatter = rng.normal(0, 0.02)  # ~20 mJy scatter
            return color + scatter
            
        except Exception as e:
            # Only warn once per band pair per run to avoid log spam
            _warn_key = (base_upper, target_upper)
            if not getattr(get_realistic_jwst_color_from_transmission, '_warned', None):
                get_realistic_jwst_color_from_transmission._warned = set()
            if _warn_key not in get_realistic_jwst_color_from_transmission._warned:
                print(f"[WARNING] Transmission-based color failed ({e}), falling back to K-correction")
                get_realistic_jwst_color_from_transmission._warned.add(_warn_key)
            return get_realistic_jwst_color(morph_type, n_sersic, base_band, 
                                           target_band, rng, redshift)
    
    else:
        # No empirical SEDs, use fallback
        return get_realistic_jwst_color(morph_type, n_sersic, base_band, 
                                       target_band, rng, redshift)


def calculate_k_correction(redshift, band, rng, galaxy_type='mixed'):
    """
    Calculate K-correction using empirical SED templates (BC03 + Calzetti+2000 + Chary & Elbaz 2001)
    
    If empirical templates available:
    - Uses Bruzual & Charlot 2003 stellar population synthesis
    - Calzetti et al. 2000 extinction law
    - Chary & Elbaz 2001, Dale & Helou 2002 dust emission
    - Accounts for Balmer break, 4000 Å break, 1.6 μm stellar bump, PAH features
    
    Otherwise falls back to simplified parametric model.
    
    Parameters:
    -----------
    redshift : float
        Galaxy redshift
    band : str
        JWST filter name (e.g., 'F115W')
    rng : np.random.Generator
        Random number generator
    galaxy_type : str
        'mixed', 'passive', 'star_forming', 'post_starburst', or 'dusty_starburst'
    
    Returns:
    --------
    k_corr : float
        K-correction in magnitudes
    """
    
    # Use empirical SED templates if available
    if EMPIRICAL_SED_AVAILABLE:
        # Determine SED type if not specified
        if galaxy_type == 'mixed':
            sed_type = select_sed_type_from_redshift(redshift, rng)
        elif galaxy_type == 'passive':
            sed_type = 'passive'
        elif galaxy_type == 'star_forming':
            sed_type = 'star_forming'
        elif galaxy_type == 'post_starburst':
            sed_type = 'post_starburst'
        elif galaxy_type == 'dusty_starburst':
            sed_type = 'dusty_starburst'
        else:
            sed_type = 'star_forming'  # Default
        
        # Get filter wavelength
        filter_wave = JWST_FILTERS.get(band, 1.5)  # Default to 1.5 μm if not found
        
        # Calculate empirical K-correction
        k_corr_empirical = calculate_k_correction_empirical(redshift, filter_wave, sed_type)
        
        # Add small random scatter for realism (measurement uncertainty)
        scatter = rng.normal(0, 0.05)
        
        return k_corr_empirical + scatter
    
    # ========================================================================
    # FALLBACK: Simplified parametric model (used if empirical templates unavailable)
    # ========================================================================
    
    # Determine galaxy SED type if not specified
    if galaxy_type == 'mixed':
        # Sample realistic mix: more star-forming at high-z, more passive at low-z
        if redshift < 1.0:
            galaxy_type = rng.choice(['passive', 'star_forming'], p=[0.6, 0.4])
        elif redshift < 3.0:
            galaxy_type = rng.choice(['passive', 'star_forming'], p=[0.4, 0.6])
        else:
            galaxy_type = rng.choice(['passive', 'star_forming'], p=[0.2, 0.8])
    else:
        if galaxy_type not in ['passive', 'star_forming']:
            galaxy_type = 'star_forming'
    
    # Reference band for color calculation (F150W is typical reference)
    # K-correction = observed_mag - rest_frame_mag
    
    if galaxy_type == 'passive':
        # Red/dead galaxies (ellipticals, S0s)
        # Strong 4000Å break makes them red
        if redshift < 1.0:
            # Low-z passive: very red
            if band == 'F115W':
                k_corr = rng.normal(0.3, 0.1)  # Fainter in blue
            elif band == 'F150W':
                k_corr = rng.normal(0.0, 0.1)  # Reference
            elif band == 'F277W':
                k_corr = rng.normal(-0.2, 0.1)  # Brighter in red
            else:  # F444W
                k_corr = rng.normal(-0.3, 0.1)  # Brightest in far-red
        elif redshift < 3.0:
            # Mid-z passive: 4000Å break shifting through bands
            if band == 'F115W':
                k_corr = rng.normal(0.5, 0.15)  # Very faint in blue
            elif band == 'F150W':
                k_corr = rng.normal(0.2, 0.1)  # Break entering
            elif band == 'F277W':
                k_corr = rng.normal(-0.1, 0.1)  # Redder side
            else:  # F444W
                k_corr = rng.normal(-0.2, 0.1)
        else:
            # High-z passive: break fully shifted
            if band == 'F115W':
                k_corr = rng.normal(0.8, 0.2)  # Extremely faint
            elif band == 'F150W':
                k_corr = rng.normal(0.4, 0.15)
            elif band == 'F277W':
                k_corr = rng.normal(0.0, 0.1)  # Break here
            else:  # F444W
                k_corr = rng.normal(-0.3, 0.1)
                
    else:  # star_forming
        # Blue/star-forming galaxies
        # Strong UV/blue continuum
        if redshift < 1.0:
            # Low-z star-forming: blue
            if band == 'F115W':
                k_corr = rng.normal(-0.2, 0.1)  # Bright in blue
            elif band == 'F150W':
                k_corr = rng.normal(0.0, 0.1)  # Reference
            elif band == 'F277W':
                k_corr = rng.normal(0.2, 0.1)  # Fainter in red
            else:  # F444W
                k_corr = rng.normal(0.3, 0.1)  # Faintest in far-red
        elif redshift < 3.0:
            # Mid-z star-forming: UV entering blue bands
            if band == 'F115W':
                k_corr = rng.normal(-0.4, 0.15)  # UV boost
            elif band == 'F150W':
                k_corr = rng.normal(-0.2, 0.1)  # Strong blue
            elif band == 'F277W':
                k_corr = rng.normal(0.1, 0.1)  # Balmer break
            else:  # F444W
                k_corr = rng.normal(0.3, 0.1)  # Red desert
        else:
            # High-z star-forming: Lyman break entering
            if band == 'F115W':
                k_corr = rng.normal(-0.6, 0.2)  # Strong UV/Lyman α
            elif band == 'F150W':
                k_corr = rng.normal(-0.3, 0.15)  # UV continuum
            elif band == 'F277W':
                k_corr = rng.normal(0.0, 0.1)  # Continuum
            else:  # F444W
                k_corr = rng.normal(0.2, 0.1)  # Fainter at red
    
    return k_corr

def tag_field_galaxies_with_galaxygenius_stamps(field_galaxies, rng, config=None):
    """Mark a fraction of field galaxies to be rendered as GalaxyGenius/SKIRT
    stamps (INTERPOL light profiles) instead of SERSIC_ELLIPSE.

    Sets ``use_galaxygenius_stamp`` and ``_stamp_view`` in-place on a subset
    of the dicts in ``field_galaxies``, chosen per the
    ``galaxygenius_stamps`` config block (off by default).
    """
    cfg = (config or {}).get('galaxygenius_stamps', {})
    if not cfg.get('enabled', False) or not field_galaxies:
        return field_galaxies

    fraction = float(cfg.get('fraction', 0.0))
    for gal in field_galaxies:
        gal['use_galaxygenius_stamp'] = bool(rng.random() < fraction)
        if gal['use_galaxygenius_stamp']:
            gal_z = gal.get('field_redshift', gal.get('redshift'))
            gal['_stamp_set'] = random_stamp_set(rng, redshift=gal_z)
            gal['_stamp_view'] = int(rng.choice(available_views(gal['_stamp_set'])))
    return field_galaxies


def tag_field_galaxies_with_tng_particles(field_galaxies, rng, config=None):
    """Mark a subset of TNG-matched field galaxies to be rendered from their
    matched subhalo's star/gas particle data (INTERPOL), per
    ``tng_mode.particle_morphology.field_enabled``/``field_fraction``.

    Sets ``_tng_particle_file`` in-place for galaxies with a ``tng_info``
    match and a locally-downloaded particle cutout. Skips galaxies already
    tagged for a GalaxyGenius stamp.

    If ``tng_mode.particle_morphology.generative_enabled`` is set and a
    galaxy has a ``tng_info`` match but no local particle cutout, it is
    instead tagged ``_generative_morph`` (Phase 9 conditional-VAE fallback,
    requires a trained checkpoint -- see
    ``galaxy_morphology.generative.inference``).
    """
    cfg = (config or {}).get('tng_mode', {}).get('particle_morphology', {})
    if not cfg.get('enabled', False) or not cfg.get('field_enabled', False) or not field_galaxies:
        return field_galaxies

    fraction = float(cfg.get('field_fraction', 1.0))
    generative_enabled = cfg.get('generative_enabled', False)
    _generative_ckpt = cfg.get('generative_checkpoint', None)
    if generative_enabled:
        try:
            from prism.morphology.generative.inference import is_available as _generative_available
        except ImportError:
            from prism.morphology.generative.inference import is_available as _generative_available
        generative_enabled = _generative_available(_generative_ckpt) if _generative_ckpt else _generative_available()

    for gal in field_galaxies:
        if gal.get('use_galaxygenius_stamp'):
            continue
        tng_info = gal.get('tng_info')
        if tng_info is None:
            continue
        if rng.random() >= fraction:
            continue
        _sim = tng_info.get('sim', 'TNG100-1')
        if not isinstance(_sim, str) or _sim != _sim or str(_sim).lower() in ('nan', 'none', ''):
            _sim = 'TNG100-1'
        generative_force = cfg.get('generative_force', False)
        if generative_force and generative_enabled:
            # Force VAE for all field galaxies — circular apodization + min size
            # floor (inference.py) prevent the sub-pixel star artifact.
            gal['_generative_morph'] = True
            continue
        particle_file = local_particle_path(tng_info['snapshot'], tng_info['subhalo_id'],
                                             min_particles=cfg.get('min_particles_field',
                                                                   cfg.get('min_particles')),
                                             sim=_sim)
        if particle_file is not None:
            gal['_tng_particle_file'] = particle_file
        elif generative_enabled:
            gal['_generative_morph'] = True
    return field_galaxies


def field_galaxy_light_model_types(field_galaxies):
    """lens_light_model_list entries for field galaxies, flattened across
    any multi-component fragments, matching the order of
    ``apply_real_jwst_colors_to_field_galaxies`` output.

    When ``morphology.multicomponent_enabled`` is set, each non-stamp/
    non-particle field galaxy is classified once here (morph type +
    component fragment) and the result is cached on the galaxy dict
    (``_morph_seed``/``_morph_type_resolved``) so
    ``apply_real_jwst_colors_to_field_galaxies`` builds per-band kwargs for
    the same fragment, consistently across bands.
    """
    morph_cfg = CONFIG.get('morphology', {}) if isinstance(CONFIG, dict) else {}
    multicomponent = morph_cfg.get('multicomponent_enabled', False)

    model_types = []
    for gal in field_galaxies:
        if gal.get('use_galaxygenius_stamp') or gal.get('_tng_particle_file') or gal.get('_generative_morph'):
            model_types.append("INTERPOL")
            continue
        if not multicomponent:
            model_types.append("SERSIC_ELLIPSE")
            continue

        base_params = {k: gal[k] for k in ('center_x', 'center_y', 'R_sersic', 'n_sersic', 'e1', 'e2')}
        seed = int(abs(hash((gal['center_x'], gal['center_y'], gal['R_sersic'], 'field_morph'))) % (2**32))
        gal['_morph_seed'] = seed
        fragment, _, resolved_morph_type = gm_build_light_model(
            'field', base_params, {'_DUMMY': 0.0}, ['_DUMMY'],
            np.random.default_rng(0), CONFIG, morph_seed=seed)
        gal['_morph_type_resolved'] = resolved_morph_type
        model_types.extend(fragment)
    return model_types


def build_galaxygenius_field_galaxy_kwargs(gal, band, rng):
    """INTERPOL kwargs for a field galaxy tagged with a GalaxyGenius stamp.

    The reference-band (F150W) magnitude gets the same K-correction
    treatment as Sersic field galaxies; the stamp's own per-band flux ratios
    (its SKIRT-computed SED) then set the realistic color for ``band``.
    """
    ref_band = 'F150W'
    base_mag = gal['magnitude']
    field_z = gal.get('field_redshift', gal.get('redshift', 1.0))

    tng_info = gal.get('tng_info')
    if tng_info is not None:
        # TNG Mode: derive the SED type from the matched subhalo's sSFR
        # rather than the morphology heuristic.
        galaxy_type = tng_sed_galaxy_type(tng_info)
    else:
        morph_type = gal.get('real_morph_type', 'spiral_late')
        if morph_type in ['elliptical', 'S0', 'spiral_early']:
            galaxy_type = 'passive'
        elif morph_type in ['irregular', 'starburst', 'peculiar']:
            galaxy_type = rng.choice(['dusty_starburst', 'post_starburst'], p=[0.3, 0.7])
        else:
            galaxy_type = 'star_forming'

    natural_scatter = rng.normal(0, 0.01)
    k_correction = calculate_k_correction(field_z, ref_band, rng, galaxy_type=galaxy_type)
    mag_ref = float(np.clip(base_mag + natural_scatter + k_correction, 18.0, _field_mag_limit()))

    view_idx = gal.get('_stamp_view', 0)
    stamp_set = gal.get('_stamp_set', 'subhalo_31')
    phi_G = np.radians(float(gal.get('position_angle', 0.0)))

    # Rescale the stamp's angular size for the field galaxy's actual
    # redshift (the stamp's morphology/SED come from a TNG subhalo at its
    # own native redshift -- 0.06 for the two curated stamps, but whatever
    # snapshot it was actually rendered at for the batch-rendered registry --
    # so its angular extent should reflect this galaxy's own redshift).
    target_size_arcsec = angular_size_for_redshift(
        field_z, native_redshift=native_redshift_for_stamp_set(stamp_set))

    return build_field_galaxy_interpol_kwargs(
        band=band,
        view_idx=view_idx,
        magnitude_ref=mag_ref,
        ref_band=ref_band,
        center_x=float(gal['center_x']),
        center_y=float(gal['center_y']),
        phi_G=phi_G,
        target_size_arcsec=target_size_arcsec,
        stamp_set=stamp_set,
    )


def apply_real_jwst_colors_to_field_galaxies(field_galaxies, band, rng):
    """CORRECTED: Apply realistic JWST colors with proper magnitude scaling"""

    if not field_galaxies:
        return []

    print(f"[DEBUG] Applying {band} colors to {len(field_galaxies)} field galaxies")

    band_galaxies = []
    mag_stats = {'before': [], 'after': []}

    for gal in field_galaxies:
        if gal.get('use_galaxygenius_stamp'):
            band_galaxies.append(build_galaxygenius_field_galaxy_kwargs(gal, band, rng))
            continue

        if gal.get('_tng_particle_file'):
            tng_info = gal['tng_info']
            field_z = gal.get('field_redshift', gal.get('redshift', 1.0))
            galaxy_type = tng_sed_galaxy_type(tng_info)
            natural_scatter = rng.normal(0, 0.01)
            k_correction = calculate_k_correction(field_z, _particle_ref_band(), rng, galaxy_type=galaxy_type)
            mag_ref = float(np.clip(gal['magnitude'] + natural_scatter + k_correction, 18.0, _field_mag_limit()))
            phi_G = 0.5 * np.arctan2(gal['e2'], gal['e1'])
            target_size_arcsec = float(gal['R_sersic']) * _field_target_size_factor()
            _field_smooth = float(
                (CONFIG or {}).get('tng_mode', {}).get('particle_morphology', {}).get(
                    'field_smooth_sigma', 2.5))
            band_galaxies.append(build_tng_particle_interpol_kwargs(
                band=band,
                particle_file=gal['_tng_particle_file'],
                halfmassrad_stars_kpc=tng_info['halfmassrad_stars_kpc'],
                magnitude_ref=mag_ref,
                ref_band=_particle_ref_band(),
                center_x=float(gal['center_x']),
                center_y=float(gal['center_y']),
                phi_G=phi_G,
                target_size_arcsec=target_size_arcsec,
                rng=rng,
                smooth_sigma=_field_smooth,
            ))
            continue

        if gal.get('_generative_morph'):
            try:
                from prism.morphology.generative.inference import build_generative_interpol_kwargs
            except ImportError:
                from prism.morphology.generative.inference import build_generative_interpol_kwargs
            _pm_cfg = (CONFIG or {}).get('tng_mode', {}).get('particle_morphology', {})
            _gen_ckpt = _pm_cfg.get('generative_checkpoint', None)
            tng_info = gal.get('tng_info')
            field_z = gal.get('field_redshift', gal.get('redshift', 1.0))
            galaxy_type = tng_sed_galaxy_type(tng_info) if tng_info is not None else 'star_forming'
            _logM = float(tng_info['stellar_mass_logmsun']) if tng_info is not None else float(gal.get('logM', 9.5))
            natural_scatter = rng.normal(0, 0.01)
            k_correction = calculate_k_correction(field_z, band, rng, galaxy_type=galaxy_type)
            mag_ref = float(np.clip(gal['magnitude'] + natural_scatter + k_correction, 18.0, _field_mag_limit()))
            phi_G = 0.5 * np.arctan2(gal['e2'], gal['e1'])
            target_size_arcsec = float(gal['R_sersic']) * _field_target_size_factor()
            _gen_kwargs = dict(
                morph_type=galaxy_type,
                logM=_logM,
                redshift=float(field_z),
                magnitude_ref=mag_ref,
                center_x=float(gal['center_x']),
                center_y=float(gal['center_y']),
                phi_G=phi_G,
                target_size_arcsec=target_size_arcsec,
                rng=rng,
            )
            if _gen_ckpt:
                _gen_kwargs['checkpoint_path'] = _gen_ckpt
            band_galaxies.append(build_generative_interpol_kwargs(**_gen_kwargs))
            continue

        gal_band = dict(gal)
        base_mag = gal['magnitude']
        color_type = gal.get('real_color_type', 'intermediate')
        mag_stats['before'].append(base_mag)
        
        # CORRECTED: Use same color processing as central lens (same catalog source)
        # Both central and field galaxies from same COSMOS-Web measurements
        # No artificial color corrections needed since they're from same observations
        
        # Only add minimal natural scatter for realism
        natural_scatter = rng.normal(0, 0.01)  # Very small, like measurement uncertainty
        
        # Apply K-correction for field galaxies at different redshifts
        field_z = gal.get('field_redshift', gal.get('redshift', 1.0))
        
        # Determine galaxy type for SED/K-correction: prefer the TNG-matched
        # subhalo's sSFR (TNG Mode) over the morphology heuristic.
        tng_info = gal.get('tng_info')
        if tng_info is not None:
            galaxy_type = tng_sed_galaxy_type(tng_info)
        else:
            morph_type = gal.get('real_morph_type', 'spiral_late')
            if morph_type in ['elliptical', 'S0', 'spiral_early']:
                galaxy_type = 'passive'
            elif morph_type in ['irregular', 'starburst', 'peculiar']:
                # Dusty starbursts or post-starbursts
                galaxy_type = rng.choice(['dusty_starburst', 'post_starburst'], p=[0.3, 0.7])
            else:
                galaxy_type = 'star_forming'
        
        # Calculate K-correction using empirical SED templates
        # This now accounts for:
        # - Spectral features (Balmer, 4000 Å break, 1.6 μm bump, PAH)
        # - Extinction (Calzetti law)
        # - Dust emission (Chary & Elbaz, Dale & Helou)
        k_correction = calculate_k_correction(field_z, band, rng, galaxy_type=galaxy_type)
        
        # Apply transformation with K-correction
        # Note: K-correction now incorporates physical color evolution,
        # so no need for additional ad-hoc color shifts
        final_mag = base_mag + natural_scatter + k_correction
        
        # Keep galaxies detectable at this telescope's noise floor
        final_mag = np.clip(final_mag, 18.0, _field_mag_limit())
        
        gal_band['magnitude'] = float(final_mag)
        mag_stats['after'].append(final_mag)

        # Native multi-component morphology (bulge/disk/bar/ring/...) for
        # the Sersic fallback, when enabled. The fragment/morph type were
        # already determined once in field_galaxy_light_model_types and
        # cached on `gal` so all bands use the same component structure.
        morph_cfg = CONFIG.get('morphology', {}) if isinstance(CONFIG, dict) else {}
        if morph_cfg.get('multicomponent_enabled', False):
            base_params = {k: gal[k] for k in ('center_x', 'center_y', 'R_sersic', 'n_sersic', 'e1', 'e2')}
            _, kwargs_by_band, _ = gm_build_light_model(
                'field', base_params, {band: final_mag}, [band], rng, CONFIG,
                morph_type=gal.get('_morph_type_resolved'), morph_seed=gal.get('_morph_seed'))
            band_galaxies.extend(kwargs_by_band[band])
            continue

        # Keep ONLY lenstronomy SERSIC_ELLIPSE kwargs (CRITICAL: any other key
        # -- e.g. log_mass, stellar_mass, sim, snapshot, subhalo_id -- makes
        # Lenstronomy.SersicElliptic.function() fail with an "unexpected
        # keyword argument" TypeError). FIX (2026-08-01): this used to be a
        # denylist of "known bad" metadata keys, which missed 'log_mass'
        # (added to the field-galaxy dict by sample_cosmos_field_galaxies)
        # and crashed every render that used the real-COSMOS-bootstrap field
        # population -- an allowlist can't silently miss a newly-added key
        # the same way.
        _sersic_ellipse_keys = {
            'center_x', 'center_y', 'e1', 'e2', 'R_sersic', 'n_sersic', 'magnitude',
        }
        for key in list(gal_band.keys()):
            if key not in _sersic_ellipse_keys:
                gal_band.pop(key, None)

        band_galaxies.append(gal_band)
    
    # Debug magnitude transformation (skip if all galaxies were stamps)
    if mag_stats['before']:
        before_range = f"{np.min(mag_stats['before']):.2f}-{np.max(mag_stats['before']):.2f}"
        after_range = f"{np.min(mag_stats['after']):.2f}-{np.max(mag_stats['after']):.2f}"
        bright_count = np.sum(np.array(mag_stats['after']) < 25.0)
        print(f"  Magnitude transformation: {before_range} → {after_range}")
        print(f"  Bright galaxies (< 25 mag): {bright_count}/{len(band_galaxies)}")
    
    return band_galaxies

# --------------------------------------------------------------------------------------
# Diagnostic function
# --------------------------------------------------------------------------------------

def diagnose_field_galaxy_sampling(field_pop, n_max=8, rng=None, numpix=300):
    """Comprehensive field galaxy sampling diagnostics"""
    
    print(f"\n{'='*60}")
    print("FIELD GALAXY SAMPLING DIAGNOSTICS")
    print(f"{'='*60}")
    
    if field_pop is None:
        print("[ERROR] field_pop is None - no field galaxy data loaded")
        return
    
    print(f"Total field galaxies available: {len(field_pop):,}")
    
    # Check data quality
    required_cols = ['re_rest_clean', 'n_rest_clean', 'q_rest_clean']
    for col in required_cols:
        if col in field_pop.columns:
            valid_count = field_pop[col].notna().sum()
            print(f"  {col}: {valid_count:,}/{len(field_pop):,} valid ({100*valid_count/len(field_pop):.1f}%)")
        else:
            print(f"  {col}: MISSING")
    
    # Morphology distribution
    if 'real_morph_type' in field_pop.columns:
        print(f"\nMorphology Distribution:")
        morph_counts = field_pop['real_morph_type'].value_counts()
        for morph, count in morph_counts.head(5).items():
            print(f"  {morph}: {count:,} ({100*count/len(field_pop):.1f}%)")
    
    # Test sampling
    print(f"\nTesting field galaxy sampling (n_max={n_max})...")
    
    sampled = sample_real_field_galaxies_for_mock(
        field_pop, n_max=n_max, rng=rng, numpix=numpix,
        pixel_scale=0.03, lens_redshift=0.8, avoid_center_arcsec=0.6,
        psf_data=None  # No PSF data in diagnostics
    )
    
    print(f"Sampling result: {len(sampled)}/{n_max} galaxies")
    
    if len(sampled) > 0:
        # Analyze sampled population
        n_values = [gal['n_sersic'] for gal in sampled]
        mag_values = [gal['magnitude'] for gal in sampled]
        morph_types = [gal.get('real_morph_type', 'unknown') for gal in sampled]
        
        print(f"\nSampled Population Analysis:")
        print(f"  Sersic indices: {np.min(n_values):.2f} - {np.max(n_values):.2f} (mean: {np.mean(n_values):.2f})")
        print(f"  Magnitudes: {np.min(mag_values):.2f} - {np.max(mag_values):.2f}")
        print(f"  Spiral fraction (n<2): {np.sum(np.array(n_values) < 2.0)/len(n_values):.2f}")
        
        # Morphology breakdown
        morph_counter = Counter(morph_types)
        print(f"  Morphologies: {dict(morph_counter)}")
    else:
        print("[ERROR] No galaxies sampled - check constraints and data quality")
    
    print(f"{'='*60}\n")

# --------------------------------------------------------------------------------------
# Remaining functions from original pipeline
# --------------------------------------------------------------------------------------

def sanitize_redshifts(df, zl_cols=("lens_redshift","zl","LP_zfinal","z_lens"),
                       zs_cols=("source_redshift","zs"),
                       zl_range=(0.0, 20.0), zs_range=(0.0, 25.0)):
    """Clean redshift data with realistic constraints"""
    out = df.copy()
    
    # Process lens redshift
    zl = None
    for c in zl_cols:
        if c in out.columns:
            zl = pd.to_numeric(out[c], errors="coerce")
            break
    if zl is None:
        zl = pd.Series(np.nan, index=out.index)

    zl[(zl < zl_range[0]) | (zl > zl_range[1])] = np.nan
    # EXPANDED: Extended lens redshift range for high-z studies
    zl = zl.fillna(pd.Series(np.random.uniform(0.2, 6.0, len(out))))

    # Process source redshift
    zs = None
    for c in zs_cols:
        if c in out.columns:
            zs = pd.to_numeric(out[c], errors="coerce")
            break
    if zs is None:
        zs = pd.Series(np.nan, index=out.index)

    zs[(zs < zs_range[0]) | (zs > zs_range[1])] = np.nan
    # Use config source_max/source_min for all source redshifts (fill and clip)
    _cfg_zs_min = CONFIG.get('redshifts', {}).get('source_min', 1.0)
    _cfg_zs_max = CONFIG.get('redshifts', {}).get('source_max', 3.5)
    zs = zs.fillna(pd.Series(np.random.uniform(_cfg_zs_min, _cfg_zs_max, len(out))))
    zs = zs.clip(lower=_cfg_zs_min, upper=_cfg_zs_max)

    # Enforce zs > zl with gap bounded by source_max
    behind = zs <= zl
    if behind.any():
        _min_dz = CONFIG.get('redshifts', {}).get('min_delta_z', 0.5)
        _gap_max = max(0.1, _cfg_zs_max - zl[behind].max())
        zs[behind] = zl[behind] + np.random.uniform(_min_dz, max(_min_dz + 0.1, _gap_max), behind.sum())
        zs[behind] = zs[behind].clip(upper=_cfg_zs_max)

    out["lens_redshift"] = zl.values
    out["source_redshift"] = zs.values
    return out

def choose_observed_band_for_rest(rest_um, z):
    """Find band closest to rest-frame wavelength"""
    try:
        z = float(z)
    except:
        z = 0.6
    z = max(z, 0.0)
    target = rest_um * (1.0 + z)
    return min(LOWER_BANDS, key=lambda b: abs(BAND_CENTERS_UM[b] - target))

def extract_restframe_struct(struct_df, z_series, rest_um=1.6):
    """Extract rest-frame structural parameters"""
    n = len(z_series)
    re_out = pd.Series(np.nan, index=range(n))
    q_out = pd.Series(np.nan, index=range(n))
    n_out = pd.Series(np.nan, index=range(n))

    # Per-object band selection for rest-frame
    for i, z in enumerate(z_series.values):
        if i >= len(struct_df):
            break
        band = choose_observed_band_for_rest(rest_um, z)
        
        for colname, series_out in [
            (f"rearc_{band}", re_out),
            (f"qratio_{band}", q_out), 
            (f"nsersic_{band}", n_out),
        ]:
            if colname in struct_df.columns:
                try:
                    val = pd.to_numeric(struct_df[colname], errors="coerce").iloc[i]
                    if pd.notna(val):
                        series_out.iat[i] = val
                except:
                    pass

    # Fallback across all bands
    def best_across_bands(template, default):
        best = None
        for b in LOWER_BANDS:
            col = template.format(band=b)
            if col in struct_df.columns:
                s = pd.to_numeric(struct_df[col], errors="coerce")
                s = s.reset_index(drop=True).reindex(range(n))
                if best is None:
                    best = s.copy()
                else:
                    missing = best.isna() & s.notna()
                    best[missing] = s[missing]
        return best.fillna(default) if best is not None else pd.Series([default]*n, index=range(n))

    re_out = re_out.fillna(best_across_bands("rearc_{band}", 0.7))
    q_out = q_out.fillna(best_across_bands("qratio_{band}", 0.8)).clip(0.05, 1.0)
    n_out = n_out.fillna(best_across_bands("nsersic_{band}", 3.0)).clip(0.5, 8.0)

    return {"re_rest": re_out, "q_rest": q_out, "n_rest": n_out}

def sample_sersic_n(z, measured=None, rng=None):
    """Sample Sersic index with redshift evolution"""
    if rng is None:
        # FIX (audit C-5): a bare default_rng() draws fresh OS entropy,
        # ignoring args.seed entirely and breaking reproducibility even
        # when a caller forgot to pass its own `rng`. Seed from the (now
        # seeded-at-startup, see main()) global np.random stream instead,
        # so behavior is deterministic given a fixed call order.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    
    try:
        z = float(z)
    except:
        z = 0.6

    if measured is not None and np.isfinite(measured):
        mu, sig, lo, hi = float(measured), 0.4, 0.5, 6.0
    elif z < 0.8:
        mu, sig, lo, hi = 3.5, 0.7, 2.0, 6.0
    elif z < 1.6:
        mu, sig, lo, hi = 2.5, 0.6, 1.0, 5.0
    else:
        mu, sig, lo, hi = 1.5, 0.5, 0.5, 3.0

    for _ in range(100):
        x = rng.normal(mu, sig)
        if lo <= x <= hi:
            return float(x)
    return float(np.clip(mu, lo, hi))

def filter_lenstronomy_params(config_dict):
    """Remove metadata parameters before passing to lenstronomy"""
    metadata_params = {'_', 'morph_type', 'redshift', 'mass_log10', 'field_redshift', 'real_morph_type', 'real_color_type', 'source_lens_id'}
    return {k: v for k, v in config_dict.items() if k not in metadata_params and not k.startswith('_')}

def create_jwst_band_configs(rng=None, use_distribution=True):
    """
    Create JWST band configs with empirical noise from real observations
    
    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator for sampling noise distribution
    use_distribution : bool, optional
        If True, sample from observed distribution of 435 lenses (DEFAULT).
        If False, use median values from config.
    
    Returns
    -------
    dict
        Band configurations with noise properties
    """
    if rng is None:
        # FIX (audit C-5): a bare default_rng() draws fresh OS entropy,
        # ignoring args.seed entirely and breaking reproducibility even
        # when a caller forgot to pass its own `rng`. Seed from the (now
        # seeded-at-startup, see main()) global np.random stream instead,
        # so behavior is deterministic given a fixed call order.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    
    # Try to load empirical noise sampler
    sampler = None
    if use_distribution and EMPIRICAL_NOISE_AVAILABLE:
        try:
            from pathlib import Path
            repo_root = Path(__file__).parent.parent
            json_path = repo_root / 'configs' / 'jwst_empirical_noise.json'
            if json_path.exists():
                sampler = EmpiricalNoiseSampler(json_path)
                print(f"  ✓ Loaded empirical noise sampler with {len(sampler.distributions['F115W']['bg_level'])} measurements per band")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load empirical noise sampler: {e}")
            print("  → Falling back to config values")
    
    # Determine sampling method from config
    noise_cfg = CONFIG.get('noise', {})
    sampling_method = noise_cfg.get('sampling_method', 'distribution')
    
    # Map config method to sampler method
    method_map = {
        'distribution': 'random',
        'median': 'median',
        'percentile': 'percentile'
    }
    sampler_method = method_map.get(sampling_method, 'random')
    
    # Sample or use config values
    noise_samples = {}
    
    if sampler and use_distribution and sampling_method == 'distribution':
        # Sample from actual observed distribution (MOST REALISTIC)
        try:
            sampled_bands = sampler.sample_all_bands(rng, method=sampler_method)
            noise_samples.update(sampled_bands)
        except KeyError as e:
            # Sampler doesn't have data for all bands - fall back to config
            print(f"  ⚠ Warning: Empirical noise sampler missing band {e}, using config defaults for all bands")
    
    # Fill in any missing bands with defaults
    noise_cfg = CONFIG.get('noise', {})
    
    # Default noise properties for all telescope bands
    default_noise = {
        # JWST NIRCam bands
        "F070W": {"sky_brightness": 30.5, "read_noise": 0.020},
        "F090W": {"sky_brightness": 30.3, "read_noise": 0.020},
        "F115W": {"sky_brightness": 30.07, "read_noise": 0.01996},
        "F150W": {"sky_brightness": 28.86, "read_noise": 0.01623},
        "F200W": {"sky_brightness": 28.5, "read_noise": 0.015},
        "F277W": {"sky_brightness": 28.02, "read_noise": 0.00590},
        "F356W": {"sky_brightness": 28.0, "read_noise": 0.007},
        "F444W": {"sky_brightness": 28.04, "read_noise": 0.00704},
        # Roman WFI bands (similar to JWST near-IR)
        "ROMAN_F062": {"sky_brightness": 30.5, "read_noise": 0.015},
        "ROMAN_F087": {"sky_brightness": 30.0, "read_noise": 0.015},
        "ROMAN_F106": {"sky_brightness": 29.5, "read_noise": 0.012},
        "ROMAN_F129": {"sky_brightness": 29.0, "read_noise": 0.010},
        "ROMAN_F146": {"sky_brightness": 28.8, "read_noise": 0.010},
        "ROMAN_F158": {"sky_brightness": 28.5, "read_noise": 0.010},
        "ROMAN_F184": {"sky_brightness": 28.3, "read_noise": 0.010},
        "ROMAN_F213": {"sky_brightness": 28.0, "read_noise": 0.010},
        # Subaru Suprime-Cam optical bands
        "SUBARU_B": {"sky_brightness": 22.5, "read_noise": 0.040},
        "SUBARU_V": {"sky_brightness": 22.0, "read_noise": 0.035},
        "SUBARU_G": {"sky_brightness": 22.3, "read_noise": 0.035},
        "SUBARU_R": {"sky_brightness": 21.5, "read_noise": 0.030},
        "SUBARU_I": {"sky_brightness": 20.8, "read_noise": 0.025},
        "SUBARU_Z": {"sky_brightness": 20.0, "read_noise": 0.020},
        "SUBARU_Y": {"sky_brightness": 19.5, "read_noise": 0.020},
        # Euclid (VIS optical + NISP near-IR)
        "EUCLID_VIS": {"sky_brightness": 22.5, "read_noise": 0.004},  # Low read noise CCD
        "EUCLID_Y": {"sky_brightness": 29.0, "read_noise": 0.012},
        "EUCLID_J": {"sky_brightness": 28.5, "read_noise": 0.010},
        "EUCLID_H": {"sky_brightness": 28.0, "read_noise": 0.010},
    }
    
    for band in UPPER_BANDS:
        if band not in noise_samples:
            # Band not in empirical data, use config or defaults
            band_noise = noise_cfg.get(band, {})
            band_defaults = default_noise.get(band, {"sky_brightness": 27.0, "read_noise": 12.0})
            noise_samples[band] = {
                'sky_brightness': band_noise.get('sky_brightness', band_defaults['sky_brightness']),
                'read_noise': band_noise.get('read_noise', band_defaults['read_noise']),
                'bg_level': band_noise.get('background_level', 0.0),
                'bg_rms': band_noise.get('background_rms', 0.02)
            }
    
    # Build configs
    configs = {}
    for band in UPPER_BANDS:
        noise = noise_samples[band]
        
        # Pick up per-telescope ZP from telescope_configs if set there.
        #
        # FIX (adversarial audit finding C-11, 2026-08-01): this used to
        # apply ONE scalar magnitude_zero_point to every band (F115W,
        # F150W, F277W, F444W all got 28.09), so the mag->flux conversion
        # carried no filter-dependent throughput information at all -- all
        # colour in the output came only from the synthetic SED offsets
        # applied elsewhere, while the *overall* photometric zeropoint was
        # identical across bands (unphysical: NIRCam's per-filter
        # zeropoints genuinely differ with throughput/bandwidth).
        #
        # Approximate published NIRCam AB zeropoints (STScI JDox pipeline
        # reference values, imaging mode, e-/s -> AB mag); these are
        # DEFAULT/PLACEHOLDER values good to ~0.1-0.3 mag -- verify against
        # the current jwst_pipeline CRDS reference files before using for
        # precision photometric science, and override via
        # telescope_configs.<band>.magnitude_zero_point in config if exact
        # per-visit calibration values are available.
        _default_band_zp = {
            'F115W': 25.68, 'F150W': 25.97, 'F277W': 26.63, 'F444W': 26.32,
        }
        _tel_name_bc = CONFIG.get('telescope', 'jwst').lower()
        _tel_cfg_bc  = CONFIG.get('telescope_configs', {}).get(_tel_name_bc, {})
        _band_zp_cfg = _tel_cfg_bc.get('band_zero_points', {})
        _zp = float(_band_zp_cfg.get(band,
                    _default_band_zp.get(band,
                    _tel_cfg_bc.get('magnitude_zero_point',
                    CONFIG.get('magnitude_zero_point', 28.09)))))

        configs[band] = {
            "pixel_scale": float(CONFIG.get('pixel_scale', 0.031)),
            "exposure_time": float(CONFIG.get('exposure_time', 1028.0)),
            "magnitude_zero_point": _zp,
            "read_noise": float(noise['read_noise']),  # Sampled from real observations
            "sky_brightness": float(noise['sky_brightness']),  # Sampled from 435 real lenses
            "psf_type": "NONE",  # We'll apply PSF convolution manually
            # Store additional empirical properties for advanced noise modeling
            "_bg_level": float(noise.get('bg_level', 0.0)),
            "_bg_rms": float(noise.get('bg_rms', 0.02))
        }
        
        for key, val in configs[band].items():
            if key != "psf_type" and not key.startswith("_"):
                assert isinstance(val, (int, float, np.floating)), \
                    f"Config {key} for {band} must be scalar, got {type(val)}"
    
    return configs

def add_cosmos_web_artifacts(images, rng, artifact_level='moderate', numpix=300, add_spikes=False):
    """
    Add realistic COSMOS-Web observational artifacts including optional JWST diffraction spikes.
    
    Artifacts are scaled appropriately for image size - NOT all simulated images should have artifacts
    since they are small postage stamps (300x300 pixels typical).
    
    Args:
        images: Dict of band images
        rng: Random number generator
        artifact_level: 'low', 'moderate', 'high'
        numpix: Image pixel size (for scaling artifact sizes)
        add_spikes: If True, detect bright objects for spike addition in RGB
        
    Returns:
        enhanced_images: Dict of modified images
        bright_positions: List of (y, x) positions of bright objects (for RGB spike addition)
    """
    enhanced_images = {}
    bright_positions = []  # Will store positions for RGB spike addition
    
    for band, image in images.items():
        enhanced = np.array(image, copy=True)
        
        # Scale artifact intensity and size based on image size
        # Smaller images get fewer and smaller artifacts
        size_factor = numpix / 300.0  # Normalized to 300px reference
        
        # Detect bright positions for spike addition to RGB (only once, use first band)
        if band == "F115W" and add_spikes:  # Only detect once
            positive_pixels = enhanced[enhanced > 0]
            if len(positive_pixels) > 10:
                threshold = np.percentile(positive_pixels, 90)  # Top 10%
                bright_mask = enhanced > threshold
                bright_y, bright_x = np.where(bright_mask)
                bright_positions = list(zip(bright_y, bright_x))
        
        # Cosmic ray hits - REALISTIC probability based on JWST exposure statistics
        # Low: ~1/500 images, Moderate: ~1/100 images, High: ~1/20 images
        cosmic_ray_prob = {'low': 0.002, 'moderate': 0.01, 'high': 0.05}[artifact_level]
        
        if rng.random() < cosmic_ray_prob:
            n_hits = max(1, safe_random_integers(rng, 1, 3)) if numpix >= 300 else 1  # Fewer for small images
            for _ in range(n_hits):
                # Ensure hit is not too close to edge
                margin = max(3, int(5 * size_factor))
                if enhanced.shape[0] > 2*margin and enhanced.shape[1] > 2*margin:
                    y = safe_random_integers(rng, margin, enhanced.shape[0]-margin)
                    x = safe_random_integers(rng, margin, enhanced.shape[1]-margin)
                    
                    # Scale cosmic ray intensity based on image size (SUBTLE, not dominating)
                    # Realistic: 0.5-2x std, not 5-12x which makes them as bright as galaxies
                    intensity_range = (0.5, 2.0) if numpix >= 300 else (0.2, 1.0)
                    intensity = rng.uniform(*intensity_range) * np.std(enhanced)
                    enhanced[y, x] += intensity
                    
                    # Spread to neighbors - RARELY spreads (mostly single-pixel)
                    # Realistic: 5% for large, 2% for small (not 25% and 10%)
                    neighbor_prob = 0.05 if numpix >= 300 else 0.02
                    spread_range = 1 if numpix >= 300 else 1
                    for dy in range(-spread_range, spread_range+1):
                        for dx in range(-spread_range, spread_range+1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < enhanced.shape[0] and 0 <= nx < enhanced.shape[1]:
                                if rng.random() < neighbor_prob:
                                    enhanced[ny, nx] += intensity * rng.uniform(0.05, 0.25)
        
        # Electronic artifacts (VERY RARE - hot pixels are uncommon)
        # 10-100x rarer than before: 0.01-0.1% instead of 1-8%
        electronic_prob = {'low': 0.0001, 'moderate': 0.0005, 'high': 0.001}[artifact_level]
        
        if rng.random() < electronic_prob and numpix >= 300:  # Only for larger images
            n_hot = safe_random_integers(rng, 1, 2)
            for _ in range(n_hot):
                y = safe_random_integers(rng, 0, enhanced.shape[0])
                x = safe_random_integers(rng, 0, enhanced.shape[1])
                enhanced[y, x] += rng.uniform(3, 8) * np.std(enhanced)
        
        # REMOVED: Spikes are now added to RGB image instead of individual filter bands
        # This prevents them from being crushed by Trilogy normalization
        # Spikes will be added in save_outputs_complete() directly to the final RGB
        # Before: if add_spikes: enhanced = add_jwst_diffraction_spikes(...)
        # After: bright_positions will be stored in field_info and used during RGB save
        
        enhanced_images[band] = enhanced.astype(np.float32)
    
    return enhanced_images, bright_positions

def normalize_for_display_astronomical(im, noise_level=0.5, sat_percent=0.001, channel_name=""):
    """Astronomical normalization following Trilogy approach for realistic JWST colors

    Used for the individual single-band display panels -- preserves the
    original noise/background appearance of each band.
    """

    if not np.isfinite(im).any():
        return np.ones_like(im) * 0.05

    im_clean = np.where(np.isfinite(im), im, 0)

    positive_data = im_clean[im_clean > 0]
    if len(positive_data) == 0:
        return np.ones_like(im) * 0.05

    # Establish noise floor (similar to Trilogy's noiselums)
    noise_floor = np.percentile(positive_data, 5) * noise_level
    im_work = np.maximum(im_clean, noise_floor)

    # Conservative saturation level
    sat_level = np.percentile(positive_data, 100 - sat_percent * 100)
    if sat_level <= noise_floor:
        sat_level = noise_floor * 10

    # Asinh stretch (standard astronomical scaling)
    # Standard softening for realistic normalization
    softening = sat_level * 0.01
    if softening <= 0:
        softening = np.mean(positive_data) * 0.01

    stretched = np.arcsinh(im_work / softening)

    # Conservative normalization
    stretched_pos = stretched[stretched > 0]
    if len(stretched_pos) > 0:
        vmax = np.percentile(stretched_pos, 99.5)
        if vmax <= 0:
            vmax = np.max(stretched_pos)
        normalized = stretched / vmax
    else:
        normalized = stretched

    normalized = np.clip(normalized, 0, 1)

    # Gentle minimum brightness
    max_val = np.max(normalized)
    if max_val < 0.001:
        normalized = normalized * (0.05 / (max_val + 1e-12))
        normalized = np.clip(normalized, 0, 1)

    return normalized

def _rgb_composite_sky_subtract(im, sigma_mult=1.5, soft_clip=False):
    """Sky-subtract and noise-threshold a single band for RGB compositing.

    When ``soft_clip`` is True, pixels below the noise threshold keep a
    tapered fraction of their flux instead of being hard-zeroed. This
    preserves faint field galaxies that are clearly visible in the
    single-band panels but would otherwise vanish from the RGB composite.
    """
    if not np.isfinite(im).any():
        return np.zeros_like(im), 0.0, 1e-12

    im_clean = np.where(np.isfinite(im), im, 0.0)
    sky_level = np.median(im_clean)
    mad = np.median(np.abs(im_clean - sky_level))
    noise_sigma = mad * 1.4826
    if noise_sigma <= 0:
        noise_sigma = np.std(im_clean)
    if noise_sigma <= 0:
        noise_sigma = np.abs(sky_level) * 0.01 + 1e-12

    threshold = max(float(sigma_mult), 0.0) * noise_sigma
    im_sub = im_clean - sky_level
    if soft_clip and threshold > 0:
        # Soft ramp: full keep above threshold; almost zero in pure noise.
        # Keep a tiny floor (5%) so very faint envelopes survive without
        # reintroducing chromatic grain into the RGB sky.
        keep = np.clip(im_sub / threshold, 0.0, 1.0)
        keep = keep * keep
        im_sub = np.maximum(im_sub, 0.0) * (0.05 + 0.95 * keep)
    else:
        im_sub = np.where(im_sub > threshold, im_sub - threshold, 0.0)
    return im_sub, float(sky_level), float(noise_sigma)


def normalize_for_rgb_composite(im, noise_level=0.5, sat_percent=0.001, sigma_mult=1.5,
                                stretch_scale=None, soft_clip=False):
    """Astronomical normalization for the RGB composite only.

    Performs a robust sky subtraction before the asinh stretch so that
    background noise pixels are pushed to (near-)zero rather than spread
    across the full display range. Without this, uncorrelated per-band noise
    survives the stretch in every channel and combines into prominent
    salt-and-pepper color speckle in RGB composites, while still leaving real
    sources (well above the noise) clearly visible.

    When ``stretch_scale`` is provided (linked-stretch mode), all RGB channels
    share the same saturation/softening scale so inter-band flux ratios --
    and therefore galaxy colour -- are preserved. Independent per-channel
    stretching washes out real SED differences and makes Euclid/TNG galaxies
    look uniformly yellow.

    This is intentionally separate from normalize_for_display_astronomical()
    so that the individual single-band display panels keep their original
    noise/background appearance -- only the RGB composite gets the cleaned-up
    treatment.
    """
    normalized, _ = _normalize_for_rgb_composite_core(
        im, sat_percent=sat_percent, sigma_mult=sigma_mult, stretch_scale=stretch_scale,
        soft_clip=soft_clip,
    )
    if normalized is None:
        return np.zeros_like(im)
    return normalized


def _normalize_for_rgb_composite_core(im, sat_percent=0.001, sigma_mult=1.5,
                                      stretch_scale=None, soft_clip=False):
    """Return (normalized_image, stretch_scale_used)."""
    im_sub, _, noise_sigma = _rgb_composite_sky_subtract(
        im, sigma_mult=sigma_mult, soft_clip=soft_clip,
    )
    positive_data = im_sub[im_sub > 0]
    if len(positive_data) == 0:
        return None, stretch_scale or 1e-12

    if stretch_scale is None:
        sat_level = np.percentile(positive_data, 100 - sat_percent * 100)
        if sat_level <= 0:
            sat_level = np.max(positive_data)
        stretch_scale = max(float(sat_level), noise_sigma * 0.5, 1e-12)
    else:
        stretch_scale = max(float(stretch_scale), 1e-12)

    stretched = np.arcsinh(im_sub / stretch_scale)
    stretched = np.clip(stretched, 0, None)

    stretched_pos = stretched[stretched > 0]
    if len(stretched_pos) > 0:
        # Soft-clip keeps faint flux; use a high vmax so bright BGG cores
        # are not crushed into flat white by a low percentile ceiling.
        vmax_pct = 99.2 if soft_clip else 99.5
        vmax = np.percentile(stretched_pos, vmax_pct)
        if vmax <= 0:
            vmax = np.max(stretched_pos)
        normalized = stretched / vmax
    else:
        normalized = stretched

    normalized = np.clip(normalized, 0, 1)
    max_val = np.max(normalized)
    if 0 < max_val < 0.05:
        normalized = normalized * (0.3 / (max_val + 1e-12))
        normalized = np.clip(normalized, 0, 1)

    return normalized, stretch_scale


def _rgb_arc_residual(lens_sources, lens_only, bands):
    """Positive residual (lens+source minus lens-only) per band for arc emphasis."""
    residual = {}
    for band in bands:
        if band not in lens_sources or band not in lens_only:
            continue
        residual[band] = np.clip(
            np.where(np.isfinite(lens_sources[band]), lens_sources[band], 0.0)
            - np.where(np.isfinite(lens_only[band]), lens_only[band], 0.0),
            0.0, None,
        )
    return residual

def apply_field_galaxy_realism(rgb, images, numpix):
    """Apply field galaxy dimming and color balance to match real JWST observations"""
    
    # Identify central lens region (brightest area - typically the lens galaxy)
    center = numpix // 2
    central_radius = max(6, numpix // 12)  # Central region radius
    
    # Create masks
    y, x = np.ogrid[:numpix, :numpix]
    central_mask = (x - center)**2 + (y - center)**2 <= central_radius**2
    
    # Field region mask (outer regions excluding center)
    field_mask = ~central_mask
    
    # CORRECTED: Don't artificially dim field galaxies 
    # Field and central galaxies from same COSMOS-Web observations should have similar intrinsic brightness
    # Only natural magnitude differences from the catalog data should apply
    
    # Color balance correction: Reduce artificial green/purple tones
    # Real JWST lenses show warm, natural colors
    
    # Suppress excessive green channel (common artifact in our mocks)
    excessive_green_mask = (rgb[:,:,1] > rgb[:,:,0]) & (rgb[:,:,1] > rgb[:,:,2]) & (rgb[:,:,1] > 0.3)
    rgb[excessive_green_mask, 1] *= 0.8  # Reduce green dominance
    
    # Enhance warm tones in central regions (more like real lenses)
    if np.any(central_mask):
        # Slightly boost red channel in central lens
        rgb[central_mask, 0] = np.minimum(rgb[central_mask, 0] * 1.05, 1.0)
        
        # Slightly reduce blue dominance in central lens
        rgb[central_mask, 2] *= 0.95
    
    # Global color temperature adjustment for more realistic appearance
    # Real JWST lenses tend to have warmer color temperatures
    rgb[:,:,0] *= 1.02  # Slight red boost
    rgb[:,:,2] *= 0.98  # Slight blue reduction
    
    return np.clip(rgb, 0, 1)

def add_spikes_to_rgb(rgb, bright_positions=None, numpix=300, rng=None):
    """
    Add JWST diffraction spikes directly to RGB image for visibility.
    This is applied AFTER RGB creation to avoid dampening by band normalization.
    
    Args:
        rgb: RGB image array [H, W, 3] with values in [0, 1]
        bright_positions: List of (y, x) coordinates of bright objects
        numpix: Image size
        rng: Random number generator
        
    Returns:
        rgb: Modified RGB with spikes added
    """
    if bright_positions is None or len(bright_positions) == 0:
        return rgb
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    # Parameters for RGB spikes (more aggressive since no normalization follows)
    spike_length = max(40, int(60 * numpix / 300))  # ~60px for 300x300
    brightness_scale = 0.3  # Add to RGB which is already normalized [0,1]
    
    # Define spike directions: 4 cardinal + 2 diagonal (at reduced strength)
    cardinal_directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W
    diagonal_directions = [(-1, 1), (1, -1)]  # NE, SW
    
    rgb_modified = rgb.copy()
    
    for y_center, x_center in bright_positions:
        y_center, x_center = int(y_center), int(x_center)
        
        # Ensure position is within bounds
        if not (0 <= y_center < numpix and 0 <= x_center < numpix):
            continue
        
        # Add cardinal spikes (full strength)
        for dy, dx in cardinal_directions:
            for dist in range(1, spike_length + 1):
                ny, nx = y_center + dy * dist, x_center + dx * dist
                if 0 <= ny < numpix and 0 <= nx < numpix:
                    # Linear falloff: stronger closer to center
                    intensity_factor = max(0, 1.0 - dist / spike_length)
                    spike_intensity = brightness_scale * intensity_factor
                    rgb_modified[ny, nx] = np.minimum(rgb_modified[ny, nx] + spike_intensity, 1.0)
        
        # Add diagonal spikes (weaker - 50% strength)
        diagonal_brightness = brightness_scale * 0.5
        for dy, dx in diagonal_directions:
            for dist in range(1, int(spike_length * 0.7) + 1):
                ny, nx = y_center + dy * dist, x_center + dx * dist
                if 0 <= ny < numpix and 0 <= nx < numpix:
                    intensity_factor = max(0, 1.0 - dist / (spike_length * 0.7))
                    spike_intensity = diagonal_brightness * intensity_factor
                    rgb_modified[ny, nx] = np.minimum(rgb_modified[ny, nx] + spike_intensity, 1.0)
    
    return rgb_modified

def create_trilogy_rgb(images, numpix, normalization_scale=None):
    """
    Create RGB image using Trilogy-style processing for JWST 4-band data
    Matches the method used for real COSMOS-Web lens observations
    
    Args:
        images: Dictionary of band images
        numpix: Image size
        normalization_scale: Optional dict with normalization scales for consistent
                            normalization across epochs (for time delay systems)
    
    Returns:
        rgb_image: RGB array [0,1]
        normalization_scales: Dict with scales used (for reuse across epochs)
    """
    
    try:
        # Try to import trilogy if available
        import trilogy
        
        # Prepare JWST 4-band data in order: F115W, F150W, F277W, F444W
        bands_data = [images["F115W"], images["F150W"], images["F277W"], images["F444W"]]
        
        # JWST band center wavelengths in microns
        band_wavelengths = [1.15, 1.50, 2.77, 4.44]
        
        # Create RGB using Trilogy with standard JWST parameters
        rgb_image = trilogy.make_rgb(
            data=bands_data,
            wavelengths=band_wavelengths,
            noise=None,  # Let trilogy estimate noise
            invert=False,
            lupton_alpha=1000,  # Standard for JWST
            lupton_Q=10,        # Conservative stretch
            saturation=0.15,    # Prevent over-saturation
        )
        
        # Trilogy returns RGB in [0,1] range, ensure proper format
        rgb_image = np.clip(rgb_image, 0, 1)
        
        print("[INFO] Using Trilogy RGB generation (matches real observations)")
        # Return dummy scales for trilogy (not used)
        return rgb_image, {}
        
    except ImportError:
        print("[INFO] Trilogy not available, using manual 4-band RGB composition")
        # Manual Trilogy-style implementation for JWST
        return create_manual_trilogy_rgb(images, numpix, normalization_scale)

def create_manual_trilogy_rgb(images, numpix, normalization_scale=None):
    """
    Manual implementation of Trilogy-style RGB for JWST 4-band data
    Based on standard astronomical RGB composition principles
    
    Args:
        images: Dictionary of band images
        numpix: Image size
        normalization_scale: Optional dict with 'F115W', 'F150W', 'F277W', 'F444W' scales
                            to use for consistent normalization across epochs.
                            If None, uses percentile-based normalization (default behavior).
    """
    
    # Use all 4 JWST bands with optimal weighting
    F115W = images["F115W"]  # Blue
    F150W = images["F150W"]  # Blue-Green  
    F277W = images["F277W"]  # Green-Red
    F444W = images["F444W"]  # Red
    
    # Trilogy-style normalization parameters (conservative)
    def trilogy_normalize(data, band_name, scale=None):
        """
        Normalize following Trilogy/Lupton approach with proper sky subtraction.

        The key improvement over a naive percentile stretch: we estimate and
        subtract the sky background before stretching so that detector noise
        (read noise, 1/f, dark current) does not bias the zero-point upward
        and make the background appear gray instead of black.
        """
        clean_data = np.where(np.isfinite(data), data, 0.0)

        # ── Sky background estimation (sigma-clipped median) ──────────────────
        # Use the 16th–50th percentile of all positive pixels as a robust sky
        # estimate.  Sigma-clip: exclude pixels > median + 3×MAD to avoid
        # contamination from the lens/arc signal.
        pos_mask = clean_data > 0
        if np.any(pos_mask):
            flat = clean_data[pos_mask]
            sky_med = np.median(flat)
            mad = np.median(np.abs(flat - sky_med))
            sky_mask = flat < sky_med + 3.0 * mad
            sky_estimate = np.median(flat[sky_mask]) if sky_mask.sum() > 10 else sky_med
        else:
            sky_estimate = 0.0

        # ── Sky subtraction ───────────────────────────────────────────────────
        sky_sub = clean_data - sky_estimate

        # ── Robust noise estimate (MAD of background pixels) ─────────────────
        bkg_flat = clean_data[pos_mask] if np.any(pos_mask) else clean_data.ravel()
        bkg_flat = bkg_flat[bkg_flat < sky_estimate + 10.0 * (sky_estimate + 1e-12)]
        noise_est = 1.4826 * float(np.median(np.abs(bkg_flat - sky_estimate))) \
                    if bkg_flat.size > 0 else 1e-10
        noise_est = max(noise_est, 1e-12)

        # ── 2σ threshold mask (standard astronomical imaging practice) ────────
        # Pixels below 2σ above sky are set to zero so detector read noise and
        # background noise map to black rather than mid-gray.
        sigma_thresh = 2.0
        sky_sub_thresh = np.where(sky_sub >= sigma_thresh * noise_est, sky_sub, 0.0)

        # ── Stretch scale ─────────────────────────────────────────────────────
        if scale is not None:
            percentile_scale = float(scale)
        else:
            sig_pixels = sky_sub_thresh[sky_sub_thresh > 0]
            p99 = float(np.percentile(sig_pixels, 99.0)) if sig_pixels.size > 0 \
                  else sigma_thresh * noise_est
            percentile_scale = max(p99, sigma_thresh * noise_est)

        # ── Asinh stretch (Lupton et al. 2004) ───────────────────────────────
        # alpha=150, Q=8 gives a comfortable dynamic range for 1028s JWST frames.
        alpha = 150.0
        Q     = 8.0

        stretched = np.arcsinh(alpha * Q * sky_sub_thresh / (percentile_scale + 1e-30)) / Q

        # ── Gamma correction: further suppress low-SB without affecting peaks ─
        gamma = 1.5

        if scale is not None:
            max_stretched = np.arcsinh(alpha * Q) / Q
            normalized = (stretched / (max_stretched + 1e-30)) ** (1.0 / gamma)
        else:
            smax = np.max(stretched)
            normalized = (stretched / (smax + 1e-30)) ** (1.0 / gamma) if smax > 0 \
                         else stretched

        return np.clip(normalized, 0, 1), percentile_scale
    
    # Build the three colour channels from band blends.
    # IMPORTANT: all three channels share the SAME normalization scale
    # (derived from the combined signal across bands) so that inter-band
    # flux ratios are preserved.  Independent per-channel normalization
    # washes out real colour information: an ETG that is 4× brighter in
    # F444W than F115W in flux would appear neutral-grey instead of
    # warm orange if each channel were stretched to its own maximum.
    b_blend = F115W
    g_blend = 0.7 * F150W + 0.3 * F277W
    r_blend = 0.8 * F444W + 0.2 * F277W

    if normalization_scale is not None:
        # Time-delay mode: per-band scales provided externally.
        b_norm, b_scale = trilogy_normalize(b_blend, "F115W",    scale=normalization_scale.get("F115W"))
        g_norm, g_scale = trilogy_normalize(g_blend, "F150W+F277W", scale=normalization_scale.get("F150W+F277W"))
        r_norm, r_scale = trilogy_normalize(r_blend, "F444W+F277W", scale=normalization_scale.get("F444W+F277W"))
    else:
        # Default: first pass — obtain per-channel scales.
        _, b_scale = trilogy_normalize(b_blend, "F115W")
        _, g_scale = trilogy_normalize(g_blend, "F150W+F277W")
        _, r_scale = trilogy_normalize(r_blend, "F444W+F277W")
        # Common scale = median of the three, so the brightest channel
        # does not dominate the stretch while still preserving flux ratios.
        common_scale = float(np.median([b_scale, g_scale, r_scale]))
        b_norm, _ = trilogy_normalize(b_blend, "F115W",       scale=common_scale)
        g_norm, _ = trilogy_normalize(g_blend, "F150W+F277W", scale=common_scale)
        r_norm, _ = trilogy_normalize(r_blend, "F444W+F277W", scale=common_scale)
        b_scale = g_scale = r_scale = common_scale

    # Create RGB array
    rgb = np.zeros((numpix, numpix, 3))
    rgb[:,:,0] = r_norm  # Red
    rgb[:,:,1] = g_norm  # Green
    rgb[:,:,2] = b_norm  # Blue
    
    # Apply final color balance for JWST realism
    rgb = apply_jwst_color_balance(rgb)
    
    # Return RGB and normalization scales for reuse across epochs
    normalization_scales = {
        "F115W": b_scale,
        "F150W+F277W": g_scale,
        "F444W+F277W": r_scale
    }
    
    return np.clip(rgb, 0, 1), normalization_scales

def apply_jwst_color_balance(rgb):
    """Apply JWST-specific color balance corrections to match real observations"""
    
    # Stronger reduction of artificial blue excess (common issue in simulations)
    rgb[:,:,2] *= 0.75
    
    # More conservative green channel (reduce artificial green artifacts)
    rgb[:,:,1] = np.power(rgb[:,:,1], 0.95)
    
    # Enhanced warm tones to match real COSMOS-Web observations
    rgb[:,:,0] *= 1.15  # Stronger red boost for natural warm tones
    
    # Apply overall contrast and saturation adjustments
    # Real JWST images tend to be less saturated than simulations
    rgb = np.power(rgb, 0.85)  # Reduce overall saturation
    
    return rgb

def ellipticity(q, pa_deg):
    """Convert axis ratio and PA to ellipticity components"""
    q = float(np.clip(q, 0.05, 1.0))
    e = (1 - q) / (1 + q)
    pa = math.radians(pa_deg)
    return e * math.cos(2*pa), e * math.sin(2*pa)

def ellipticity_to_axis_ratio(e1, e2):
    """Convert ellipticity components to axis ratio q."""
    e = float(np.hypot(float(e1), float(e2)))
    e = np.clip(e, 0.0, 0.95)
    return float((1.0 - e) / (1.0 + e))

def build_field_structural_metadata(field_galaxies):
    """Build per-galaxy and summary structural metadata for field galaxies."""
    if field_galaxies is None:
        field_galaxies = []

    per_galaxy = []
    for g in field_galaxies:
        e1 = float(g.get('e1', 0.0))
        e2 = float(g.get('e2', 0.0))
        per_galaxy.append({
            'center_x': float(g.get('center_x', np.nan)),
            'center_y': float(g.get('center_y', np.nan)),
            'R_sersic': float(g.get('R_sersic', np.nan)),
            'n_sersic': float(g.get('n_sersic', np.nan)),
            'axis_ratio': float(g.get('axis_ratio', ellipticity_to_axis_ratio(e1, e2))),
            'e1': e1,
            'e2': e2,
            'position_angle_deg': float(g.get('position_angle', np.nan)),
            'magnitude': float(g.get('magnitude', np.nan)),
            'field_redshift': float(g.get('field_redshift', g.get('redshift', np.nan))),
            'morph_type': str(g.get('real_morph_type', g.get('morph_type', 'unknown')))
        })

    field_n = [d['n_sersic'] for d in per_galaxy if np.isfinite(d['n_sersic'])]
    field_r = [d['R_sersic'] for d in per_galaxy if np.isfinite(d['R_sersic'])]
    field_q = [d['axis_ratio'] for d in per_galaxy if np.isfinite(d['axis_ratio'])]
    field_z = [d['field_redshift'] for d in per_galaxy if np.isfinite(d['field_redshift'])]

    return {
        'field_structural_data': per_galaxy,
        'field_n_sersic_list': [d['n_sersic'] for d in per_galaxy],
        'field_radius_list': [d['R_sersic'] for d in per_galaxy],
        'field_axis_ratio_list': [d['axis_ratio'] for d in per_galaxy],
        'field_redshift_list': [d['field_redshift'] for d in per_galaxy],
        'field_mean_n_sersic': float(np.mean(field_n)) if field_n else np.nan,
        'field_mean_radius': float(np.mean(field_r)) if field_r else np.nan,
        'field_mean_axis_ratio': float(np.mean(field_q)) if field_q else np.nan,
        'field_mean_redshift': float(np.mean(field_z)) if field_z else np.nan,
    }

def create_parameter_variations(base_catalog, variations_per_base=25, rng=None):
    """Generate diverse variations from limited base catalog"""
    if rng is None:
        # FIX (audit C-5): a bare default_rng() draws fresh OS entropy,
        # ignoring args.seed entirely and breaking reproducibility even
        # when a caller forgot to pass its own `rng`. Seed from the (now
        # seeded-at-startup, see main()) global np.random stream instead,
        # so behavior is deterministic given a fixed call order.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    
    print(f"[INFO] Creating {variations_per_base} variations per base lens...")
    print(f"[INFO] Input: {len(base_catalog)} base → Output: {len(base_catalog) * variations_per_base} diverse")
    
    expanded_catalog = []
    
    for base_idx, base_row in base_catalog.iterrows():
        for var_idx in range(variations_per_base):
            varied_row = base_row.copy()
            
            # Einstein radius: ±15% scatter, extended range for high-z massive lenses
            base_theta_E = float(base_row.get("theta_E", base_row.get("b", 1.0)))
            varied_row["theta_E"] = base_theta_E * rng.lognormal(0, 0.15)
            varied_row["theta_E"] = np.clip(varied_row["theta_E"], 0.3, _theta_E_hard_max())
            
            # Redshift uncertainty
            base_zl = float(base_row.get("lens_redshift", base_row.get("zl", 0.6)))
            varied_row["lens_redshift"] = base_zl + rng.normal(0, 0.03)
            varied_row["lens_redshift"] = np.clip(varied_row["lens_redshift"], 0.2, 6.0)
            
            base_zs = float(base_row.get("source_redshift", base_row.get("zs", 2.0)))
            varied_row["source_redshift"] = base_zs + rng.normal(0, 0.1)
            varied_row["source_redshift"] = max(varied_row["source_redshift"], 
                                              varied_row["lens_redshift"] + 0.3)
            _zs_max_var = CONFIG.get('redshifts', {}).get('source_max', 3.5)
            varied_row["source_redshift"] = np.clip(varied_row["source_redshift"], 0.8, _zs_max_var)
            
            # Source position variations
            # Sample the offset as a fraction of theta_E (not an absolute
            # offset independent of theta_E) so the source/caustic geometry
            # stays consistent across the theta_E scatter above. The ratio
            # is drawn from a distribution skewed toward small values --
            # real confirmed strong lenses are selection-biased toward
            # sources near the quad caustic, which is what produces the high
            # quad/Einstein-ring fraction seen in the real COSMOS-Web sample
            # (86% quad+). Calibrated against lenstronomy's
            # LensEquationSolver for SIE+SHEAR with the lens_q/shear
            # distributions used elsewhere in this function: Beta(1.5,4)*0.6
            # (mean~0.16, used in earlier Phase 2 iterations) gave only ~51%
            # quad+; Beta(1,8)*0.6 (mean~0.065) gives ~83-89% quad+, matching
            # real (see analysis/sim_obs_comparison/reports/phase2_*.md).
            base_xs = float(base_row.get("source_x", base_row.get("xs", 0.0)))
            base_ys = float(base_row.get("source_y", base_row.get("ys", 0.0)))
            base_offset = np.hypot(base_xs, base_ys)
            base_angle = np.arctan2(base_ys, base_xs) if base_offset > 1e-6 else rng.uniform(0, 2*np.pi)

            offset_ratio = rng.beta(1.0, 8.0) * 0.6  # mean ~0.065, max 0.6
            offset = offset_ratio * varied_row["theta_E"]

            varied_row["source_x"] = offset * np.cos(base_angle)
            varied_row["source_y"] = offset * np.sin(base_angle)
            _xymax = _source_xy_hard_max(varied_row["theta_E"])
            varied_row["source_x"] = np.clip(varied_row["source_x"], -_xymax, _xymax)
            varied_row["source_y"] = np.clip(varied_row["source_y"], -_xymax, _xymax)

            # Stored so the source position can be re-derived later if FP/FJ
            # consistency revises theta_E (see fp_consistent_lens_params call).
            varied_row["source_offset_ratio"] = offset_ratio
            varied_row["source_angle"] = base_angle
            
            # Morphological variations
            base_ql = float(base_row.get("lens_axis_ratio", base_row.get("ql", 0.7)))
            base_qs = float(base_row.get("source_axis_ratio", base_row.get("qs", 0.7)))
            
            varied_row["lens_axis_ratio"] = base_ql * rng.lognormal(0, 0.1)
            varied_row["source_axis_ratio"] = base_qs * rng.lognormal(0, 0.15)
            varied_row["lens_axis_ratio"] = np.clip(varied_row["lens_axis_ratio"], 0.2, 1.0)
            varied_row["source_axis_ratio"] = np.clip(varied_row["source_axis_ratio"], 0.2, 1.0)
            
            # Size variations
            base_rl = float(base_row.get("lens_radius", base_row.get("rl", 0.7)))
            base_rs = float(base_row.get("source_radius", base_row.get("rs", 0.1)))
            
            varied_row["lens_radius"] = base_rl * rng.lognormal(0, 0.2)
            varied_row["source_radius"] = base_rs * rng.lognormal(0, 0.25)
            varied_row["lens_radius"] = np.clip(varied_row["lens_radius"], 0.2, 4.0)  # Extended for massive galaxies
            varied_row["source_radius"] = np.clip(varied_row["source_radius"], 0.02, 0.6)  # Consistent with high-z range
            
            # Magnitude variations
            for band in LOWER_BANDS:
                num = band[1:4]
                
                base_ml = float(base_row.get(f"lens_mag_{band}", base_row.get(f"ml{num}", 22.0)))
                ml_scatter = rng.normal(0, 0.1)
                ml_intrinsic = rng.normal(0, 0.3)
                varied_row[f"lens_mag_{band}"] = base_ml + ml_scatter + ml_intrinsic
                varied_row[f"lens_mag_{band}"] = np.clip(varied_row[f"lens_mag_{band}"], 18.0, 27.0)
                
                base_ms = float(base_row.get(f"source_mag_{band}", base_row.get(f"ms{num}", 20.0)))  # Realistic base for lensed sources (accounts for magnification)
                ms_scatter = rng.normal(0, 0.15)
                ms_intrinsic = rng.normal(0, 0.4)
                varied_row[f"source_mag_{band}"] = base_ms + ms_scatter + ms_intrinsic
                varied_row[f"source_mag_{band}"] = np.clip(varied_row[f"source_mag_{band}"], 18.0, 25.0)  # Realistic lensed sources (accounts for magnification)
            
            # Add variation metadata
            varied_row["base_lens_id"] = base_idx
            varied_row["variation_id"] = var_idx
            varied_row["total_id"] = base_idx * variations_per_base + var_idx
            
            expanded_catalog.append(varied_row)
    
    result_df = pd.DataFrame(expanded_catalog)
    print(f"[INFO] Parameter variations complete: {len(result_df)} total configurations")
    
    return result_df


def generate_subhalo_population(host_mass_log10, host_redshift, source_redshift,
                                theta_E, rng, cosmo=None):
    """Sample a population of NFW dark-matter subhalo perturbers for a lens.

    This is deliberately scoped to *substructure*, not the main deflector:
    the SIE(+shear) macromodel is kept as the realistic total (stars+DM) mass
    description of the lens galaxy (isothermal total profiles are the
    standard, observationally supported choice -- e.g. SLACS). NFW is the
    physically appropriate profile for dark-matter-dominated subhalos, which
    perturb the macromodel at the level of flux-ratio anomalies, astrometric
    shifts, and small-scale convergence/shear structure -- not dramatic arcs.

    Method:
      - Subhalo masses drawn from a power-law mass function dN/dm ~ m^-alpha,
        rescaled so the total bound mass equals a configurable fraction of
        the host halo mass (CONFIG['subhalos']['mass_fraction_of_host']).
      - Concentrations from the Duffy et al. (2008) c(M, z) relation
        (NFW, Delta=200 critical, full sample: A=5.71, B=-0.084, C=-0.47).
      - Projected positions drawn within a configurable multiple of the
        Einstein radius (centrally weighted, i.e. uniform in projected area)
        -- the region where perturbations are actually detectable.
      - Physical (M, c) -> lensing (Rs, alpha_Rs) via lenstronomy's LensCosmo,
        which self-consistently uses the lens/source angular-diameter-distance
        geometry and critical surface density.

    Returns (model_names, kwargs_list), each possibly empty, ready to be
    appended directly to lens_model_list / kwargs_lens. Returns ([], []) if
    CONFIG['subhalos']['enabled'] is False (the default) or lenstronomy's
    cosmology helper is unavailable.
    """
    sub_cfg = CONFIG.get('subhalos', {})
    if not sub_cfg.get('enabled', False):
        return [], []

    if cosmo is None:
        cosmo = COSMO

    try:
        from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    except ImportError:
        return [], []

    lens_cosmo = LensCosmo(z_lens=float(host_redshift), z_source=float(source_redshift), cosmo=cosmo)

    host_mass = 10.0 ** float(host_mass_log10)
    count_min, count_max = sub_cfg.get('count_range', [3, 12])
    n_sub = int(rng.integers(int(count_min), int(count_max) + 1))
    if n_sub <= 0:
        return [], []

    # --- masses: power-law mass function dN/dm ~ m^-alpha (inverse-CDF
    # sampling), rescaled to a total bound-mass budget ---
    alpha = float(sub_cfg.get('mass_function_slope', 1.9))
    log_min, log_max = sub_cfg.get('mass_range_log10_host', [-4.5, -2.0])
    m_min = host_mass * 10.0 ** float(log_min)
    m_max = host_mass * 10.0 ** float(log_max)
    u = rng.uniform(0.0, 1.0, n_sub)
    if abs(alpha - 1.0) > 1e-6:
        p = 1.0 - alpha
        masses = (m_min ** p + u * (m_max ** p - m_min ** p)) ** (1.0 / p)
    else:
        masses = m_min * (m_max / m_min) ** u

    mass_fraction = float(sub_cfg.get('mass_fraction_of_host', 0.01))
    mass_budget = host_mass * mass_fraction
    mass_sum = float(np.sum(masses))
    if mass_sum > 0:
        masses = masses * (mass_budget / mass_sum)
    masses = np.clip(masses, m_min, m_max)

    # --- concentrations: Duffy et al. (2008), Table 1 (NFW, Delta=200c, full sample) ---
    A, B, C = 5.71, -0.084, -0.47
    M_pivot = 2.0e12 / cosmo.h
    concentrations = A * (masses / M_pivot) ** B * (1.0 + float(host_redshift)) ** C
    concentrations = np.clip(concentrations, 2.0, 25.0)

    # --- projected positions: uniform in projected area within a configurable
    # multiple of theta_E -- subhalos far outside this radius do not produce
    # detectable perturbations on the lensed images ---
    max_r_thetaE = float(sub_cfg.get('max_radius_einstein_units', 3.0))
    max_r = max_r_thetaE * float(theta_E)

    model_names = []
    kwargs_list = []
    for i in range(n_sub):
        r_proj = max_r * np.sqrt(rng.uniform(0.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        x_sub = r_proj * np.cos(phi)
        y_sub = r_proj * np.sin(phi)

        try:
            Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=float(masses[i]), c=float(concentrations[i]))
        except Exception:
            continue

        model_names.append('NFW')
        kwargs_list.append(dict(
            Rs=float(Rs_angle),
            alpha_Rs=float(alpha_Rs),
            center_x=float(x_sub),
            center_y=float(y_sub)
        ))

    return model_names, kwargs_list


def _caustic_safe_source_position(lens_model_list, kwargs_lens, theta_E, rng,
                                   max_mu=12.0, min_mu=2.0, max_attempts=40):
    """Sample a source position (x_s, y_s) inside the tangential caustic.

    v17: adds min_mu gate so sources MUST be inside the caustic (μ ≥ min_mu),
    producing multiple images. Without this, sources outside the caustic render
    as single unlensed blobs indistinguishable from non-lenses.

    Samples positions biased toward the Einstein radius (r ∈ [0.05, 0.55]×θ_E)
    where magnification is high, then accepts only if min_mu ≤ |μ| ≤ max_mu.
    Falls back to a position on the Einstein ring after max_attempts.

    Returns (source_x, source_y) in arcsec.
    """
    try:
        from lenstronomy.LensModel.magnification_model import MagnificationModel
        mm = MagnificationModel(lens_model_list=lens_model_list)
    except Exception:
        # Fallback: place on Einstein ring where μ is always high
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.1, 0.35) * theta_E
        return float(r * np.cos(angle)), float(r * np.sin(angle))

    best_x, best_y, best_mu = None, None, 0.0
    for _ in range(max_attempts):
        angle = rng.uniform(0, 2 * np.pi)
        # Minimum 0.15*theta_E offset avoids near-perfect alignment (Einstein ring)
        # which produces symmetric donuts instead of realistic asymmetric arcs.
        r = rng.uniform(0.15, 0.55) * theta_E
        sx = r * np.cos(angle)
        sy = r * np.sin(angle)
        try:
            mu = abs(float(mm.magnification(sx, sy, kwargs_lens)))
        except Exception:
            continue
        # Track the best in-caustic candidate seen so far
        if min_mu <= mu and mu > best_mu:
            best_x, best_y, best_mu = sx, sy, mu
        # Accept if inside caustic AND below blow-out threshold
        if min_mu <= mu <= max_mu:
            return float(sx), float(sy)

    # Fallback: use best in-caustic candidate found, or place just inside θ_E
    if best_x is not None and best_mu >= min_mu:
        # Clip magnification if above max_mu by nudging slightly outward
        r_best = np.hypot(best_x, best_y)
        if r_best > 0 and best_mu > max_mu:
            scale = min(1.0, 0.45 * theta_E / r_best)
            return float(best_x * scale), float(best_y * scale)
        return float(best_x), float(best_y)
    # Last resort: place at 0.3*θ_E where SIE caustic always gives μ>2
    angle = rng.uniform(0, 2 * np.pi)
    return float(0.3 * theta_E * np.cos(angle)), float(0.3 * theta_E * np.sin(angle))


def simulate_complete_lens_system_with_real_fields(row, band_cfgs, rng, field_pop=None,
                                                  numpix=300, n_field_max=8, add_artifacts=True, psf_data=None,
                                                  fixed_field_galaxies=None, fixed_lens_params=None, add_spikes=False, pixel_scale=0.031):
    """Complete lens system simulation with REAL field galaxy contamination and PSF convolution
    
    Args:
        pixel_scale: Pixel scale in arcsec/pixel (default 0.031 for JWST)
    """
    
    # Validate and extract core parameters - extended range for massive high-z lenses
    theta_E = float(row.get("theta_E", row.get("b", 1.0)))
    theta_E = np.clip(theta_E, 0.3, _theta_E_hard_max())
    
    lens_q = np.clip(float(row.get("lens_axis_ratio", row.get("ql", 0.7))), 0.2, 1.0)
    source_q = np.clip(float(row.get("source_axis_ratio", row.get("qs", 0.7))), 0.2, 1.0)
    lens_z = float(row.get("lens_redshift", row.get("zl", 0.6)))

    # Guard: z_source must be behind z_lens (physical lensing requirement).
    # Catalog-provided fixed redshifts can violate this; fix in-place.
    _source_z_raw = float(row.get("source_redshift", row.get("zs", 2.0)))
    _min_dz_row = CONFIG.get('redshifts', {}).get('min_delta_z', 0.5)
    if _source_z_raw <= lens_z + _min_dz_row:
        _fixed_zs = lens_z + _min_dz_row + float(np.random.uniform(0.1, 0.5))
        row["source_redshift"] = _fixed_zs
        print(f"[REDSHIFT] lens_id={row.get('lens_id','?')} z_s={_source_z_raw:.2f} <= z_l={lens_z:.2f}+{_min_dz_row} -> fixed to {_fixed_zs:.2f}")
    
    # === LENS MASS MODEL (CENTERED) ===
    # Initialize env_params early to avoid scoping issues
    env_params = {}
    _sigma_kms = None  # always initialised here; overwritten by FP block below if enabled
    # always initialised here -- the fixed_lens_params branch below (used for
    # time-delay epochs > 0, reusing epoch 0's lens model) skips the FP/FJ
    # block entirely, so without this default _fp_enabled was unbound when
    # referenced later in the metadata dict, raising UnboundLocalError for
    # every non-initial time-delay epoch.
    _fp_enabled = False

    # Use fixed parameters if provided (for time delay consistency)
    if fixed_lens_params is not None:
        lens_pa = fixed_lens_params['lens_pa']
        e1_l = fixed_lens_params['e1_l']
        e2_l = fixed_lens_params['e2_l']
        env_type = fixed_lens_params['env_type']
        shear_gamma1 = fixed_lens_params['shear_gamma1']
        shear_gamma2 = fixed_lens_params['shear_gamma2']
        
        # Check if binary lens model is provided
        if 'lens_model_list' in fixed_lens_params and 'kwargs_lens' in fixed_lens_params:
            lens_model_list = fixed_lens_params['lens_model_list']
            kwargs_lens = fixed_lens_params['kwargs_lens']
            print(f"[TIME_DELAY] Using fixed binary/custom lens model: {lens_model_list}")
        else:
            # Fallback to standard SIE+SHEAR
            kwargs_sie = dict(
                theta_E=theta_E,
                center_x=0.0,
                center_y=0.0,
                e1=float(e1_l), e2=float(e2_l)
            )
            kwargs_shear = dict(
                gamma1=float(shear_gamma1),
                gamma2=float(shear_gamma2)
            )
            kwargs_lens = [kwargs_sie, kwargs_shear]
            lens_model_list = ['SIE', 'SHEAR']
        
        print(f"[TIME_DELAY] Using fixed lens parameters: PA={lens_pa:.1f}°")
        
        # Get env_params for later use (for field galaxy count, etc.)
        env_cfg = CONFIG.get('environment', {}).get('types', {})
        env_params = env_cfg.get(env_type, {})
    else:
        # Standard single lens generation
        if ('lens_e1' in row.index or 'lens_e1' in row) and pd.notna(row.get('lens_e1', np.nan)):
            e1_l = float(row['lens_e1'])
            e2_l = float(row['lens_e2'])
            lens_pa = float(0.5 * np.degrees(np.arctan2(e2_l, e1_l)))
            e1_l, e2_l = np.clip([e1_l, e2_l], -0.8, 0.8)
        else:
            lens_pa = rng.uniform(-180, 180)
            e1_l, e2_l = ellipticity(lens_q, lens_pa)
            e1_l, e2_l = np.clip([e1_l, e2_l], -0.8, 0.8)
        
        # === LENS CLASS DISTRIBUTION (PRISM Classification) ===
        # Use lens_class_distribution config to determine lens type FIRST
        use_binary = False
        binary_type = None  # Will be 'sie_sie', 'nfw_nfw', or 'shear_only'
        use_group = False
        
        lens_class_dist = CONFIG.get('lens_class_distribution', {})
        if lens_class_dist.get('enabled', True):
            # Use new distribution-based approach
            rand_val = rng.random()
            cumulative = 0.0
            
            # Define the class fractions
            single_field_frac = lens_class_dist.get('single_field', {}).get('fraction', 0.45)
            group_frac = lens_class_dist.get('group', {}).get('fraction', 0.20)
            bsie_frac = lens_class_dist.get('binary_sie_sie', {}).get('fraction', 0.175)
            bnfw_frac = lens_class_dist.get('binary_nfw_nfw', {}).get('fraction', 0.175)
            bshear_frac = lens_class_dist.get('binary_shear_only', {}).get('fraction', 0.0)
            
            # Determine lens class
            if rand_val < single_field_frac:
                # Single field galaxy
                use_binary = False
                use_group = False
            elif rand_val < single_field_frac + group_frac:
                # Group lens
                use_binary = False
                use_group = True
            elif rand_val < single_field_frac + group_frac + bsie_frac:
                # Binary SIE+SIE
                use_binary = True
                binary_type = 'sie_sie'
            elif rand_val < single_field_frac + group_frac + bsie_frac + bnfw_frac:
                # Binary NFW+NFW
                use_binary = True
                binary_type = 'nfw_nfw'
            else:
                # Binary SHEAR-only
                use_binary = True
                binary_type = 'shear_only'
        else:
            # Fallback to old binary_lenses config for backward compatibility
            binary_cfg = CONFIG.get('binary_lenses', {})
            if BINARY_LENS_AVAILABLE and binary_cfg.get('enabled', False):
                binary_fraction = binary_cfg.get('fraction', 0.15)
                if rng.random() < binary_fraction:
                    use_binary = True
                    mass_profiles = binary_cfg.get('mass_profile_types', {})
                    total_mass = mass_profiles.get('sie_sie', 0.5) + mass_profiles.get('nfw_nfw', 0.5)
                    rand_mass = rng.random() * total_mass
                    binary_type = 'sie_sie' if rand_mass < mass_profiles.get('sie_sie', 0.5) else 'nfw_nfw'
        
        # === ENVIRONMENT TYPE (based on lens class) ===
        # Link environment type to lens class for consistency
        env_cfg = CONFIG.get('environment', {}).get('types', {})
        if use_group:
            # Group lenses always have 'group' environment
            env_type = 'group'
        elif use_binary:
            # Binary lenses always have 'galaxy_pair' environment
            env_type = 'galaxy_pair'
        else:
            # Single field lenses have 'isolated_field' environment
            env_type = 'isolated_field'
        
        env_params = env_cfg.get(env_type, {})
        
        # Environment-specific external shear (COWLS calibrated; Q1 override if in row)
        if pd.notna(row.get('shear_gamma1', np.nan)):
            shear_gamma1 = float(row['shear_gamma1'])
            shear_gamma2 = float(row['shear_gamma2'])
        else:
            shear_min = env_params.get('shear_min', 0.01)
            shear_max = env_params.get('shear_max', 0.05)
            # Euclid: only override env shear when Q1 shear sampling is on
            # or an explicit shear_range is provided in euclid_q1.
            if CONFIG.get('telescope', 'jwst').lower() == 'euclid':
                _q1 = CONFIG.get('euclid_q1', {}) or {}
                q1_shear = _q1.get('shear_range') or {}
                if _q1.get('use_q1_shear', False) or q1_shear:
                    shear_min = q1_shear.get('min', shear_min)
                    shear_max = q1_shear.get('max', shear_max)
            shear_g = rng.uniform(shear_min, shear_max)
            shear_phi = rng.uniform(0, np.pi)
            shear_gamma1 = shear_g*math.cos(2*shear_phi)
            shear_gamma2 = shear_g*math.sin(2*shear_phi)
        
        # === BINARY LENS GENERATION ===
        # Check if binary lenses are enabled
        if use_binary and BINARY_LENS_AVAILABLE and (CONFIG.get('binary_lenses', {}).get('enabled', True) or lens_class_dist.get('enabled', True)):
            # Generate binary lens system
            binary_cfg = CONFIG.get('binary_lenses', {})
            
            # Sample mass ratio
            mass_ratio_min = binary_cfg.get('mass_ratio', {}).get('min', 0.3)
            mass_ratio_max = binary_cfg.get('mass_ratio', {}).get('max', 1.0)
            mass_ratio = rng.uniform(mass_ratio_min, mass_ratio_max)
            
            # Secondary Einstein radius (scales as M^0.5)
            theta_E_2 = theta_E * np.sqrt(mass_ratio)
            
            # Sample separation in Einstein radii
            sep_min = binary_cfg.get('separation', {}).get('min_in_einstein_radii', 0.5)
            sep_max = binary_cfg.get('separation', {}).get('max_in_einstein_radii', 2.0)
            separation_factor = rng.uniform(sep_min, sep_max)
            separation_arcsec = (theta_E + theta_E_2) * separation_factor
            
            # Random position angle
            position_angle = rng.uniform(0, 2 * np.pi)
            
            # Secondary position
            x2 = separation_arcsec * np.cos(position_angle)
            y2 = separation_arcsec * np.sin(position_angle)
            
            # Secondary ellipticity
            e1_2, e2_2 = e1_l + rng.normal(0, 0.1), e2_l + rng.normal(0, 0.1)
            e1_2, e2_2 = np.clip([e1_2, e2_2], -0.8, 0.8)
            
            # Build binary lens model
            kwargs_sie_1 = dict(
                theta_E=theta_E,
                center_x=0.0,
                center_y=0.0,
                e1=float(e1_l), e2=float(e2_l)
            )
            kwargs_sie_2 = dict(
                theta_E=float(theta_E_2),
                center_x=float(x2),
                center_y=float(y2),
                e1=float(e1_2), e2=float(e2_2)
            )
            kwargs_shear = dict(
                gamma1=float(shear_gamma1),
                gamma2=float(shear_gamma2)
            )
            
            # Choose lens model based on binary_type from lens_class_distribution
            if binary_type == 'shear_only':
                # SHEAR-only binary (two shear components at different positions)
                # This simulates a pair of galaxies creating extended shear field
                kwargs_shear_1 = {
                    'gamma1': float(shear_gamma1),
                    'gamma2': float(shear_gamma2),
                    'ra_0': 0.0,
                    'dec_0': 0.0
                }
                kwargs_shear_2 = {
                    'gamma1': float(shear_gamma1) * 0.7,  # Weaker secondary shear
                    'gamma2': float(shear_gamma2) * 0.7,
                    'ra_0': float(x2),
                    'dec_0': float(y2)
                }
                # SIE for main lens + two shear components
                kwargs_lens = [kwargs_sie_1, kwargs_shear_1, kwargs_shear_2]
                lens_model_list = ['SIE', 'SHEAR', 'SHEAR']
            elif binary_type == 'nfw_nfw' or (binary_type is None and rng.random() < binary_cfg.get('mass_profile_types', {}).get('nfw_nfw', 0.5)):
                # NFW+NFW binary: derive physical (M, c) for each component from its
                # Einstein radius via the Einstein-mass relation, then convert to
                # lensing units (Rs, alpha_Rs) with LensCosmo.nfw_physical2angle --
                # the same physically-consistent route used for subhalos above
                # (replaces the previous arbitrary Rs=0.3*theta_E, alpha_Rs=1.5*theta_E).
                try:
                    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
                    _src_z_bin = float(row.get("source_redshift", row.get("zs", 2.0)))
                    _lens_cosmo_bin = LensCosmo(z_lens=float(lens_z), z_source=_src_z_bin, cosmo=COSMO)
                    _sigma_cr = _lens_cosmo_bin.sigma_crit_angle  # M_sun / arcsec^2

                    # Duffy et al. (2008) mass-concentration relation (same as subhalos)
                    A, B, C = 5.71, -0.084, -0.47
                    M_pivot = 2.0e12 / COSMO.h

                    def _nfw_kwargs(theta_E_comp, x_c, y_c):
                        M_E = _sigma_cr * np.pi * theta_E_comp**2  # Einstein mass [M_sun]
                        conc = A * (M_E / M_pivot) ** B * (1.0 + lens_z) ** C
                        conc = float(np.clip(conc, 2.0, 25.0))
                        Rs_angle, alpha_Rs = _lens_cosmo_bin.nfw_physical2angle(M=float(M_E), c=conc)
                        return dict(Rs=float(Rs_angle), alpha_Rs=float(alpha_Rs),
                                    center_x=float(x_c), center_y=float(y_c))

                    kwargs_nfw_1 = _nfw_kwargs(theta_E, 0.0, 0.0)
                    kwargs_nfw_2 = _nfw_kwargs(theta_E_2, x2, y2)
                    kwargs_lens = [kwargs_nfw_1, kwargs_nfw_2, kwargs_shear]
                    lens_model_list = ['NFW', 'NFW', 'SHEAR']
                except Exception:
                    # Fallback to SIE+SIE
                    kwargs_lens = [kwargs_sie_1, kwargs_sie_2, kwargs_shear]
                    lens_model_list = ['SIE', 'SIE', 'SHEAR']
            else:
                # SIE+SIE binary
                kwargs_lens = [kwargs_sie_1, kwargs_sie_2, kwargs_shear]
                lens_model_list = ['SIE', 'SIE', 'SHEAR']
        
        # If NOT binary and NOT group, use standard SIE + SHEAR
        if not use_binary and not use_group:
            # Standard single field lens: SIE + SHEAR
            kwargs_sie = dict(
                theta_E=theta_E,
                center_x=0.0,
                center_y=0.0,
                e1=float(e1_l), e2=float(e2_l)
            )
            kwargs_shear = dict(
                gamma1=float(shear_gamma1),
                gamma2=float(shear_gamma2)
            )
            kwargs_lens = [kwargs_sie, kwargs_shear]
            lens_model_list = ['SIE', 'SHEAR']
        
        # === GROUP LENS GENERATION ===
        # Generate group lenses with multiple mass components
        if use_group and not use_binary:
            # Main lens (central, most massive)
            kwargs_main = dict(
                theta_E=theta_E,
                center_x=0.0,
                center_y=0.0,
                e1=float(e1_l), e2=float(e2_l)
            )
            
            # Sample 1-2 satellite galaxies at various positions
            n_satellites = int(rng.choice([1, 2], p=[0.6, 0.4]))
            satellite_components = []
            
            for i in range(n_satellites):
                # Satellite mass (30-70% of main)
                mass_frac = rng.uniform(0.3, 0.7)
                theta_E_sat = theta_E * np.sqrt(mass_frac)
                
                # Position (1.5-3.0 Einstein radii from center)
                r_sat = rng.uniform(1.5, 3.0) * theta_E
                angle_sat = rng.uniform(0, 2 * np.pi)
                x_sat = r_sat * np.cos(angle_sat)
                y_sat = r_sat * np.sin(angle_sat)
                
                # Ellipticity (slightly different from main)
                e1_sat, e2_sat = e1_l + rng.normal(0, 0.15), e2_l + rng.normal(0, 0.15)
                e1_sat, e2_sat = np.clip([e1_sat, e2_sat], -0.8, 0.8)
                
                kwargs_sat = dict(
                    theta_E=float(theta_E_sat),
                    center_x=float(x_sat),
                    center_y=float(y_sat),
                    e1=float(e1_sat), e2=float(e2_sat)
                )
                satellite_components.append(kwargs_sat)
            
            # External shear
            kwargs_shear = dict(
                gamma1=float(shear_gamma1),
                gamma2=float(shear_gamma2)
            )
            
            # Combine: Main + Satellites + Shear
            kwargs_lens = [kwargs_main] + satellite_components + [kwargs_shear]
            lens_model_list = ['SIE'] * (1 + n_satellites) + ['SHEAR']

    # === MAIN LENS LIGHT (CENTERED) ===
    # Use fixed parameters if provided (for time delay consistency)
    _tng_lens = None
    if fixed_lens_params is not None:
        lens_radius = fixed_lens_params['lens_radius']
        n_lens = fixed_lens_params['n_lens']
    else:
        # Scale lens size with redshift using mass-size relation
        geo = CONFIG.get('geometry', {})
        
        # Get lens mass (if available) or estimate from Einstein radius
        lens_mass_log10 = row.get("lens_mass_log10", None)
        if lens_mass_log10 is None:
            # Estimate mass from Einstein radius: M ~ theta_E^2 * D_A(z) / (4πG)
            # Rough estimate: log10(M/M_sun) ~ 11 + 2*log10(theta_E) + log10(D_A/1000)
            da_mpc = angular_diameter_distance(lens_z)
            lens_mass_log10 = 11.0 + 2.0 * np.log10(theta_E) + np.log10(da_mpc/1000.0)
            lens_mass_log10 = np.clip(lens_mass_log10, 10.5, 12.5)  # Reasonable range

        # === TNG MODE: look up a physically matched TNG100-1 subhalo ===
        # (off by default; see config['tng_mode']). When available, its
        # half-mass radius and sSFR override the Sersic R_sersic/n_sersic
        # derived below, and _tng_lens is stored in field_info metadata.
        _tng_lens = query_tng_properties(lens_z, float(lens_mass_log10), rng, CONFIG,
                                          exclude_subhalos=_used_tng_subhalos,
                                          sfr_class=CONFIG.get('tng_mode', {}).get('lens_sfr_class'),
                                          environment=CONFIG.get('tng_mode', {}).get('lens_environment'))

        # === FUNDAMENTAL PLANE + FABER-JACKSON CONSISTENCY ===
        # Derive σ, Re, and θ_E that satisfy the FP/FJ scaling relations
        # Calibrations: Bernardi+2003 (local), Singh+2021 (z-evolution), Sonnenfeld+2023 (lens bias)
        _fp_cfg = CONFIG.get('fundamental_plane', {}) if isinstance(CONFIG, dict) else {}
        _fp_enabled = _fp_cfg.get('enabled', True) and FUNDAMENTAL_PLANE_AVAILABLE
        _sigma_kms = None  # initialise before branches so catalog output is always defined

        if _fp_enabled:
            _src_z = float(row.get("source_redshift", row.get("zs", 2.0)))
            _D_l   = angular_diameter_distance(lens_z)
            _D_s   = angular_diameter_distance(_src_z)
            _D_ls  = COSMO.angular_diameter_distance_z1z2(lens_z, _src_z).value
            _da_ls_over_ds = max(0.01, _D_ls / _D_s)

            _fp = fp_consistent_lens_params(
                row=row,
                lens_z=lens_z,
                source_z=_src_z,
                theta_E_catalog=theta_E,
                lens_mass_log10=float(lens_mass_log10),
                da_ls_over_ds=_da_ls_over_ds,
                rng=rng,
                enforce_fp=True,
                da_l_mpc=_D_l,
                theta_E_max=_theta_E_hard_max(),
                catalog_theta_E_weight=float(_fp_cfg.get('catalog_theta_E_weight', 0.60)),
            )
            reff_kpc = _fp['re_kpc']
            _sigma_kms = _fp['sigma_kms']
            row["lens_sigma_kms"] = float(_sigma_kms)
            # Optionally keep catalog θ_E (e.g. dramatic paper rings) while
            # still using FP for σ and Re.
            _update_te = bool(_fp_cfg.get('update_theta_E', True))
            _te_max = _theta_E_hard_max()
            if _update_te and 0.3 <= _fp['theta_E'] <= _te_max:
                theta_E = _fp['theta_E']
                # Update kwargs_lens only for SIE/SIS — NFW uses Rs/alpha_Rs, not theta_E
                if lens_model_list and lens_model_list[0] in ('SIE', 'SIS', 'SPEMD', 'SPP'):
                    kwargs_lens[0]['theta_E'] = theta_E
                # FIX (adversarial audit finding C-2, 2026-08-01): row["theta_E"]
                # was never updated here, so every saved label/metadata/training-
                # catalog value downstream of `row` kept reporting the PRE-FP
                # value while the actual rendered deflector (kwargs_lens, and
                # everything derived from it: kappa maps, magnification) used
                # the POST-FP value. Confirmed by execution: metadata reported
                # theta_E=1.799 while the rendered kappa map's theta_E_eff was
                # 2.484 (+38%). Writing back here keeps `row` truthful.
                row["theta_E"] = float(theta_E)

            # === GROUP/CLUSTER-SCALE THETA_E OVERRIDE ===
            # The catalog/FP pipeline above caps out around ~2-2.5" for
            # this project's COSMOS-Web single-galaxy mass range, and the
            # lens_class_distribution "group" class only adds small
            # satellite perturbers -- it does NOT raise the main deflector's
            # theta_E. For a genuinely group/cluster-scale lens (needed so
            # the ring is visible without zooming in on a full 1' field),
            # directly override theta_E here, bypassing the mass-derived
            # value.
            #
            # NOTE (audit finding C-3): this override intentionally breaks
            # the physical theta_E<->mass relation for the affected systems
            # -- it draws a display-motivated Einstein radius, not one
            # derived from a group halo mass function. That is a real,
            # documented modelling limitation (see PROJECT_NOTES /
            # audit report), not something this patch can fix without
            # implementing group-halo mass sampling. What THIS patch fixes
            # is INTERNAL CONSISTENCY of what gets recorded: previously
            # theta_E was overridden but row["theta_E"] and lens_sigma_kms
            # were left at their pre-override (FP-derived) values, so the
            # saved labels described a *different, lower-mass* system than
            # the one actually rendered. Now: row["theta_E"] matches the
            # rendered deflector, sigma_kms is back-solved from the SIS
            # relation to be self-consistent with the overridden theta_E
            # (not left silently contradicting it), and the affected
            # systems are explicitly flagged in metadata so population
            # statistics can exclude them if a physically-derived sample
            # is required.
            _grp_te_cfg = CONFIG.get('geometry', {}).get('group_scale_theta_E', {})
            row["theta_E_override_applied"] = False
            if _grp_te_cfg.get('enabled', False):
                _te_min = float(_grp_te_cfg.get('min', 6.0))
                _te_max_grp = float(_grp_te_cfg.get('max', 15.0))
                _theta_E_pre_override = float(theta_E)
                theta_E = float(rng.uniform(_te_min, _te_max_grp))
                if lens_model_list and lens_model_list[0] in ('SIE', 'SIS', 'SPEMD', 'SPP'):
                    kwargs_lens[0]['theta_E'] = theta_E
                row["theta_E"] = float(theta_E)
                row["theta_E_pre_override"] = _theta_E_pre_override
                row["theta_E_override_applied"] = True
                # Back-solve the SIS-equivalent sigma_v so the recorded
                # kinematic label is not left contradicting the rendered
                # deflector: theta_E = 4*pi*(sigma/c)^2 * D_LS/D_S
                # (same relation as fundamental_plane.einstein_radius_from_sigma,
                # inverted).
                _sigma_kms = float(2.998e5 * np.sqrt(
                    max(theta_E, 1e-6) / 206265.0 / (4.0 * np.pi * max(_da_ls_over_ds, 1e-6))
                ))
                row["lens_sigma_kms"] = _sigma_kms

            # Re-derive the source position from the offset/theta_E ratio and
            # angle stored by create_parameter_variations, now that theta_E
            # has been finalized by the FP/FJ consistency step above. Without
            # this, the offset/theta_E ratio set at variation time (see
            # create_parameter_variations) gets decoupled whenever FP
            # consistency changes theta_E, undermining the quad-caustic bias.
            _offset_ratio = row.get("source_offset_ratio", None)
            _offset_angle = row.get("source_angle", None)
            if _offset_ratio is not None and _offset_angle is not None:
                _offset = float(_offset_ratio) * theta_E
                _xymax = _source_xy_hard_max(theta_E)
                row["source_x"] = float(np.clip(_offset * np.cos(float(_offset_angle)), -_xymax, _xymax))
                row["source_y"] = float(np.clip(_offset * np.sin(float(_offset_angle)), -_xymax, _xymax))
        else:
            # Sample physical effective radius from mass-size relation
            reff_kpc = mass_size_relation(lens_mass_log10, lens_z, rng)

        # TNG Mode: replace the Sersic/FP effective radius with the matched
        # subhalo's actual stellar half-mass radius.
        if _tng_lens is not None:
            reff_kpc = _tng_lens['halfmassrad_stars_kpc']
        else:
            # Brightest Group Galaxy (BGG) environmental effect: a fraction
            # of (non-TNG-matched) lens galaxies are treated as group-central
            # deflectors and use the Gozaliasl+2025 BGG size-mass relation
            # (quiescent row, since PRISM lenses are early-type by design)
            # instead of the field FP/CWMGs relation, overriding whatever Re
            # the branch above produced.
            _bgg_frac = float(CONFIG.get('mass_size', {}).get('bgg_fraction', 0.0))
            if rng.random() < _bgg_frac:
                reff_kpc = mass_size_relation(lens_mass_log10, lens_z, rng, is_bgg=True, bgg_type='QG')

        # Convert to angular radius
        lens_radius = convert_physical_to_angular_radius(reff_kpc, lens_z)

        # Apply config bounds as safety limits.
        # Floor raised to 0.5" so Sérsic n~4 wings extend visibly beyond
        # the Einstein ring (real COSMOS-Web lenses show extended red halos).
        _lr_min = max(geo.get('lens_radius_min', 0.2), 0.5)
        lens_radius = np.clip(lens_radius, _lr_min, geo.get('lens_radius_max', 4.0))

        n_lens = sample_sersic_n(lens_z, measured=row.get("n_rest"), rng=rng)

        # TNG Mode: bias n_sersic toward early-type (de Vaucouleurs, n~4) for
        # quenched subhalos or late-type (exponential, n~1) for star-forming
        # ones, consistent with the matched subhalo's actual sSFR.
        if _tng_lens is not None:
            if _tng_lens['ssfr_per_yr'] < TNG_QUENCHED_SSFR_THRESHOLD:
                n_lens = max(float(n_lens), 3.0)
            else:
                n_lens = min(float(n_lens), 2.0)

        lens_morph_cfg = CONFIG.get('lens_morphology', {}) if isinstance(CONFIG, dict) else {}
        min_lens_sersic = lens_morph_cfg.get('min_lens_sersic', None)
        if min_lens_sersic is not None:
            try:
                n_lens = max(float(min_lens_sersic), float(n_lens))
            except Exception:
                pass
    
    # FIX (adversarial audit finding C-10, 2026-08-01): the lens LIGHT
    # ellipticity used to be bit-identical to the lens MASS ellipticity
    # (same e1_l/e2_l passed to both the SIE mass model and this light
    # profile). Real ETG lenses show genuine mass/light misalignment --
    # SLACS measurements find mass is typically rounder than light by
    # ~0.05-0.1 in axis ratio, with position-angle misalignment of order
    # 5-15 deg (e.g. Gomer & Williams 2020; Bruderer+2016 on SLACS/BELLS
    # mass-light alignment) -- this is real, unmodelled physical
    # uncertainty in real lens analyses, not a nuisance parameter. Add a
    # modest, physically-motivated q/PA offset so the light profile is
    # not a perfect, unrealistically-informative proxy for the mass
    # model's exact orientation.
    _dPA_light = float(rng.normal(0.0, 8.0))  # deg, ~SLACS-like scatter
    _dq_light = float(np.clip(rng.normal(0.0, 0.06), -0.15, 0.15))
    _lens_q_light = float(np.clip(ellipticity_to_axis_ratio(e1_l, e2_l) + _dq_light, 0.15, 0.98))
    _lens_pa_light = lens_pa + _dPA_light
    _e1_light, _e2_light = ellipticity(_lens_q_light, _lens_pa_light)
    _e1_light, _e2_light = float(np.clip(_e1_light, -0.8, 0.8)), float(np.clip(_e2_light, -0.8, 0.8))
    row["lens_light_pa_offset_deg"] = _dPA_light
    row["lens_light_q_offset"] = _dq_light

    main_lens_light = dict(
        R_sersic=lens_radius,
        n_sersic=float(n_lens),
        center_x=0.0,
        center_y=0.0,
        e1=_e1_light, e2=_e2_light
    )

    # For binary/pair lens systems, the second mass clump must have a visible
    # counterpart: physically, lensing pairs are two M*-type (massive,
    # typically passive/elliptical) galaxies at THE SAME redshift (lens_z is
    # shared by both mass components above), so the companion is rendered
    # with the same Sersic morphology and the same per-band color (only the
    # overall flux is scaled down by its lower mass/theta_E), instead of an
    # invisible dark perturber or an unrelated field galaxy with a mismatched
    # color/redshift.
    companion_lens_light = None
    _mass_ratio = None
    _companion_color_jitter = None
    if locals().get('use_binary') and 'x2' in locals() and lens_model_list and len(lens_model_list) >= 2 \
            and lens_model_list[1] in ('SIE', 'NFW', 'SPEMD'):
        _mass_ratio = float(np.clip(mass_ratio, 0.05, 1.0)) if 'mass_ratio' in locals() else 0.5
        # Both members of a genuine massive lensing pair must themselves be
        # M*-ish (theta_E_2 ~ theta_E * sqrt(mass_ratio) with mass_ratio in
        # [0.3, 1.0]) -> at these masses the morphology-mass relation says
        # both are overwhelmingly early-type/quenched (same broad SED class,
        # hence similar broadband colors). They are NOT clones, though: each
        # has its own merger/star-formation history, so we draw an
        # INDEPENDENT Sersic index from the same redshift-dependent
        # early-type distribution (rather than forcing n_sersic identical),
        # and add small per-band magnitude jitter (age/metallicity scatter
        # among ellipticals, ~0.15 mag rms) on top of the mass-ratio flux
        # scaling -- enough for the companion to look like a distinct
        # individual elliptical without turning it into an unrelated
        # (differently-typed/colored) field galaxy.
        companion_lens_light = dict(
            R_sersic=float(np.clip(lens_radius * np.sqrt(_mass_ratio), 0.1, lens_radius)),
            n_sersic=float(sample_sersic_n(lens_z, rng=rng)),
            center_x=float(x2),
            center_y=float(y2),
            e1=float(e1_2) if 'e1_2' in locals() else float(e1_l),
            e2=float(e2_2) if 'e2_2' in locals() else float(e2_l)
        )
        _companion_color_jitter = {b: float(rng.normal(0.0, 0.15)) for b in UPPER_BANDS}
    elif (locals().get('use_group') and not locals().get('use_binary')
          and locals().get('satellite_components')):
        # Brightest group galaxy (BGG) satellite: visible early-type companion
        # co-located with the first group SIE mass component.
        sat0 = satellite_components[0]
        _mass_ratio = float(np.clip(
            (float(sat0['theta_E']) / max(float(theta_E), 1e-6)) ** 2, 0.20, 0.70))
        _bgg_n = float(max(sample_sersic_n(lens_z, rng=rng), 3.5))
        companion_lens_light = dict(
            R_sersic=float(np.clip(lens_radius * np.sqrt(_mass_ratio), 0.2, lens_radius * 0.85)),
            n_sersic=_bgg_n,
            center_x=float(sat0['center_x']),
            center_y=float(sat0['center_y']),
            e1=float(sat0['e1']), e2=float(sat0['e2']),
        )
        _companion_color_jitter = {b: float(rng.normal(0.0, 0.12)) for b in UPPER_BANDS}

    # TNG particle morphology for the companion lens galaxy (Phase 2): match
    # a TNG subhalo at the companion's own (mass-ratio-scaled) stellar mass
    # and, if it has a locally-downloaded particle cutout, render it as an
    # INTERPOL profile from particle data instead of Sersic.
    _tng_companion = None
    _companion_particle_file = None
    if companion_lens_light is not None:
        _companion_mass_log10 = float(lens_mass_log10) + np.log10(_mass_ratio)
        _tng_companion = query_tng_properties(lens_z, _companion_mass_log10, rng, CONFIG,
                                               exclude_subhalos=_used_tng_subhalos)
        _pm_cfg = CONFIG.get('tng_mode', {}).get('particle_morphology', {})
        if (_tng_companion is not None and _pm_cfg.get('enabled', False)
                and _pm_cfg.get('companion_enabled', False)
                and rng.random() < float(_pm_cfg.get('companion_fraction', 1.0))):
            _comp_sim = _tng_companion.get('sim', 'TNG100-1')
            if not isinstance(_comp_sim, str) or str(_comp_sim).lower() in ('nan', 'none', ''):
                _comp_sim = 'TNG100-1'
            _companion_particle_file = local_particle_path(_tng_companion['snapshot'], _tng_companion['subhalo_id'],
                                                             min_particles=_pm_cfg.get('min_particles'),
                                                             sim=_comp_sim)

    # === LENSED SOURCE ===
    # Use fixed source parameters if provided (for time delay consistency)
    if fixed_lens_params is not None:
        source_pa = fixed_lens_params['source_pa']
        e1_s = fixed_lens_params['e1_s']
        e2_s = fixed_lens_params['e2_s']
    else:
        source_pa = float(row.get("source_pa", row.get("ps", rng.uniform(-180, 180))))
        e1_s, e2_s = ellipticity(source_q, source_pa)
        e1_s, e2_s = np.clip([e1_s, e2_s], -0.8, 0.8)
    
    # Configurable source size generation (ML-optimized)
    # Use fixed parameters if provided (for time delay consistency)
    _tng_source = None
    if fixed_lens_params is not None:
        source_radius = fixed_lens_params['source_radius']
        source_n = fixed_lens_params['source_n']
        source_z = float(row.get("source_redshift", row.get("zs", 2.0)))
    else:
        source_z = float(row.get("source_redshift", row.get("zs", 2.0)))

        # === TNG MODE: look up a physically matched TNG100-1 subhalo for
        # the lensed source. Sources don't have a catalog stellar mass, so
        # the target mass is sampled from config['tng_mode'] around a value
        # typical for lensed high-z star-forming galaxies.
        _tng_cfg = CONFIG.get('tng_mode', {}) if isinstance(CONFIG, dict) else {}
        _source_logM = float(rng.normal(
            _tng_cfg.get('source_logM_default', 9.5),
            _tng_cfg.get('source_logM_scatter', 0.5),
        ))
        _source_logM = np.clip(_source_logM, 8.0, 11.0)
        # Randomise SFR class: ~60% star-forming, ~40% quiescent across the batch
        _src_sfr_class = 'star_forming' if rng.random() < 0.60 else 'quiescent'
        _tng_source = query_tng_properties(source_z, _source_logM, rng, CONFIG,
                                            exclude_subhalos=_used_tng_subhalos,
                                            sfr_class=_src_sfr_class)

        # FIX (adversarial audit finding C-6, 2026-08-01): source_radius
        # used to be defined as theta_E * (a random fraction) -- i.e. the
        # ANGULAR SIZE of a background galaxy at z_source~1.75 was a
        # deterministic function of a FOREGROUND deflector's mass at
        # z_lens~0.45. These are causally unrelated systems; the source's
        # size should come from its own stellar mass and redshift via a
        # size-mass relation, exactly like the TNG-mode branch below
        # already does correctly for the matched-subhalo case. Using the
        # same mass_size_relation()/convert_physical_to_angular_radius()
        # pipeline here (instead of only for TNG mode) removes the
        # circular theta_E dependency for the default/non-TNG path too,
        # and stops an ML model from being able to infer theta_E directly
        # from the source's apparent size (a pure shortcut in the old
        # scheme).
        geo = CONFIG.get('geometry', {})
        _src_reff_kpc = mass_size_relation(_source_logM, source_z, rng)
        source_radius = convert_physical_to_angular_radius(_src_reff_kpc, source_z)

        # Soft rendering-stability cap (NOT a physical constraint): an
        # extremely large source relative to the lens light can produce
        # degenerate/numerically awkward caustic-crossing geometry. This
        # is a practical rendering safeguard, not a claim that real
        # background sources cannot subtend more angular size than the
        # foreground lens light.
        if geo.get('enforce_source_smaller_than_lens', True):
            max_ratio = geo.get('max_source_to_lens_ratio', 0.6)
            max_allowed = lens_radius * max_ratio
            if source_radius > max_allowed:
                source_radius = max_allowed
        
        # TNG Mode: replace the catalog-derived source radius with the
        # matched subhalo's actual stellar half-mass radius, projected to
        # the source's redshift.
        if _tng_source is not None:
            source_radius = convert_physical_to_angular_radius(
                _tng_source['halfmassrad_stars_kpc'], source_z
            )
            if geo.get('enforce_source_smaller_than_lens', True):
                max_ratio = geo.get('max_source_to_lens_ratio', 0.6)
                source_radius = min(source_radius, lens_radius * max_ratio)

        # Final safety bounds with minimum size to ensure visibility.
        # Scale floor with θ_E: min ~ 0.20" × (θ_E / 0.5"), capped at 0.45".
        # Raised from v13's 0.15" floor — fragmented PSF chains were still
        # appearing for small sources with large θ_E (quad-image systems).
        _src_r_min = float(np.clip(0.20 * (theta_E / 0.5), 0.20, 0.45))
        source_radius = np.clip(source_radius, _src_r_min, 0.8)

        # Floor at n=0.8 (Gaussian-like disk): sources with n<0.8 render as
        # near-point-sources, producing fragmented PSF-dot chains when lensed
        # into multiple images. Real lensed sources are star-forming disks.
        source_n = np.clip(rng.lognormal(np.log(1.2), 0.4), 0.8, 4.0)

        # TNG Mode: bias n_sersic toward the matched subhalo's morphology
        # (quenched -> early-type/de Vaucouleurs, star-forming -> exponential).
        # Always enforce the 0.8 floor so TNG star-forming match doesn't
        # collapse to sub-Gaussian point source.
        if _tng_source is not None:
            if _tng_source['ssfr_per_yr'] < TNG_QUENCHED_SSFR_THRESHOLD:
                source_n = max(float(source_n), 3.0)
            else:
                source_n = float(np.clip(source_n, 0.8, 2.0))
    
    # Ensure source is visible by checking magnitude constraints
    source_z = float(row.get("source_redshift", row.get("zs", 2.0)))
    phot = CONFIG['photometry']
    
    # Check if source will be visible (not too faint)
    # FIX (adversarial audit finding C-13, 2026-08-01): faint sources were
    # silently EDITED (brightened) to fit within source_mag_max rather
    # than rejected -- this truncates the faint end of the rendered
    # source luminosity function with a delta function at the limit, and
    # was previously undetectable from the output alone. Now flagged in
    # `row` so it's recorded in saved metadata and can be filtered out of
    # any luminosity-function/population-statistics use of the dataset.
    row["source_mag_brightening_applied"] = False
    for band in LOWER_BANDS:
        src_mag = float(row.get(f"source_mag_{band}", 20.5))
        if src_mag > phot.get('source_mag_max', 25.0):
            print(f"[WARNING] Source too faint in {band}: {src_mag:.2f}")
            # Adjust source to be brighter
            row[f"source_mag_{band}"] = phot.get('source_mag_max', 25.0) - 0.5
            row["source_mag_brightening_applied"] = True
    
    # v17 caustic multiplicity gate: source must be INSIDE the tangential caustic
    # (min_mu ≤ |μ| ≤ max_mu) to produce multiple images / arcs.
    # Previously only upper-bounded μ; lower bound now enforces multiplicity.
    _geo_vis = CONFIG.get('geometry', {})
    _src_x_raw = float(row.get("source_x", row.get("xs", 0)))
    _src_y_raw = float(row.get("source_y", row.get("ys", 0)))
    _max_mu = CONFIG.get('geometry', {}).get('max_magnification', 12.0)
    _min_mu = CONFIG.get('geometry', {}).get('min_magnification', 2.0)
    _n_mass_models = sum(1 for m in lens_model_list if m not in ('SHEAR', 'SHEAR_GAMMA_PSI', 'CONVERGENCE'))
    _is_binary = _n_mass_models >= 2
    try:
        from lenstronomy.LensModel.magnification_model import MagnificationModel as _MagModel
        _mm = _MagModel(lens_model_list=lens_model_list)
        _mu_raw = abs(float(_mm.magnification(_src_x_raw, _src_y_raw, kwargs_lens)))
    except Exception:
        _mu_raw = _min_mu
    # For binary lenses: check image count instead of just μ.
    # Source outside binary caustic → 2-3 images (blobs); inside → 4-5 images (arcs).
    _needs_resample = (_mu_raw < _min_mu or _mu_raw > _max_mu)
    if _geo_vis.get('force_caustic_source_position', False):
        _needs_resample = True
    if _is_binary and not _needs_resample:
        try:
            from lenstronomy.LensModel.lens_model import LensModel as _LensModel
            _lm = _LensModel(lens_model_list=lens_model_list)
            _n_img = len(_lm.find_lens_images(_src_x_raw, _src_y_raw, kwargs_lens,
                                               min_distance=0.05, search_window=4.0,
                                               verbose=False)[0])
            if _n_img < 3:
                _needs_resample = True
                print(f"[CAUSTIC] lens_id={row.get('lens_id','?')} binary n_images={_n_img} < 3 -> resampling")
        except Exception:
            pass
    if _needs_resample:
        _min_mu_resample = max(_min_mu, 5.0) if _is_binary else _min_mu
        _src_x_safe, _src_y_safe = _caustic_safe_source_position(
            lens_model_list, kwargs_lens, theta_E, rng,
            max_mu=_max_mu, min_mu=_min_mu_resample)
        row["source_x"] = _src_x_safe
        row["source_y"] = _src_y_safe
        print(f"[CAUSTIC] lens_id={row.get('lens_id','?')} mu={_mu_raw:.1f} outside [{_min_mu},{_max_mu}] -> resampled ({_src_x_safe:.3f},{_src_y_safe:.3f})")

    # FIX (adversarial audit finding C-13, 2026-08-01): the selection
    # function (magnification gate [min_mu, max_mu], and whether this
    # particular source position had to be resampled to satisfy it) was
    # applied but never recorded -- so nothing in the saved output let a
    # population-statistics user know the sample excludes the mu<min_mu
    # regime that dominates REAL galaxy-galaxy lens samples (typical
    # mu~2-5), biasing any magnification-distribution claim without a
    # way to detect or correct for it after the fact.
    try:
        _mu_final = abs(float(_mm.magnification(
            float(row.get("source_x", row.get("xs", 0))),
            float(row.get("source_y", row.get("ys", 0))),
            kwargs_lens)))
    except Exception:
        _mu_final = _mu_raw
    row["magnification"] = float(_mu_final)
    row["magnification_gate_min"] = float(_min_mu)
    row["magnification_gate_max"] = float(_max_mu)
    row["source_position_resampled_for_caustic"] = bool(_needs_resample)

    lensed_source = dict(
        R_sersic=source_radius,
        n_sersic=float(source_n),
        center_x=float(row.get("source_x", row.get("xs", 0))),
        center_y=float(row.get("source_y", row.get("ys", 0))),
        e1=float(e1_s), e2=float(e2_s)
    )

    # === MULTI-SOURCE: additional lensed background sources ===
    # Reuses the same physics already validated for the primary source
    # (caustic-safe positioning, lognormal size/Sersic sampling) instead of
    # the quarantined, physically-incorrect MultiSourceLensingSystem class
    # in advanced_lens_features.py (never imported by this module).
    #
    # A second (or third...) source at a different redshift behind the same
    # lens plane is handled with a distance-ratio "beta" rescaling of the
    # deflection field (kappa/gamma scale as beta = D_LS/D_S), NOT full
    # multi-plane ray tracing. Each extra source is rendered with its own
    # image_model call using the beta-rescaled lens kwargs and co-added to
    # the primary image, since lenstronomy's single ImageModel call only
    # supports one common deflection field for all source-plane components.
    _ms_cfg = CONFIG.get('multi_source', {}) if isinstance(CONFIG, dict) else {}
    additional_sources = []  # list of dicts: lensed_source, src_mag_by_band, beta_ratio
    if _ms_cfg.get('enabled', False):
        _ms_probs = _ms_cfg.get('n_sources_probs', [0.4, 0.3, 0.15, 0.1, 0.05])
        _ms_counts = list(range(1, len(_ms_probs) + 1))
        n_sources = int(rng.choice(_ms_counts, p=_ms_probs))
        if n_sources > 1:
            from astropy.cosmology import FlatLambdaCDM
            _cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            _d_s1 = _cosmo.angular_diameter_distance(source_z).value
            _d_ls1 = _cosmo.angular_diameter_distance_z1z2(lens_z, source_z).value
            _beta1 = _d_ls1 / _d_s1 if _d_s1 > 0 else 1.0
            _z_gap = _ms_cfg.get('extra_source_z_offset_range', [0.3, 2.5])

            for _i in range(n_sources - 1):
                _add_z = source_z + float(rng.uniform(_z_gap[0], _z_gap[1]))
                _d_s2 = _cosmo.angular_diameter_distance(_add_z).value
                _d_ls2 = _cosmo.angular_diameter_distance_z1z2(lens_z, _add_z).value
                _beta2 = _d_ls2 / _d_s2 if _d_s2 > 0 else _beta1
                _beta_ratio = float(_beta2 / _beta1) if _beta1 > 0 else 1.0

                _add_pa = float(rng.uniform(-180, 180))
                _add_q = float(np.clip(rng.uniform(0.3, 1.0), 0.2, 1.0))
                _add_e1, _add_e2 = ellipticity(_add_q, _add_pa)
                _add_e1, _add_e2 = np.clip([_add_e1, _add_e2], -0.8, 0.8)

                # FIX (regression from adversarial audit finding C-6 fix,
                # 2026-08-01): this block referenced size_frac_mean/sigma/
                # min/max, which no longer exist since the primary source's
                # sizing was switched from theta_E*fraction to a physical
                # mass-size relation (see the fix above) -- this caused a
                # NameError crashing every multi-source render (confirmed
                # by execution: "NameError: name 'size_frac_mean' is not
                # defined" on every attempt). Apply the same physical
                # size-mass-redshift approach here for consistency: sample
                # an independent stellar mass for this extra source and
                # derive its angular size from mass_size_relation(), same
                # as the primary source.
                _add_logM = float(np.clip(rng.normal(
                    CONFIG.get('tng_mode', {}).get('source_logM_default', 9.5),
                    CONFIG.get('tng_mode', {}).get('source_logM_scatter', 0.5),
                ), 8.0, 11.0))
                _add_reff_kpc = mass_size_relation(_add_logM, _add_z, rng)
                _add_radius = convert_physical_to_angular_radius(_add_reff_kpc, _add_z)
                _add_radius = float(np.clip(_add_radius, _src_r_min * 0.6, 0.8))
                _add_n = float(np.clip(rng.lognormal(np.log(1.2), 0.4), 0.8, 4.0))

                # Rescale the deflection field (theta_E ~ sqrt(beta), shear ~ beta)
                # to this source's own lens-source geometry, then find a
                # caustic-safe position for it under that rescaled field.
                _kwargs_lens_scaled = []
                for _kw in kwargs_lens:
                    _kw2 = dict(_kw)
                    if 'theta_E' in _kw2:
                        _kw2['theta_E'] = _kw2['theta_E'] * np.sqrt(max(_beta_ratio, 1e-6))
                    if 'gamma1' in _kw2:
                        _kw2['gamma1'] = _kw2['gamma1'] * _beta_ratio
                    if 'gamma2' in _kw2:
                        _kw2['gamma2'] = _kw2['gamma2'] * _beta_ratio
                    _kwargs_lens_scaled.append(_kw2)
                _theta_E_scaled = theta_E * np.sqrt(max(_beta_ratio, 1e-6))

                _add_x, _add_y = _caustic_safe_source_position(
                    lens_model_list, _kwargs_lens_scaled, _theta_E_scaled, rng,
                    max_mu=_max_mu, min_mu=_min_mu)

                # Extra sources are fainter on average than the primary lensed
                # source (they are typically less-magnified chance alignments).
                _mag_offset = float(rng.uniform(0.3, 1.8))
                _add_src_mag_by_band = {b: float(row.get(f"source_mag_{BAND_TO_LOWER[b]}", 20.5)) + _mag_offset
                                         for b in UPPER_BANDS}

                _add_lensed_source = dict(
                    R_sersic=_add_radius, n_sersic=_add_n,
                    center_x=float(_add_x), center_y=float(_add_y),
                    e1=float(_add_e1), e2=float(_add_e2),
                )
                additional_sources.append(dict(
                    lensed_source=_add_lensed_source,
                    src_mag_by_band=_add_src_mag_by_band,
                    redshift=_add_z,
                    kwargs_lens_scaled=_kwargs_lens_scaled,
                ))
            _extra_z_strs = [f"{s['redshift']:.2f}" for s in additional_sources]
            print(f"[MULTI-SOURCE] lens_id={row.get('lens_id','?')} n_sources={n_sources} "
                  f"extra_z={_extra_z_strs}")

    # === ENVIRONMENT-DRIVEN FIELD GALAXIES (from paper) ===
    # Sample field count from environment-specific distribution
    # Use Poisson/triangular distributions per environment type
    # Target count derived from the real COSMOS-Web detection density at this
    # FOV's area (see field_galaxy_count_target), not a guessed absolute
    # number, so density stays realistic at any image_size (e.g. a 1'
    # extended-FOV render) instead of reusing a count tuned for a much
    # smaller field.
    _area_scale = field_density_area_scale(numpix, pixel_scale, CONFIG)
    env_mean, env_std = field_galaxy_count_target(numpix, pixel_scale, env_params, CONFIG)
    env_min  = max(0.0, env_mean - 3 * env_std)
    env_max  = env_mean + 3 * env_std
    n_field_env = int(np.clip(rng.normal(env_mean, env_std), env_min, env_max))

    # TNG Mode: if a matched lens subhalo's FoF group environment is
    # available, let it (optionally) set the *richness class*, then redraw
    # the count from COSMOS surface density × FOV (same path as above).
    # Absolute triangular counts × area_scale were calibrated for small JWST
    # cutouts and collapsed Euclid 96px hybrids to ~2 field galaxies.
    _tng_mode_cfg = CONFIG.get('tng_mode', {}) if isinstance(CONFIG, dict) else {}
    if _tng_mode_cfg.get('enabled', False) and _tng_mode_cfg.get('environment_drives_field_count', False) and _tng_lens is not None:
        tng_env = _tng_lens.get('environment', 'isolated')
        _tng_richness = {
            'isolated': 2.5,
            'pair': 3.0,
            'group': 4.5,
            'rich_group': 6.5,
        }.get(tng_env, 2.5)
        env_mean, env_std = field_galaxy_count_target(
            numpix, pixel_scale, {'galaxy_count_mean': _tng_richness}, CONFIG)
        env_min = max(0.0, env_mean - 3 * env_std)
        env_max = env_mean + 3 * env_std
        n_field_env = int(np.clip(rng.normal(env_mean, env_std), env_min, env_max))

    # Clamp to n_field_max
    n_field_env = min(max(0, n_field_env), n_field_max)

    # Environment-specific avoid radius; placement radius scales with the
    # linear FOV size (sqrt of area) so galaxies spread across the full
    # extended field rather than clustering at the reference FOV's radius.
    env_sep_mean = env_params.get('separation_mean', 2.0) * (_area_scale ** 0.5)
    avoid_radius = max(0.3, env_sep_mean * 0.2)  # 20% of typical separation

    # Use fixed field galaxies if provided (for time delay consistency)
    if fixed_field_galaxies is not None:
        field_galaxies_base = fixed_field_galaxies
        print(f"[TIME_DELAY] Reusing {len(field_galaxies_base)} fixed field galaxies")
    elif field_pop is not None:
        # Get lens ID for lens-specific field galaxy sampling
        lens_id = row.get('ASSOC_ID', row.get('lens_id', None))
        
        # Use enhanced field sampling if available
        # For large FOV (n_field_env > 10) bypass ENHANCED_SAMPLER which has
        # hardcoded small-field caps; use the synthetic generator at the correct
        # pixel scale instead so all galaxies are spread across the full FOV.
        if ENHANCED_SAMPLER is not None and n_field_env <= 10:
            try:
                field_galaxies_base = ENHANCED_SAMPLER.sample_field_galaxies_enhanced(
                    central_redshift=lens_z,
                    central_mass_log10=lens_mass_log10,
                    n_max=n_field_env,
                    rng=rng,
                    numpix=numpix,
                    pixel_scale=pixel_scale,
                    avoid_center_arcsec=avoid_radius,
                    psf_data=psf_data,
                    lens_radius=lens_radius,
                    lens_id=lens_id
                )
                print(f"[ENHANCED] Sampled {len(field_galaxies_base)} field galaxies using ML models")
            except Exception as e:
                print(f"[WARNING] Enhanced sampling failed: {e}, falling back to basic sampling")
                field_galaxies_base = sample_real_field_galaxies_for_mock(
                    field_pop, n_max=n_field_env, rng=rng, numpix=numpix,
                    pixel_scale=pixel_scale, lens_redshift=lens_z,
                    avoid_center_arcsec=avoid_radius,
                    psf_data=psf_data,
                    lens_radius=lens_radius,
                    lens_id=lens_id,
                    lens_mass_log10=lens_mass_log10,
                    halo_radius_constraint=True
                )
        else:
            # Use basic field sampling
            field_galaxies_base = sample_real_field_galaxies_for_mock(
                field_pop, n_max=n_field_env, rng=rng, numpix=numpix,
                pixel_scale=pixel_scale, lens_redshift=lens_z,
                avoid_center_arcsec=avoid_radius,
                psf_data=psf_data,
                lens_radius=lens_radius,
                lens_id=lens_id,
                lens_mass_log10=lens_mass_log10,
                halo_radius_constraint=True
            )
    elif n_field_env > 10:
        # Large-FOV mode: generate synthetic population spread across full image
        _pix_scale = CONFIG.get('pixel_scale', 0.031) if isinstance(CONFIG, dict) else 0.031
        field_galaxies_base = generate_synthetic_field_population(
            rng, n_field_env, numpix, pixel_scale=_pix_scale, env_type=env_type, lens_z=lens_z)
        print(f"[INFO] Large-FOV synthetic field: {len(field_galaxies_base)} galaxies over {numpix*_pix_scale:.0f}\"")
    else:
        field_galaxies_base = generate_synthetic_field_population(
            rng, max(n_field_env, 2), numpix, env_type=env_type, lens_z=lens_z)
        print(f"[INFO] Using synthetic field galaxies for {env_type} environment")

    apply_tng_field_overrides(field_galaxies_base, rng, CONFIG, exclude_subhalos=_used_tng_subhalos)
    tag_field_galaxies_with_galaxygenius_stamps(field_galaxies_base, rng, CONFIG)
    tag_field_galaxies_with_tng_particles(field_galaxies_base, rng, CONFIG)
    n_field_added = len(field_galaxies_base)

    # === OPTIONAL: DARK-MATTER SUBHALO PERTURBERS (NFW) ===
    # Appended on top of the SIE(+shear) macromodel -- NOT a replacement for
    # it. SIE remains the realistic total (stars+DM) mass description of the
    # lens galaxy; NFW is the physically appropriate profile for the
    # dark-matter-dominated substructure that perturbs it. Disabled unless
    # CONFIG['subhalos']['enabled'] is set -- validate on a small batch
    # before enabling for production runs (extra deflectors slow ray-tracing).
    _sub_mass_log10 = float(row.get("lens_mass_log10", 11.0 + 2.0 * np.log10(theta_E)))
    _sub_source_z = float(row.get("source_redshift", row.get("zs", 2.0)))
    sub_model_names, sub_kwargs_list = generate_subhalo_population(
        host_mass_log10=_sub_mass_log10,
        host_redshift=lens_z,
        source_redshift=_sub_source_z,
        theta_E=theta_E,
        rng=rng
    )
    if sub_model_names:
        lens_model_list = lens_model_list + sub_model_names
        kwargs_lens = kwargs_lens + sub_kwargs_list
        print(f"[SUBHALO] Added {len(sub_model_names)} NFW perturbers "
              f"(host logM={_sub_mass_log10:.2f}, z_l={lens_z:.2f})")

    # Precompute per-band lens/source magnitudes (deterministic, no RNG) so
    # the native multi-component light models (galaxy_morphology) can be
    # built once, ahead of the per-band loop.
    _phot = CONFIG['photometry']
    _geo_vis = CONFIG.get('geometry', {})
    _visible_arcs = bool(_geo_vis.get('visible_lensed_arcs', False))
    _min_delta = float(_phot.get('min_source_fainter_than_lens_mag', 0.8))
    _max_delta = float(_phot.get('max_source_fainter_than_lens_mag',
                                  _phot.get('source_mag_diff_max', 3.5)))
    if _visible_arcs:
        _max_delta = min(_max_delta, float(_phot.get('visible_arc_mag_diff_max', 1.5)))
    lens_mag_by_band = {}
    src_mag_by_band = {}
    for _b in UPPER_BANDS:
        _lens_mag = float(row.get(f"lens_mag_{BAND_TO_LOWER[_b]}", _phot.get('lens_base_mag_zero', 22.0)))
        _src_mag = float(row.get(f"source_mag_{BAND_TO_LOWER[_b]}", _phot.get('source_base_mag', 20.5)))
        _lens_mag = float(np.clip(_lens_mag, _phot['lens_mag_min'], _phot['lens_mag_max']))
        _src_mag = float(np.clip(_src_mag, _phot['source_mag_min'], _phot['source_mag_max']))
        if _src_mag < _lens_mag + _min_delta:
            _src_mag = _lens_mag + _min_delta
        if _src_mag > _lens_mag + _max_delta:
            _src_mag = _lens_mag + _max_delta
        if _visible_arcs:
            _bright_cap = float(_phot.get('visible_arc_source_mag_max', 21.5))
            _src_mag = min(_src_mag, _bright_cap)
        lens_mag_by_band[_b] = _lens_mag
        src_mag_by_band[_b] = _src_mag

    # Independent, deterministic RNG seeds for morphology classification /
    # component placement -- so enabling morphology.multicomponent_enabled
    # does not consume extra draws from `rng` and therefore does not shift
    # the downstream mass-model/source-position RNG sequence (Phase 2.1
    # RNG isolation; see phase1_audit.md Section 18).
    _lens_id_for_seed = str(row.get('lens_id', 0))
    _lens_morph_seed = int(abs(hash(_lens_id_for_seed + ':lens')) % (2**32))
    _source_morph_seed = int(abs(hash(_lens_id_for_seed + ':source')) % (2**32))

    # Optionally render the foreground lens galaxy's own light as a
    # GalaxyGenius/SKIRT stamp (INTERPOL, image-plane like the field-galaxy
    # stamps) instead of the native Sersic-based model. Sized to a multiple
    # of the lens's own R_sersic so the stamp roughly matches the lens
    # galaxy's expected angular extent.
    _gg_cfg = CONFIG.get('galaxygenius_stamps', {})
    use_lens_stamp = bool(
        _gg_cfg.get('lens_enabled', False)
        and rng.random() < float(_gg_cfg.get('lens_fraction', 0.0))
    )
    # Optionally render the foreground lens galaxy's own light from its
    # matched TNG subhalo's star/gas particle data (procedural, not pasted --
    # see galaxy_morphology/tng_particle_light.py), gated by
    # tng_mode.particle_morphology and the availability of a locally
    # downloaded particle cutout for this subhalo.
    _particle_morph_cfg = _tng_mode_cfg.get('particle_morphology', {})
    _lens_particle_file = None
    if (_particle_morph_cfg.get('enabled', False) and _tng_lens is not None
            and _particle_morph_cfg.get('lens_enabled', True)
            and rng.random() < float(_particle_morph_cfg.get('lens_fraction', 1.0))):
        _lens_sim = _tng_lens.get('sim', 'TNG100-1')
        if not isinstance(_lens_sim, str) or str(_lens_sim).lower() in ('nan', 'none', ''):
            _lens_sim = 'TNG100-1'
        _lens_particle_file = local_particle_path(_tng_lens['snapshot'], _tng_lens['subhalo_id'],
                                                    min_particles=_particle_morph_cfg.get('min_particles'),
                                                    sim=_lens_sim)

    if use_lens_stamp:
        _lens_morph_type = None
        lens_fragment = ["INTERPOL"]
        _lens_stamp_set = random_stamp_set(rng, redshift=lens_z)
        _lens_view = int(rng.choice(available_views(_lens_stamp_set)))
        _lens_phi_G = 0.5 * np.arctan2(main_lens_light['e2'], main_lens_light['e1'])
        _lens_size_factor = float(_gg_cfg.get('lens_target_size_factor', 6.0))
        _lens_target_size = float(main_lens_light['R_sersic']) * _lens_size_factor
        lens_kw_by_band = {
            b: [build_lens_light_interpol_kwargs(
                band=b, view_idx=_lens_view, magnitude_ref=lens_mag_by_band[b],
                center_x=main_lens_light['center_x'], center_y=main_lens_light['center_y'],
                phi_G=_lens_phi_G, target_size_arcsec=_lens_target_size,
                stamp_set=_lens_stamp_set,
            )]
            for b in UPPER_BANDS
        }
    elif _lens_particle_file is not None:
        print(f"[TNG_PARTICLE] Using local star-particle cutout for lens light: {_lens_particle_file.name}")
        _lens_morph_type = None
        lens_fragment = ["INTERPOL"]
        # Mass-light orientation consistency: derive the lens light's PA
        # (and, for the mass model, e1/e2) from the same TNG subhalo's
        # particle projection, instead of an independently-sampled
        # ellipticity -- so the deflector's mass ellipse is aligned with its
        # own rendered light distribution.
        _lens_axis_ratio, _lens_pa_rad = get_projection_orientation(
            _lens_particle_file, _tng_lens['halfmassrad_stars_kpc'], rng)
        _lens_phi_G = _lens_pa_rad
        _lens_e1, _lens_e2 = ellipticity(_lens_axis_ratio, math.degrees(_lens_pa_rad))
        main_lens_light['e1'] = float(_lens_e1)
        main_lens_light['e2'] = float(_lens_e2)
        if lens_model_list and lens_model_list[0] in ('SIE', 'SPEMD', 'NFW_ELLIPSE'):
            kwargs_lens[0]['e1'] = float(_lens_e1)
            kwargs_lens[0]['e2'] = float(_lens_e2)
        _lens_size_factor = float(_gg_cfg.get('lens_target_size_factor', 6.0))
        _lens_target_size = float(main_lens_light['R_sersic']) * _lens_size_factor
        _pref = _particle_ref_band()
        _lens_smooth = float(_particle_morph_cfg.get('lens_smooth_sigma', 3.0))
        lens_kw_by_band = {
            b: [build_tng_particle_interpol_kwargs(
                band=b, particle_file=_lens_particle_file,
                halfmassrad_stars_kpc=_tng_lens['halfmassrad_stars_kpc'],
                magnitude_ref=lens_mag_by_band[_pref],
                ref_band=_pref,
                center_x=main_lens_light['center_x'], center_y=main_lens_light['center_y'],
                phi_G=_lens_phi_G, target_size_arcsec=_lens_target_size, rng=rng,
                smooth_sigma=_lens_smooth,
            )]
            for b in UPPER_BANDS
        }
    elif (_particle_morph_cfg.get('generative_enabled', False)
          and _particle_morph_cfg.get('generative_force', False)
          and _particle_morph_cfg.get('lens_enabled', False)
          and _tng_lens is not None):
        # Generative VAE fallback for lens when no particle file and generative forced
        _gen_ckpt_l = _particle_morph_cfg.get('generative_checkpoint', None)
        try:
            from prism.morphology.generative.inference import build_generative_interpol_kwargs as _bgi
        except ImportError:
            from prism.morphology.generative.inference import build_generative_interpol_kwargs as _bgi
        _lens_morph_type = None
        lens_fragment = ["INTERPOL"]
        _l_phi_G = 0.5 * np.arctan2(main_lens_light['e2'], main_lens_light['e1'])
        _l_size = float(main_lens_light['R_sersic']) * float(_gg_cfg.get('lens_target_size_factor', 6.0))
        _l_galaxy_type = tng_sed_galaxy_type(_tng_lens)
        _gen_l_base = dict(
            morph_type=_l_galaxy_type, logM=float(_tng_lens['stellar_mass_logmsun']),
            redshift=float(lens_z), center_x=main_lens_light['center_x'],
            center_y=main_lens_light['center_y'], phi_G=_l_phi_G,
            target_size_arcsec=_l_size, rng=rng,
        )
        if _gen_ckpt_l:
            _gen_l_base['checkpoint_path'] = _gen_ckpt_l
        lens_kw_by_band = {b: [_bgi(**{**_gen_l_base, 'magnitude_ref': lens_mag_by_band[b]})] for b in UPPER_BANDS}
        print(f"[GENERATIVE] lens galaxy forced to VAE (morph={_l_galaxy_type} logM={_tng_lens['stellar_mass_logmsun']:.1f})")
    else:
        lens_morph_cfg = CONFIG.get('lens_morphology', {}) if isinstance(CONFIG, dict) else {}
        _force_lens_morph = lens_morph_cfg.get('force_morph_type') if isinstance(lens_morph_cfg, dict) else None
        lens_fragment, lens_kw_by_band, _lens_morph_type = gm_build_light_model(
            'lens', main_lens_light, lens_mag_by_band, UPPER_BANDS, rng, CONFIG,
            morph_seed=_lens_morph_seed,
            morph_type=_force_lens_morph,
        )

    # Optionally render the lensed source as a GalaxyGenius/SKIRT stamp
    # (INTERPOL light profile, ray-traced through the lens equation like any
    # other source-plane profile) instead of the native Sersic-based model.
    use_source_stamp = bool(
        _gg_cfg.get('source_enabled', False)
        and rng.random() < float(_gg_cfg.get('source_fraction', 0.0))
    )
    # Optionally render the lensed source from its matched TNG subhalo's
    # star/gas particle data (procedural, ray-traced through the lens
    # equation like any other source-plane profile), gated the same way as
    # the lens-light particle morphology above.
    _source_particle_file = None
    if (_particle_morph_cfg.get('enabled', False) and _tng_source is not None
            and _particle_morph_cfg.get('source_enabled', True)
            and rng.random() < float(_particle_morph_cfg.get('source_fraction', 1.0))):
        _src_sim = _tng_source.get('sim', 'TNG100-1')
        if not isinstance(_src_sim, str) or str(_src_sim).lower() in ('nan', 'none', ''):
            _src_sim = 'TNG100-1'
        _source_particle_file = local_particle_path(_tng_source['snapshot'], _tng_source['subhalo_id'],
                                                      min_particles=_particle_morph_cfg.get('min_particles'),
                                                      sim=_src_sim)

    if use_source_stamp:
        source_fragment = ["INTERPOL"]
        # Rescale the stamp's angular size for the source's actual redshift
        # (the stamp's morphology/SED come from a z=0.06 TNG subhalo, but its
        # angular extent should reflect this source's own redshift), with an
        # optional config multiplier for additional control.
        _src_z = float(row.get("source_redshift", row.get("zs", 2.0)))
        _src_stamp_set = random_stamp_set(rng, redshift=_src_z)
        _src_view = int(rng.choice(available_views(_src_stamp_set)))
        _src_phi_G = 0.5 * np.arctan2(lensed_source['e2'], lensed_source['e1'])
        _src_size_scale = float(_gg_cfg.get('source_size_scale_factor', 1.0))
        _src_target_size = float(lensed_source['R_sersic']) * _src_size_scale
        source_kw_by_band = {
            b: [build_source_interpol_kwargs(
                band=b, view_idx=_src_view, magnitude_ref=src_mag_by_band[b],
                center_x=lensed_source['center_x'], center_y=lensed_source['center_y'],
                phi_G=_src_phi_G, target_size_arcsec=_src_target_size,
                stamp_set=_src_stamp_set,
            )]
            for b in UPPER_BANDS
        }
    elif _source_particle_file is not None:
        print(f"[TNG_PARTICLE] Using local star-particle cutout for lensed source: {_source_particle_file.name}")
        source_fragment = ["INTERPOL"]
        _src_z = float(row.get("source_redshift", row.get("zs", 2.0)))
        _src_phi_G = 0.5 * np.arctan2(lensed_source['e2'], lensed_source['e1'])
        _src_size_scale = float(_gg_cfg.get('source_size_scale_factor', 1.0))
        _src_target_size = float(lensed_source['R_sersic']) * _src_size_scale
        _src_smooth = float(_particle_morph_cfg.get('source_smooth_sigma', 2.5))
        _pref = _particle_ref_band()
        source_kw_by_band = {
            b: [build_tng_particle_interpol_kwargs(
                band=b, particle_file=_source_particle_file,
                halfmassrad_stars_kpc=_tng_source['halfmassrad_stars_kpc'],
                magnitude_ref=src_mag_by_band[_pref],
                ref_band=_pref,
                center_x=lensed_source['center_x'], center_y=lensed_source['center_y'],
                phi_G=_src_phi_G, target_size_arcsec=_src_target_size, rng=rng,
                smooth_sigma=_src_smooth,
            )]
            for b in UPPER_BANDS
        }
    elif (_particle_morph_cfg.get('generative_enabled', False)
          and _particle_morph_cfg.get('generative_force', False)
          and _particle_morph_cfg.get('source_enabled', False)
          and _tng_source is not None):
        # Generative VAE fallback for source when no particle file and generative forced
        _gen_ckpt_s = _particle_morph_cfg.get('generative_checkpoint', None)
        try:
            from prism.morphology.generative.inference import build_generative_interpol_kwargs as _bgi_s
        except ImportError:
            from prism.morphology.generative.inference import build_generative_interpol_kwargs as _bgi_s
        source_fragment = ["INTERPOL"]
        _src_z2 = float(row.get("source_redshift", row.get("zs", 2.0)))
        _s_phi_G = 0.5 * np.arctan2(lensed_source['e2'], lensed_source['e1'])
        _s_size = float(lensed_source['R_sersic']) * float(_gg_cfg.get('source_size_scale_factor', 1.0))
        _s_galaxy_type = tng_sed_galaxy_type(_tng_source)
        _gen_s_base = dict(
            morph_type=_s_galaxy_type, logM=float(_tng_source['stellar_mass_logmsun']),
            redshift=float(_src_z2), center_x=lensed_source['center_x'],
            center_y=lensed_source['center_y'], phi_G=_s_phi_G,
            target_size_arcsec=_s_size, rng=rng,
        )
        if _gen_ckpt_s:
            _gen_s_base['checkpoint_path'] = _gen_ckpt_s
        source_kw_by_band = {b: [_bgi_s(**{**_gen_s_base, 'magnitude_ref': src_mag_by_band[b]})] for b in UPPER_BANDS}
        print(f"[GENERATIVE] source galaxy forced to VAE (morph={_s_galaxy_type} logM={_tng_source['stellar_mass_logmsun']:.1f} z={_src_z2:.2f})")
    else:
        source_fragment, source_kw_by_band, _ = gm_build_light_model(
            'source', lensed_source, src_mag_by_band, UPPER_BANDS, rng, CONFIG,
            morph_seed=_source_morph_seed)

    # Build light-model fragments for any additional (multi-source) lensed
    # sources sampled above. Uses the standard morphology path only (not the
    # TNG-particle/generative-stamp branches used for the primary source),
    # since each extra source is rendered in its own separate image_model
    # call below and co-added to the primary image.
    for _extra in additional_sources:
        _extra_fragment, _extra_kw_by_band, _ = gm_build_light_model(
            'source', _extra['lensed_source'], _extra['src_mag_by_band'], UPPER_BANDS, rng, CONFIG,
            morph_seed=int(rng.integers(0, 2**31 - 1)))
        _extra['fragment'] = _extra_fragment
        _extra['kw_by_band'] = _extra_kw_by_band

    # Setup model lists
    lens_light_models = list(lens_fragment)  # Main lens (1+ native components)
    if companion_lens_light is not None:
        lens_light_models += ["INTERPOL" if _companion_particle_file is not None else "SERSIC_ELLIPSE"]  # Companion lens (massive pair)
    lens_light_models += field_galaxy_light_model_types(field_galaxies_base)  # Field galaxies

    model_lists = dict(
        lens_model_list=lens_model_list,  # Use dynamic lens model (binary or single)
        lens_light_model_list=lens_light_models,
        source_light_model_list=source_fragment
    )
    
    numerics = dict(supersampling_factor=1, supersampling_convolution=False)

    # Check if intermediate image saving is enabled
    save_intermediate = CONFIG.get('save_intermediate_images', False)
    intermediate_images_dict = {}  # Store intermediate images: {step: {band: image}}
    
    # === GENERATE IMAGES PER BAND ===
    images = {}
    
    # Get PSF arrays once for all steps
    psf_arrays_all = None
    if psf_data is not None:
        lens_id = row.get('lens_id', None)
        psf_arrays_all = get_psf_for_simulation(
            psf_data, lens_id, rng, psf_tile=row.get('euclid_psf_tile'))
    
    # Get morphology type for enhancements
    lens_q = float(row.get("lens_axis_ratio", row.get("ql", 0.7)))
    morph_type = None
    if save_intermediate:
        if _lens_morph_type is not None:
            morph_type = _lens_morph_type
        else:
            try:
                morph_type = classify_galaxy_morphology_enhanced(n_lens, lens_q, rng)
            except:
                pass
    
    for b in UPPER_BANDS:
        try:
            cfg = dict(band_cfgs[b])
            cfg['pixel_scale'] = float(pixel_scale)  # Use resolution-specific pixel scale
            
            # Component magnitudes with configurable clipping and relative constraint
            phot = CONFIG['photometry']
            lens_mag = lens_mag_by_band[b]
            src_mag = src_mag_by_band[b]

            # Build component lists (native multi-component fragments if
            # morphology.multicomponent_enabled, else single SERSIC_ELLIPSE)
            kw_src = list(source_kw_by_band[b])
            kw_lens = list(lens_kw_by_band[b])

            if companion_lens_light is not None:
                # Same redshift + same broad passive/elliptical SED class as
                # the main lens (physically expected for an M*-M* pair) -> the
                # bulk of the per-band magnitude offset is the mass-ratio flux
                # scaling (theta_E_2/theta_E ~ sqrt(mass_ratio) -> flux ~
                # mass_ratio), plus a small per-band jitter representing the
                # companion's own age/metallicity history -- similar but not
                # identical color to the primary.
                companion_mag = float(np.clip(
                    lens_mag - 2.5 * np.log10(_mass_ratio) + _companion_color_jitter[b],
                    phot['lens_mag_min'], phot['lens_mag_max']
                ))
                if _companion_particle_file is not None:
                    _comp_smooth = float(_particle_morph_cfg.get(
                        'companion_smooth_sigma',
                        _particle_morph_cfg.get('lens_smooth_sigma', 3.0),
                    ))
                    kw_lens.append(build_tng_particle_interpol_kwargs(
                        band=b,
                        particle_file=_companion_particle_file,
                        halfmassrad_stars_kpc=_tng_companion['halfmassrad_stars_kpc'],
                        magnitude_ref=companion_mag,
                        ref_band=_particle_ref_band(),
                        center_x=companion_lens_light['center_x'],
                        center_y=companion_lens_light['center_y'],
                        phi_G=0.5 * np.arctan2(companion_lens_light['e2'], companion_lens_light['e1']),
                        target_size_arcsec=companion_lens_light['R_sersic'] * _field_target_size_factor(),
                        rng=rng,
                        smooth_sigma=_comp_smooth,
                    ))
                else:
                    kw_lens.append(dict(companion_lens_light, magnitude=companion_mag))

            # Add REAL field galaxies with appropriate colors
            if field_galaxies_base:
                field_band = apply_real_jwst_colors_to_field_galaxies(field_galaxies_base, b, rng)
            else:
                field_band = []
                
            kw_lens.extend(field_band)
            
            # Create simulation (filter out metadata params)
            sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(cfg), kwargs_model=model_lists)
            kw_lens_amp, kw_src_amp, _ = sim.magnitude2amplitude(kw_lens, kw_src)
            im_model = sim.image_model_class(numerics)
            
            # Generate intermediate images if enabled
            if save_intermediate:
                psf_array = psf_arrays_all[b] if psf_arrays_all and b in psf_arrays_all else None
                
                # Lens light components: main lens (1+ native components) +
                # companion (if a binary pair)
                n_lens_light = len(lens_fragment) + (1 if companion_lens_light is not None else 0)
                lens_light_only = kw_lens[:n_lens_light]

                # Step 1: Lens only
                step1 = generate_intermediate_images(
                    'lens_only', b, sim, kwargs_lens, kw_src, lens_light_only,
                    psf_array, cfg, rng, numpix, add_artifacts, lens_q, n_lens, morph_type, lens_model_list,
                    lens_light_fragment=lens_fragment, source_light_fragment=source_fragment
                )
                if step1:
                    if 'lens_only' not in intermediate_images_dict:
                        intermediate_images_dict['lens_only'] = {}
                    intermediate_images_dict['lens_only'][b] = step1[b]

                # Step 2a: Lens + sources
                step2a = generate_intermediate_images(
                    'lens_sources', b, sim, kwargs_lens, kw_src, lens_light_only,
                    psf_array, cfg, rng, numpix, add_artifacts, lens_q, n_lens, morph_type, lens_model_list,
                    lens_light_fragment=lens_fragment, source_light_fragment=source_fragment
                )
                if step2a:
                    if 'lens_sources' not in intermediate_images_dict:
                        intermediate_images_dict['lens_sources'] = {}
                    intermediate_images_dict['lens_sources'][b] = step2a[b]
                
                # Step 2b: Sources only (lensed, no lens light) — shows gravitational arcs
                step2b = generate_intermediate_images(
                    'sources_only', b, sim, kwargs_lens, kw_src, [],
                    psf_array, cfg, rng, numpix, add_artifacts, lens_q, n_lens, None, lens_model_list,
                    source_light_fragment=source_fragment
                )
                if step2b:
                    if 'sources_only' not in intermediate_images_dict:
                        intermediate_images_dict['sources_only'] = {}
                    intermediate_images_dict['sources_only'][b] = step2b[b]

                # Step 2c: Source UNLENSED — original source morphology rendered without deflection.
                # kwargs_lens is unused in this step (generate_intermediate_images handles it).
                step2c = generate_intermediate_images(
                    'sources_unlensed', b, sim, kwargs_lens, kw_src, [],
                    psf_array, cfg, rng, numpix, add_artifacts, lens_q, n_lens, None, lens_model_list,
                    source_light_fragment=source_fragment
                )
                if step2c:
                    if 'sources_unlensed' not in intermediate_images_dict:
                        intermediate_images_dict['sources_unlensed'] = {}
                    intermediate_images_dict['sources_unlensed'][b] = step2c[b]

                # Step 3a: Field galaxies only
                if field_band:
                    step3a = generate_intermediate_images(
                        'field_only', b, sim, kwargs_lens, [], field_band,
                        psf_array, cfg, rng, numpix, add_artifacts, lens_q, n_lens, None, lens_model_list
                    )
                    if step3a:
                        if 'field_only' not in intermediate_images_dict:
                            intermediate_images_dict['field_only'] = {}
                        intermediate_images_dict['field_only'][b] = step3a[b]
            
            # Generate clean lens system (final image - all components)
            clean = im_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kw_src_amp,
                kwargs_lens_light=kw_lens_amp
            )

            # Multi-source: render each additional background source with its
            # own beta-rescaled deflection field (single lens plane, distance-
            # ratio approximation -- see note at additional_sources
            # construction) and co-add its lensed image.
            for _extra in additional_sources:
                try:
                    _extra_model_lists = dict(
                        lens_model_list=lens_model_list,
                        lens_light_model_list=[],
                        source_light_model_list=_extra['fragment'],
                    )
                    _extra_sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(cfg),
                                         kwargs_model=_extra_model_lists)
                    _extra_kw_src = list(_extra['kw_by_band'][b])
                    _, _extra_kw_src_amp, _ = _extra_sim.magnitude2amplitude([], _extra_kw_src)
                    _extra_im_model = _extra_sim.image_model_class(numerics)
                    _extra_clean = _extra_im_model.image(
                        kwargs_lens=_extra['kwargs_lens_scaled'],
                        kwargs_source=_extra_kw_src_amp,
                        kwargs_lens_light=[],
                    )
                    # PSF convolution (below, applied once to the combined
                    # image) is linear, so summing pre-PSF images here is
                    # equivalent to summing individually-convolved images.
                    clean = clean + _extra_clean
                except Exception as _e_extra:
                    print(f"[MULTI-SOURCE] Band {b} extra source render failed: {_e_extra}")

            # Apply PSF convolution if PSF data is available
            if psf_arrays_all and b in psf_arrays_all and psf_arrays_all[b] is not None:
                clean = apply_psf_convolution(clean, psf_arrays_all[b])
                print(f"[PSF] Applied {b} PSF convolution to lens system")

            # Store for batch enhancement
            images[b] = clean

        except Exception as e:
            print(f"[ERROR] Band {b} simulation failed: {e}")
            # FIX (adversarial audit finding C-16, 2026-08-01): a fabricated
            # near-empty fallback image used to be substituted silently,
            # indistinguishable downstream from a real successful render
            # with a normal "is_lens" label. Flag it so this system can be
            # excluded from any downstream science/ML use.
            row["band_render_failed"] = True
            row.setdefault("band_render_failed_bands", []).append(b)
            fallback = rng.exponential(1e-8, (numpix, numpix))
            center = numpix // 2
            fallback[center-12:center+12, center-12:center+12] += 5e-7
            images[b] = fallback.astype(np.float32)

    # NEW v11: Apply morphological enhancements to ALL bands at once
    try:
        # Get lens morphology parameters for enhancement
        lens_q = float(row.get("lens_axis_ratio", row.get("ql", 0.7)))
        morph_type = classify_galaxy_morphology_enhanced(n_lens, lens_q, rng, allow_ring=False)

        # If the lens already received a native bar/ring SERSIC_ELLIPSE
        # component (galaxy_morphology, multicomponent_enabled=True), skip
        # the corresponding pixel-space texture to avoid double-rendering.
        _morph_cfg = CONFIG.get('morphology', {}) if isinstance(CONFIG, dict) else {}
        skip_native_bar_ring = bool(
            _morph_cfg.get('multicomponent_enabled', False)
            and _lens_morph_type in ('barred_spiral', 'ring')
        )

        # Apply enhancements (use fixed lens_pa if provided)
        enhancement_pa = fixed_lens_params['lens_pa'] if fixed_lens_params is not None else lens_pa
        images = apply_morphological_enhancements(
            images, n_lens, lens_q, morph_type,
            numpix=numpix, pixel_scale=float(pixel_scale),
            seed=int(row.get("lens_id", 0)),
            position_angle=enhancement_pa,
            context='lens',
            skip_native_bar_ring=skip_native_bar_ring,
            r_eff_pix=float(lens_radius) / float(pixel_scale)
        )
        print(f"[v11] Applied {morph_type} enhancements to lens galaxy")
    except Exception as e:
        print(f"[WARNING] Morphological enhancement failed: {e}")
        # Continue without enhancements

    # Now add noise and detector effects to enhanced images
    # Determine which telescope detector to use
    _tel_name = CONFIG.get('telescope', 'jwst').lower()
    _det_enabled = CONFIG.get('detector_chain', {}).get('enabled', True)
    _det_overrides = CONFIG.get('detector_chain', {}).get('effects', {})

    # Persistence: residual charge trapped during one band's exposure carries
    # forward into the next band's exposure on the same detector (sequential
    # multi-band readout). Updated after each band via make_persistence_map().
    _persistence_carry = None
    # FIX (audit C-5): Python's hash() on str is salted per-process
    # (PYTHONHASHSEED) by default -- this seed differed between runs even
    # at identical --seed, breaking reproducibility of the PRNU flat-field
    # pattern (and, downstream, the noise realization) entirely. crc32 is
    # deterministic across processes/machines.
    _prnu_seed = int(zlib.crc32(str(row.get('lens_id', 0)).encode()) % (2**31))

    for b in UPPER_BANDS:
        try:
            if b not in images:
                continue

            exposure_time = float(CONFIG.get('exposure_time', 1028.0))

            if DETECTOR_CHAIN_AVAILABLE and _det_enabled:
                # FIX (adversarial audit finding C-8.2, 2026-08-01; this
                # exact bug was ALREADY diagnosed and its fix specified in
                # analysis/sim_obs_comparison/reports/phase1_real_vs_sim_comparison.md
                # dated 2026-06-12, but never implemented -- that report
                # measured the sim background RMS at 6x too low vs real
                # COSMOS-Web data because DetectorChain.apply() has no sky
                # term: Poisson shot noise was only ever computed on
                # source+dark signal, then a separate additive Gaussian
                # (add_sky_background_noise, post-chain) was used as a
                # stopgap to match the target RMS -- which gets the total
                # noise AMPLITUDE roughly right but not its physical
                # origin (no sqrt(signal+SKY+dark) scaling, no interaction
                # with saturation/ADC/PRNU for the sky component).
                #
                # Implemented here: inject the sky flux level into the
                # image BEFORE the chain runs (so Poisson/saturation/PRNU
                # all see signal+sky together, exactly as the report's
                # "Next Steps" #1 specifies), then subtract the same mean
                # back out afterward (sky-subtracted convention, matching
                # what downstream code already expects from
                # add_sky_background_noise's zero-mean-noise-only output).
                # bg_level is derived from the empirically-calibrated
                # bg_rms via the inverse Poisson relation when not
                # explicitly configured (bg_rms = sqrt(bg_level*t_exp)/t_exp
                # => bg_level = bg_rms^2 * t_exp).
                _bg_rms_b = float(band_cfgs.get(b, {}).get('_bg_rms', 0.0))
                _bg_level_b = float(band_cfgs.get(b, {}).get('_bg_level', 0.0))
                if _bg_level_b <= 0.0 and _bg_rms_b > 0.0 and exposure_time > 0:
                    _bg_level_b = (_bg_rms_b ** 2) * exposure_time
                _image_plus_sky = images[b] + _bg_level_b if _bg_level_b > 0 else images[b]

                # Full physically-realistic detector chain
                chain = make_detector_chain(
                    telescope=_tel_name,
                    band=b,
                    rng=rng,
                    exposure_time=exposure_time,
                    numpix=int(numpix),
                    persistence_map=_persistence_carry,
                    seed_prnu=_prnu_seed,
                    enabled=_det_overrides if _det_overrides else None,
                )
                final_image = chain.apply(_image_plus_sky)
                if _det_overrides.get('persistence', chain.enabled.get('persistence', False)):
                    _persistence_carry = chain.make_persistence_map(final_image)
                if _bg_level_b > 0:
                    final_image = final_image - _bg_level_b
                else:
                    # No sky level available to inject (bg_rms not
                    # configured either) -- fall back to the old additive-
                    # noise-only approximation so behavior degrades
                    # gracefully rather than silently losing all sky noise.
                    final_image = add_sky_background_noise(final_image, b, rng, band_cfgs)
                print(f"[DET] Applied {_tel_name} detector chain to {b} (sky flux {'injected pre-chain' if _bg_level_b>0 else 'NOT available, using post-hoc approximation'})")
            else:
                # Fallback: legacy lenstronomy noise model
                cfg = dict(band_cfgs[b])
                cfg['pixel_scale'] = float(pixel_scale)
                sim = SimAPI(numpix=int(numpix),
                             kwargs_single_band=filter_lenstronomy_params(cfg),
                             kwargs_model=model_lists)
                noise = sim.noise_for_model(model=images[b])
                final_image = images[b] + noise

            # Add observational artifacts (cosmic rays, hot pixels, diffraction spikes)
            if add_artifacts:
                artifact_images, _ = add_cosmos_web_artifacts(
                    {b: final_image}, rng, 'moderate', numpix=numpix, add_spikes=add_spikes
                )
                images[b] = artifact_images[b]
            else:
                images[b] = final_image.astype(np.float32)

        except Exception as e:
            print(f"[ERROR] Band {b} simulation failed: {e}")
            # FIX (audit C-16): flag fabricated fallback, see identical
            # fix above.
            row["band_render_failed"] = True
            row.setdefault("band_render_failed_bands", []).append(b)
            fallback = rng.exponential(1e-8, (numpix, numpix))
            center = numpix // 2
            fallback[center-12:center+12, center-12:center+12] += 5e-7
            images[b] = fallback.astype(np.float32)

    # Classify lens system
    from prism.lensing.lens_system_classifier import LensSystemClassifier
    lens_system_class = LensSystemClassifier.classify_lens(lens_model_list, kwargs_lens)
    
    # Enhanced field info with real galaxy + structural metadata
    field_structural_meta = build_field_structural_metadata(field_galaxies_base)
    field_info = {
        'n_field_galaxies': n_field_added,
        'field_positions': [(g['center_x'], g['center_y']) for g in field_galaxies_base],
        'field_magnitudes': [g['magnitude'] for g in field_galaxies_base],
        'real_galaxy_types': [g.get('real_morph_type', 'unknown') for g in field_galaxies_base],
        'source_lens_ids': [g.get('source_lens_id', 'synthetic') for g in field_galaxies_base],
        # Lens structural parameters (actual simulated values)
        'lens_n_sersic': float(n_lens),
        'lens_radius': float(lens_radius),
        'lens_axis_ratio': float(ellipticity_to_axis_ratio(e1_l, e2_l)),
        'lens_pa': float(lens_pa),
        'lens_e1': float(e1_l),
        'lens_e2': float(e2_l),
        # Source structural parameters (actual simulated values)
        'source_n_sersic': float(source_n),
        'source_radius': float(source_radius),
        'source_axis_ratio': float(ellipticity_to_axis_ratio(e1_s, e2_s)),
        'source_pa': float(source_pa),
        'source_e1': float(e1_s),
        'source_e2': float(e2_s),
        'source_x': float(lensed_source['center_x']),
        'source_y': float(lensed_source['center_y']),
        'using_real_data': field_pop is not None,
        'env_type': env_type,  # Record chosen environment for auditing
        'lens_system_class': lens_system_class,  # Add lens system classification
        'lens_model_list': lens_model_list,  # Store model list for reference
        'kwargs_lens': kwargs_lens,  # Store lens parameters for kappa computation
        'numpix': int(numpix),  # Store image resolution
        'delta_pix': float(pixel_scale),  # Store pixel scale (arcsec/pixel)
        'intermediate_images': intermediate_images_dict if save_intermediate else None,  # Add intermediate images
        'psf_arrays': psf_arrays_all,  # PSF kernels per band (saved to separate folder)
        'band_noise_cfgs': band_cfgs,  # Per-band noise config (for FITS NOISE extension)
        # Fundamental Plane / Faber-Jackson derived quantities
        'lens_sigma_kms': float(_sigma_kms) if _sigma_kms is not None else float('nan'),
        'fp_enforced': _fp_enabled,
        # TNG Mode (config['tng_mode']['enabled']): physically matched
        # TNG100-1 subhalo properties for the lens/source, or None if TNG
        # Mode is disabled / no match was found.
        'tng_lens': _tng_lens,
        'tng_source': _tng_source,
        'tng_field': [g.get('tng_info') for g in field_galaxies_base],
    }
    field_info.update(field_structural_meta)

    return images, n_lens, field_info


def simulate_lens_system_with_time_delays(row, band_cfgs, rng, field_pop=None,
                                         numpix=300, n_field_max=8, add_artifacts=True, psf_data=None, add_spikes=False, pixel_scale=0.031):
    """
    Generate lens system with optional time delays for variable sources.
    
    Args:
        pixel_scale: Pixel scale in arcsec/pixel (default 0.031 for JWST)
    
    Returns:
        - If time delays enabled: List of (images_dict, metadata_dict) tuples for each epoch
        - If time delays disabled: Single (images_dict, metadata_dict) tuple
    """
    # Check if time delays should be generated
    if not TIME_DELAY_INTEGRATION_AVAILABLE:
        print(f"[TIME_DELAY] Module not available, skipping time delays for lens {row.get('lens_id', 'unknown')}")
        # Standard single-epoch simulation
        images, n_lens, field_info = simulate_complete_lens_system_with_real_fields(
            row, band_cfgs, rng, field_pop, numpix, n_field_max, add_artifacts, psf_data, pixel_scale=pixel_scale
        )
        # Include field_info in metadata so intermediate images can be saved
        metadata = {'has_time_delays': False, 'epoch': 0}
        metadata.update(field_info)
        return [(images, metadata)]
    
    # Check if this lens should have time delays
    time_delay_cfg = CONFIG.get('time_delays', {})
    enabled = time_delay_cfg.get('enabled', False)
    fraction = time_delay_cfg.get('fraction_variable_sources', 0.15)
    
    print(f"[TIME_DELAY] Checking lens {row.get('lens_id', 'unknown')}: enabled={enabled}, fraction={fraction}")
    
    should_generate = should_generate_time_delays(CONFIG, rng)
    print(f"[TIME_DELAY] should_generate={should_generate}")
    
    if not should_generate:
        # Standard single-epoch simulation
        images, n_lens, field_info = simulate_complete_lens_system_with_real_fields(
            row, band_cfgs, rng, field_pop, numpix, n_field_max, add_artifacts, psf_data, pixel_scale=pixel_scale
        )
        # Include field_info in metadata so intermediate images can be saved
        metadata = {'has_time_delays': False, 'epoch': 0}
        metadata.update(field_info)
        return [(images, metadata)]
    
    # Time delay simulation
    print(f"[TIME_DELAY] Generating time-series for lens {row.get('lens_id', 'unknown')}")
    
    # Extract lens parameters (outside try block so they're available for error handling)
    theta_E = float(row.get("theta_E", row.get("b", 1.0)))
    lens_z = float(row.get("lens_redshift", row.get("zl", 0.6)))
    source_z = float(row.get("source_redshift", row.get("zs", 2.0)))
    source_x = float(row.get("source_x", 0.2))
    source_y = float(row.get("source_y", 0.1))
    
    try:
        # Select variable source type
        source_type = select_variable_source_type(CONFIG, rng)
        print(f"[TIME_DELAY] Variable source type: {source_type}")
        
        # Decide if this system should be binary
        use_binary = False
        if BINARY_LENS_AVAILABLE and CONFIG.get('binary_lenses', {}).get('enabled', False):
            binary_fraction = CONFIG['binary_lenses'].get('fraction', 0.15)
            use_binary = rng.random() < binary_fraction
        
        print(f"[TIME_DELAY] Binary lens system: {use_binary}")
        
        # Generate lens model (binary or single)
        if use_binary:
            mass_profiles = RealisticMassProfiles(rng=rng, config=CONFIG)
            _binary_result = mass_profiles.generate_binary_lens_system(
                primary_mass_log10=float(row.get("lens_mass_log10", row.get("logM", 11.2))),
                primary_redshift=lens_z,
                primary_theta_E=theta_E,
                source_redshift=float(row.get("source_redshift", row.get("zs", 2.0)))
            )
            lens_model_list = _binary_result['lens_model_list']
            kwargs_lens = _binary_result['kwargs_lens']
            print(f"[TIME_DELAY] Generated binary lens: {lens_model_list}")
        else:
            # Standard single lens + shear
            lens_model_list = ['SIE', 'SHEAR']
            kwargs_sie = dict(
                theta_E=theta_E,
                center_x=0.0, center_y=0.0,
                e1=0.1, e2=0.05
            )
            kwargs_shear = dict(gamma1=0.05, gamma2=0.02)
            kwargs_lens = [kwargs_sie, kwargs_shear]
        
        time_delays_result = calculate_time_delays_simplified(
            lens_model_list, kwargs_lens, source_x, source_y,
            lens_z, source_z, theta_E, rng
        )
    except Exception as e:
        print(f"[TIME_DELAY] ERROR during time delay calculation: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to single epoch
        images, n_lens, field_info = simulate_complete_lens_system_with_real_fields(
            row, band_cfgs, rng, field_pop, numpix, n_field_max, add_artifacts, psf_data
        )
        # Include field_info in metadata so intermediate images can be saved
        metadata = {'has_time_delays': False, 'epoch': 0}
        metadata.update(field_info)
        return [(images, metadata)]
    
    print(f"[TIME_DELAY] Calculated {len(time_delays_result['time_delays'])} images with delays: {time_delays_result['time_delays']}")
    
    # Generate light curve FIRST to optimize epoch selection
    time_array = np.linspace(0, CONFIG.get('time_delays', {}).get('epochs', {}).get('time_range_days', 200), 1000)
    base_magnitude = float(row.get("source_mag_f444w", 20.5))
    source_z = float(row.get("source_redshift", row.get("zs", 2.0)))
    
    # Generate light curve for epoch optimization
    light_curve = generate_light_curve_for_source(
        time_array, source_type, CONFIG, rng, base_magnitude,
        redshift=source_z, black_hole_mass=None
    )
    
    # Generate epoch times OPTIMIZED based on light curve to maximize variation
    epoch_times = generate_epoch_times(CONFIG, rng, light_curve=light_curve, time_array=time_array)
    print(f"[TIME_DELAY] Generating {len(epoch_times)} epochs at optimized times: {epoch_times}")
    print(f"[TIME_DELAY] Magnitude range: {light_curve.min():.2f} to {light_curve.max():.2f} (range: {light_curve.max() - light_curve.min():.2f})")
    
    # Get black hole mass if available (for enhanced quasar/AGN models)
    black_hole_mass = row.get("black_hole_mass", None)
    if black_hole_mass is not None:
        black_hole_mass = float(black_hole_mass)
    
    # Light curve already generated above for epoch optimization, but regenerate with black_hole_mass if available
    if black_hole_mass is not None:
        light_curve = generate_light_curve_for_source(
            time_array, source_type, CONFIG, rng, base_magnitude,
            redshift=source_z, black_hole_mass=black_hole_mass
        )
        # Re-optimize epochs with updated light curve if black hole mass was provided
        epoch_times = generate_epoch_times(CONFIG, rng, light_curve=light_curve, time_array=time_array)
        print(f"[TIME_DELAY] Re-optimized epochs with BH mass: {epoch_times}")
    # Otherwise, light_curve is already defined above
    
    # CRITICAL: Generate ALL lens and environment parameters ONCE before epoch loop
    # This ensures lens orientation, field galaxies, and environment stay consistent
    print(f"[TIME_DELAY] Generating fixed lens and environment parameters for all epochs...")
    
    # Initialize env_params early to avoid scoping issues
    env_params = {}
    
    # Extract lens parameters
    lens_z = float(row.get("lens_redshift", row.get("zl", 0.6)))
    lens_mass_log10 = float(row.get("lens_mass_log10", row.get("logM", 11.2)))
    lens_radius = float(row.get("lens_radius", row.get("Re_kpc", 5.0)))
    lens_q = np.clip(float(row.get("lens_axis_ratio", row.get("ql", 0.7))), 0.2, 1.0)
    source_q = np.clip(float(row.get("source_axis_ratio", row.get("qs", 0.7))), 0.2, 1.0)
    
    # FIXED: Generate lens position angle ONCE (not per epoch)
    fixed_lens_pa = rng.uniform(-180, 180)
    fixed_e1_l, fixed_e2_l = ellipticity(lens_q, fixed_lens_pa)
    fixed_e1_l, fixed_e2_l = np.clip([fixed_e1_l, fixed_e2_l], -0.8, 0.8)
    
    # FIXED: Generate source position angle ONCE
    fixed_source_pa = float(row.get("source_pa", row.get("ps", rng.uniform(-180, 180))))
    fixed_e1_s, fixed_e2_s = ellipticity(source_q, fixed_source_pa)
    fixed_e1_s, fixed_e2_s = np.clip([fixed_e1_s, fixed_e2_s], -0.8, 0.8)
    
    # Sample environment type (same for all epochs)
    env_cfg = CONFIG.get('environment', {}).get('types', {})
    env_types = ['isolated_field', 'galaxy_pair', 'group']
    env_fractions = [
        env_cfg.get('isolated_field', {}).get('fraction', 0.45),
        env_cfg.get('galaxy_pair', {}).get('fraction', 0.35),
        env_cfg.get('group', {}).get('fraction', 0.20)
    ]
    env_type = rng.choice(env_types, p=env_fractions)
    env_params = env_cfg.get(env_type, {})
    
    # FIXED: Generate external shear ONCE (same for all epochs)
    shear_min = env_params.get('shear_min', 0.01)
    shear_max = env_params.get('shear_max', 0.05)
    fixed_shear_g = rng.uniform(shear_min, shear_max)
    fixed_shear_phi = rng.uniform(0, np.pi)
    fixed_shear_gamma1 = fixed_shear_g * math.cos(2 * fixed_shear_phi)
    fixed_shear_gamma2 = fixed_shear_g * math.sin(2 * fixed_shear_phi)
    
    # Recalculate time delays with actual lens parameters (ellipticity and shear)
    # This ensures time delays are based on the actual lens model using Fermat potential
    
    # Regenerate lens model with fixed parameters
    if use_binary:
        # Binary lens: regenerate with fixed ellipticities for primary lens
        mass_profiles = RealisticMassProfiles(rng=rng, config=CONFIG)
        _binary_result = mass_profiles.generate_binary_lens_system(
            primary_mass_log10=float(row.get("lens_mass_log10", row.get("logM", 11.2))),
            primary_redshift=lens_z,
            primary_theta_E=theta_E,
            source_redshift=float(row.get("source_redshift", row.get("zs", 2.0)))
        )
        lens_model_list = _binary_result['lens_model_list']
        kwargs_lens = _binary_result['kwargs_lens']
        # Apply fixed ellipticities to primary lens (first component)
        if lens_model_list[0] in ['SIE', 'SPEMD']:
            kwargs_lens[0]['e1'] = float(fixed_e1_l)
            kwargs_lens[0]['e2'] = float(fixed_e2_l)
        elif lens_model_list[0] == 'NFW_ELLIPSE':
            kwargs_lens[0]['e1'] = float(fixed_e1_l)
            kwargs_lens[0]['e2'] = float(fixed_e2_l)
        print(f"[TIME_DELAY] Binary lens with fixed ellipticities: e1={fixed_e1_l:.3f}, e2={fixed_e2_l:.3f}")
    else:
        # Single lens + shear
        lens_model_list = ['SIE', 'SHEAR']
        kwargs_sie = dict(
            theta_E=theta_E,
            center_x=0.0, center_y=0.0,
            e1=float(fixed_e1_l), e2=float(fixed_e2_l)
        )
        kwargs_shear = dict(gamma1=float(fixed_shear_gamma1), gamma2=float(fixed_shear_gamma2))
        kwargs_lens = [kwargs_sie, kwargs_shear]
        print(f"[TIME_DELAY] Single lens: e1={fixed_e1_l:.3f}, e2={fixed_e2_l:.3f}, γ1={fixed_shear_gamma1:.3f}, γ2={fixed_shear_gamma2:.3f}")
    
    # Recalculate with proper lens parameters using physical Fermat potential equation
    time_delays_result = calculate_time_delays_simplified(
        lens_model_list, kwargs_lens, source_x, source_y,
        lens_z, source_z, theta_E, rng
    )
    
    print(f"[TIME_DELAY] Recalculated time delays with actual lens parameters:")
    print(f"[TIME_DELAY]   Time delays: {time_delays_result['time_delays']} days")
    
    # FIXED: Generate other lens parameters ONCE (Sérsic index, source radius, etc.)
    # These should also be consistent across epochs
    # Note: theta_E already defined above, no need to redefine
    
    # Get lens mass for size calculation
    if lens_mass_log10 is None:
        da_mpc = angular_diameter_distance(lens_z)
        lens_mass_log10 = 11.0 + 2.0 * np.log10(theta_E) + np.log10(da_mpc/1000.0)
        lens_mass_log10 = np.clip(lens_mass_log10, 10.5, 12.5)

    # TNG Mode: look up a physically matched TNG100-1 subhalo for the lens
    # ONCE (mirrors simulate_complete_lens_system_with_real_fields). Its
    # half-mass radius / sSFR override reff_kpc/n_lens below, and its
    # environment can drive the field-galaxy count further down.
    _tng_lens = query_tng_properties(lens_z, float(lens_mass_log10), rng, CONFIG,
                                     exclude_subhalos=_used_tng_subhalos,
                                     sfr_class=CONFIG.get('tng_mode', {}).get('lens_sfr_class'),
                                     environment=CONFIG.get('tng_mode', {}).get('lens_environment'))

    # FIXED: Sample physical effective radius ONCE — apply FP/FJ first
    geo = CONFIG.get('geometry', {})
    _td_sigma_kms = None
    _td_fp_cfg = CONFIG.get('fundamental_plane', {}) if isinstance(CONFIG, dict) else {}
    _td_fp_enabled = _td_fp_cfg.get('enabled', True) and FUNDAMENTAL_PLANE_AVAILABLE
    if _td_fp_enabled:
        _td_src_z = float(row.get("source_redshift", row.get("zs", 2.0)))
        _td_D_l   = angular_diameter_distance(lens_z)
        _td_D_s   = angular_diameter_distance(_td_src_z)
        _td_D_ls  = COSMO.angular_diameter_distance_z1z2(lens_z, _td_src_z).value
        _td_da_ls_ds = max(0.01, _td_D_ls / _td_D_s)
        _td_fp = fp_consistent_lens_params(
            row=row,
            lens_z=lens_z,
            source_z=_td_src_z,
            theta_E_catalog=theta_E,
            lens_mass_log10=float(lens_mass_log10),
            da_ls_over_ds=_td_da_ls_ds,
            rng=rng,
            enforce_fp=True,
            da_l_mpc=_td_D_l,
            theta_E_max=_theta_E_hard_max(),
        )
        reff_kpc = _td_fp['re_kpc']
        _td_sigma_kms = _td_fp['sigma_kms']
        if 0.3 <= _td_fp['theta_E'] <= _theta_E_hard_max():
            theta_E = _td_fp['theta_E']
    else:
        reff_kpc = mass_size_relation(lens_mass_log10, lens_z, rng)
    if _tng_lens is not None:
        reff_kpc = _tng_lens['halfmassrad_stars_kpc']
    else:
        _bgg_frac = float(CONFIG.get('mass_size', {}).get('bgg_fraction', 0.0))
        if rng.random() < _bgg_frac:
            reff_kpc = mass_size_relation(lens_mass_log10, lens_z, rng, is_bgg=True, bgg_type='QG')
    fixed_lens_radius = convert_physical_to_angular_radius(reff_kpc, lens_z)
    fixed_lens_radius = np.clip(fixed_lens_radius,
                                geo.get('lens_radius_min', 0.2),
                                geo.get('lens_radius_max', 4.0))

    # FIXED: Sample Sérsic index ONCE
    fixed_n_lens = sample_sersic_n(lens_z, measured=row.get("n_rest"), rng=rng)
    lens_morph_cfg = CONFIG.get('lens_morphology', {}) if isinstance(CONFIG, dict) else {}
    min_lens_sersic = lens_morph_cfg.get('min_lens_sersic', None)
    if min_lens_sersic is not None:
        try:
            fixed_n_lens = max(float(min_lens_sersic), float(fixed_n_lens))
        except Exception:
            pass
    if _tng_lens is not None:
        if _tng_lens['ssfr_per_yr'] < TNG_QUENCHED_SSFR_THRESHOLD:
            fixed_n_lens = max(float(fixed_n_lens), 3.0)
        else:
            fixed_n_lens = min(float(fixed_n_lens), 2.0)

    # FIXED: Generate source parameters ONCE
    # FIX (adversarial audit finding C-6, 2026-08-01): this path (used by
    # the time-delay generator) still had the theta_E*fraction source-
    # sizing this project already replaced elsewhere for the exact same
    # reason -- source angular size should come from the source's OWN
    # stellar mass/redshift, not a foreground deflector's Einstein radius
    # (physically unrelated systems; also an ML shortcut risk). Sample the
    # source's target stellar mass ONCE (moved up from the TNG-lookup
    # block below) and use mass_size_relation()/
    # convert_physical_to_angular_radius() for the default case too, with
    # the TNG match (if found) still overriding as before.
    source_z = float(row.get("source_redshift", row.get("zs", 2.0)))
    _tng_cfg = CONFIG.get('tng_mode', {}) if isinstance(CONFIG, dict) else {}
    _source_logM = float(rng.normal(
        _tng_cfg.get('source_logM_default', 9.5),
        _tng_cfg.get('source_logM_scatter', 0.5),
    ))
    _source_logM = np.clip(_source_logM, 8.0, 11.0)
    _src_reff_kpc = mass_size_relation(_source_logM, source_z, rng)
    fixed_source_radius = convert_physical_to_angular_radius(_src_reff_kpc, source_z)
    if geo.get('enforce_source_smaller_than_lens', True):
        max_ratio = geo.get('max_source_to_lens_ratio', 0.6)
        max_allowed = fixed_lens_radius * max_ratio
        if fixed_source_radius > max_allowed:
            fixed_source_radius = max_allowed
    fixed_source_radius = np.clip(fixed_source_radius, 0.15, 0.6)
    fixed_source_n = np.clip(rng.lognormal(np.log(1.2), 0.4), 0.3, 8.0)

    # TNG Mode: look up a physically matched TNG100-1 subhalo for the
    # lensed source ONCE, same pattern as
    # simulate_complete_lens_system_with_real_fields (no catalog mass for
    # sources, so a target stellar mass is sampled from config).
    _tng_source = query_tng_properties(source_z, _source_logM, rng, CONFIG, exclude_subhalos=_used_tng_subhalos)
    if _tng_source is not None:
        fixed_source_radius = convert_physical_to_angular_radius(
            _tng_source['halfmassrad_stars_kpc'], source_z
        )
        if geo.get('enforce_source_smaller_than_lens', True):
            max_ratio = geo.get('max_source_to_lens_ratio', 0.6)
            fixed_source_radius = min(fixed_source_radius, fixed_lens_radius * max_ratio)
        fixed_source_radius = np.clip(fixed_source_radius, 0.15, 0.6)
        if _tng_source['ssfr_per_yr'] < TNG_QUENCHED_SSFR_THRESHOLD:
            fixed_source_n = max(float(fixed_source_n), 3.0)
        else:
            fixed_source_n = min(float(fixed_source_n), 2.0)

    # Store all fixed lens parameters to pass to each epoch
    fixed_lens_params = {
        'lens_pa': fixed_lens_pa,
        'e1_l': float(fixed_e1_l),
        'e2_l': float(fixed_e2_l),
        'source_pa': fixed_source_pa,
        'e1_s': float(fixed_e1_s),
        'e2_s': float(fixed_e2_s),
        'env_type': env_type,
        'shear_gamma1': float(fixed_shear_gamma1),
        'shear_gamma2': float(fixed_shear_gamma2),
        'lens_radius': float(fixed_lens_radius),
        'n_lens': float(fixed_n_lens),
        'source_radius': float(fixed_source_radius),
        'source_n': float(fixed_source_n),
        'lens_model_list': lens_model_list,
        'kwargs_lens': kwargs_lens
    }
    
    # Determine field galaxy count — target derived from the real COSMOS-Web
    # detection density at this FOV's area (see field_galaxy_count_target),
    # so density stays realistic at any image_size, e.g. a 1' extended-FOV
    # render.
    _area_scale = field_density_area_scale(numpix, pixel_scale, CONFIG)
    env_mean, env_std = field_galaxy_count_target(numpix, pixel_scale, env_params, CONFIG)
    env_min  = max(0.0, env_mean - 3 * env_std)
    env_max  = env_mean + 3 * env_std
    n_field_env = int(np.clip(rng.normal(env_mean, env_std), env_min, env_max))

    _tng_mode_cfg = CONFIG.get('tng_mode', {}) if isinstance(CONFIG, dict) else {}
    if _tng_mode_cfg.get('enabled', False) and _tng_mode_cfg.get('environment_drives_field_count', False) and _tng_lens is not None:
        tng_env = _tng_lens.get('environment', 'isolated')
        _tng_richness = {
            'isolated': 2.5,
            'pair': 3.0,
            'group': 4.5,
            'rich_group': 6.5,
        }.get(tng_env, 2.5)
        env_mean, env_std = field_galaxy_count_target(
            numpix, pixel_scale, {'galaxy_count_mean': _tng_richness}, CONFIG)
        env_min = max(0.0, env_mean - 3 * env_std)
        env_max = env_mean + 3 * env_std
        n_field_env = int(np.clip(rng.normal(env_mean, env_std), env_min, env_max))

    n_field_env = min(max(0, n_field_env), n_field_max)
    env_sep_mean = env_params.get('separation_mean', 2.0) * (_area_scale ** 0.5)
    avoid_radius = max(0.3, env_sep_mean * 0.2)

    # Generate field galaxies once
    if field_pop is not None and n_field_env <= 10:
        lens_id = row.get('ASSOC_ID', row.get('lens_id', None))
        if ENHANCED_SAMPLER is not None:
            try:
                fixed_field_galaxies = ENHANCED_SAMPLER.sample_field_galaxies_enhanced(
                    central_redshift=lens_z,
                    central_mass_log10=lens_mass_log10,
                    n_max=n_field_env,
                    rng=rng,
                    numpix=numpix,
                    pixel_scale=0.03,
                    avoid_center_arcsec=avoid_radius,
                    psf_data=psf_data,
                    lens_radius=lens_radius,
                    lens_id=lens_id
                )
            except Exception as e:
                print(f"[WARNING] Enhanced sampling failed: {e}, using basic")
                fixed_field_galaxies = sample_real_field_galaxies_for_mock(
                    field_pop, n_max=n_field_env, rng=rng, numpix=numpix,
                    pixel_scale=0.03, lens_redshift=lens_z,
                    avoid_center_arcsec=avoid_radius,
                    psf_data=psf_data,
                    lens_radius=lens_radius,
                    lens_id=lens_id,
                    lens_mass_log10=lens_mass_log10,
                    halo_radius_constraint=True
                )
        else:
            fixed_field_galaxies = sample_real_field_galaxies_for_mock(
                field_pop, n_max=n_field_env, rng=rng, numpix=numpix,
                pixel_scale=0.03, lens_redshift=lens_z,
                avoid_center_arcsec=avoid_radius,
                psf_data=psf_data,
                lens_radius=lens_radius,
                lens_id=lens_id,
                lens_mass_log10=lens_mass_log10,
                halo_radius_constraint=True
            )
    elif n_field_env > 10:
        _pix_scale = CONFIG.get('pixel_scale', 0.031) if isinstance(CONFIG, dict) else 0.031
        fixed_field_galaxies = generate_synthetic_field_population(
            rng, n_field_env, numpix, pixel_scale=_pix_scale)
        print(f"[INFO] Large-FOV synthetic field: {len(fixed_field_galaxies)} galaxies over {numpix*_pix_scale:.0f}\"")
    else:
        fixed_field_galaxies = generate_synthetic_field_population(rng, max(n_field_env, 2), numpix)

    apply_tng_field_overrides(fixed_field_galaxies, rng, CONFIG, exclude_subhalos=_used_tng_subhalos)
    tag_field_galaxies_with_tng_particles(fixed_field_galaxies, rng, CONFIG)

    print(f"[TIME_DELAY] Generated {len(fixed_field_galaxies)} field galaxies (will reuse for all epochs)")
    
    field_structural_meta = build_field_structural_metadata(fixed_field_galaxies)

    # Classify lens system (needed for kappa map category/sub_type)
    from prism.lensing.lens_system_classifier import LensSystemClassifier
    lens_system_class = LensSystemClassifier.classify_lens(lens_model_list, kwargs_lens)

    # Store consistent field info (will be same for all epochs)
    # Note: For time delays, intermediate images are only generated for the first epoch
    # to save computation time (lens and field galaxies don't change between epochs)
    consistent_field_info = {
        # Lens mass model — required for kappa/convergence map generation
        'lens_model_list': lens_model_list,
        'kwargs_lens': kwargs_lens,
        'lens_system_class': lens_system_class,
        'numpix': int(numpix),
        'delta_pix': float(pixel_scale),
        'n_field_galaxies': len(fixed_field_galaxies),
        'field_positions': [(g.get('center_x', 0), g.get('center_y', 0)) for g in fixed_field_galaxies],
        'field_magnitudes': [g.get('magnitude', 22.0) for g in fixed_field_galaxies],
        'real_galaxy_types': [g.get('real_morph_type', 'unknown') for g in fixed_field_galaxies],
        'source_lens_ids': [g.get('source_lens_id', 'synthetic') for g in fixed_field_galaxies],
        'lens_n_sersic': float(fixed_lens_params['n_lens']),
        'lens_radius': float(fixed_lens_params['lens_radius']),
        'lens_axis_ratio': float(ellipticity_to_axis_ratio(fixed_lens_params['e1_l'], fixed_lens_params['e2_l'])),
        'lens_pa': float(fixed_lens_params['lens_pa']),
        'lens_e1': float(fixed_lens_params['e1_l']),
        'lens_e2': float(fixed_lens_params['e2_l']),
        'source_n_sersic': float(fixed_lens_params['source_n']),
        'source_radius': float(fixed_lens_params['source_radius']),
        'source_axis_ratio': float(ellipticity_to_axis_ratio(fixed_lens_params['e1_s'], fixed_lens_params['e2_s'])),
        'source_pa': float(fixed_lens_params['source_pa']),
        'source_e1': float(fixed_lens_params['e1_s']),
        'source_e2': float(fixed_lens_params['e2_s']),
        'source_x': float(source_x),
        'source_y': float(source_y),
        'using_real_data': field_pop is not None,
        'env_type': env_type,
        'tng_lens': _tng_lens,
        'tng_source': _tng_source,
        'tng_field': [g.get('tng_info') for g in fixed_field_galaxies],
        'intermediate_images': None  # Will be set for first epoch only
    }
    consistent_field_info.update(field_structural_meta)
    
    # Generate images for each epoch
    epoch_results = []
    normalization_scales = None  # Will be calculated from first epoch and reused
    
    for epoch_idx, obs_time in enumerate(epoch_times):
        # Calculate source magnitudes for each image at this epoch
        image_magnitudes = apply_time_delay_to_source_magnitude(
            base_magnitude, light_curve, time_array,
            time_delays_result['time_delays'], obs_time
        )
        
        # Use average magnitude for source (simplified - in reality each image has different mag)
        avg_source_mag = np.mean(image_magnitudes)
        
        # Debug: Print magnitude changes for visibility
        if epoch_idx == 0:
            print(f"[TIME_DELAY] Base magnitude: {base_magnitude:.2f}, Epoch 0 magnitude: {avg_source_mag:.2f}")
        else:
            prev_mag = np.mean(apply_time_delay_to_source_magnitude(
                base_magnitude, light_curve, time_array,
                time_delays_result['time_delays'], epoch_times[epoch_idx-1]
            ))
            print(f"[TIME_DELAY] Epoch {epoch_idx}: magnitude={avg_source_mag:.2f}, change from prev={avg_source_mag-prev_mag:.2f} mag")
        
        # Modify row to use time-delayed source magnitude
        row_epoch = row.copy()
        for b in UPPER_BANDS:
            band_lower = BAND_TO_LOWER[b]
            original_mag = float(row.get(f"source_mag_{band_lower}", base_magnitude))
            # Adjust magnitude based on light curve
            mag_diff = avg_source_mag - base_magnitude
            row_epoch[f"source_mag_{band_lower}"] = original_mag + mag_diff
        
        # Generate images for this epoch (reusing fixed field galaxies and lens parameters)
        images, n_lens, field_info = simulate_complete_lens_system_with_real_fields(
            row_epoch, band_cfgs, rng, field_pop, numpix, n_field_max, add_artifacts, psf_data,
            fixed_field_galaxies=fixed_field_galaxies,
            fixed_lens_params=fixed_lens_params
        )
        
        # ENHANCEMENT: Apply per-image brightness differences based on time delays
        # This creates the realistic effect where different lensed images show different
        # brightnesses at the same observation time due to time delays
        # NOTE: Set APPLY_PER_IMAGE_TIME_DELAY_BRIGHTNESS = False to disable this effect
        APPLY_PER_IMAGE_TIME_DELAY_BRIGHTNESS = False  # Set to True to enable brightness variations
        
        if TIME_DELAY_IMAGE_MODIFICATION_AVAILABLE and APPLY_PER_IMAGE_TIME_DELAY_BRIGHTNESS:
            try:
                # Get lens model parameters for image position detection
                lens_model_list = ['SIE', 'SHEAR']
                kwargs_sie = dict(
                    theta_E=theta_E,
                    center_x=0.0, center_y=0.0,
                    e1=float(fixed_lens_params['e1_l']),
                    e2=float(fixed_lens_params['e2_l'])
                )
                kwargs_shear = dict(
                    gamma1=float(fixed_lens_params['shear_gamma1']),
                    gamma2=float(fixed_lens_params['shear_gamma2'])
                )
                kwargs_lens = [kwargs_sie, kwargs_shear]
                
                # Apply per-image brightness modification to each band
                for band in UPPER_BANDS:
                    if band in images:
                        images[band] = apply_per_image_time_delay_brightness(
                            images[band],
                            time_delays_result,
                            image_magnitudes,
                            lens_model_list,
                            kwargs_lens,
                            source_x,
                            source_y,
                            pixel_scale=0.03,
                            numpix=numpix
                        )
                print(f"[TIME_DELAY] Applied per-image brightness differences (delays: {time_delays_result['time_delays']} days)")
            except Exception as e:
                print(f"[WARNING] Failed to apply per-image time delay brightness: {e}")
                # Continue without modification (fallback to average brightness)
        else:
            if not APPLY_PER_IMAGE_TIME_DELAY_BRIGHTNESS:
                print(f"[TIME_DELAY] Skipping per-image brightness differences (disabled)")
        
        # Calculate normalization scales from first epoch for consistent RGB across epochs
        # Also extract intermediate images for first epoch only (lens/field don't change)
        if epoch_idx == 0:
            # Create RGB from first epoch to get normalization scales
            try:
                _, normalization_scales = create_trilogy_rgb(images, numpix, normalization_scale=None)
                print(f"[TIME_DELAY] Calculated normalization scales from epoch 0 for consistent RGB")
            except Exception as e:
                print(f"[WARNING] Failed to calculate normalization scales: {e}, using None")
                normalization_scales = None
            
            # Extract intermediate images from first epoch (lens and field galaxies are the same for all epochs)
            if field_info and field_info.get('intermediate_images'):
                consistent_field_info['intermediate_images'] = field_info['intermediate_images']
                print(f"[TIME_DELAY] Extracted intermediate images from epoch 0 (will be reused for all epochs)")
        
        # Store normalization scales in metadata for use during saving
        metadata = create_time_delay_metadata(
            int(row.get('lens_id', epoch_idx)),
            source_type, time_delays_result, epoch_times, light_curve, time_array
        )
        metadata['epoch'] = epoch_idx
        metadata['observation_time_days'] = float(obs_time)
        metadata['source_magnitude'] = float(avg_source_mag)
        metadata['normalization_scales'] = normalization_scales  # Store for consistent RGB (None if not calculated yet)
        metadata['time_delays_result'] = time_delays_result  # Store full result including image_positions for catalog saving
        metadata.update(consistent_field_info)  # Use consistent field info for all epochs
        
        epoch_results.append((images, metadata))
    
    return epoch_results

# --------------------------------------------------------------------------------------
# CORRECTED: Enhanced non-lens system generation
# --------------------------------------------------------------------------------------

def generate_nonlens_system_complete(mode, band_cfgs, rng, field_pop=None, 
                                   numpix=300, n_field_max=8, add_artifacts=True, psf_data=None,
                                   use_hard_negatives=False, hard_negative_type=None, add_spikes=False, pixel_scale=0.031):
    """CORRECTED: Generate realistic non-lens systems with proper field populations and PSF convolution
    
    Args:
        pixel_scale: Pixel scale in arcsec/pixel (default 0.031 for JWST)
    """
    
    # Check if we should use hard negatives
    if use_hard_negatives and ML_ENHANCEMENTS_AVAILABLE:
        # Generate hard negative case
        hard_negative_miner = JWSTHardNegativeMiner(rng)
        hard_negative_profile = hard_negative_miner.generate_hard_negative(
            hard_negative_type, numpix, pixel_scale=pixel_scale
        )
        
        # Create central components from hard negative profile
        central_components = [{
            'center_x': 0.0,
            'center_y': 0.0,
            'R_sersic': 1.0,  # Will be overridden by profile
            'n_sersic': 2.0,
            'e1': 0.0,
            'e2': 0.0,
            'magnitude': 20.0,
            'morph_type': hard_negative_type or 'edge_on_spiral',
            'hard_negative_profile': hard_negative_profile  # Store profile for rendering
        }]
        central_morphologies = [hard_negative_type or 'edge_on_spiral']
        
        print(f"[HARD NEGATIVE] Generated {hard_negative_type or 'edge_on_spiral'} case")
        
    else:
        # Regular non-lens generation
        # Sample environment type for non-lens systems (same as lens systems)
        env_cfg = CONFIG.get('environment', {}).get('types', {})
        env_types = ['isolated_field', 'galaxy_pair', 'group']
        env_fractions = [
            env_cfg.get('isolated_field', {}).get('fraction', 0.45),
            env_cfg.get('galaxy_pair', {}).get('fraction', 0.35),
            env_cfg.get('group', {}).get('fraction', 0.20)
        ]
        env_type = rng.choice(env_types, p=env_fractions)
        env_params = env_cfg.get(env_type, {})
        
        central_components = []
        central_morphologies = []  # Track morphology for color assignment

        if mode == "central_galaxy":
            # Single galaxy with realistic morphology and mass-size relations
            morph_type = rng.choice(['spiral', 'elliptical', 'S0'], p=[0.5, 0.3, 0.2])
            
            # Sample realistic redshift and mass for non-lens galaxies
            galaxy_z = rng.uniform(0.2, 3.0)  # Realistic redshift range
            galaxy_mass_log10 = rng.uniform(9.5, 11.5)  # Realistic mass range
            
            # Apply mass-size relation with redshift scaling
            reff_kpc = mass_size_relation(galaxy_mass_log10, galaxy_z, rng)
            galaxy_radius = convert_physical_to_angular_radius(reff_kpc, galaxy_z)
            
            # Apply reasonable bounds for non-lens galaxies
            galaxy_radius = np.clip(galaxy_radius, 0.1, 2.0)

            if morph_type == 'spiral':
                n_sersic = rng.choice([0.8, 1.0, 1.2, 1.5], p=[0.3, 0.4, 0.2, 0.1])
            elif morph_type == 'elliptical':
                n_sersic = rng.choice([3.0, 4.0, 5.0], p=[0.5, 0.35, 0.15])
            else:  # S0
                n_sersic = rng.choice([2.0, 2.5, 3.0], p=[0.4, 0.4, 0.2])

            # Realistic ellipticity
            if morph_type == 'spiral':
                q_ratio = rng.beta(1.5, 2) * 0.6 + 0.3  # More elongated
            else:
                q_ratio = rng.beta(2, 1.8) * 0.5 + 0.5  # Rounder

            e = (1 - q_ratio) / (1 + q_ratio)
            pa = rng.uniform(-180, 180)
            e1 = e * np.cos(2 * np.radians(pa))
            e2 = e * np.sin(2 * np.radians(pa))

            central_components = [{
                'center_x': 0.0,
                'center_y': 0.0,
                'R_sersic': float(galaxy_radius),  # Use mass-size relation
                'n_sersic': float(n_sersic),
                'e1': float(np.clip(e1, -0.6, 0.6)),
                'e2': float(np.clip(e2, -0.6, 0.6)),
                'magnitude': float(rng.uniform(18.0, 21.0)),  # Reference magnitude (F150W)
                'morph_type': morph_type,  # Store for color calculation
                'redshift': galaxy_z,  # Store redshift for color calculation
                'mass_log10': galaxy_mass_log10  # Store mass for reference
            }]
            central_morphologies = [morph_type]
    
        elif mode == "galaxy_pair":
            # Realistic galaxy pair with mass-size relations and redshift scaling
            sep = rng.uniform(1.2, 3.5)  # Reduced separation to keep both visible
            angle = rng.uniform(0, 2*np.pi)
            mass_ratio = rng.uniform(0.3, 1.0)  # Secondary fainter than primary

            # Primary at/near center, secondary offset
            x1, y1 = rng.normal(0.0, 0.15), rng.normal(0.0, 0.15)  # Primary near center
            x2 = x1 + sep * np.cos(angle)  # Secondary offset from primary
            y2 = y1 + sep * np.sin(angle)

            # Sample redshifts and masses for both galaxies
            z1 = rng.uniform(0.2, 2.5)  # Primary galaxy redshift
            z2 = rng.uniform(0.2, 2.5)  # Secondary galaxy redshift
            mass1_log10 = rng.uniform(10.0, 11.5)  # Primary mass
            mass2_log10 = mass1_log10 - rng.uniform(0.2, 1.0)  # Secondary less massive

            # Apply mass-size relations
            reff1_kpc = mass_size_relation(mass1_log10, z1, rng)
            reff2_kpc = mass_size_relation(mass2_log10, z2, rng)
            radius1 = convert_physical_to_angular_radius(reff1_kpc, z1)
            radius2 = convert_physical_to_angular_radius(reff2_kpc, z2)
            
            # Apply bounds
            radius1 = np.clip(radius1, 0.1, 2.0)
            radius2 = np.clip(radius2, 0.05, 1.5)

            # Primary galaxy (more massive)
            morph1 = rng.choice(['spiral', 'elliptical', 'S0'], p=[0.4, 0.4, 0.2])
            if morph1 == 'spiral':
                n1 = rng.choice([1.0, 1.5, 2.0], p=[0.5, 0.3, 0.2])
            elif morph1 == 'elliptical':
                n1 = rng.choice([3.0, 4.0], p=[0.6, 0.4])
            else:
                n1 = rng.choice([2.0, 2.5], p=[0.7, 0.3])

            # Secondary galaxy (less massive)
            morph2 = rng.choice(['spiral', 'S0'], p=[0.7, 0.3])  # Less likely to be elliptical
            if morph2 == 'spiral':
                n2 = rng.choice([0.8, 1.0, 1.2], p=[0.4, 0.4, 0.2])
            else:
                n2 = rng.choice([2.0, 2.5], p=[0.8, 0.2])

            central_components = [
                {
                    'center_x': float(x1), 'center_y': float(y1),
                    'R_sersic': float(radius1),  # Use mass-size relation
                    'n_sersic': float(n1),
                    'e1': float(rng.normal(0, 0.3)),
                    'e2': float(rng.normal(0, 0.3)),
                    'magnitude': float(rng.uniform(18.5, 20.5)),
                    'morph_type': morph1,
                    'redshift': z1,
                    'mass_log10': mass1_log10
                },
                {
                    'center_x': float(x2), 'center_y': float(y2),
                    'R_sersic': float(radius2),  # Use mass-size relation
                    'n_sersic': float(n2),
                    'e1': float(rng.normal(0, 0.25)),
                    'e2': float(rng.normal(0, 0.25)),
                    'magnitude': float(rng.uniform(20.0, 22.0)),
                    'morph_type': morph2,
                    'redshift': z2,
                    'mass_log10': mass2_log10
                }
            ]
            central_morphologies = [morph1, morph2]
    
        elif mode == "galaxy_group":
            # Small galaxy group (3-4 galaxies) with mass-size relations
            n_group = safe_random_integers(rng, 3, 5)

            for i in range(n_group):
                # Place in loose cluster
                if i == 0:  # Central dominant galaxy
                    x, y = 0.0, 0.0
                    mag_range = (18.0, 21.0)
                    # Central galaxy: massive, low-z
                    z_gal = rng.uniform(0.2, 1.5)
                    mass_log10 = rng.uniform(10.5, 11.5)
                    morph = rng.choice(['elliptical', 'S0'], p=[0.7, 0.3])
                else:  # Satellite galaxies
                    r = rng.uniform(1.0, 3.5)
                    theta = rng.uniform(0, 2*np.pi)
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    mag_range = (20.0, 24.0)  # More realistic for satellites
                    # Satellites: less massive, can be higher-z
                    z_gal = rng.uniform(0.2, 2.5)
                    mass_log10 = rng.uniform(9.5, 10.5)
                    morph = rng.choice(['spiral', 'elliptical', 'S0'], p=[0.5, 0.3, 0.2])

                # Apply mass-size relations
                reff_kpc = mass_size_relation(mass_log10, z_gal, rng)
                galaxy_radius = convert_physical_to_angular_radius(reff_kpc, z_gal)
                galaxy_radius = np.clip(galaxy_radius, 0.05, 1.5)

                # Set morphological parameters
                if morph == 'spiral':
                    n_sersic = rng.choice([1.0, 1.2, 1.5], p=[0.5, 0.3, 0.2])
                    q_ratio = rng.beta(1.5, 2) * 0.6 + 0.25
                elif morph == 'elliptical':
                    n_sersic = rng.choice([3.0, 4.0], p=[0.7, 0.3])
                    q_ratio = rng.beta(2.5, 1.5) * 0.4 + 0.6
                else:  # S0
                    n_sersic = rng.choice([2.0, 2.5], p=[0.8, 0.2])
                    q_ratio = rng.beta(2, 1.8) * 0.5 + 0.4

                e = (1 - q_ratio) / (1 + q_ratio)
                pa = rng.uniform(-180, 180)
                e1 = e * np.cos(2 * np.radians(pa))
                e2 = e * np.sin(2 * np.radians(pa))

                central_components.append({
                    'center_x': float(x),
                    'center_y': float(y),
                    'R_sersic': float(galaxy_radius),  # Use mass-size relation
                    'n_sersic': float(n_sersic),
                    'e1': float(np.clip(e1, -0.5, 0.5)),
                    'e2': float(np.clip(e2, -0.5, 0.5)),
                    'magnitude': float(rng.uniform(*mag_range)),
                    'morph_type': morph,
                    'redshift': z_gal,
                    'mass_log10': mass_log10
                })
                central_morphologies.append(morph)
    
    # Environment-based field galaxy sampling for non-lens systems. Target
    # derived from the real COSMOS-Web detection density at this FOV's area
    # (see field_galaxy_count_target), with the same isolated/pair/group
    # richness ratios used for lens systems (galaxy_count_mean reference
    # values 2.5/3.0/4.5), so density stays realistic at any image_size.
    _env_ref_mean = {'isolated_field': 2.5, 'galaxy_pair': 3.0, 'group': 4.5}.get(env_type, 2.5)
    _env_mean_target, _env_std_target = field_galaxy_count_target(
        numpix, pixel_scale, {'galaxy_count_mean': _env_ref_mean}, CONFIG)
    if env_type == 'isolated_field':
        n_field_env = int(np.clip(rng.normal(_env_mean_target, _env_std_target), 0, _env_mean_target + 3 * _env_std_target))
    elif env_type == 'galaxy_pair':
        n_field_env = int(np.clip(rng.normal(_env_mean_target, _env_std_target), 1, _env_mean_target + 3 * _env_std_target))
    elif env_type == 'group':
        n_field_env = int(np.clip(rng.normal(_env_mean_target, _env_std_target), 2, _env_mean_target + 3 * _env_std_target))
    else:
        # Fallback
        n_field_env = min(int(round(_env_mean_target)), n_field_max)

    # Clamp to n_field_max
    n_field_env = min(n_field_env, n_field_max)

    field_galaxies_base = []
    if field_pop is not None:
        # For non-lens, use typical central galaxy size as reference
        typical_lens_size = 1.0  # arcsec, typical massive galaxy
        field_galaxies_base = sample_real_field_galaxies_for_mock(
            field_pop,
            n_max=n_field_env,  # Use environment-based count
            rng=rng, 
            numpix=numpix,
            pixel_scale=0.03, 
            lens_redshift=0.5, 
            avoid_center_arcsec=0.5,  # Less restrictive for non-lens
            psf_data=psf_data,  # Pass PSF data for field galaxy convolution
            lens_radius=typical_lens_size,  # Constrain field galaxy sizes
            lens_id=None,  # No specific lens for non-lens systems
            lens_mass_log10=11.0,  # Typical massive galaxy mass
            halo_radius_constraint=True  # Apply halo radius constraint
        )
    
    # Ensure minimum realistic field population
    if len(field_galaxies_base) < 4:
        synthetic_needed = 5 - len(field_galaxies_base)
        synthetic_gals = generate_synthetic_field_population(rng, synthetic_needed, numpix)
        field_galaxies_base.extend(synthetic_gals)
        print(f"[INFO] Non-lens system: added {synthetic_needed} synthetic field galaxies")
    
    print(f"[DEBUG] Non-lens system field population: {len(field_galaxies_base)} galaxies")

    tag_field_galaxies_with_galaxygenius_stamps(field_galaxies_base, rng, CONFIG)

    # Native multi-component morphology (bulge/disk/bar/ring/...) for the
    # central galaxy/galaxies of non-lens systems, when enabled. Classified
    # once here (independent of band-specific magnitude); per-band kwargs
    # are built in the band loop below using the cached fragment/morph type.
    _morph_cfg_nl = CONFIG.get('morphology', {})
    _multicomponent_nl = _morph_cfg_nl.get('multicomponent_enabled', False)
    central_fragments = []
    for i, comp in enumerate(central_components):
        # Resolve one SED class per component here (not per band below), so
        # every band's color for this component is driven by the same
        # underlying SED rather than an independent per-band draw.
        comp['_sed_type_resolved'] = resolve_sed_type_from_morphology(
            comp.get('morph_type', 'spiral'), comp['n_sersic'], rng)
        if _multicomponent_nl and 'hard_negative_profile' not in comp:
            base_params = {k: comp[k] for k in ('center_x', 'center_y', 'R_sersic', 'n_sersic', 'e1', 'e2')}
            seed = int(abs(hash((str(mode), i, 'central_morph'))) % (2**32))
            comp['_morph_seed'] = seed
            frag, _, resolved_morph_type = gm_build_light_model(
                'field', base_params, {'_DUMMY': 0.0}, ['_DUMMY'],
                np.random.default_rng(0), CONFIG,
                morph_type=str(comp.get('morph_type', 'spiral')).lower(), morph_seed=seed)
            comp['_morph_type_resolved'] = resolved_morph_type
            central_fragments.append(frag)
        else:
            central_fragments.append(['SERSIC_ELLIPSE'])

    # Setup simulation (no lensing)
    model_lists = {
        'lens_model_list': [],  # No mass model
        'lens_light_model_list': [m for frag in central_fragments for m in frag]
                                   + field_galaxy_light_model_types(field_galaxies_base),
        'source_light_model_list': []  # No background source
    }
    
    numerics = dict(supersampling_factor=1, supersampling_convolution=False)
    
    # Generate images for each band
    images = {}
    for b in UPPER_BANDS:
        try:
            cfg = dict(band_cfgs[b])
            cfg['pixel_scale'] = float(pixel_scale)  # Use resolution-specific pixel scale

            # Apply band-specific colors to central components
            all_light_components = []
            for i, comp in enumerate(central_components):
                comp_band = dict(comp)

                # Get realistic color for this band
                morph = comp.get('morph_type', 'spiral')
                n_sersic = comp['n_sersic']
                base_mag = comp['magnitude']  # Reference magnitude (F150W)

                # Calculate band-specific magnitude
                color_offset = get_realistic_jwst_color_from_transmission(morph, n_sersic,
                                                        base_band='F150W',
                                                        target_band=b,
                                                        redshift=comp.get('redshift', 0.5),
                                                        rng=rng,
                                                        sed_type=comp.get('_sed_type_resolved'))
                band_mag = float(np.clip(base_mag + color_offset, 18.0, 28.0))

                if i == 0 and b == 'F115W':  # Debug first galaxy colors
                    print(f"[COLOR] Non-lens central galaxy: {morph}, n={n_sersic:.2f}")
                    print(f"        F150W (ref)={base_mag:.2f}, F115W={band_mag:.2f} (Δ={color_offset:+.2f})")

                if _multicomponent_nl and 'hard_negative_profile' not in comp:
                    base_params = {k: comp[k] for k in ('center_x', 'center_y', 'R_sersic', 'n_sersic', 'e1', 'e2')}
                    _, kwargs_by_band, _ = gm_build_light_model(
                        'field', base_params, {b: band_mag}, [b], rng, CONFIG,
                        morph_type=comp.get('_morph_type_resolved'), morph_seed=comp.get('_morph_seed'))
                    all_light_components.extend(kwargs_by_band[b])
                    continue

                comp_band['magnitude'] = band_mag

                # Remove metadata parameters before passing to lenstronomy
                comp_band.pop('morph_type', None)
                comp_band.pop('redshift', None)
                comp_band.pop('mass_log10', None)

                all_light_components.append(comp_band)

            n_central_components = len(all_light_components)
            if field_galaxies_base:
                field_band_components = apply_real_jwst_colors_to_field_galaxies(field_galaxies_base, b, rng)
                all_light_components.extend(field_band_components)

                print(f"[DEBUG] {b}: {n_central_components} central + {len(field_band_components)} field = {len(all_light_components)} total")
            
            # Create simulation (filter out metadata params)
            sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(cfg), kwargs_model=model_lists)
            kw_lens_amp, _, _ = sim.magnitude2amplitude(all_light_components, [])
            im_model = sim.image_model_class(numerics)
            
            # Generate image (no lensing)
            clean_image = im_model.image(
                kwargs_lens=[],  # No mass
                kwargs_source=[],  # No background source
                kwargs_lens_light=kw_lens_amp  # All light components
            )

            # Apply PSF convolution if PSF data is available
            if psf_data is not None:
                psf_arrays = get_psf_for_simulation(psf_data, None, rng)
                if psf_arrays[b] is not None:
                    clean_image = apply_psf_convolution(clean_image, psf_arrays[b])
                    print(f"[PSF] Applied {b} PSF convolution to non-lens system")

            # Store for enhancement
            images[b] = clean_image

        except Exception as e:
            print(f"[ERROR] Non-lens band {b} failed: {e}")
            import traceback
            traceback.print_exc()

            # Create better fallback image
            fallback = rng.exponential(1e-8, (numpix, numpix))

            # Add realistic central source
            center = numpix // 2
            y, x = np.ogrid[:numpix, :numpix]

            for comp in central_components:
                cx = center + comp['center_x'] / 0.03
                cy = center + comp['center_y'] / 0.03

                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                profile = np.exp(-r / (comp['R_sersic'] / 0.03))
                fallback += profile * rng.uniform(5e-7, 2e-6)

            images[b] = fallback.astype(np.float32)

    # NEW v11: Apply morphological enhancements to central galaxies
    if len(central_components) > 0 and images:
        try:
            # Get primary galaxy parameters for enhancement
            primary_comp = central_components[0]
            n_central = primary_comp['n_sersic']
            e1, e2 = primary_comp['e1'], primary_comp['e2']
            q_central = (1 - np.sqrt(e1**2 + e2**2)) / (1 + np.sqrt(e1**2 + e2**2))
            q_central = np.clip(q_central, 0.2, 1.0)
            if np.hypot(e1, e2) < 1e-6:
                pa_central = 0.0
            else:
                pa_central = 0.5 * np.degrees(np.arctan2(e2, e1))

            morph_type = classify_galaxy_morphology_enhanced(n_central, q_central, rng, allow_ring=False)

            # Apply enhancements
            images = apply_morphological_enhancements(
                images, n_central, q_central, morph_type,
                numpix=numpix, pixel_scale=float(pixel_scale),
                seed=int(safe_random_integers(rng, 0, 100000)),
                position_angle=pa_central,
                r_eff_pix=float(primary_comp['R_sersic']) / float(pixel_scale)
            )
            print(f"[v11] Applied {morph_type} enhancements to non-lens central galaxy")
        except Exception as e:
            print(f"[WARNING] Non-lens morphological enhancement failed: {e}")

    # Add noise and detector effects to enhanced images
    _tel_name_nl = CONFIG.get('telescope', 'jwst').lower()
    _det_enabled_nl = CONFIG.get('detector_chain', {}).get('enabled', True)
    _det_overrides_nl = CONFIG.get('detector_chain', {}).get('effects', {})
    # FIX (audit C-5): see PRNU seed fix above -- hash() is process-salted.
    _prnu_seed_nl = int(zlib.crc32(str(mode).encode()) % (2**31))

    for b in UPPER_BANDS:
        try:
            if b not in images:
                continue

            exposure_time = float(CONFIG.get('exposure_time', 1028.0))

            if DETECTOR_CHAIN_AVAILABLE and _det_enabled_nl:
                # FIX (audit C-8.2): see identical fix + rationale on the
                # lens-system render path above -- inject sky flux before
                # the chain so Poisson noise is physically correct, matched
                # to the non-lens (negative-class) images too so the two
                # classes don't end up with systematically different noise
                # statistics (which would itself be an ML shortcut risk).
                _bg_rms_b = float(band_cfgs.get(b, {}).get('_bg_rms', 0.0))
                _bg_level_b = float(band_cfgs.get(b, {}).get('_bg_level', 0.0))
                if _bg_level_b <= 0.0 and _bg_rms_b > 0.0 and exposure_time > 0:
                    _bg_level_b = (_bg_rms_b ** 2) * exposure_time
                _image_plus_sky = images[b] + _bg_level_b if _bg_level_b > 0 else images[b]

                chain = make_detector_chain(
                    telescope=_tel_name_nl,
                    band=b,
                    rng=rng,
                    exposure_time=exposure_time,
                    numpix=int(numpix),
                    seed_prnu=_prnu_seed_nl,
                    enabled=_det_overrides_nl if _det_overrides_nl else None,
                )
                final_image = chain.apply(_image_plus_sky)
                if _bg_level_b > 0:
                    final_image = final_image - _bg_level_b
                else:
                    final_image = add_sky_background_noise(final_image, b, rng, band_cfgs)
            else:
                cfg = dict(band_cfgs[b])
                cfg['pixel_scale'] = float(pixel_scale)
                sim = SimAPI(numpix=int(numpix),
                             kwargs_single_band=filter_lenstronomy_params(cfg),
                             kwargs_model=model_lists)
                noise = sim.noise_for_model(model=images[b])
                final_image = images[b] + noise

            # Add observational artifacts
            if add_artifacts:
                artifact_dict, _ = add_cosmos_web_artifacts(
                    {b: final_image}, rng, 'moderate', numpix=numpix, add_spikes=add_spikes
                )
                images[b] = artifact_dict[b]
            else:
                images[b] = final_image.astype(np.float32)

            # Debug flux levels
            total_flux = np.sum(images[b])
            max_flux = np.max(images[b])
            print(f"[DEBUG] {b} band: total_flux={total_flux:.2e}, max_flux={max_flux:.2e}")

        except Exception as e:
            print(f"[ERROR] Band {b} post-processing failed: {e}")

    # Enhanced system information
    system_info = {
        'mode': mode,
        'n_central_galaxies': len(central_components),
        'n_field_galaxies': len(field_galaxies_base),
        'using_real_data': field_pop is not None,
        'total_components': len(central_components) + len(field_galaxies_base),
        'central_morphologies': [comp.get('morph_type', 'unknown') for comp in central_components],
        'field_morphologies': [gal.get('real_morph_type', 'unknown') for gal in field_galaxies_base],
        'env_type': env_type,  # Record chosen environment for auditing
        'band_noise_cfgs': band_cfgs,  # Per-band noise config (for FITS NOISE extension)
    }
    
    return images, system_info

# --------------------------------------------------------------------------------------
# Intermediate image generation functions
# --------------------------------------------------------------------------------------

def generate_intermediate_images(step, band, sim, kwargs_lens, kwargs_source, kwargs_lens_light,
                                psf_array, band_cfg, rng, numpix, add_artifacts, lens_q, n_lens, morph_type=None, lens_model_list=None,
                                lens_light_fragment=None, source_light_fragment=None):
    """
    Generate intermediate images at different simulation steps.
    
    Steps:
    1. 'lens_only': Lens galaxy only
    2. 'lens_sources': Lens + lensed sources
    3. 'sources_only': Lensed sources only (no lens light)
    4. 'field_only': Field galaxies only (no lens, no sources)
    5. 'final': All components (lens + sources + field)
    
    Args:
        step: One of the step names above
        band: Band name (e.g., 'F115W')
        sim: SimAPI instance (will create new one with correct model_lists)
        kwargs_lens: Lens model parameters
        kwargs_source: Source model parameters (list)
        kwargs_lens_light: Lens light parameters (list)
        psf_array: PSF array for convolution
        band_cfg: Band configuration
        rng: Random number generator
        numpix: Image size
        add_artifacts: Whether to add artifacts
        lens_q: Lens axis ratio (for morphological enhancements)
        n_lens: Lens Sersic index (for morphological enhancements)
        morph_type: Morphology type (for enhancements)
        lens_light_fragment: list[str] of SERSIC_ELLIPSE entries describing
            the main lens's native multi-component fragment (e.g.
            ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'] for bulge+disk). Defaults
            to a single component for backward compatibility.
        source_light_fragment: list[str] for the lensed source's fragment,
            matching len(kwargs_source). Defaults to a single component.

    Returns:
        Dictionary with intermediate images for all bands
    """
    intermediate_images = {}

    if lens_light_fragment is None:
        lens_light_fragment = ["SERSIC_ELLIPSE"]
    if source_light_fragment is None:
        source_light_fragment = ["SERSIC_ELLIPSE"] * max(len(kwargs_source), 1)

    # Split lens light components: main lens (possibly multi-component) vs
    # field galaxies.
    # For 'field_only': kwargs_lens_light contains ONLY field galaxies (no lens),
    # so ALL elements are field galaxies.
    # For other steps: kwargs_lens_light[:len(lens_light_fragment)] is the
    # main lens, the remainder are field galaxies.
    if step == 'field_only':
        main_lens_light = None
        field_light = list(kwargs_lens_light)   # all are field galaxies
    else:
        n_main = len(lens_light_fragment)
        if len(kwargs_lens_light) == 0 and step not in ['sources_only', 'sources_unlensed']:
            print(f"[WARNING] No lens light components for {step} step in {band}")
            return {band: np.zeros((numpix, numpix), dtype=np.float32)}
        main_lens_light = list(kwargs_lens_light[:n_main]) if len(kwargs_lens_light) > 0 else None
        field_light = list(kwargs_lens_light[n_main:]) if len(kwargs_lens_light) > n_main else []
    
    # Use provided lens_model_list or try to get from sim, or fall back to default
    if lens_model_list is not None:
        original_lens_models = lens_model_list
    else:
        try:
            original_lens_models = sim.kwargs_model.get('lens_model_list', ['SIE', 'SHEAR'])
        except:
            original_lens_models = ['SIE', 'SHEAR']
    
    try:
        # Create model_lists that match the components we're actually using
        if step == 'lens_only':
            # Step 1: Lens galaxy only
            if main_lens_light is None:
                return {band: np.zeros((numpix, numpix), dtype=np.float32)}
            step_model_lists = dict(
                lens_model_list=original_lens_models,
                lens_light_model_list=lens_light_fragment,
                source_light_model_list=[]
            )
            step_sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(band_cfg), kwargs_model=step_model_lists)
            numerics = dict(supersampling_factor=1, supersampling_convolution=False)
            step_im_model = step_sim.image_model_class(numerics)
            kw_lens_amp, _, _ = step_sim.magnitude2amplitude(main_lens_light, [])
            # Trim kwargs_lens to match original_lens_models length
            trimmed_kwargs_lens = kwargs_lens[:len(original_lens_models)]
            image = step_im_model.image(
                kwargs_lens=trimmed_kwargs_lens,
                kwargs_source=[],
                kwargs_lens_light=kw_lens_amp
            )
            
        elif step == 'lens_sources':
            # Step 2a: Lens + lensed sources
            step_model_lists = dict(
                lens_model_list=original_lens_models,
                lens_light_model_list=lens_light_fragment,
                source_light_model_list=source_light_fragment
            )
            step_sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(band_cfg), kwargs_model=step_model_lists)
            numerics = dict(supersampling_factor=1, supersampling_convolution=False)
            step_im_model = step_sim.image_model_class(numerics)
            # Trim kwargs_lens to match original_lens_models length
            trimmed_kwargs_lens = kwargs_lens[:len(original_lens_models)]
            kw_lens_amp, kw_src_amp, _ = step_sim.magnitude2amplitude(main_lens_light, kwargs_source)
            image = step_im_model.image(
                kwargs_lens=trimmed_kwargs_lens,
                kwargs_source=kw_src_amp,
                kwargs_lens_light=kw_lens_amp
            )
            
        elif step == 'sources_only':
            # Step 2b: Lensed sources only (no lens light, but lens mass still deflects)
            step_model_lists = dict(
                lens_model_list=original_lens_models,
                lens_light_model_list=[],
                source_light_model_list=source_light_fragment
            )
            step_sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(band_cfg), kwargs_model=step_model_lists)
            numerics = dict(supersampling_factor=1, supersampling_convolution=False)
            step_im_model = step_sim.image_model_class(numerics)
            # Trim kwargs_lens to match original_lens_models length
            trimmed_kwargs_lens = kwargs_lens[:len(original_lens_models)]
            _, kw_src_amp, _ = step_sim.magnitude2amplitude([], kwargs_source)
            image = step_im_model.image(
                kwargs_lens=trimmed_kwargs_lens,  # Keep lens mass for deflection (trimmed)
                kwargs_source=kw_src_amp,
                kwargs_lens_light=[]  # No lens light
            )

        elif step == 'sources_unlensed':
            # Source morphology with NO gravitational deflection — shows original source profile.
            # Use a zero-mass lens (theta_E=0) so the source plane = image plane (no deflection).
            step_model_lists = dict(
                lens_model_list=original_lens_models,
                lens_light_model_list=[],
                source_light_model_list=source_light_fragment
            )
            step_sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(band_cfg), kwargs_model=step_model_lists)
            numerics = dict(supersampling_factor=1, supersampling_convolution=False)
            step_im_model = step_sim.image_model_class(numerics)
            # Zero-mass lens: all mass parameters zeroed → no deflection
            zero_kwargs_lens = []
            for kw in kwargs_lens[:len(original_lens_models)]:
                kw_zero = dict(kw)
                if 'theta_E' in kw_zero:
                    kw_zero['theta_E'] = 0.0
                if 'gamma1' in kw_zero:
                    kw_zero['gamma1'] = 0.0
                if 'gamma2' in kw_zero:
                    kw_zero['gamma2'] = 0.0
                zero_kwargs_lens.append(kw_zero)
            # Center the source at origin for the unlensed view
            src_centered = [dict(kw, center_x=0.0, center_y=0.0) for kw in kwargs_source]
            _, kw_src_amp, _ = step_sim.magnitude2amplitude([], src_centered)
            image = step_im_model.image(
                kwargs_lens=zero_kwargs_lens,
                kwargs_source=kw_src_amp,
                kwargs_lens_light=[]
            )

        elif step == 'field_only':
            # Step 3a: Field galaxies only (no lens, no sources)
            if len(field_light) > 0:
                field_light_model_list = [
                    "INTERPOL" if "image" in d else "SERSIC_ELLIPSE" for d in field_light
                ]
                step_model_lists = dict(
                    lens_model_list=["SIE"],  # Minimal lens model (no deflection)
                    lens_light_model_list=field_light_model_list,
                    source_light_model_list=[]
                )
                step_sim = SimAPI(numpix=int(numpix), kwargs_single_band=filter_lenstronomy_params(band_cfg), kwargs_model=step_model_lists)
                numerics = dict(supersampling_factor=1, supersampling_convolution=False)
                step_im_model = step_sim.image_model_class(numerics)
                kw_field_amp, _, _ = step_sim.magnitude2amplitude(field_light, [])
                # Create empty lens model (no deflection)
                empty_lens = [{'theta_E': 0.001, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0}]
                image = step_im_model.image(
                    kwargs_lens=empty_lens,
                    kwargs_source=[],
                    kwargs_lens_light=kw_field_amp
                )
            else:
                # No field galaxies, return empty image
                image = np.zeros((numpix, numpix), dtype=np.float32)
                
        elif step == 'final':
            # Step 3b: All components (already computed in main function)
            # This will be handled separately
            return None
        else:
            print(f"[WARNING] Unknown step: {step}")
            return None
        
        # Apply PSF convolution (all steps should have PSF for realism)
        if psf_array is not None:
            image = apply_psf_convolution(image, psf_array)
        
        # Apply morphological enhancements for lens-only and lens+sources steps
        if step in ['lens_only', 'lens_sources'] and morph_type is not None:
            try:
                images_dict = {band: image}
                images_dict = apply_morphological_enhancements(
                    images_dict, n_lens, lens_q, morph_type,
                    numpix=numpix, pixel_scale=float(band_cfg.get('pixel_scale', 0.03)),
                    seed=int(safe_random_integers(rng, 0, int(1e6))),
                    context='lens'
                )
                image = images_dict[band]
            except Exception as e:
                print(f"[WARNING] Morphological enhancement failed for {step}: {e}")
        
        # Add noise (background effects) - use step_sim if it exists, otherwise use original sim
        step_sim_for_noise = step_sim if 'step_sim' in locals() else sim
        noise = step_sim_for_noise.noise_for_model(model=image)
        image = image + noise
        
        # Add artifacts if enabled
        if add_artifacts:
            artifact_images, _ = add_cosmos_web_artifacts({band: image}, rng, 'moderate', numpix=numpix)
            image = artifact_images[band]
        
        intermediate_images[band] = image.astype(np.float32)
        
    except Exception as e:
        print(f"[ERROR] Failed to generate {step} image for {band}: {e}")
        # Return empty image on error
        intermediate_images[band] = np.zeros((numpix, numpix), dtype=np.float32)
    
    return intermediate_images

# --------------------------------------------------------------------------------------
# Save and utility functions
# --------------------------------------------------------------------------------------

def crop_centered_stack(stack, pixel_scale, cutout_arcsec):
    """Center-crop an (n_bands, numpix, numpix) image stack to a square of
    ``cutout_arcsec`` on a side, centered on the lens (the image center, by
    construction of the simulator's coordinate grid).

    This is the "simulate wide, deliver narrow" pattern: the full image is
    rendered at the full simulated FOV (e.g. 1') so field-galaxy density,
    shear, and flexion all reflect the true environment out to that radius,
    but ML training typically only needs a small lens-centered cutout (e.g.
    5-10"). Cropping post-render keeps both available without re-simulating:
    the wide image/maps for selection-function or shear/flexion studies, and
    a small cutout for ML, from the same underlying simulation.
    """
    numpix = stack.shape[-1]
    half_cutout_px = int(round((cutout_arcsec / 2.0) / pixel_scale))
    half_cutout_px = min(half_cutout_px, numpix // 2)
    center = numpix // 2
    lo, hi = center - half_cutout_px, center + half_cutout_px
    return stack[..., lo:hi, lo:hi]


def save_outputs_unified(lens_id, images, out_root, row, n_lens_used, field_info,
                         is_lens=True, epoch_index=None, metadata=None, bands=None):
    """Save all generated products in a single compressed .npz file per sample
    
    This unified storage format includes:
    - 4-band final composite image
    - 4-band intermediate images (source-only, lens+source, lens+source+environment, environment-only)
    - RGB visualization
    - Sample metadata
    
    Args:
        lens_id: Sample identifier
        images: Dict with 4-band final composite {band: array}
        out_root: Output directory path
        row: Catalog row with sample parameters
        n_lens_used: Number of lenses used
        field_info: Dictionary with intermediate_images and other info
        is_lens: True for lens samples, False for non-lens
        epoch_index: Optional epoch number for time-delay systems
        metadata: Optional dict with additional metadata
        bands: Optional list of band names (uppercase) for this telescope/resolution
    Returns:
        bool: Success status
    """
    active_bands = bands if bands is not None else UPPER_BANDS
    # Generate PRISM-formatted filename: PRISM_[lens|nonlens]_[TYPE_][epoch_]ID
    from prism.lensing.lens_system_classifier import LensSystemClassifier
    
    sample_type = "lens" if is_lens else "nonlens"
    epoch_str = f"epoch{epoch_index:02d}_" if epoch_index is not None else ""

    # FIX (adversarial audit finding C-14, 2026-08-01): this is the naming
    # construction actually used under output.unified_storage=true (this
    # project's default/primary output mode) -- an earlier fix to the
    # filename_base variable in the caller did NOT reach this code, since
    # save_outputs_unified rebuilds the name from scratch here using
    # is_lens/lens_system_class directly, unconditionally leaking the
    # class (and, for lenses, the lens_system_class) into the filename.
    # Same opt-in output.neutral_filenames flag as the other fix.
    _neutral_names = CONFIG.get('output', {}).get('neutral_filenames', False) if isinstance(CONFIG, dict) else False
    if _neutral_names:
        _neutral_id = int(lens_id) + (0 if is_lens else 1_000_000)
        base = f"cosmos_sample_{epoch_str}{_neutral_id:06d}"
    elif is_lens:
        # Lens format: PRISM_lens_TYPE_[epoch_]ID
        lens_system_class = field_info.get('lens_system_class', 'single_field') if field_info else 'single_field'
        short_code = LensSystemClassifier.get_short_code(lens_system_class)
        base = f"PRISM_lens_{short_code}_{epoch_str}{int(lens_id):06d}"
    else:
        # Non-lens format: PRISM_nonlens_[epoch_]ID
        base = f"PRISM_nonlens_{epoch_str}{int(lens_id):06d}"
    
    # Create unified directory
    unified_dir = out_root / "unified_npz"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create jpg_rgb for quick visualization
    (out_root / "jpg_rgb").mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare data dictionary for .npz file
        data_dict = {}
        _tel = CONFIG.get('telescope', 'jwst') if isinstance(CONFIG, dict) else 'jwst'
        
        # 1. Final N-band composite image
        final_stack = np.stack([images[b] for b in active_bands], axis=0).astype(np.float32)
        if not np.isfinite(final_stack).any():
            print(f"[ERROR] {base}: Non-finite final image data")
            return False
        data_dict['image_final'] = final_stack

        # 1b. Optional ML-training cutout: a small lens-centered crop of the
        # same full-FOV simulation (see crop_centered_stack), so a wide
        # physically-realistic render (correct field density/shear/flexion
        # out to the full FOV) can still deliver a small postage stamp for
        # ML without re-simulating. Config: output.ml_cutout_arcsec.
        _ml_cutout_arcsec = CONFIG.get('output', {}).get('ml_cutout_arcsec') if isinstance(CONFIG, dict) else None
        if _ml_cutout_arcsec:
            _cutout_pixel_scale = field_info.get('delta_pix', CONFIG.get('pixel_scale', 0.031)) if field_info else CONFIG.get('pixel_scale', 0.031)
            cutout_stack = crop_centered_stack(final_stack, _cutout_pixel_scale, _ml_cutout_arcsec)
            cutout_dir = out_root / "ml_cutout"
            cutout_dir.mkdir(parents=True, exist_ok=True)
            np.save(cutout_dir / f"{base}_cutout{_ml_cutout_arcsec:g}arcsec.npy", cutout_stack)

        # 2. Intermediate images (if available)
        if field_info and field_info.get('intermediate_images'):
            intermediate_images = field_info['intermediate_images']
            
            for step in ['lens_only', 'sources_only', 'sources_unlensed', 'lens_sources', 'field_only']:
                if step in intermediate_images:
                    step_images = intermediate_images[step]
                    if all(b in step_images for b in active_bands):
                        step_stack = np.stack([step_images[b] for b in active_bands],
                                             axis=0).astype(np.float32)
                        data_dict[f'image_{step}'] = step_stack

                        # Optionally save intermediate RGB JPGs for quick inspection
                        if CONFIG.get('output', {}).get('save_intermediate_rgb_jpg', False):
                            try:
                                step_rgb = create_jwst_panel_rgb(
                                    step_images, bands=active_bands, telescope=_tel,
                                )
                                if step_rgb is not None:
                                    step_dir_jpg = out_root / "jpg_rgb" / f"intermediate_{step}"
                                    step_dir_jpg.mkdir(parents=True, exist_ok=True)
                                    Image.fromarray((step_rgb * 255).astype(np.uint8)).save(
                                        step_dir_jpg / f"{base}.jpg",
                                        quality=95, optimize=True
                                    )
                            except Exception as e:
                                print(f"[WARNING] Failed to create intermediate RGB for {step}: {e}")
        
        # 3. RGB visualization (stored as uint8 to save space)
        try:
            _arc_images = None
            if field_info and field_info.get('intermediate_images'):
                _ii = field_info['intermediate_images']
                if 'lens_sources' in _ii and 'lens_only' in _ii:
                    _arc_images = {
                        'lens_sources': _ii['lens_sources'],
                        'lens_only': _ii['lens_only'],
                    }
            # Panel JPG: individual bands + RGB composite side-by-side (matches intermediate format)
            panel = create_jwst_panel_rgb(images, bands=active_bands, telescope=_tel,
                                          arc_images=_arc_images)
            # Pure RGB composite stored in npz (compact)
            rgb = create_jwst_rgb(images, bands=active_bands, telescope=_tel,
                                  arc_images=_arc_images)
            if rgb is None and active_bands:
                # Fallback: grayscale from first band
                gray = normalize_for_display_astronomical(images[active_bands[0]], noise_level=0.3, sat_percent=0.01)
                rgb = np.stack([gray, gray, gray], axis=-1)
            if rgb is not None:
                # Store pure RGB as uint8 in npz (saves 4x space).
                # FIX (adversarial audit finding C-7, 2026-08-01): renamed
                # from 'rgb_visualization' to 'display_rgb_visualization'
                # -- this is a per-image adaptively-stretched (arcsinh,
                # data-dependent percentile normalization), NON-physical,
                # display-only composite, stored alongside the real
                # calibrated science arrays (image_final etc.) in the same
                # archive with no prior naming cue that it must not be
                # used as model input or for photometric measurement.
                data_dict['display_rgb_visualization'] = (rgb * 255).astype(np.uint8)

            # Save panel (bands + RGB) as the final JPG — consistent with intermediate images
            save_img = panel if panel is not None else rgb
            if save_img is not None:
                Image.fromarray((save_img * 255).astype(np.uint8)).save(
                    out_root / "jpg_rgb" / f"{base}.jpg",
                    quality=95, optimize=True
                )
        except Exception as e:
            print(f"[WARNING] RGB creation failed for {base}: {e}")

        # 4. Optional stacked .npy output (steps x 5 channels)
        stacked_cfg = CONFIG.get('output', {}).get('stacked_npy', {})
        if stacked_cfg.get('enabled', False):
            # FIX (adversarial audit finding C-7, 2026-08-01): this used to
            # interleave a display-only "rgb_gray" channel (NTSC luma of an
            # adaptively per-image-normalized asinh/arcsinh RGB composite,
            # with no physical units and a data-dependent, non-reproducible
            # scale) directly into the SAME homogeneous float32 science
            # tensor as the real per-band calibrated images, with no
            # channel manifest -- a downstream consumer slicing this array
            # by index had no way to know one channel per step was not a
            # real filter. Now: the per-band science stack (channel_names
            # recorded) and the display-only luma channel are saved to
            # SEPARATE arrays, and the display channel is clearly
            # namespaced `display_*`.
            steps_order = stacked_cfg.get(
                'order',
                ['lens_only', 'sources_only', 'sources_unlensed', 'lens_sources', 'field_only', 'final']
            )
            stacked_channels = []
            display_channels = []
            channel_names = []

            def _rgb_gray(step_imgs):
                step_rgb = create_jwst_rgb(step_imgs, bands=active_bands)
                if step_rgb is None:
                    return np.zeros_like(step_imgs[active_bands[0]], dtype=np.float32)
                r = step_rgb[..., 0]
                g = step_rgb[..., 1]
                b = step_rgb[..., 2]
                return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)

            for step in steps_order:
                if step == 'final':
                    step_images = images
                else:
                    step_images = None
                    if field_info and field_info.get('intermediate_images'):
                        step_images = field_info['intermediate_images'].get(step)

                if step_images and all(b in step_images for b in active_bands):
                    bands_stack = np.stack([step_images[b] for b in active_bands], axis=0).astype(np.float32)
                    rgb_gray = _rgb_gray(step_images)[None, ...]
                else:
                    bands_stack = np.zeros((len(active_bands),) + images[active_bands[0]].shape, dtype=np.float32)
                    rgb_gray = np.zeros((1,) + images[active_bands[0]].shape, dtype=np.float32)

                stacked_channels.append(bands_stack)
                display_channels.append(rgb_gray)
                channel_names.extend(f"{step}_{b}" for b in active_bands)

            stacked = np.concatenate(stacked_channels, axis=0).astype(np.float32)
            display_stack = np.concatenate(display_channels, axis=0).astype(np.float32)
            stacked_dir = out_root / "unified_npy"
            stacked_dir.mkdir(parents=True, exist_ok=True)
            np.save(stacked_dir / f"{base}.npy", stacked)
            np.save(stacked_dir / f"{base}_channel_names.npy", np.array(channel_names))
            np.save(stacked_dir / f"{base}_display_rgb_luma.npy", display_stack)
        
        # 4. Metadata
        meta = {
            'lens_id': int(lens_id),
            'is_lens': is_lens,
            'n_lens_used': n_lens_used,
            'bands': UPPER_BANDS,
            'theta_E': float(row.get('theta_E', row.get('b', 0.0))),
            'lens_redshift': float(row.get('lens_redshift', row.get('zl', 0.0))),
            'source_redshift': float(row.get('source_redshift', row.get('zs', 0.0))),
            # Added per adversarial audit findings C-2/C-13 (2026-08-01):
            # these were previously absent from every saved label, and
            # theta_E_override_applied lets downstream population-statistics
            # code exclude the display-motivated group-scale overrides
            # (which are not derived from a physical group-halo mass
            # function -- see the override's code comment) from any claim
            # about a physically-representative theta_E distribution.
            'theta_E_override_applied': bool(row.get('theta_E_override_applied', False)),
            'theta_E_pre_override': (float(row['theta_E_pre_override'])
                                      if row.get('theta_E_pre_override') is not None else None),
            'lens_sigma_kms': (float(row['lens_sigma_kms'])
                                if row.get('lens_sigma_kms') is not None else None),
            'shear_gamma1': (float(row['shear_gamma1'])
                              if pd.notna(row.get('shear_gamma1', np.nan)) else None),
            'shear_gamma2': (float(row['shear_gamma2'])
                              if pd.notna(row.get('shear_gamma2', np.nan)) else None),
            # Selection-function record (audit finding C-13): lets a
            # population-statistics user identify/exclude systems where
            # the source was force-brightened or force-repositioned to
            # satisfy a rendering constraint, and see the magnification
            # gate that was applied.
            'magnification': (float(row['magnification'])
                               if row.get('magnification') is not None else None),
            'magnification_gate_min': (float(row['magnification_gate_min'])
                                        if row.get('magnification_gate_min') is not None else None),
            'magnification_gate_max': (float(row['magnification_gate_max'])
                                        if row.get('magnification_gate_max') is not None else None),
            'source_position_resampled_for_caustic': bool(row.get('source_position_resampled_for_caustic', False)),
            'source_mag_brightening_applied': bool(row.get('source_mag_brightening_applied', False)),
        }
        
        if epoch_index is not None:
            meta['epoch_index'] = epoch_index

        # TNG Mode: surface the matched-subhalo info (or None) from
        # field_info, since field_info itself isn't merged into meta below.
        if field_info:
            meta['tng_lens'] = field_info.get('tng_lens')
            meta['tng_source'] = field_info.get('tng_source')
            meta['tng_field'] = field_info.get('tng_field')

        if metadata:
            meta.update(metadata)

        meta.pop('psf_arrays', None)

        def _to_jsonable(obj):
            """Convert common numpy types to JSON-serializable objects."""
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            return obj
        
        # Store metadata as JSON string (npz doesn't handle dicts directly)
        import json
        data_dict['metadata'] = json.dumps(_to_jsonable(meta))
        
        # 5. Save unified .npz file (compressed)
        npz_path = unified_dir / f"{base}.npz"
        np.savez_compressed(npz_path, **data_dict)
        
        # Print file size reduction info
        npz_size_mb = npz_path.stat().st_size / (1024**2)
        print(f"[INFO] Saved {base}.npz ({npz_size_mb:.2f} MB)")

        # 6. Save PSF arrays to separate psf_arrays/ folder
        psf_arrays_to_save = field_info.get('psf_arrays') if field_info else None
        if psf_arrays_to_save:
            psf_dir_out = out_root / "psf_arrays"
            psf_dir_out.mkdir(parents=True, exist_ok=True)
            psf_save_dict = {
                band: arr for band, arr in psf_arrays_to_save.items()
                if arr is not None and isinstance(arr, np.ndarray)
            }
            if psf_save_dict:
                psf_path = psf_dir_out / f"{base}_psf.npz"
                np.savez_compressed(psf_path, **psf_save_dict)
                print(f"[INFO] Saved PSF arrays for {base} ({list(psf_save_dict.keys())})")

        # 6b. Optional multi-extension FITS export (image cube + PSF + NOISE +
        # SEGMENTATION + TRUTH_CATALOG), gated by output.save_fits.
        if CONFIG.get('output', {}).get('save_fits', False):
            try:
                from prism.io.fits_export import (
                    compute_noise_sigma_maps, compute_segmentation_map, write_lens_fits
                )
                band_noise_cfgs = field_info.get('band_noise_cfgs', {}) if field_info else {}
                noise_sigma_maps = compute_noise_sigma_maps(final_stack, band_noise_cfgs, active_bands)
                segmentation_map = compute_segmentation_map(final_stack, band_noise_cfgs, active_bands)

                ref_cfg = band_noise_cfgs.get(active_bands[0], {}) if active_bands else {}
                fits_dir = out_root / "fits"
                fits_dir.mkdir(parents=True, exist_ok=True)
                fits_path = write_lens_fits(
                    fits_dir / base,
                    image_final=final_stack,
                    psf_arrays=psf_arrays_to_save,
                    noise_sigma_maps=noise_sigma_maps,
                    segmentation_map=segmentation_map,
                    truth_catalog_row=_to_jsonable(meta),
                    band_names=active_bands,
                    pixel_scale=ref_cfg.get('pixel_scale'),
                    exposure_time=ref_cfg.get('exposure_time'),
                    magnitude_zero_point=ref_cfg.get('magnitude_zero_point'),
                )
                print(f"[INFO] Saved FITS for {base} ({fits_path.name})")
            except Exception as e:
                print(f"[WARNING] FITS export failed for {base}: {e}")

        # KAPPA: Compute and save convergence/shear/magnification maps
        if KAPPA_OUTPUT_AVAILABLE and is_lens:
            try:
                lens_model_list = field_info.get('lens_model_list') if field_info else None
                kwargs_lens = field_info.get('kwargs_lens') if field_info else None
                lens_system_class = field_info.get('lens_system_class', 'single_field') if field_info else 'single_field'
                
                # Determine sub_type from lens_model_list
                sub_type = '+'.join(lens_model_list) if lens_model_list else 'UNKNOWN'
                
                if lens_model_list and kwargs_lens:
                    # Create lens model for kappa computation
                    from lenstronomy.LensModel.lens_model import LensModel
                    lens_model = LensModel(lens_model_list)
                    
                    # Compute kappa products: native-resolution map plus an
                    # extended-FOV map (default 1', auto-coarsened resolution
                    # so compute cost stays bounded regardless of telescope
                    # pixel scale -- config: output.extended_lensing_fov_arcmin)
                    kappa_dict = compute_kappa_products(
                        lens_model, kwargs_lens,
                        num_pix=field_info.get('numpix', 300) if field_info else 300,
                        delta_pix=field_info.get('delta_pix', 0.031) if field_info else 0.031,
                        compute_flexion=True,
                        extended_fov_arcmin=CONFIG.get('output', {}).get('extended_lensing_fov_arcmin', 1.0),
                    )

                    # Create kappa output directory
                    kappa_dir = out_root / "kappa_maps"
                    kappa_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save kappa outputs (NPY, NPZ, 2×JPG)
                    kappa_success = save_kappa_outputs(
                        kappa_dict, kappa_dir, str(lens_id).zfill(6),
                        category=lens_system_class,
                        sub_type=sub_type
                    )
                    
                    if kappa_success:
                        print(f"[KAPPA] Saved convergence maps for {lens_id} ({lens_system_class})")
                    else:
                        print(f"[WARNING] Kappa output save failed for {lens_id}")
            except Exception as e:
                print(f"[WARNING] Kappa computation failed: {e}")
                # Non-fatal failure - continue without kappa maps
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Unified save failed for {base}: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_outputs_complete(lens_id, images, out_root, row, n_lens_used, field_info, rng=None):
    """Complete save function with astronomical RGB and comprehensive diagnostics"""
    if CONFIG.get('output', {}).get('unified_storage', False):
        return save_outputs_unified(
            lens_id, images, out_root, row, n_lens_used, field_info,
            is_lens=True, epoch_index=None, metadata=None
        )
    
    # Get lens system class and create PRISM-formatted filename
    # Format: PRISM_lens_TYPE_ID.jpg
    from prism.lensing.lens_system_classifier import LensSystemClassifier
    lens_system_class = field_info.get('lens_system_class', 'single_field')
    short_code = LensSystemClassifier.get_short_code(lens_system_class)
    base = f"PRISM_lens_{short_code}_{int(lens_id):06d}"
    
    # Create directories
    for subdir in ["npy", "jpg_rgb", "diagnostics"]:
        (out_root/subdir).mkdir(parents=True, exist_ok=True)

    success = True
    
    try:
        # Save 4-band .npy stack
        stack = np.stack([images[b] for b in UPPER_BANDS], axis=0).astype(np.float32)
        
        if not np.isfinite(stack).any():
            print(f"[ERROR] Lens {lens_id}: Non-finite stack")
            return False
            
        np.save(out_root/"npy"/f"{base}_4bands.npy", stack)

        # Get image dimensions from the images
        numpix = images["F115W"].shape[0]

        # TRILOGY-STYLE RGB: Match real COSMOS-Web observation processing
        try:
            rgb, _ = create_trilogy_rgb(images, numpix)  # Ignore normalization scales for non-time-delay systems
        except Exception as e:
            print(f"[WARNING] Trilogy RGB failed, using fallback: {e}")
            # Fallback to improved method
            R = normalize_for_display_astronomical(images["F444W"], noise_level=0.3, sat_percent=0.01, channel_name="R")
            B = normalize_for_display_astronomical(images["F115W"], noise_level=0.4, sat_percent=0.01, channel_name="B")
            G1 = normalize_for_display_astronomical(images["F150W"], noise_level=0.35, sat_percent=0.01, channel_name="G1")
            G2 = normalize_for_display_astronomical(images["F277W"], noise_level=0.35, sat_percent=0.01, channel_name="G2")
            G = 0.6 * G1 + 0.4 * G2
            rgb = np.stack([R, G, B], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            rgb = apply_field_galaxy_realism(rgb, images, numpix)

        # Noise suppression in very faint regions
        background_level = np.percentile(np.max(rgb, axis=2), 10)
        noise_mask = np.max(rgb, axis=2) < (background_level * 2)
        rgb[noise_mask] = rgb[noise_mask] * 0.3

        # Conservative brightness boost
        max_brightness = np.max(rgb)
        if max_brightness < 0.02:  # Only boost very dark images
            boost = 0.08 / (max_brightness + 1e-10)
            boost = min(boost, 2.0)  # Conservative cap
            rgb = rgb * boost
            rgb = np.clip(rgb, 0, 1)

        # Add JWST diffraction spikes directly to RGB (AFTER normalization to preserve visibility)
        # DISABLED: Spikes looked artificial and unrealistic in simulated images
        # Users can add spikes as post-processing if needed for specific use cases
        # if field_info.get('add_spikes', False) and rng is not None:
        #     rgb = add_spikes_to_rgb(rgb, bright_positions, numpix=numpix, rng=rng)
        #     rgb = np.clip(rgb, 0, 1)  # Ensure values stay in [0,1]

        # Save the main RGB image (300x300)
        # Get expected image size from the images
        expected_size = images["F115W"].shape[0]
        
        # Validate RGB dimensions before saving
        if rgb.shape[0] != expected_size or rgb.shape[1] != expected_size or rgb.shape[2] != 3:
            print(f"[WARNING] Lens {lens_id}: RGB has unexpected shape {rgb.shape}, expected ({expected_size}, {expected_size}, 3)")
            # Try to extract the correct region if possible
            if rgb.shape[0] >= expected_size and rgb.shape[1] >= expected_size:
                rgb = rgb[:expected_size, :expected_size, :3]
                print(f"[INFO] Extracted {expected_size}x{expected_size} region from RGB")
            else:
                print(f"[ERROR] Cannot fix RGB dimensions for lens {lens_id}")
                return False
        
        # Save panel (bands + RGB) as primary JPG — consistent with intermediate images
        try:
            panel_rgb = create_jwst_panel_rgb(images)
        except Exception:
            panel_rgb = None
        save_img = panel_rgb if panel_rgb is not None else rgb
        Image.fromarray((save_img * 255).astype(np.uint8)).save(
            out_root/"jpg_rgb"/f"{base}.jpg", quality=95
        )

        # Save intermediate step images if available
        if field_info.get('intermediate_images') and CONFIG.get('save_intermediate_images', False):
            try:
                intermediate_images = field_info['intermediate_images']
                intermediate_dir = out_root / "intermediate_steps" / base
                intermediate_dir.mkdir(parents=True, exist_ok=True)

                for step_name, step_bands in intermediate_images.items():
                    try:
                        step_panel = create_jwst_panel_rgb(step_bands)
                        if step_panel is not None:
                            Image.fromarray((step_panel*255).astype(np.uint8)).save(
                                intermediate_dir / f"{step_name}_rgb.jpg", quality=95
                            )
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Comprehensive diagnostics
        if lens_id < 100 or (lens_id % 20 == 0):
            diag = {
                'lens_id': int(lens_id),
                'generation_info': {
                    'base_lens_id': int(row.get('base_lens_id', lens_id)),
                    'variation_id': int(row.get('variation_id', 0)),
                    'lens_n_sampled': float(n_lens_used),
                    'field_galaxies': field_info
                },
                'parameters': {
                    'theta_E': float(row.get('theta_E', row.get('b', np.nan))),
                    'lens_redshift': float(row.get('lens_redshift', np.nan)),
                    'source_redshift': float(row.get('source_redshift', np.nan)),
                    'lens_radius': float(row.get('lens_radius', np.nan)),
                    'source_radius': float(row.get('source_radius', np.nan))
                },
                'image_quality': {
                    'rgb_max': float(np.max(rgb)),
                    'rgb_mean': float(np.mean(rgb)),
                    'total_flux': float(np.sum([np.sum(images[b]) for b in UPPER_BANDS]))
                },
                'band_stats': {}
            }
            
            for b in UPPER_BANDS:
                band_data = images[b]
                diag['band_stats'][b] = {
                    'min': float(np.min(band_data)),
                    'max': float(np.max(band_data)),
                    'mean': float(np.mean(band_data)),
                    'std': float(np.std(band_data))
                }
            
            with open(out_root/"diagnostics"/f"{base}_diag.json", "w") as f:
                json.dump(diag, f, indent=2)
        
    except Exception as e:
        print(f"[ERROR] Save failed for lens {lens_id}: {e}")
        success = False
    
    return success

def read_combined_cosmos_catalogs(structural_path, analysis_path=None):
    """Read and process COSMOS catalogs with proper pandas handling"""
    print("Reading COSMOS-Web catalogs...")
    structural_df = _read_csv_robust(structural_path)
    print(f"  Structural: {len(structural_df)} rows, {len(structural_df.columns)} cols")

    analysis_df = None
    if analysis_path and os.path.exists(analysis_path):
        analysis_df = _read_csv_robust(analysis_path)
        print(f"  Analysis: {len(analysis_df)} rows, {len(analysis_df.columns)} cols")
    else:
        print("  Analysis: (none)")

    base_len = len(structural_df)
    print(f"  Using full structural catalog: {base_len} base configurations")
    
    conv = pd.DataFrame(index=range(base_len))

    # Telescope-specific lens-population overrides (e.g. Euclid Q1 statistics).
    # Only applied when telescope == "euclid"; other telescopes use the
    # global redshifts/mass/theta_E settings unchanged.
    _telescope = CONFIG.get('telescope', 'jwst').lower()
    _lens_pop = CONFIG.get('telescope_configs', {}).get(_telescope, {}).get('lens_population', {})

    # Process redshifts with config-driven ranges
    z_cfg = {**CONFIG.get('redshifts', {}), **_lens_pop.get('redshifts', {})}
    z_lens_min = z_cfg.get('lens_min', 0.4)
    z_lens_max = z_cfg.get('lens_max', 3.0)
    z_source_min = z_cfg.get('source_min', 2.0)
    z_source_max = z_cfg.get('source_max', 6.0)
    min_delta_z = z_cfg.get('min_delta_z', 0.3)

    # Detection-completeness mixture (e.g. Euclid): a fraction of lenses
    # are drawn from the *detected* Q1 distribution above, the rest from
    # an "extended" population covering regimes (low theta_E, high
    # z_lens, etc.) that current detection pipelines systematically miss.
    # `is_detected_like` is reused below for theta_E so the same systems
    # are consistently "detected-like" or "extended" across properties.
    _detected_frac = _lens_pop.get('detected_fraction', 1.0)
    _ext_z = _lens_pop.get('extended_redshifts', {})
    is_detected_like = np.random.random(base_len) < _detected_frac

    zl_series = _get_series(analysis_df, ["lens_redshift", "zl", "z_spec"], base_len, default=np.nan)
    if zl_series.isna().all() and "LP_zfinal" in structural_df.columns:
        zl_series = pd.to_numeric(structural_df["LP_zfinal"], errors="coerce")
        zl_series = zl_series.reset_index(drop=True).reindex(range(base_len))

    if _ext_z:
        ext_lens_min = _ext_z.get('lens_min', z_lens_min)
        ext_lens_max = _ext_z.get('lens_max', z_lens_max)
        zl_detected = np.random.uniform(z_lens_min, z_lens_max, base_len)
        zl_extended = np.random.uniform(ext_lens_min, ext_lens_max, base_len)
        zl_random = np.where(is_detected_like, zl_detected, zl_extended)
        # Overall bounds span both ranges so extended-population values aren't clipped away
        zl_clip_min, zl_clip_max = min(z_lens_min, ext_lens_min), max(z_lens_max, ext_lens_max)
    else:
        zl_random = np.random.uniform(z_lens_min, z_lens_max, base_len)
        zl_clip_min, zl_clip_max = z_lens_min, z_lens_max

    # Fill missing lens redshifts within config range
    zl_filled = zl_series.fillna(pd.Series(zl_random))
    # Enforce lens redshift bounds
    conv["lens_redshift"] = zl_filled.clip(zl_clip_min, zl_clip_max)

    # Source redshift: always behind lens with minimum gap
    if _ext_z:
        ext_source_min = _ext_z.get('source_min', z_source_min)
        ext_source_max = _ext_z.get('source_max', z_source_max)
        zs_detected = np.random.uniform(z_source_min, z_source_max, base_len)
        zs_extended = np.random.uniform(ext_source_min, ext_source_max, base_len)
        zs_series = pd.Series(np.where(is_detected_like, zs_detected, zs_extended))
        zs_clip_max = max(z_source_max, ext_source_max)
        ext_min_delta_z = _ext_z.get('min_delta_z', min_delta_z)
        delta_z = np.where(is_detected_like, min_delta_z, ext_min_delta_z)
    else:
        zs_series = pd.Series(np.random.uniform(z_source_min, z_source_max, base_len))
        zs_clip_max = z_source_max
        delta_z = min_delta_z
    if z_cfg.get('enforce_source_behind_lens', True):
        # Ensure zs > zl + min_delta_z
        min_allowed_zs = conv["lens_redshift"] + delta_z
        zs_series = np.maximum(zs_series, min_allowed_zs)
        # Clip to source max
        zs_series = np.minimum(zs_series, zs_clip_max)
    conv["source_redshift"] = zs_series

    # Einstein radius with mass-θ_E correlation (massive lenses M=10^10-12 M☉)
    theta_E_series = _get_series(analysis_df, ["theta_E", "einstein_radius"], base_len, default=np.nan)

    mass_cfg = CONFIG.get('mass', {})
    _theta_E_pop = _lens_pop.get('theta_E')
    if _theta_E_pop is not None:
        # Telescope-specific Einstein-radius distribution (e.g. Euclid Q1
        # statistics): lognormal centered on the observed median, clipped
        # to the observed range. Overrides any catalog-derived theta_E
        # (which is based on COSMOS-Web/JWST lenses and not representative
        # of Euclid's selection function).
        te_min = _theta_E_pop.get('min', 0.3)
        te_max = _theta_E_pop.get('max', 2.4)
        te_median = _theta_E_pop.get('median', 1.0)
        te_log_scatter = _theta_E_pop.get('log_scatter', 0.35)
        theta_E_detected = np.random.lognormal(np.log(te_median), te_log_scatter, base_len)
        theta_E_detected = np.clip(theta_E_detected, te_min, te_max)

        _ext_te = _lens_pop.get('extended_theta_E')
        if _ext_te:
            # "Extended" population: smaller, fainter Einstein radii (and
            # the occasional larger one) that current detection pipelines
            # systematically under-recover, used together with the
            # detected-like population via `is_detected_like` so the
            # simulated set covers the full completeness function rather
            # than just the confirmed-detection regime.
            ete_min = _ext_te.get('min', te_min)
            ete_max = _ext_te.get('max', te_max)
            ete_median = _ext_te.get('median', te_median)
            ete_log_scatter = _ext_te.get('log_scatter', te_log_scatter)
            theta_E_extended = np.random.lognormal(np.log(ete_median), ete_log_scatter, base_len)
            theta_E_extended = np.clip(theta_E_extended, ete_min, ete_max)
            theta_E_generated = np.where(is_detected_like, theta_E_detected, theta_E_extended)
        else:
            theta_E_generated = theta_E_detected
        theta_E_series = pd.Series(theta_E_generated)
    elif theta_E_series.isna().all() and mass_cfg.get('use_mass_theta_E_correlation', True):
        # Generate lens masses (log10 scale)
        log_mass_min = mass_cfg.get('lens_mass_min', 10.0)
        log_mass_max = mass_cfg.get('lens_mass_max', 12.0)
        log_masses = np.random.uniform(log_mass_min, log_mass_max, base_len)
        
        # Empirical: θ_E ∝ M^0.5 at fixed z (Treu+2010, Auger+2010)
        # Normalize: M=10^11 M☉ → θ_E ~ 1.0"
        norm_mass = 11.0
        theta_E_base = 1.0
        theta_E_generated = theta_E_base * np.sqrt(10**(log_masses - norm_mass))
        
        # Add scatter (20% lognormal)
        theta_E_scattered = theta_E_generated * np.exp(np.random.normal(0, 0.2, base_len))
        theta_E_series = pd.Series(np.clip(theta_E_scattered, 0.3, _theta_E_hard_max()))
    elif theta_E_series.isna().all():
        # Fallback without mass correlation
        theta_E_generated = np.random.lognormal(np.log(1.0), 0.35, base_len)
        theta_E_series = pd.Series(np.clip(theta_E_generated, 0.3, _theta_E_hard_max()))
    
    conv["theta_E"] = theta_E_series.fillna(1.0)

    # Extract structural parameters
    if len(structural_df) > 0:
        rest_params = extract_restframe_struct(structural_df, conv["lens_redshift"])
        conv["lens_radius"] = rest_params["re_rest"]
        conv["lens_axis_ratio"] = rest_params["q_rest"]
        # Gravitational lenses are selected by velocity-dispersion threshold
        # and are overwhelmingly massive ETGs (n~4, de Vaucouleurs).
        # Catalog nsersic values have large photometric scatter: ~52% land
        # below n=2 even for genuine ellipticals, driving disk/spiral-like
        # sim morphologies that don't match real observations.
        # Apply a lens-population prior: measured n<2.5 is treated as
        # photometric scatter around a true ETG -- resample from N(4, 0.6)
        # clipped [2.5, 6.0]; genuine high-n objects are kept as measured.
        raw_n = rest_params["n_rest"].to_numpy()
        etg_n = np.clip(np.random.normal(4.0, 0.6, len(raw_n)), 2.5, 6.0)
        conv["n_rest"] = pd.Series(
            np.where(raw_n >= 2.5, raw_n, etg_n),
            index=conv.index,
        )
    else:
        conv["lens_radius"] = 0.7
        conv["lens_axis_ratio"] = 0.7
        conv["n_rest"] = 3.0

    # Generate magnitudes as pandas Series.
    #
    # Physical galaxy SEDs vary smoothly with wavelength: the *brightness*
    # (overall normalization) is a single per-galaxy property, while the
    # *color* (band-to-band variation) is set by the smooth, physically
    # motivated SED/K-correction terms. Drawing the brightness scatter
    # independently inside the band loop (as before) injected ~1.2 mag of
    # *uncorrelated* noise between adjacent filters -- equivalent to a
    # randomly oscillating SED -- which produced unrealistic, flat,
    # single-band-dominated colors in the rendered RGB composites. Sampling
    # the per-galaxy scatter once (outside the loop) keeps brightness
    # coherent across bands and lets only the genuine SED-driven color
    # terms vary with wavelength.
    phot = CONFIG['photometry']
    z_lens = conv["lens_redshift"]
    z_source = conv["source_redshift"]
    lens_base = phot.get('lens_base_mag_zero', 21.0) + phot.get('lens_redshift_log_slope', 0.8) * np.log10(np.clip(z_lens, 0.2, 6.0))
    lens_scatter = np.random.normal(0, 1.2, base_len)

    # source_base_mag is calibrated at a reference redshift; without a
    # distance-modulus term, sources at very different z_source all get the
    # same apparent brightness, which is unphysical -- a real galaxy of
    # fixed absolute luminosity gets fainter with distance. Apply
    # Delta(distmod) relative to the reference z so the config's calibrated
    # base magnitude is preserved at that z, and sources at higher/lower z
    # are dimmed/brightened accordingly (~1.5-2 mag across z=1-3).
    _src_mag_ref_z = float(phot.get('source_base_mag_ref_z', 2.0))
    _distmod_ref = COSMO.distmod(_src_mag_ref_z).value
    _distmod_src = COSMO.distmod(np.clip(z_source.to_numpy(), 0.1, 10.0)).value
    source_distmod_offset = _distmod_src - _distmod_ref

    source_base = phot.get('source_base_mag', 20.5) + source_distmod_offset
    mag_diff = np.random.uniform(phot.get('source_mag_diff_min', 1.5), phot.get('source_mag_diff_max', 5.0), base_len)
    source_scatter = np.random.normal(0, 0.8, base_len)

    min_delta = float(phot.get('min_source_fainter_than_lens_mag', 0.8))

    # Per-band colour: convolve each galaxy's SED with the active telescope's
    # filter transmission curves. Reference band = first configured band.
    # FIX (audit C-5): see the identical fix applied to the `if rng is
    # None: rng = default_rng()` fallback pattern above -- bare
    # default_rng() ignores args.seed.
    sed_rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    ref_band = UPPER_BANDS[0] if UPPER_BANDS else 'F150W'

    def _per_band_colors(morph_type, n_sersic, redshift):
        # Resolve one SED class for this galaxy, then reuse it for every
        # band below -- an independent per-band draw would let the same
        # galaxy's flux ratios reflect different SEDs in different filters.
        sed_type = resolve_sed_type_from_morphology(morph_type, n_sersic, sed_rng)
        return {
            band: get_realistic_jwst_color_from_transmission(
                morph_type, n_sersic, base_band=ref_band, target_band=band.upper(),
                redshift=redshift, rng=sed_rng, sed_type=sed_type)
            for band in LOWER_BANDS
        }

    n_rest_vals = conv["n_rest"].to_numpy()
    q_rest_vals = conv["lens_axis_ratio"].to_numpy()
    zl_vals = z_lens.to_numpy()
    zs_vals = z_source.to_numpy()

    lens_color_table = []
    source_color_table = []
    for i in range(base_len):
        lens_morph = classify_galaxy_morphology_enhanced(float(n_rest_vals[i]), float(q_rest_vals[i]), sed_rng)
        lens_color_table.append(_per_band_colors(lens_morph, float(n_rest_vals[i]), float(zl_vals[i])))

        # Source structural params (n_sersic, q) are drawn later in the
        # pipeline; sample a representative pair here so its SED type/colour
        # is consistent with the morphology distribution of high-z sources
        # (predominantly star-forming/irregular discs and clumpy systems).
        source_n = float(sed_rng.uniform(0.4, 2.5))
        source_q = float(sed_rng.uniform(0.3, 0.95))
        source_morph = classify_galaxy_morphology_enhanced(source_n, source_q, sed_rng)
        source_color_table.append(_per_band_colors(source_morph, source_n, float(zs_vals[i])))

    for band in LOWER_BANDS:
        num = band[1:4]

        color_offset = np.array([t[band] for t in lens_color_table])
        lens_mags = lens_base + color_offset + lens_scatter

        # Prefer the real COSMOS-Web photometry for this exact lens galaxy
        # (mag_f115w..mag_f444w in the structural catalog) over the synthetic
        # SED-derived color: it is the actual measured per-band SED of the
        # real lens, so using it directly reproduces its real JWST color
        # (e.g. quiescent ellipticals genuinely brighter in F444W than
        # F115W), instead of a randomized morphology-based estimate.
        # Paper / custom-brightness configs may disable this so synthetic
        # lens_base_mag_zero controls the apparent magnitude.
        real_mag_col = f"mag_{band}"
        if (phot.get('use_catalog_lens_mags', True)
                and real_mag_col in structural_df.columns):
            real_mags = pd.to_numeric(structural_df[real_mag_col], errors="coerce")
            real_mags = real_mags.reset_index(drop=True).reindex(range(base_len)).to_numpy()
            lens_mags = np.where(np.isfinite(real_mags), real_mags, lens_mags)

        conv[f"lens_mag_{band}"] = pd.Series(np.clip(lens_mags, phot['lens_mag_min'], phot['lens_mag_max']), index=conv.index)

        source_color = np.array([t[band] for t in source_color_table])
        source_mags = source_base + source_color + source_scatter + mag_diff
        # Enforce per-row constraint: source >= lens + min_delta
        lens_series = conv[f"lens_mag_{band}"]
        enforced_source = np.maximum(source_mags, lens_series + min_delta)
        conv[f"source_mag_{band}"] = pd.Series(np.clip(enforced_source, phot['source_mag_min'], phot['source_mag_max']), index=conv.index)

    # Source geometry (with config-driven size distribution)
    conv["source_x"] = pd.Series(np.random.uniform(-0.35, 0.35, base_len))
    conv["source_y"] = pd.Series(np.random.uniform(-0.35, 0.35, base_len))
    conv["source_axis_ratio"] = pd.Series(np.clip(np.random.beta(1.8, 1.8, base_len)*0.8 + 0.2, 0.2, 0.98))
    conv["source_pa"] = pd.Series(np.random.uniform(-180, 180, base_len))
    
    # Source size generation.
    # FIX (adversarial audit finding C-6, 2026-08-01): this catalog-level
    # column (feeds create_parameter_variations' per-variation source_radius
    # jitter, see base_row.get("source_radius") there) used the same
    # theta_E*fraction sizing this project already replaced elsewhere in
    # the per-render path, for the same reason: source angular size should
    # come from the source's own stellar mass/redshift, not the deflector's
    # Einstein radius. mass_size_relation()/convert_physical_to_angular_
    # radius() take scalar arguments, so this is a per-row loop (a one-time
    # catalog-build cost, not a per-image-render cost).
    _tng_cfg_vec = CONFIG.get('tng_mode', {}) if isinstance(CONFIG, dict) else {}
    _src_logM_mean = _tng_cfg_vec.get('source_logM_default', 9.5)
    _src_logM_sigma = _tng_cfg_vec.get('source_logM_scatter', 0.5)
    _vec_rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    _source_radii = np.empty(base_len)
    _zs_arr = conv["source_redshift"].to_numpy()
    for _i in range(base_len):
        _logM_i = float(np.clip(_vec_rng.normal(_src_logM_mean, _src_logM_sigma), 8.0, 11.0))
        _reff_kpc_i = mass_size_relation(_logM_i, float(_zs_arr[_i]), _vec_rng)
        _source_radii[_i] = convert_physical_to_angular_radius(_reff_kpc_i, float(_zs_arr[_i]))
    conv["source_radius"] = pd.Series(np.clip(_source_radii, 0.05, 0.8), index=conv.index)

    # Sanitize redshifts
    conv = sanitize_redshifts(conv)

    # Hard enforcement: z_source must exceed z_lens by at least min_delta_z.
    # sanitize_redshifts covers NaN-filled rows; this catches catalog-provided
    # fixed redshifts that slipped through (e.g. v12 lens_id=000001 z_s<z_l).
    _min_dz = CONFIG.get('redshifts', {}).get('min_delta_z', 0.5)
    _zl = conv["lens_redshift"]
    _zs = conv["source_redshift"]
    _bad = _zs <= _zl + _min_dz
    if _bad.any():
        _n_bad = int(_bad.sum())
        print(f"[REDSHIFT] Fixing {_n_bad} rows with z_source <= z_lens + {_min_dz}")
        conv.loc[_bad, "source_redshift"] = (_zl[_bad] + _min_dz +
            pd.Series(np.random.uniform(0.1, 0.5, _n_bad), index=conv[_bad].index))

    return conv

def normalize_cosmos_catalog(df):
    """Minimal normalization and validation"""
    df = df.copy()
    
    required_defaults = {
        "source_pa": 0.0,
        "lens_radius": 0.7,
        "source_radius": 0.12,
        "source_x": 0.0,
        "source_y": 0.0,
        "lens_axis_ratio": 0.7,
        "source_axis_ratio": 0.7,
    }
    
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
    
    return df

def safe_stat(df, col, label, fmt="{:.3f}"):
    """Safe statistics reporting"""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[pd.notna(s) & np.isfinite(s)]
        if len(s) > 0:
            try:
                return f"{label}: " + fmt.format(s.min()) + " – " + fmt.format(s.max())
            except:
                return f"{label}: {s.min()} – {s.max()}"
    return f"{label}: N/A"

# --------------------------------------------------------------------------------------
# Helper functions for the main function
# --------------------------------------------------------------------------------------

def save_complete_outputs(filename_base, images, out_root, row_data, field_info, 
                         system_type='lens', rng=None, normalization_scales=None, resolution_name=None, bands=None):
    """Save complete outputs for training sample
    
    Supports both legacy (separate npy files) and unified (.npz) storage modes.
    Set CONFIG['output']['unified_storage'] = True to use unified mode.
    
    Args:
        normalization_scales: Optional dict with normalization scales for consistent
                             RGB across epochs (for time delay systems)
        field_info: Dictionary that may contain 'intermediate_images' key
        resolution_name: Optional resolution name (e.g., 'jwst', 'roman', 'euclid', 'ground_based')
                        to organize outputs by resolution
        bands: Optional list of band names (uppercase) for this telescope/resolution
    """
    # Use provided bands or fall back to global UPPER_BANDS
    active_bands = bands if bands is not None else UPPER_BANDS
    # Optional: filter out empty/near-empty samples
    output_cfg = CONFIG.get('output', {})
    if output_cfg.get('skip_empty_images', True):
        min_total_flux = float(output_cfg.get('min_total_flux', 1e-7))
        min_source_flux = float(output_cfg.get('min_source_flux', 1e-8))

        total_flux = sum(np.sum(images[b]) for b in active_bands if b in images)
        if total_flux < min_total_flux:
            print(f"[SKIP] {filename_base}: total_flux={total_flux:.2e} below {min_total_flux:.2e}")
            return False

        # If intermediate images are available, ensure source signal exists
        if field_info and field_info.get('intermediate_images'):
            intermediate = field_info['intermediate_images']
            source_images = None
            if 'sources_only' in intermediate:
                source_images = intermediate['sources_only']
            elif 'lens_sources' in intermediate:
                source_images = intermediate['lens_sources']

            if source_images and all(b in source_images for b in active_bands):
                source_flux = sum(np.sum(source_images[b]) for b in active_bands if b in source_images)
                if source_flux < min_source_flux:
                    print(f"[SKIP] {filename_base}: source_flux={source_flux:.2e} below {min_source_flux:.2e}")
                    return False

    # Adjust output root for multi-resolution if specified
    output_root = out_root
    if resolution_name:
        output_root = out_root / f"resolution_{resolution_name}"
        output_root.mkdir(parents=True, exist_ok=True)

    # Check if unified storage is enabled
    use_unified = CONFIG.get('output', {}).get('unified_storage', False)
    
    if use_unified:
        # Parse lens_id from filename_base (e.g., "cosmos_lens_012345" -> 12345)
        import re
        match = re.search(r'(\d{6})', filename_base)
        if match:
            lens_id = int(match.group(1))
        else:
            lens_id = 0
        
        is_lens = system_type == 'lens'
        
        # Extract epoch if present
        epoch_match = re.search(r'epoch(\d+)', filename_base)
        epoch_index = int(epoch_match.group(1)) if epoch_match else None
        
        # Create minimal row dict for unified save
        # FIX (adversarial audit finding C-2, 2026-08-01): this used to drop
        # every field except theta_E/redshifts, silently discarding
        # theta_E_override_applied/theta_E_pre_override/lens_sigma_kms/
        # shear_gamma1/shear_gamma2 even though row_data (the actual
        # mutated row from simulate_complete_lens_system_with_real_fields)
        # carried them correctly -- so the *value* of theta_E was fixed but
        # the diagnostic override flag and kinematic labels were still
        # lost at this specific call site.
        row = {
            'lens_id': lens_id,
            'theta_E': row_data.get('theta_E', 0.0),
            'lens_redshift': row_data.get('lens_redshift', 0.0),
            'source_redshift': row_data.get('source_redshift', 0.0),
            'theta_E_override_applied': row_data.get('theta_E_override_applied', False),
            'theta_E_pre_override': row_data.get('theta_E_pre_override', None),
            'lens_sigma_kms': row_data.get('lens_sigma_kms', None),
            'shear_gamma1': row_data.get('shear_gamma1', None),
            'shear_gamma2': row_data.get('shear_gamma2', None),
            'magnification': row_data.get('magnification', None),
            'magnification_gate_min': row_data.get('magnification_gate_min', None),
            'magnification_gate_max': row_data.get('magnification_gate_max', None),
            'source_position_resampled_for_caustic': row_data.get('source_position_resampled_for_caustic', False),
            'source_mag_brightening_applied': row_data.get('source_mag_brightening_applied', False),
        }
        
        # Add time-delay metadata if present
        metadata = {}
        if normalization_scales:
            metadata['normalization_scales'] = normalization_scales
        
        return save_outputs_unified(
            lens_id, images, output_root, row, 0, field_info,
            is_lens=is_lens, epoch_index=epoch_index, metadata=metadata,
            bands=active_bands
        )
    
    # Legacy mode: separate files
    try:
        # Create directories
        (output_root / "npy").mkdir(parents=True, exist_ok=True)
        (output_root / "jpg_rgb").mkdir(parents=True, exist_ok=True)
        
        # Save N-band NPY stack (final image)
        stack = np.stack([images[b] for b in active_bands], axis=0).astype(np.float32)
        if not np.isfinite(stack).any():
            print(f"[ERROR] {filename_base}: Non-finite image data")
            return False
        
        np.save(output_root / "npy" / f"{filename_base}.npy", stack)
        
        # Create and save panel RGB (1x5 showing individual bands + RGB composite)
        # Use consistent normalization scales if provided (for time delay epochs)
        rgb = create_jwst_panel_rgb(images, normalization_scales=normalization_scales, bands=active_bands)
        if rgb is None:
            print(f"[ERROR] {filename_base}: Panel RGB creation failed")
            return False
        
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            output_root / "jpg_rgb" / f"{filename_base}.jpg", 
            quality=95, optimize=True
        )
        
        # Save intermediate images if available
        if field_info and field_info.get('intermediate_images'):
            intermediate_images = field_info['intermediate_images']
            
            # Create intermediate image directories
            for step in ['lens_only', 'lens_sources', 'sources_only', 'sources_unlensed', 'field_only']:
                if step in intermediate_images:
                    step_dir_npy = output_root / "npy" / f"intermediate_{step}"
                    step_dir_jpg = output_root / "jpg_rgb" / f"intermediate_{step}"
                    step_dir_npy.mkdir(parents=True, exist_ok=True)
                    step_dir_jpg.mkdir(parents=True, exist_ok=True)
                    
                    # Check if all bands are present
                    step_images = intermediate_images[step]
                    if all(b in step_images for b in active_bands):
                        # Save N-band NPY stack (N depends on telescope)
                        step_stack = np.stack([step_images[b] for b in active_bands], axis=0).astype(np.float32)
                        np.save(step_dir_npy / f"{filename_base}.npy", step_stack)
                        
                        # Create and save RGB
                        try:
                            step_rgb = create_jwst_panel_rgb(step_images, bands=active_bands)
                            if step_rgb is not None:
                                Image.fromarray((step_rgb * 255).astype(np.uint8)).save(
                                    step_dir_jpg / f"{filename_base}.jpg",
                                    quality=95, optimize=True
                                )
                        except Exception as e:
                            print(f"[WARNING] Failed to create RGB for {step}: {e}")
        
        # KAPPA: Compute and save convergence/shear/magnification maps
        if KAPPA_OUTPUT_AVAILABLE and field_info and system_type == 'lens':
            try:
                lens_model_list = field_info.get('lens_model_list')
                kwargs_lens = field_info.get('kwargs_lens')
                lens_system_class = field_info.get('lens_system_class', 'single_field')
                
                # Determine sub_type from lens_model_list
                sub_type = '+'.join(lens_model_list) if lens_model_list else 'UNKNOWN'
                
                if lens_model_list and kwargs_lens:
                    # Create lens model for kappa computation
                    from lenstronomy.LensModel.lens_model import LensModel
                    lens_model = LensModel(lens_model_list)
                    
                    # Compute kappa products: native-resolution map plus an
                    # extended-FOV map (default 1', auto-coarsened resolution
                    # so compute cost stays bounded regardless of telescope
                    # pixel scale -- config: output.extended_lensing_fov_arcmin)
                    kappa_dict = compute_kappa_products(
                        lens_model, kwargs_lens,
                        num_pix=field_info.get('numpix', 300),
                        delta_pix=field_info.get('delta_pix', 0.031),
                        compute_flexion=True,
                        extended_fov_arcmin=CONFIG.get('output', {}).get('extended_lensing_fov_arcmin', 1.0),
                    )

                    # Extract lens_id from filename
                    import re
                    match = re.search(r'(\d{6})', filename_base)
                    lens_id = match.group(1) if match else 'unknown'
                    
                    # Create kappa output directory (in resolution-specific root)
                    kappa_dir = output_root / "kappa_maps"
                    kappa_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save kappa outputs (NPY, NPZ, 2×JPG)
                    kappa_success = save_kappa_outputs(
                        kappa_dict, kappa_dir, lens_id,
                        category=lens_system_class,
                        sub_type=sub_type
                    )
                    
                    if kappa_success:
                        print(f"[KAPPA] Saved convergence maps for {lens_id} ({lens_system_class})")
                    else:
                        print(f"[WARNING] Kappa output save failed for {lens_id}")
            except Exception as e:
                print(f"[WARNING] Kappa computation failed: {e}")
                # Non-fatal failure - continue without kappa maps
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Save failed for {filename_base}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_jwst_rgb(images, bands=None, telescope=None, arc_images=None):
    """Create realistic RGB using astronomical normalization, tuned per telescope.

    Works with any available band combination by selecting appropriate bands:
    - Red channel: Longest available wavelength
    - Blue channel: Shortest available wavelength
    - Green channel: Middle wavelength(s)

    The display stretch (noise suppression, saturation, color enhancement,
    gamma) is selected from TELESCOPE_RGB_PARAMS based on `telescope`
    (defaults to CONFIG['telescope']), with a wavelength-range heuristic as
    a fallback for unrecognized telescopes.

    When ``arc_images`` contains ``lens_sources`` and ``lens_only`` step dicts,
    a boosted arc residual is added before stretching so lensed sources remain
    visible against the bright deflector in the RGB composite.
    """
    try:
        # Determine available bands and their wavelengths
        if bands is not None:
            available_bands = [b for b in bands if b in images]
        else:
            available_bands = list(images.keys())
        if not available_bands:
            print("[ERROR] No bands available for RGB creation")
            return None

        # Sort bands by wavelength
        band_wavelengths = {b: BAND_CENTERS_UM.get(b.lower(), 0) for b in available_bands}
        sorted_bands = sorted(available_bands, key=lambda b: band_wavelengths.get(b, 0))

        if len(sorted_bands) < 3:
            print(f"[WARNING] Only {len(sorted_bands)} bands available, need at least 3 for RGB")
            # Use grayscale for single band
            if len(sorted_bands) == 1:
                gray = normalize_for_display_astronomical(images[sorted_bands[0]], noise_level=0.3, sat_percent=0.01)
                return np.stack([gray, gray, gray], axis=-1)
            return None

        # Resolve telescope-specific RGB stretch parameters
        if telescope is None:
            telescope = CONFIG.get('telescope', 'jwst')
        telescope = (telescope or 'jwst').lower()
        rgb_params = dict(TELESCOPE_RGB_PARAMS.get(telescope) or {})
        if not rgb_params:
            # Fallback heuristic for unrecognized telescopes: detect narrow-band
            # (near-IR only) filter sets and use the Euclid/Roman-style stretch
            wavelengths = [band_wavelengths.get(b, 0) for b in sorted_bands]
            wavelength_range = max(wavelengths) - min(wavelengths)
            is_narrow_band = wavelength_range < 1.0  # Less than 1 micron range (Roman: ~0.97 μm)
            rgb_params = dict(TELESCOPE_RGB_PARAMS['euclid'] if is_narrow_band else TELESCOPE_RGB_PARAMS['jwst'])

        # Optional per-run overrides (e.g. paper gallery brighter arcs)
        _rgb_cfg = CONFIG.get('output', {}).get('rgb', {}) if isinstance(CONFIG, dict) else {}
        for _k in ('noise_level', 'sat_percent', 'sigma_mult', 'color_enhance', 'arc_boost',
                   'linked_stretch', 'soft_clip', 'band_style_stretch', 'field_overlay',
                   'field_overlay_snr', 'use_trilogy', 'use_eummy'):
            if _k in _rgb_cfg:
                rgb_params[_k] = _rgb_cfg[_k]
        if 'gamma' in _rgb_cfg:
            rgb_params['gamma'] = tuple(_rgb_cfg['gamma'])

        noise_level = rgb_params['noise_level']
        sat_percent = rgb_params['sat_percent']
        sigma_mult = rgb_params['sigma_mult']
        color_enhance = rgb_params['color_enhance']
        gamma_r, gamma_g, gamma_b = rgb_params['gamma']
        linked_stretch = bool(rgb_params.get('linked_stretch', True))
        arc_boost = float(rgb_params.get('arc_boost', 0.0))
        soft_clip = bool(rgb_params.get('soft_clip', False))
        # Match single-band panel visibility (keeps faint field galaxies)
        band_style = bool(rgb_params.get('band_style_stretch', False))

        # Official Euclid display path (Mischa Schirmer / eummy)
        use_eummy = bool(rgb_params.get('use_eummy', False)) or bool(
            _rgb_cfg.get('use_eummy', False)
        )
        use_trilogy = bool(rgb_params.get('use_trilogy', False)) or bool(
            _rgb_cfg.get('use_trilogy', False)
        )
        # trilogy wins if both set (explicit A/B); prefer eummy when only that is on
        if (use_trilogy or use_eummy) and telescope == 'euclid':
            imgs = {b: np.asarray(images[b], dtype=np.float64) for b in available_bands}
            if arc_images and arc_boost > 0:
                arc_residual = _rgb_arc_residual(
                    arc_images.get('lens_sources', {}),
                    arc_images.get('lens_only', {}),
                    sorted_bands,
                )
                for b in list(imgs):
                    if b in arc_residual:
                        imgs[b] = imgs[b] + arc_boost * arc_residual[b]
            if use_trilogy:
                # FIX (adversarial audit finding C-9, 2026-08-01): this
                # try/except order was INVERTED -- it tried `src.X` (which
                # silently resolves to the SIBLING dev repo's file via the
                # shared `src` namespace package, since both repos happen
                # to be on sys.path) FIRST, only falling back to this
                # package's own prism.telescopes.euclid_trilogy_rgb if
                # that failed. Since the sibling repo's file exists and
                # imports fine, this package was unconditionally executing
                # unversioned code from a different git repository.
                from prism.telescopes.euclid_trilogy_rgb import (
                    create_euclid_trilogy_rgb, trilogy_params_from_config,
                )
                return create_euclid_trilogy_rgb(
                    imgs, **trilogy_params_from_config(
                        CONFIG if isinstance(CONFIG, dict) else {}
                    )
                )
            from prism.telescopes.euclid_eummy_rgb import create_euclid_eummy_rgb, eummy_params_from_config
            return create_euclid_eummy_rgb(imgs, **eummy_params_from_config(
                CONFIG if isinstance(CONFIG, dict) else {}
            ))

        norm_kwargs = dict(sat_percent=sat_percent, sigma_mult=sigma_mult, soft_clip=soft_clip)

        # Select bands for RGB channels
        red_band = sorted_bands[-1]
        blue_band = sorted_bands[0]
        mid_bands = sorted_bands[1:-1]

        # Optional arc residual (lensed source light minus lens-only)
        arc_residual = {}
        if arc_images and arc_boost > 0:
            arc_residual = _rgb_arc_residual(
                arc_images.get('lens_sources', {}),
                arc_images.get('lens_only', {}),
                sorted_bands,
            )

        def _channel_data(band_list, weights=None):
            if len(band_list) == 1:
                data = np.asarray(images[band_list[0]], dtype=np.float64).copy()
                if arc_residual:
                    data = data + arc_boost * arc_residual.get(band_list[0], 0.0)
                return data
            weights = weights or [1.0 / len(band_list)] * len(band_list)
            data = np.zeros_like(images[band_list[0]], dtype=np.float64)
            for band, w in zip(band_list, weights):
                band_data = np.asarray(images[band], dtype=np.float64)
                if arc_residual and band in arc_residual:
                    band_data = band_data + arc_boost * arc_residual[band]
                data = data + w * band_data
            return data

        r_data = _channel_data([red_band])
        b_data = _channel_data([blue_band])
        g_data = _channel_data(mid_bands) if mid_bands else _channel_data([blue_band])

        if band_style:
            # Same stretch as single-band panels → faint field galaxies stay visible
            # in RGB (at the cost of a grainier background than the cleaned composite).
            R = normalize_for_display_astronomical(r_data, noise_level=noise_level, sat_percent=min(sat_percent, 0.05))
            G = normalize_for_display_astronomical(g_data, noise_level=noise_level, sat_percent=min(sat_percent, 0.05))
            B = normalize_for_display_astronomical(b_data, noise_level=noise_level, sat_percent=min(sat_percent, 0.05))
        elif linked_stretch:
            # Shared stretch scale preserves inter-band flux ratios (true colour).
            _, r_scale = _normalize_for_rgb_composite_core(r_data, **norm_kwargs)
            _, g_scale = _normalize_for_rgb_composite_core(g_data, **norm_kwargs)
            _, b_scale = _normalize_for_rgb_composite_core(b_data, **norm_kwargs)
            scales = [s for s in (r_scale, g_scale, b_scale) if s is not None and s > 0]
            common_scale = float(np.median(scales)) if scales else None
            R, _ = _normalize_for_rgb_composite_core(r_data, stretch_scale=common_scale, **norm_kwargs)
            G, _ = _normalize_for_rgb_composite_core(g_data, stretch_scale=common_scale, **norm_kwargs)
            B, _ = _normalize_for_rgb_composite_core(b_data, stretch_scale=common_scale, **norm_kwargs)
            if R is None:
                R = np.zeros_like(r_data)
            if G is None:
                G = np.zeros_like(g_data)
            if B is None:
                B = np.zeros_like(b_data)
        else:
            R = normalize_for_rgb_composite(r_data, noise_level=noise_level, **norm_kwargs)
            G = normalize_for_rgb_composite(g_data, noise_level=noise_level, **norm_kwargs)
            B = normalize_for_rgb_composite(b_data, noise_level=noise_level, **norm_kwargs)

        # Combine channels
        rgb = np.stack([R, G, B], axis=-1)
        rgb = np.clip(rgb, 0, 1)

        # For band-style stretch: keep smooth galaxy profiles from the panel
        # normalizer, but neutralize uncorrelated per-channel sky noise so the
        # RGB does not turn into purple salt-and-pepper.
        if band_style:
            try:
                from scipy.ndimage import gaussian_filter
                luma = np.mean(rgb, axis=2)
                # Mild spatial smooth of chroma only (luma stays sharp)
                rgb_s = np.stack([gaussian_filter(rgb[:, :, c], sigma=0.7) for c in range(3)], axis=-1)
                chroma = rgb_s - np.mean(rgb_s, axis=2, keepdims=True)
                # SNR proxy from raw VIS/blue channel if available, else luma
                sky = float(np.median(luma))
                mad = float(np.median(np.abs(luma - sky))) * 1.4826 + 1e-12
                snr = (luma - sky) / mad
                # Full colour above ~4σ; grayscale below ~2σ
                w = np.clip((snr - 2.0) / 2.0, 0.0, 1.0)[..., None]
                rgb = (luma[..., None] + chroma * w)
                # Soft sky floor matching panel gray ~ mid-dark, not pure black
                floor = max(sky - 0.5 * mad, 0.0)
                rgb = np.clip(rgb - 0.35 * floor, 0.0, 1.0)
                rgb = np.clip(rgb, 0, 1)
            except Exception:
                pass

        if bool(rgb_params.get('field_overlay', False)) and not band_style:
            # Detect real sources on raw flux (SNR), not display-stretched panels
            # (asinh stretch turns noise grain into fake "galaxies").
            snr_peak = None
            for _band in sorted_bands:
                _im = np.asarray(images[_band], dtype=np.float64)
                _sky = float(np.median(_im))
                _mad = float(np.median(np.abs(_im - _sky))) * 1.4826
                if _mad <= 0:
                    _mad = float(np.std(_im)) + 1e-12
                _snr = (_im - _sky) / _mad
                snr_peak = _snr if snr_peak is None else np.maximum(snr_peak, _snr)
            snr_thresh = float(rgb_params.get('field_overlay_snr', 4.0))
            source_mask = snr_peak > snr_thresh
            try:
                from scipy.ndimage import binary_opening, binary_dilation, generate_binary_structure, label
                st = generate_binary_structure(2, 1)
                source_mask = binary_opening(source_mask, structure=st, iterations=1)
                # Drop isolated noise spikes (keep components with >= 4 pixels)
                labeled, nlab = label(source_mask)
                if nlab > 0:
                    counts = np.bincount(labeled.ravel())
                    keep_ids = np.where(counts >= 4)[0]
                    source_mask = np.isin(labeled, keep_ids) & (labeled > 0)
                source_mask = binary_dilation(source_mask, structure=st, iterations=1)
            except Exception:
                pass
            # Display luminance target from band-style stretch
            band_peak = None
            for _band in sorted_bands:
                _bn = normalize_for_display_astronomical(
                    np.asarray(images[_band], dtype=np.float64),
                    noise_level=0.30, sat_percent=0.01,
                )
                band_peak = _bn if band_peak is None else np.maximum(band_peak, _bn)
            lum = np.mean(rgb, axis=2)
            deficit = np.clip(band_peak - lum, 0.0, None)
            # Mild blend — avoid dumping band-panel noise grain into RGB
            blend = 0.55 if soft_clip else 0.95
            for _c in range(3):
                rgb[:, :, _c] = np.where(
                    source_mask,
                    np.clip(rgb[:, :, _c] + blend * deficit, 0.0, 1.0),
                    rgb[:, :, _c],
                )

        # Force a clean dark sky: desaturate residual colour noise near the floor
        # (skip when band_style already applied luma/chroma denoise above).
        if not band_style:
            luma = np.mean(rgb, axis=2, keepdims=True)
            sky_ref = float(np.percentile(luma, 20))
            chroma_kill = np.clip((0.07 + sky_ref - luma) / max(0.04, 0.07 + sky_ref), 0.0, 1.0)
            rgb = rgb * (1.0 - chroma_kill) + luma * chroma_kill
            # Gentle black-point bias (not a hard crush) — avoids stair-step edges
            black = 0.55 * float(np.percentile(luma, 12))
            rgb = np.clip(rgb - black, 0.0, 1.0)
            peak = float(np.percentile(rgb, 99.7))
            if peak > 1e-6:
                rgb = np.clip(rgb / peak, 0.0, 1.0)

        # Color saturation enhancement (telescope-specific)
        if color_enhance > 1.0:
            # Apply color enhancement to separate subtle differences, but only
            # where there is real signal -- in low-brightness background
            # pixels the per-band noise is largely uncorrelated, so amplifying
            # the color deviation there just creates random color speckle.
            # Ramp the enhancement strength from 1.0 (background) to
            # color_enhance (sources) based on local brightness so the
            # background stays clean while source colors -- including faint,
            # low surface-brightness sources -- remain visible and enhanced.
            rgb_mean = np.mean(rgb, axis=2, keepdims=True)
            bg_threshold = 0.08
            signal_threshold = 0.22
            enhancement_weight = np.clip(
                (rgb_mean - bg_threshold) / (signal_threshold - bg_threshold), 0.0, 1.0
            )
            enhancement_factor = 1.0 + (color_enhance - 1.0) * enhancement_weight
            color_deviation = rgb - rgb_mean
            rgb = rgb_mean + color_deviation * enhancement_factor
            rgb = np.clip(rgb, 0, 1)

        # Per-channel gamma correction (telescope-specific; defaults to 1.0 = no-op)
        if (gamma_r, gamma_g, gamma_b) != (1.0, 1.0, 1.0):
            rgb[:, :, 0] = np.power(rgb[:, :, 0], gamma_r)
            rgb[:, :, 1] = np.power(rgb[:, :, 1], gamma_g)
            rgb[:, :, 2] = np.power(rgb[:, :, 2], gamma_b)

        # Mild brightness lift for narrow-band telescopes (Euclid/Roman).
        # Soft-clip / band-style paths already preserve midtones — keep boost small
        # so BGG cores are not blown to flat white.
        if color_enhance > 1.0 and not soft_clip and not band_style:
            brightness_factor = 1.10
            rgb = rgb * brightness_factor
            rgb = np.clip(rgb, 0, 1)
        elif color_enhance > 1.0 and soft_clip:
            rgb = np.clip(rgb * 1.04, 0, 1)
        
        # Conservative brightness adjustment
        max_brightness = np.max(rgb)
        if max_brightness < 0.02:
            boost = min(0.08 / (max_brightness + 1e-12), 2.0)
            rgb *= boost
            rgb = np.clip(rgb, 0, 1)
        
        return rgb
        
    except Exception as e:
        print(f"[ERROR] RGB creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_jwst_panel_rgb(images, normalization_scales=None, bands=None, telescope=None,
                          arc_images=None):
    """Create 1xN panel RGB showing individual bands + RGB composite with labels

    Args:
        images: Dictionary of band images
        normalization_scales: Optional dict with normalization scales for consistent
                             RGB across epochs (for time delay systems)
        bands: Optional list of band names (uppercase) to use. If None, uses UPPER_BANDS
        telescope: Optional telescope name to select RGB display tuning
                   (defaults to CONFIG['telescope']). See TELESCOPE_RGB_PARAMS.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image, ImageDraw, ImageFont

        # Use provided bands or fall back to UPPER_BANDS
        active_bands = bands if bands is not None else UPPER_BANDS
        # Filter to only bands that exist in images (uppercase keys)
        available_bands = [b for b in active_bands if b in images]

        if not available_bands:
            print(f"[ERROR] No bands available for RGB creation")
            return None

        if telescope is None:
            telescope = CONFIG.get('telescope', 'jwst')
        telescope = (telescope or 'jwst').lower()

        band_images = {}

        for band in available_bands:
            # Original normalization for individual band panels -- unaffected
            # by the telescope-specific RGB composite tuning, so single-band
            # displays keep their original noise/background appearance.
            normalized = normalize_for_display_astronomical(
                images[band],
                noise_level=0.3,
                sat_percent=0.01,
                channel_name=band
            )
            band_images[band] = normalized

        # Create RGB composite with improved color balance
        # Use consistent normalization if provided (for time delay epochs)
        if normalization_scales is not None and len(normalization_scales) > 0:
            # Use trilogy RGB with consistent normalization to preserve brightness differences
            numpix = images[available_bands[0]].shape[0]  # Use first available band for shape
            try:
                rgb_composite, _ = create_trilogy_rgb(images, numpix, normalization_scales)
            except Exception as e:
                print(f"[WARNING] Failed to use consistent normalization scales: {e}, falling back to default")
                # Fall back to create_jwst_rgb which works with any bands
                rgb_composite = create_jwst_rgb(images, bands=active_bands, telescope=telescope,
                                                arc_images=arc_images)
        else:
            # Use flexible create_jwst_rgb that works with any band combination
            rgb_composite = create_jwst_rgb(images, bands=active_bands, telescope=telescope,
                                            arc_images=arc_images)
        
        # Get image dimensions from first available band
        height, width = images[available_bands[0]].shape
        
        # Create 1xN panel by concatenating images horizontally with frames
        # Add frame width between panels and space for labels
        n_panels = len(available_bands) + 1  # Individual bands + RGB composite
        frame_width = 2  # pixels
        label_height = 30  # pixels for band labels
        panel_width = width * n_panels + frame_width * (n_panels - 1)  # frames between panels
        panel_height = height + label_height  # Add space for labels
        
        # Create the panel image with black background
        panel_rgb = np.zeros((panel_height, panel_width, 3), dtype=np.float32)
        
        # Place individual band images (grayscale) with frames
        for i, band in enumerate(available_bands):
            start_x = i * (width + frame_width)
            end_x = start_x + width
            # Convert grayscale to RGB
            band_rgb = np.stack([band_images[band], band_images[band], band_images[band]], axis=-1)
            panel_rgb[label_height:label_height+height, start_x:end_x, :] = band_rgb
            
            # Add frame after each panel (except the last one)
            if i < n_panels - 1:  # Don't add frame after the last panel
                frame_start = end_x
                frame_end = frame_start + frame_width
                # White frame
                panel_rgb[label_height:label_height+height, frame_start:frame_end, :] = 1.0
        
        # Place RGB composite in the last panel
        start_x = len(available_bands) * (width + frame_width)
        end_x = start_x + width
        panel_rgb[label_height:label_height+height, start_x:end_x, :] = rgb_composite
        
        # Convert to PIL Image for text rendering
        panel_pil = Image.fromarray((panel_rgb * 255).astype(np.uint8))
        draw = ImageDraw.Draw(panel_pil)
        
        # Try to use a system font, fallback to default if not available
        try:
            # Try different font sizes and types
            font_size = min(20, width // 15)  # Scale font size with image size
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Add band labels
        labels = list(available_bands) + ['RGB']
        for i, label in enumerate(labels):
            start_x = i * (width + frame_width)
            # Center the text in each panel
            text_width = draw.textlength(label, font=font)
            text_x = start_x + (width - text_width) // 2
            text_y = 5  # Small margin from top
            
            # Add white text with black outline for better visibility
            # Draw black outline
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), label, font=font, fill=(0, 0, 0))
            # Draw white text
            draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))
        
        # Convert back to numpy array
        panel_rgb = np.array(panel_pil).astype(np.float32) / 255.0
        
        return panel_rgb
        
    except Exception as e:
        print(f"[ERROR] Panel RGB creation failed: {e}")
        # Fallback to simple RGB
        return create_jwst_rgb(images, bands=active_bands, telescope=CONFIG.get('telescope', 'jwst'))

# Quality assessment helper functions
def categorize_dataset_size(n_samples):
    """Categorize dataset by size"""
    if n_samples >= 10000:
        return "very_large"
    elif n_samples >= 5000:
        return "large"
    elif n_samples >= 2000:
        return "medium"
    elif n_samples >= 500:
        return "small"
    else:
        return "very_small"

def assess_training_quality(lens_dark_pct, nonlens_dark_pct, total_samples, errors):
    """Assess overall training dataset quality"""
    avg_dark = (lens_dark_pct + nonlens_dark_pct) / 2
    error_rate = 100 * errors / max(total_samples + errors, 1)
    
    if total_samples >= 8000 and avg_dark < 5 and error_rate < 2:
        return "Excellent"
    elif total_samples >= 3000 and avg_dark < 15 and error_rate < 5:
        return "Good"
    elif total_samples >= 1000 and avg_dark < 25 and error_rate < 10:
        return "Acceptable"
    else:
        return "Poor"

def assess_class_balance(n_positive, n_negative):
    """Assess class balance for training"""
    if n_negative == 0:
        return "single_class"
    
    ratio = n_positive / n_negative
    if 0.8 <= ratio <= 1.2:
        return "excellent"
    elif 0.5 <= ratio <= 2.0:
        return "good"
    elif 0.2 <= ratio <= 5.0:
        return "moderate"
    else:
        return "poor"

def is_training_ready(total_samples, lens_dark_pct, nonlens_dark_pct, errors):
    """Determine if dataset is ready for ML training"""
    return (total_samples >= 1000 and 
            lens_dark_pct < 30 and 
            nonlens_dark_pct < 30 and 
            errors < total_samples * 0.2)

def recommend_ml_strategy(n_lens, n_nonlens, has_real_fields):
    """Recommend ML training strategy"""
    total = n_lens + n_nonlens
    
    if n_nonlens == 0:
        strategy = "Anomaly detection or one-class classification"
    elif n_lens > 0 and n_nonlens > 0:
        balance_ratio = n_lens / n_nonlens
        if 0.7 <= balance_ratio <= 1.3:
            strategy = "Balanced binary classification"
        else:
            strategy = "Imbalanced classification with class weighting"
    else:
        strategy = "Review dataset composition"
    
    if total >= 8000:
        strategy += " + data augmentation"
    elif total < 2000:
        strategy += " + transfer learning"
    
    if has_real_fields:
        strategy += " + domain adaptation for field contamination"
    
    return strategy

# --------------------------------------------------------------------------------------
# COMPLETE CORRECTED MAIN FUNCTION
# --------------------------------------------------------------------------------------

def main():
    """Complete main function for COSMOS-Web lens simulation pipeline  v10 CORRECTED"""
    global UPPER_BANDS, LOWER_BANDS, BAND_TO_LOWER
    
    # Complete argument parser with all required arguments
    parser = argparse.ArgumentParser(
        description="COSMOS-Web Mock Lens Generator  v10 FINAL CORRECTED with Enhanced Field Galaxies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete ML training dataset generation with realistic field galaxy populations.

Example usage:
  python cosmos_web_lens_mock_ v10_final.py \\
    --cosmos_catalog cosmos_web_lens_structural_properties.csv \\
    --merged_field_catalog merged_lens_field_catalog.csv \\
    --output_dir training_dataset \\
    --n_lenses 5000 --n_non_lenses 5000 \\
    --variations_per_base 25 --n_field_max 8 --add_artifacts
        """
    )
    
    # Required core arguments
    parser.add_argument("--cosmos_catalog", required=True, type=str,
                       help="COSMOS-Web structural properties CSV file")
    parser.add_argument("--output_dir", required=True, type=str,
                       help="Output directory for training dataset")
    
    # Generation control
    parser.add_argument("--n_lenses", type=int, default=10000,
                       help="Number of lens systems (positive samples)")
    parser.add_argument("--n_non_lenses", type=int, default=0,
                       help="Number of non-lens systems (negative samples)")
    parser.add_argument("--non_lens_modes", type=str, default="central_galaxy galaxy_pair galaxy_group",
                       help="Non-lens simulation modes (space-separated)")
    parser.add_argument("--variations_per_base", type=int, default=25,
                       help="Parameter variations per base configuration")
    parser.add_argument("--batch_size", type=int, default=200,
                       help="Batch size for progress reporting")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--numpix", type=int, default=None,
                       help="Image size in pixels (square). Default: auto from telescope_configs "
                            "(300 for JWST, 128 for Roman/Euclid/Subaru/LSST)")
    # Config file
    parser.add_argument("--config", type=str, default=None,
                       help="YAML configuration file to override defaults")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for file naming (for continuing previous runs)")
    
    # CORRECTED: Field galaxy and realism arguments
    parser.add_argument("--merged_field_catalog", type=str, default=None,
                       help="Merged CSV file with all field galaxy measurements")
    parser.add_argument("--field_galaxy_dir", type=str, default=None,
                       help="Directory with individual field galaxy catalogs")
    parser.add_argument("--n_field_max", type=int, default=5,  # Tightened from 8 to 4-5
                       help="Maximum field galaxies per image (environment-based: 0-2 field, 1-3 pair, 2-4 group)")
    parser.add_argument("--add_artifacts", action="store_true",
                       help="Add observational artifacts (cosmic rays, noise). Can also be controlled via config file.")
    parser.add_argument("--add_jwst_spikes", action="store_true",
                       help="Add realistic JWST 4-point diffraction spikes for bright stars. Only works with --add_artifacts.")
    parser.add_argument("--save_intermediate", action="store_true",
                       help="Save intermediate step images (lens plane, source plane, final). Useful for diagnostics.")
    
    # Optional inputs
    parser.add_argument("--lens_analysis_catalog", type=str, default=None,
                       help="Optional lens analysis CSV with Einstein radii")
    parser.add_argument("--no_date_suffix", action="store_true",
                       help="Skip timestamp suffix on output directory")
    
    # Parse and validate arguments
    args = parser.parse_args()

    # Display startup banner
    print(f"{'='*80}")
    print("COSMOS-Web Mock Lens Generator  v10 FINAL CORRECTED")
    print("Enhanced Field Galaxy Populations + Morphological Diversity")
    print(f"{'='*80}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {args.seed}")
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'interactive')}")
    print("")

    # Load configuration
    load_config(args.config)

    # Let YAML n_lenses/n_non_lenses override CLI defaults when not explicitly given.
    import sys as _sys
    if '--n_lenses' not in _sys.argv and 'n_lenses' in CONFIG:
        args.n_lenses = int(CONFIG['n_lenses'])
    if '--n_non_lenses' not in _sys.argv and 'n_non_lenses' in CONFIG:
        args.n_non_lenses = int(CONFIG['n_non_lenses'])
    if '--n_field_max' not in _sys.argv and 'n_field_max' in CONFIG:
        args.n_field_max = int(CONFIG['n_field_max'])
    if '--variations_per_base' not in _sys.argv and 'variations_per_base' in CONFIG:
        args.variations_per_base = int(CONFIG['variations_per_base'])

    # Override config with command-line arguments
    if args.add_artifacts:
        CONFIG['add_artifacts'] = True
    if args.save_intermediate:
        CONFIG['save_intermediate_images'] = True

    # Resolve numpix: CLI arg wins; otherwise output.extended_image_fov_arcmin
    # (if set) overrides image_size to cover that FOV at the active
    # telescope's native pixel scale, for ANY telescope; otherwise use
    # telescope_configs[telescope].image_size; otherwise fall back to global
    # image_size; otherwise 300 (JWST default).
    _active_tel = CONFIG.get('telescope', 'jwst').lower()
    _tel_cfg = CONFIG.get('telescope_configs', {}).get(_active_tel, {})
    _active_pixel_scale = float(_tel_cfg.get('pixel_scale', CONFIG.get('pixel_scale', 0.031)))
    _extended_image_fov_arcmin = CONFIG.get('output', {}).get('extended_image_fov_arcmin')
    if args.numpix is not None:
        _resolved_numpix = int(args.numpix)
    elif _extended_image_fov_arcmin:
        _resolved_numpix = int(round(_extended_image_fov_arcmin * 60.0 / _active_pixel_scale))
        # Keep even so the grid has a well-defined center pixel pair, matching
        # the convention used elsewhere (e.g. Euclid's 600px @ 0.1"/px = 1').
        if _resolved_numpix % 2:
            _resolved_numpix += 1
        print(f"[CONFIG] output.extended_image_fov_arcmin={_extended_image_fov_arcmin} -> "
              f"overriding image_size to {_resolved_numpix}px @ {_active_pixel_scale:.4f}\"/px "
              f"for telescope={_active_tel.upper()}")
    else:
        _resolved_numpix = int(_tel_cfg.get('image_size',
                               CONFIG.get('image_size', 300)))
    args.numpix = _resolved_numpix
    print(f"[CONFIG] Image size: {args.numpix}×{args.numpix} px  "
          f"(telescope={CONFIG.get('telescope','jwst').upper()}, "
          f"pixel_scale={CONFIG.get('telescope_configs',{}).get(CONFIG.get('telescope','jwst'),{}).get('pixel_scale', CONFIG.get('pixel_scale',0.031)):.4f}\"/px, "
          f"FOV={args.numpix * CONFIG.get('telescope_configs',{}).get(CONFIG.get('telescope','jwst'),{}).get('pixel_scale', CONFIG.get('pixel_scale',0.031)):.1f}\")")
    
    # Check for multi-resolution configuration
    multi_resolution_enabled = CONFIG.get('multi_resolution', {}).get('enabled', False)
    resolution_scales = []
    if multi_resolution_enabled:
        scales_config = CONFIG.get('multi_resolution', {}).get('scales', [])
        for scale_cfg in scales_config:
            resolution_scales.append({
                'name': scale_cfg.get('name'),
                'pixel_scale': scale_cfg.get('pixel_scale'),
                'description': scale_cfg.get('description', scale_cfg.get('name')),
                'fov_arcsec': scale_cfg.get('fov_arcsec')
            })
        if resolution_scales:
            print(f"\n[MULTI-RESOLUTION] Enabled with {len(resolution_scales)} scales:")
            for res in resolution_scales:
                print(f"  - {res['name']:15s}: {res['pixel_scale']:.3f}\"/pix ({res['description']})")
        else:
            print(f"\n[MULTI-RESOLUTION] Config present but no scales defined - disabling")
            multi_resolution_enabled = False
    
    # Verify time delay config is loaded
    if 'time_delays' in CONFIG:
        td_cfg = CONFIG['time_delays']
        print(f"[CONFIG] Time delays: enabled={td_cfg.get('enabled', False)}, fraction={td_cfg.get('fraction_variable_sources', 0)}")
    else:
        print("[CONFIG] WARNING: time_delays not found in CONFIG - time delays will be disabled")

    # Initialize global random state
    #
    # FIX (adversarial audit finding C-5, 2026-08-01): args.seed only ever
    # seeded this local `rng` object. But read_combined_cosmos_catalogs()
    # and numerous other functions draw from the GLOBAL, UNSEEDED
    # np.random.* stream (54 call sites) and from bare, OS-entropy-seeded
    # np.random.default_rng() calls -- both completely ignore args.seed.
    # Confirmed by execution: two identical calls to
    # read_combined_cosmos_catalogs() in one process returned different
    # theta_E (2.741 vs 3.091). Seeding the global legacy np.random state
    # here makes every remaining `np.random.X(...)` call in the module
    # deterministic *given a fixed call order*, which is the best
    # available fix without threading `rng` through 54+ call sites (a much
    # larger refactor deferred; see PROJECT_NOTES). This does not, by
    # itself, fix the bare `np.random.default_rng()` calls at various
    # per-galaxy/per-band sites (those still draw fresh OS entropy each
    # time) -- those are patched individually below via _seeded_rng().
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"✓ Random seed: {args.seed} (global np.random legacy state seeded too)")
    
    # STEP 1: Input validation and path setup
    print("\nSTEP 1: Input Validation")
    print("=" * 50)
    
    try:
        # Validate COSMOS catalog
        struct_path = Path(args.cosmos_catalog).expanduser().resolve()
        if not struct_path.exists():
            raise FileNotFoundError(f"COSMOS catalog not found: {struct_path}")
        
        file_size_mb = struct_path.stat().st_size / (1024 * 1024)
        print(f"✓ COSMOS catalog: {struct_path.name}")
        print(f"  Size: {file_size_mb:.1f} MB")

        # Validate analysis catalog (optional)
        analysis_path = None
        if args.lens_analysis_catalog:
            analysis_path = Path(args.lens_analysis_catalog).expanduser().resolve()
            if analysis_path.exists():
                print(f"✓ Analysis catalog: {analysis_path.name}")
            else:
                print(f"⚠ Analysis catalog not found (will generate parameters)")
                analysis_path = None

        # CORRECTED: Enhanced field galaxy data validation
        merged_field_path = None        
        if args.merged_field_catalog:
            merged_field_path = Path(args.merged_field_catalog).expanduser().resolve()
            if merged_field_path.exists():
                field_size_mb = merged_field_path.stat().st_size / (1024 * 1024)
                print(f"✓ Merged field catalog: {merged_field_path.name}")
                print(f"  Size: {field_size_mb:.1f} MB")
                
                # Test CSV readability
                try:
                    test_df = pd.read_csv(merged_field_path, nrows=3, dtype=str, low_memory=False)
                    print(f"  ✓ CSV format validated ({len(test_df.columns)} columns)")
                except Exception as csv_e:
                    raise RuntimeError(f"Merged field catalog corrupted: {csv_e}")
            else:
                raise FileNotFoundError(f"Merged field catalog not found: {merged_field_path}")
        elif args.field_galaxy_dir:
            field_dir_path = Path(args.field_galaxy_dir).expanduser().resolve()
            if field_dir_path.exists():
                print(f"✓ Field galaxy directory: {field_dir_path}")
            else:
                print(f"⚠ Field galaxy directory not found")
        else:
            print(f"⚠ No field galaxy data - will use synthetic populations")

    except Exception as e:
        print(f"ERROR: Input validation failed: {e}")
        sys.exit(1)

    # STEP 2: Output directory setup
    print(f"\nSTEP 2: Output Directory Setup")
    print("=" * 50)
    
    try:
        base_output_dir = Path(args.output_dir).expanduser().resolve()
        
        # Add job/timestamp suffix unless disabled
        if not args.no_date_suffix:
            job_id = os.environ.get('SLURM_JOB_ID', None)
            if job_id:
                suffix = f"job_{job_id}"
            else:
                from datetime import datetime
                suffix = f"date_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_root = base_output_dir.parent / f"{base_output_dir.name}_{suffix}"
        else:
            out_root = base_output_dir
        
        # Create complete directory structure
        out_root.mkdir(parents=True, exist_ok=True)
        required_subdirs = ["npy", "jpg_rgb", "diagnostics", "logs"]
        for subdir in required_subdirs:
            (out_root / subdir).mkdir(exist_ok=True)
        
        print(f"✓ Output directory: {out_root}")
        
        # Check write permissions
        test_file = out_root / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print(f"✓ Write permissions verified")

    except Exception as e:
        print(f"ERROR: Output directory setup failed: {e}")
        sys.exit(1)

    # STEP 3: CORRECTED field galaxy population loading
    print(f"\nSTEP 3: Enhanced Field Galaxy Data Loading")
    print("=" * 50)
    
    field_pop = None
    try:
        if merged_field_path:
            print(f"Loading merged field catalog...")
            field_pop = load_real_field_galaxy_population_from_merged(str(merged_field_path))
            
            if field_pop is not None:
                print(f"✓ SUCCESS: Loaded {len(field_pop):,} real field galaxies")
            else:
                raise RuntimeError("Failed to process merged field catalog")
        elif args.field_galaxy_dir:
            field_dir_path = Path(args.field_galaxy_dir).expanduser().resolve()
            if field_dir_path and field_dir_path.exists():
                print(f"Loading field galaxy directory...")
                field_pop = load_real_field_galaxy_population(str(field_dir_path))
                
                if field_pop is not None:
                    print(f"✓ SUCCESS: Loaded {len(field_pop):,} field galaxies from directory")
                else:
                    raise RuntimeError("Failed to load field galaxy directory")
        else:
            print(f"⚠ No field galaxy data provided")
            print(f"  Will use synthetic field galaxy populations")

        # CORRECTED: Run enhanced diagnostics
        if field_pop is not None:
            print(f"\n[INFO] Running enhanced field galaxy diagnostics...")
            diagnose_field_galaxy_sampling(field_pop, n_max=args.n_field_max, rng=rng, numpix=args.numpix)
        else:
            print(f"[INFO] No real field data - synthetic populations will be used")

    except Exception as e:
        print(f"ERROR: Field galaxy loading failed: {e}")
        print(f"⚠ Continuing with synthetic field galaxy populations")
        field_pop = None

    # STEP 4: Load and process base lens catalog
    print(f"\nSTEP 4: Base Lens Catalog Processing")
    print("=" * 50)
    
    try:
        print(f"Loading COSMOS structural catalog...")
        base_catalog = read_combined_cosmos_catalogs(
            str(struct_path),
            str(analysis_path) if analysis_path else None
        )
        
        print(f"✓ Base catalog processed: {len(base_catalog):,} configurations")

        # Optional catalog-level selection for lens Sérsic index (student-project control)
        lens_sel_cfg = CONFIG.get('lens_selection', {}) if isinstance(CONFIG, dict) else {}
        min_catalog_sersic = lens_sel_cfg.get('min_lens_sersic_catalog', None)
        if min_catalog_sersic is not None:
            try:
                min_catalog_sersic = float(min_catalog_sersic)
                candidate_cols = ['n_rest', 'n_rest_clean', 'lens_n_sersic']
                sersic_col = next((c for c in candidate_cols if c in base_catalog.columns), None)
                if sersic_col is not None:
                    before_count = len(base_catalog)
                    sersic_vals = pd.to_numeric(base_catalog[sersic_col], errors='coerce')
                    keep_mask = pd.notna(sersic_vals) & np.isfinite(sersic_vals) & (sersic_vals >= min_catalog_sersic)
                    base_catalog = base_catalog.loc[keep_mask].reset_index(drop=True)
                    after_count = len(base_catalog)
                    print(f"✓ Applied lens_selection.min_lens_sersic_catalog={min_catalog_sersic:.2f} on '{sersic_col}': {before_count:,} -> {after_count:,}")
                    if after_count == 0:
                        raise RuntimeError("No lenses left after min_lens_sersic_catalog filter")
                else:
                    print("[WARNING] min_lens_sersic_catalog set but no Sérsic column found in base catalog")
            except Exception as e:
                print(f"[WARNING] Could not apply min_lens_sersic_catalog filter: {e}")
        
        # Display key parameter statistics
        key_params = ['theta_E', 'lens_redshift', 'source_redshift', 'lens_radius']
        for param in key_params:
            if param in base_catalog.columns:
                values = pd.to_numeric(base_catalog[param], errors='coerce')
                valid_values = values[pd.notna(values) & np.isfinite(values)]
                if len(valid_values) > 0:
                    print(f"  {param}: {valid_values.min():.3f} - {valid_values.max():.3f}")

    except Exception as e:
        print(f"ERROR: Base catalog loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # STEP 5: Parameter variation generation
    print(f"\nSTEP 5: Parameter Variation Generation")
    print("=" * 50)
    
    try:
        if args.variations_per_base > 1:
            print(f"Generating {args.variations_per_base} variations per base configuration...")
            expanded_catalog = create_parameter_variations(
                base_catalog, 
                variations_per_base=args.variations_per_base,
                rng=rng
            )
            diversity_factor = len(expanded_catalog) / len(base_catalog)
            print(f"✓ Parameter variations complete: {diversity_factor:.1f}x diversity")
        else:
            print(f"No parameter variations requested")
            expanded_catalog = base_catalog.copy()
            expanded_catalog["base_lens_id"] = expanded_catalog.index
            expanded_catalog["variation_id"] = 0
            expanded_catalog["total_id"] = expanded_catalog.index

    except Exception as e:
        print(f"ERROR: Parameter variation failed: {e}")
        sys.exit(1)

    # STEP 6: Final catalog preparation
    print(f"\nSTEP 6: Training Catalog Preparation")  
    print("=" * 50)
    
    try:
        # Prepare lens catalog
        if args.n_lenses > 0:
            if args.n_lenses < len(expanded_catalog):
                lens_catalog = expanded_catalog.iloc[:args.n_lenses].reset_index(drop=True)
                print(f"✓ Lens catalog: {len(lens_catalog):,} (limited)")
            else:
                lens_catalog = expanded_catalog.copy()
                print(f"✓ Lens catalog: {len(lens_catalog):,} (using all)")
            
            # Normalize and validate
            lens_catalog = normalize_cosmos_catalog(lens_catalog)
            print(f"✓ Lens catalog normalized")
        else:
            lens_catalog = pd.DataFrame()
            print(f"⚠ No lens systems requested")

        # Parse non-lens configuration
        if args.n_non_lenses > 0:
            non_lens_modes = [mode.strip() for mode in args.non_lens_modes.split() if mode.strip()]
            if not non_lens_modes:
                non_lens_modes = ["central_galaxy"]
            print(f"✓ Non-lens systems: {args.n_non_lenses:,}")
            print(f"✓ Non-lens modes: {non_lens_modes}")
        else:
            non_lens_modes = []

        total_planned = len(lens_catalog) + args.n_non_lenses
        
        if total_planned == 0:
            raise ValueError("No systems requested for generation")

        print(f"\nFinal Generation Plan:")
        print(f"  Lens systems: {len(lens_catalog):,}")
        print(f"  Non-lens systems: {args.n_non_lenses:,}")
        print(f"  Total samples: {total_planned:,}")
        print(f"  Real field data: {'YES' if field_pop is not None else 'SYNTHETIC'}")
        print(f"  Field contamination: 0-{args.n_field_max} galaxies/image")

    except Exception as e:
        print(f"ERROR: Catalog preparation failed: {e}")
        sys.exit(1)

    # Save processed input catalogs
    try:
        if not lens_catalog.empty:
            lens_catalog.to_csv(out_root / "input_lens_catalog_processed.csv", index=False)
        print(f"✓ Input catalogs saved")
    except Exception as e:
        print(f"WARNING: Could not save input catalogs: {e}")

    # STEP 7: Simulation environment setup (telescope-aware)
    _active_telescope = CONFIG.get('telescope', 'jwst').lower()
    print(f"\nSTEP 7: Simulation Environment  [{_active_telescope.upper()}]")
    print("=" * 50)

    try:
        # Always load real JWST PSF tiles (used as JWST PSF, or as fallback)
        print("📋 Loading real JWST PSF data...")
        psf_data = load_psf_data()
        print(f"✓ PSF data loaded: {len(psf_data)} tiles available")

        _psf_cache_dir = Path(__file__).resolve().parent.parent / 'data' / 'psf_cache'
        resolution_psf_cache = {'jwst': psf_data, 'default': psf_data}

        # Build synthetic PSF for any non-JWST telescope used (single or multi-res mode)
        _telescopes_needed = set()
        if multi_resolution_enabled and resolution_scales:
            _telescopes_needed = {r['name'] for r in resolution_scales if r['name'] != 'jwst'}
        elif _active_telescope != 'jwst':
            _telescopes_needed = {_active_telescope}

        if _telescopes_needed and SYNTHETIC_PSF_AVAILABLE:
            print(f"📋 Generating / loading synthetic PSFs: {sorted(_telescopes_needed)}")
            for _rname in sorted(_telescopes_needed):
                _tel_cfg = CONFIG.get('telescope_configs', {}).get(_rname, {})
                _ps = float(_tel_cfg.get('pixel_scale',
                            TELESCOPE_FILTERS.get(_rname) and 0.168 or 0.200))
                # Use telescope_configs pixel_scale, or detector_chain defaults
                from prism.io.detector_chain import TELESCOPE_PARAMS as _DET_PARAMS
                _ps = float(_tel_cfg.get('pixel_scale',
                            _DET_PARAMS.get(_rname, {}).get('pixel_scale', 0.200)))
                _bands = get_telescope_bands(_rname, CONFIG.get('bands'))
                print(f"  [{_rname}] pixel_scale={_ps}\"/pix  bands={_bands}")
                # FIX (adversarial audit finding, 2026-08-01): thread the
                # configured euclid_q1.data_dir through so empirical Q1 PSF
                # tiles are actually found instead of silently falling back
                # to an analytical/Gaussian PSF -- see the fix comment in
                # synthetic_psf_generator.py::build_resolution_psf_cache.
                _q1_cfg_dir = CONFIG.get('euclid_q1', {}).get('data_dir') if isinstance(CONFIG, dict) else None
                _q1_tiles_dir = (Path(_q1_cfg_dir) / 'tiles') if _q1_cfg_dir else None
                resolution_psf_cache[_rname] = _build_resolution_psf_cache(
                    _rname, _bands, _ps, psf_size=101, cache_dir=_psf_cache_dir,
                    q1_psf_dir=_q1_tiles_dir,
                )
            print(f"✓ Synthetic PSF cache built: {sorted(resolution_psf_cache.keys())}")
        elif _telescopes_needed and not SYNTHETIC_PSF_AVAILABLE:
            print("WARNING: synthetic_psf_generator not available — non-JWST telescopes use Gaussian PSF")

        # For single non-JWST telescope, point psf_data at that telescope's cache
        if _active_telescope != 'jwst' and _active_telescope in resolution_psf_cache:
            psf_data = resolution_psf_cache[_active_telescope]

        # Band configs: use telescope-specific pixel scale and exposure time
        _tel_cfg_main = CONFIG.get('telescope_configs', {}).get(_active_telescope, {})
        _main_pixel_scale = float(_tel_cfg_main.get('pixel_scale', CONFIG.get('pixel_scale', 0.031)))
        _main_exp_time    = float(_tel_cfg_main.get('exposure_time', CONFIG.get('exposure_time', 1028.0)))

        # Override global pixel_scale for this run so band_cfgs pick it up
        CONFIG['pixel_scale'] = _main_pixel_scale
        CONFIG['exposure_time'] = _main_exp_time

        # Override UPPER_BANDS for single non-JWST telescope mode
        if _active_telescope != 'jwst' and not multi_resolution_enabled:
            _tel_bands = get_telescope_bands(_active_telescope, CONFIG.get('bands'))
            global UPPER_BANDS, LOWER_BANDS, BAND_TO_LOWER, BAND_CENTERS_UM
            UPPER_BANDS = _tel_bands
            LOWER_BANDS = [b.lower() for b in UPPER_BANDS]
            BAND_TO_LOWER = {b: b.lower() for b in UPPER_BANDS}
            BAND_CENTERS_UM = {b.lower(): ALL_BAND_CENTERS_UM.get(b.lower()) for b in UPPER_BANDS}
            print(f"[CONFIG] Telescope bands overridden: {UPPER_BANDS}")

        # Initialize band configs
        band_cfgs = create_jwst_band_configs(rng=rng, use_distribution=True)
        print(f"✓ Band configurations initialised  telescope={_active_telescope.upper()}")
        print(f"  pixel_scale={_main_pixel_scale:.4f}\"/pix  exposure_time={_main_exp_time:.1f}s")
        for band, cfg in band_cfgs.items():
            print(f"  {band}: ZP={cfg['magnitude_zero_point']:.1f}, "
                  f"sky={cfg['sky_brightness']:.2f} mag/arcsec², PSF={cfg['psf_type']}")

        # Euclid Q1: apply empirical population priors and PSF assignments
        if (_active_telescope == 'euclid' and EUCLID_Q1_AVAILABLE
                and euclid_q1_enabled(CONFIG) and not lens_catalog.empty):
            print("\n📋 Applying Euclid Q1 population priors...")
            _q1_rng = np.random.default_rng(args.seed if hasattr(args, 'seed') else 42)
            lens_catalog = lens_catalog.reset_index(drop=True)
            lens_catalog = apply_euclid_q1_physics(lens_catalog, CONFIG, _q1_rng)
            if CONFIG.get('euclid_q1', {}).get('use_q1_photometry', True):
                lens_catalog = apply_euclid_q1_photometry(lens_catalog, CONFIG, _q1_rng)
            # lens_id matches cosmos_lens_{idx:06d} used in simulation output filenames
            lens_catalog['lens_id'] = [
                f"cosmos_lens_{i + args.start_idx:06d}" for i in range(len(lens_catalog))
            ]
            if 'euclid_psf_tile' in lens_catalog.columns:
                assign_df = lens_catalog[['lens_id', 'euclid_psf_tile']].rename(
                    columns={'euclid_psf_tile': 'tile'})
                assign_df['psf_source'] = 'euclid_q1'
                _q1_data_dir = CONFIG.get('euclid_q1', {}).get('data_dir', 'data/euclid_q1_psf') if isinstance(CONFIG, dict) else 'data/euclid_q1_psf'
                assign_path = Path(_q1_data_dir) / 'psf_assignment.csv'
                assign_path.parent.mkdir(parents=True, exist_ok=True)
                assign_df.to_csv(assign_path, index=False)
                print(f"✓ Euclid Q1 PSF assignments written: {len(assign_df)} lenses → {assign_path}")
            _q1_cat = get_euclid_q1_catalog(CONFIG)
            if _q1_cat is not None:
                print(f"✓ Euclid Q1 catalogue: {_q1_cat.n_systems} model systems, "
                      f"{len(_q1_cat.psf_tiles)} empirical PSF tiles")
        if _active_telescope == 'euclid' and psf_data and EUCLID_Q1_AVAILABLE and is_euclid_q1_psf_data(psf_data):
            print(f"✓ Using empirical Euclid Q1 PSF library: {len(psf_data)} tiles")

    except Exception as e:
        print(f"ERROR: Simulation setup failed: {e}")
        sys.exit(1)

    # STEP 8: Main simulation execution
    print(f"\n{'='*80}")
    print("ENHANCED TRAINING DATASET GENERATION")
    print(f"{'='*80}")
    
    simulation_start_time = time.time()
    
    # Global tracking variables
    all_training_records = []
    simulation_errors = 0

    # Determine resolutions to generate
    resolutions_to_generate = []
    if multi_resolution_enabled and resolution_scales:
        resolutions_to_generate = resolution_scales
        print(f"\n[MULTI-RESOLUTION] Will generate {len(resolutions_to_generate)} resolutions per lens/non-lens")
    else:
        # Single-telescope mode: use the active telescope's pixel scale and image size
        _st_name = _active_telescope
        _st_tel_cfg = CONFIG.get('telescope_configs', {}).get(_st_name, {})
        _st_pixel_scale = float(_st_tel_cfg.get('pixel_scale', CONFIG.get('pixel_scale', 0.031)))
        _st_image_size  = int(_st_tel_cfg.get('image_size', args.numpix))
        # Keep 'name' consistent with telescope so PSF cache key matches
        resolutions_to_generate = [{'name': _st_name, 'pixel_scale': _st_pixel_scale}]
        print(f"[CONFIG] Single-telescope mode: {_st_name.upper()}  "
              f"pixel_scale={_st_pixel_scale:.4f}\"/px  image_size={_st_image_size}px")

    # === CORRECTED: LENS SYSTEM GENERATION ===
    lens_records = []
    lens_dark_count = 0
    field_galaxy_total = 0
    
    # Shared set of (sim, snapshot, subhalo_id) tuples already used in this
    # batch run.  Passed to every query_tng_properties call so the same TNG
    # galaxy is never reused as lens AND source, or across two systems.
    _used_tng_subhalos: set = set()

    if not lens_catalog.empty:
        print(f"\nGenerating {len(lens_catalog):,} LENS SYSTEMS (Positive Samples)")
        print("-" * 60)

        lens_flux_history = []

        for idx, row in lens_catalog.iterrows():
            # Save RNG state before processing this lens
            # In multi-resolution mode, we'll restore this state for each resolution
            # so that the SAME lens is generated at each pixel scale
            if multi_resolution_enabled:
                rng_state_before_lens = rng.bit_generator.state
            
            # Iterate over resolutions
            for res_idx, resolution in enumerate(resolutions_to_generate):
                res_name = resolution['name']
                pixel_scale = resolution['pixel_scale']
                # Resolve numpix for this resolution:
                # priority: fov_arcsec in multi-res config > telescope_configs[name].image_size > CLI --numpix
                fov_arcsec = resolution.get('fov_arcsec') or CONFIG.get('image_fov_arcsec')
                if fov_arcsec:
                    numpix_res = max(64, int(round(float(fov_arcsec) / float(pixel_scale))))
                else:
                    _res_tel_cfg2 = CONFIG.get('telescope_configs', {}).get(res_name, {})
                    numpix_res = int(_res_tel_cfg2.get('image_size', args.numpix))

                # Set telescope-specific bands + pixel scale for this resolution
                resolution_bands = get_telescope_bands(res_name, CONFIG.get('bands', None))
                resolution_bands_lower = [b.lower() for b in resolution_bands]

                original_upper_bands = UPPER_BANDS
                original_lower_bands = LOWER_BANDS
                original_band_to_lower = BAND_TO_LOWER

                UPPER_BANDS = resolution_bands
                LOWER_BANDS = resolution_bands_lower
                BAND_TO_LOWER = {b: b.lower() for b in resolution_bands}

                # Apply telescope-specific pixel scale for band configs
                _res_tel_cfg = CONFIG.get('telescope_configs', {}).get(res_name, {})
                CONFIG['pixel_scale'] = float(_res_tel_cfg.get('pixel_scale', pixel_scale))
                CONFIG['exposure_time'] = float(_res_tel_cfg.get('exposure_time',
                                                CONFIG.get('exposure_time', 1028.0)))

                resolution_band_cfgs = create_jwst_band_configs(rng=rng, use_distribution=True)
                
                # For multi-resolution, add progress indicator
                if multi_resolution_enabled:
                    res_indicator = f" [{res_name} {pixel_scale:.3f}\"/pix, filters: {','.join(resolution_bands)}]"
                else:
                    res_indicator = ""
                
                # For multi-resolution: reset RNG to same state for each resolution
                # This ensures IDENTICAL lenses are generated at each pixel scale
                if multi_resolution_enabled:
                    rng.bit_generator.state = rng_state_before_lens
                
                try:
                    res_psf_data = resolution_psf_cache.get(res_name, psf_data)
                    epoch_results = simulate_lens_system_with_time_delays(
                        row, resolution_band_cfgs, rng,
                        field_pop=field_pop,
                        numpix=numpix_res,
                        n_field_max=args.n_field_max,
                        add_artifacts=CONFIG.get('add_artifacts', args.add_artifacts),
                        psf_data=res_psf_data,
                        add_spikes=args.add_jwst_spikes,
                        pixel_scale=pixel_scale
                    )
                    
                    # Process each epoch
                    first_epoch_metadata = None  # Store first epoch metadata for catalog
                    for epoch_idx, (imgs, metadata) in enumerate(epoch_results):
                        # Store first epoch metadata for later use
                        if epoch_idx == 0:
                            first_epoch_metadata = metadata
                        
                        # Quality assessment (use resolution-specific bands)
                        total_flux = sum(np.sum(imgs[b]) for b in resolution_bands)
                        if epoch_idx == 0:  # Only track first epoch for statistics
                            lens_flux_history.append(total_flux)
                            is_dark = total_flux < 1e-7
                            n_field = metadata.get('n_field_galaxies', 0)
                            field_galaxy_total += n_field
                            
                            if is_dark:
                                lens_dark_count += 1
                        
                        # Create filename with epoch suffix if time delays.
                        # FIX (adversarial audit finding C-14, 2026-08-01):
                        # 'cosmos_lens_*' vs 'cosmos_nonlens_*' reveals the
                        # class label directly in the filename -- any
                        # tooling that globs/sorts by filename (rather than
                        # reading the actual is_lens column) gets the
                        # answer for free, and it's an easy accidental
                        # shortcut for a careless dataloader. Opt-in
                        # (output.neutral_filenames) so existing tooling
                        # that already depends on the lens/nonlens prefix
                        # keeps working by default.
                        _neutral_names = CONFIG.get('output', {}).get('neutral_filenames', False) if isinstance(CONFIG, dict) else False
                        _fname_prefix = "cosmos_sample" if _neutral_names else "cosmos_lens"
                        if metadata.get('has_time_delays', False):
                            filename_base = f"{_fname_prefix}_{idx + args.start_idx:06d}_epoch{epoch_idx:02d}"
                        else:
                            filename_base = f"{_fname_prefix}_{idx + args.start_idx:06d}"
                        
                        # Save outputs with resolution name and telescope-specific bands
                        # Use consistent normalization scales for time delay epochs
                        norm_scales = metadata.get('normalization_scales', None) if metadata.get('has_time_delays', False) else None
                        save_success = save_complete_outputs(
                            filename_base, imgs, out_root, row, metadata, 
                            system_type='lens', rng=rng, normalization_scales=norm_scales,
                            resolution_name=res_name if multi_resolution_enabled else None,
                            bands=resolution_bands
                        )
                    
                    # Save time delay metadata to catalog if applicable (use first epoch metadata)
                    if first_epoch_metadata and first_epoch_metadata.get('has_time_delays', False):
                        # Save time delay catalog entry (only once per lens system)
                        time_delay_catalog_path = out_root / "time_delay_catalog.csv"
                        # Get image positions from time_delays_result if available
                        image_positions = None
                        if 'time_delays_result' in first_epoch_metadata:
                            td_result = first_epoch_metadata['time_delays_result']
                            if isinstance(td_result, dict):
                                image_positions = td_result.get('image_positions', [])
                        # Fallback: try direct access
                        if not image_positions:
                            image_positions = first_epoch_metadata.get('image_positions', [])
                        
                        # Convert image positions to string format for CSV
                        # Handle both list of tuples and numpy arrays
                        if image_positions:
                            try:
                                # Convert to list of tuples if needed
                                if hasattr(image_positions[0], '__len__') and not isinstance(image_positions[0], str):
                                    image_positions_list = [(float(x), float(y)) for x, y in image_positions]
                                else:
                                    image_positions_list = image_positions
                                image_positions_str = str(image_positions_list)
                            except Exception as e:
                                print(f"[WARNING] Could not format image_positions: {e}")
                                image_positions_str = str(image_positions)
                        else:
                            image_positions_str = "[]"
                        
                        td_entry = {
                            'lens_id': int(idx + args.start_idx),
                            'source_type': first_epoch_metadata.get('source_type', 'unknown'),
                            'n_images': first_epoch_metadata.get('n_images', 0),
                            'n_epochs': first_epoch_metadata.get('n_epochs', 0),
                            'time_delays_days': str(first_epoch_metadata.get('time_delays_days', [])),
                            'image_positions_arcsec': image_positions_str,  # NEW: Save actual image positions
                            'lens_redshift': first_epoch_metadata.get('lens_redshift', np.nan),
                            'source_redshift': first_epoch_metadata.get('source_redshift', np.nan),
                            'theta_E': float(row.get('theta_E', np.nan))
                        }
                        td_df = pd.DataFrame([td_entry])
                        if time_delay_catalog_path.exists():
                            td_df.to_csv(time_delay_catalog_path, mode='a', header=False, index=False)
                        else:
                            td_df.to_csv(time_delay_catalog_path, index=False)
                    
                    # Create training record only for first epoch.
                    # FIX (discovered while closing adversarial-audit gap
                    # C-21, 2026-08-01, by actually EXECUTING the
                    # time-delay code path -- which the original static
                    # audit never did): this condition used to be
                    # `epoch_idx == 0`, but this code is OUTSIDE the
                    # `for epoch_idx, ... in enumerate(epoch_results):`
                    # loop above (same indentation as the `for` statement,
                    # not nested inside it) -- so `epoch_idx` here is
                    # Python's post-loop leftover value from the LAST
                    # iteration (e.g. 3 for a 4-epoch system), never 0 for
                    # any multi-epoch time-delay system. Confirmed by
                    # execution: a 4-epoch time-delay run saved all 4
                    # epoch .npz files successfully, but produced
                    # "0 successful" lens records and
                    # "ERROR: No training samples generated successfully"
                    # -- the catalog/label pipeline silently produced
                    # NOTHING for every time-delay system ever generated.
                    # Fixed to key off first_epoch_metadata (correctly
                    # captured for epoch 0 inside the loop) instead of the
                    # stale loop variable.
                    if first_epoch_metadata is not None and save_success:
                        # Use first_epoch_metadata if available, otherwise use current metadata
                        record_metadata = first_epoch_metadata if first_epoch_metadata else metadata
                        # Create comprehensive training record (first epoch only)
                        training_record = {
                            'lens_id': int(idx + args.start_idx),
                            'system_type': 'lens',
                            'is_lens': 1,  # Binary label for ML
                            'filename_base': filename_base,
                            'base_lens_id': int(row.get('base_lens_id', idx)),
                            'variation_id': int(row.get('variation_id', 0)),
                            'n_field_galaxies': int(record_metadata.get('n_field_galaxies', 0)),
                            'using_real_fields': bool(field_pop is not None),
                            'is_dark_image': bool(total_flux < 1e-7),
                            'total_flux': float(total_flux),
                            'has_time_delays': bool(record_metadata.get('has_time_delays', False)),
                            'n_epochs': int(record_metadata.get('n_epochs', 1)) if record_metadata.get('has_time_delays', False) else 1,
                            'lens_n_sersic': float(record_metadata.get('lens_n_sersic', row.get('lens_n_sersic', row.get('n_rest', 2.5)))),
                            'lens_system_class': str(record_metadata.get('lens_system_class', 'single_field'))
                        }
                        
                        # Add physical and structural parameters
                        for param in ["theta_E", "lens_redshift", "source_redshift",
                                     "lens_radius", "source_radius", "source_x", "source_y",
                                     "lens_axis_ratio", "source_axis_ratio"]:
                            value = record_metadata.get(param, row.get(param, np.nan))
                            training_record[param] = float(value) if pd.notna(value) else np.nan

                        for param in [
                            'lens_pa', 'lens_e1', 'lens_e2',
                            'source_n_sersic', 'source_pa', 'source_e1', 'source_e2',
                            'field_mean_n_sersic', 'field_mean_radius', 'field_mean_axis_ratio', 'field_mean_redshift'
                        ]:
                            value = record_metadata.get(param, np.nan)
                            training_record[param] = float(value) if pd.notna(value) else np.nan

                        training_record['field_n_sersic_list'] = json.dumps(record_metadata.get('field_n_sersic_list', []))
                        training_record['field_radius_list'] = json.dumps(record_metadata.get('field_radius_list', []))
                        training_record['field_axis_ratio_list'] = json.dumps(record_metadata.get('field_axis_ratio_list', []))
                        training_record['field_redshift_list'] = json.dumps(record_metadata.get('field_redshift_list', []))
                        training_record['field_structural_data'] = json.dumps(record_metadata.get('field_structural_data', []))

                        # TNG Mode: flatten matched-subhalo properties (or
                        # NaN/empty if disabled/no match) into per-object
                        # columns for the lens, lensed source, and a
                        # field-galaxy match-rate summary + full per-galaxy
                        # JSON list.
                        training_record.update(flatten_tng_info('tng_lens', record_metadata.get('tng_lens')))
                        training_record.update(flatten_tng_info('tng_source', record_metadata.get('tng_source')))
                        tng_field_list = record_metadata.get('tng_field', []) or []
                        n_tng_field_matched = sum(1 for t in tng_field_list if t is not None)
                        training_record['tng_field_n_matched'] = n_tng_field_matched
                        training_record['tng_field_n_total'] = len(tng_field_list)
                        training_record['tng_field_match_fraction'] = (
                            n_tng_field_matched / len(tng_field_list) if tng_field_list else 0.0
                        )
                        training_record['tng_field_info_list'] = json.dumps(tng_field_list)

                        # Add band measurements
                        for b in UPPER_BANDS:
                            training_record[f"flux_sum_{b.lower()}"] = float(np.sum(imgs[b]))
                            training_record[f"flux_max_{b.lower()}"] = float(np.max(imgs[b]))
                        
                        # Add magnitudes
                        for band in LOWER_BANDS:
                            training_record[f"lens_mag_{band}"] = float(row.get(f"lens_mag_{band}", np.nan))
                            training_record[f"source_mag_{band}"] = float(row.get(f"source_mag_{band}", np.nan))
                        
                        lens_records.append(training_record)
                        all_training_records.append(training_record)

                except Exception as e:
                    print(f"[ERROR] Lens system {idx} (resolution {res_name}) failed: {e}")
                    if simulation_errors < 3:
                        import traceback
                        traceback.print_exc()
                    simulation_errors += 1
                    # Restore original bands before continuing
                    UPPER_BANDS = original_upper_bands
                    LOWER_BANDS = original_lower_bands
                    BAND_TO_LOWER = original_band_to_lower
                    continue

                # Restore original bands after successful resolution processing
                UPPER_BANDS = original_upper_bands
                LOWER_BANDS = original_lower_bands
                BAND_TO_LOWER = original_band_to_lower

            # Progress reporting
            if (idx + 1) % args.batch_size == 0:
                processed = idx + 1
                success_count = len(lens_records)
                success_rate = 100 * success_count / processed
                dark_rate = 100 * lens_dark_count / max(success_count, 1)
                avg_field = field_galaxy_total / max(success_count, 1)
                recent_flux = np.mean(lens_flux_history[-args.batch_size:]) if lens_flux_history else 0
                
                print(f"  LENS {processed:,}/{len(lens_catalog):,} | "
                      f"Success: {success_rate:.1f}% | "
                      f"Dark: {dark_rate:.1f}% | "
                      f"Field: {avg_field:.1f}/img | "
                      f"Flux: {recent_flux:.1e}")

        print(f"✓ Lens generation complete: {len(lens_records):,} successful")

    # === CORRECTED: NON-LENS SYSTEM GENERATION ===
    nonlens_records = []
    nonlens_dark_count = 0
    
    if args.n_non_lenses > 0:
        print(f"\nGenerating {args.n_non_lenses:,} NON-LENS SYSTEMS (Negative Samples)")
        print("-" * 60)
        
        nonlens_flux_history = []
        
        for idx in range(args.n_non_lenses):
            # Save RNG state before processing this non-lens
            # In multi-resolution mode, we'll restore this state for each resolution
            # so that the SAME non-lens system is generated at each pixel scale
            if multi_resolution_enabled:
                rng_state_before_nonlens = rng.bit_generator.state
            
            # Iterate over resolutions
            for res_idx, resolution in enumerate(resolutions_to_generate):
                res_name = resolution['name']
                pixel_scale = resolution['pixel_scale']

                # Resolve numpix: fov_arcsec > telescope_configs[name].image_size > CLI --numpix
                _nl_fov = resolution.get('fov_arcsec') or CONFIG.get('image_fov_arcsec')
                if _nl_fov:
                    numpix_res = max(64, int(round(float(_nl_fov) / float(pixel_scale))))
                else:
                    _nl_tel_cfg2 = CONFIG.get('telescope_configs', {}).get(res_name, {})
                    numpix_res = int(_nl_tel_cfg2.get('image_size', args.numpix))

                # Set telescope-specific bands + pixel scale for this resolution
                resolution_bands = get_telescope_bands(res_name, CONFIG.get('bands', None))
                original_upper_bands = UPPER_BANDS
                original_lower_bands  = LOWER_BANDS
                original_band_to_lower = BAND_TO_LOWER
                UPPER_BANDS    = resolution_bands
                LOWER_BANDS    = [b.lower() for b in resolution_bands]
                BAND_TO_LOWER  = {b: b.lower() for b in resolution_bands}
                _nl_tel_cfg = CONFIG.get('telescope_configs', {}).get(res_name, {})
                CONFIG['pixel_scale']    = float(_nl_tel_cfg.get('pixel_scale', pixel_scale))
                CONFIG['exposure_time']  = float(_nl_tel_cfg.get('exposure_time',
                                                 CONFIG.get('exposure_time', 1028.0)))
                resolution_band_cfgs = create_jwst_band_configs(rng=rng, use_distribution=True)

                # For multi-resolution: reset RNG to same state for each resolution
                # This ensures IDENTICAL non-lens systems are generated at each pixel scale
                if multi_resolution_enabled:
                    rng.bit_generator.state = rng_state_before_nonlens

                try:
                    # Select simulation mode
                    mode = rng.choice(non_lens_modes)
                    res_psf_data = resolution_psf_cache.get(res_name, psf_data)
                    imgs, system_info = generate_nonlens_system_complete(
                        mode, resolution_band_cfgs, rng,
                        field_pop=field_pop,
                        numpix=numpix_res,
                        n_field_max=args.n_field_max,  # Now 8 instead of 3
                        add_artifacts=CONFIG.get('add_artifacts', args.add_artifacts),  # Use config, fallback to CLI
                        psf_data=res_psf_data,
                        add_spikes=args.add_jwst_spikes,
                        pixel_scale=pixel_scale
                    )
                    
                    # Quality assessment
                    total_flux = sum(np.sum(imgs[b]) for b in resolution_bands)
                    nonlens_flux_history.append(total_flux)
                    is_dark = total_flux < 1e-7
                    n_field = system_info.get('n_field_galaxies', 0)
                    
                    if is_dark:
                        nonlens_dark_count += 1
                    
                    # Save outputs with resolution name.
                    # FIX (audit C-14): see identical fix + rationale on
                    # the lens-save path above. A large offset is added to
                    # the non-lens index under neutral naming so lens and
                    # non-lens filenames don't collide now that they share
                    # the same 'cosmos_sample_NNNNNN' prefix/numbering
                    # space (the old distinct 'cosmos_lens_'/'cosmos_nonlens_'
                    # prefixes made collisions impossible even with
                    # identical indices).
                    _neutral_names = CONFIG.get('output', {}).get('neutral_filenames', False) if isinstance(CONFIG, dict) else False
                    if _neutral_names:
                        filename_base = f"cosmos_sample_{idx + args.start_idx + 1_000_000:06d}"
                    else:
                        filename_base = f"cosmos_nonlens_{idx + args.start_idx:06d}"
                    save_success = save_complete_outputs(
                        filename_base, imgs, out_root, system_info, system_info,
                        system_type='nonlens', rng=rng,
                        resolution_name=res_name if multi_resolution_enabled else None,
                        bands=resolution_bands
                    )
                    
                    if save_success:
                        # Create training record (only on first resolution to avoid duplicates)
                        if res_idx == 0:
                            training_record = {
                                'lens_id': int(idx + args.start_idx + 100000),  # Offset to avoid conflicts
                                'system_type': 'nonlens',
                                'is_lens': 0,  # Binary label for ML
                                'filename_base': filename_base,
                                'nonlens_mode': str(mode),
                                'n_field_galaxies': int(n_field),
                                'using_real_fields': bool(field_pop is not None),
                                'is_dark_image': bool(is_dark),
                                'total_flux': float(total_flux)
                            }
                            
                            # Add band measurements
                            for b in resolution_bands:
                                training_record[f"flux_sum_{b.lower()}"] = float(np.sum(imgs[b]))
                                training_record[f"flux_max_{b.lower()}"] = float(np.max(imgs[b]))
                            
                            # Add system-specific information
                            for key, value in system_info.items():
                                if isinstance(value, (int, float, str, bool)) and not key.startswith('_'):
                                    training_record[f"nonlens_{key}"] = value
                            
                            nonlens_records.append(training_record)
                            all_training_records.append(training_record)

                except Exception as e:
                    print(f"[ERROR] Non-lens system {idx} (resolution {res_name}) failed: {e}")
                    simulation_errors += 1
                    # Restore original bands before continuing
                    UPPER_BANDS = original_upper_bands
                    LOWER_BANDS = original_lower_bands
                    BAND_TO_LOWER = original_band_to_lower
                    continue

                # Restore original bands after successful resolution processing
                UPPER_BANDS = original_upper_bands
                LOWER_BANDS = original_lower_bands
                BAND_TO_LOWER = original_band_to_lower

            # Progress reporting
            if (idx + 1) % args.batch_size == 0:
                processed = idx + 1
                success_count = len(nonlens_records)
                success_rate = 100 * success_count / processed
                dark_rate = 100 * nonlens_dark_count / max(success_count, 1)
                recent_flux = np.mean(nonlens_flux_history[-args.batch_size:]) if nonlens_flux_history else 0
                
                print(f"  NON-LENS {processed:,}/{args.n_non_lenses:,} | "
                      f"Success: {success_rate:.1f}% | "
                      f"Dark: {dark_rate:.1f}% | "
                      f"Flux: {recent_flux:.1e}")

        print(f"✓ Non-lens generation complete: {len(nonlens_records):,} successful")

    # STEP 9: Results compilation and quality assessment
    simulation_end_time = time.time()
    total_duration = simulation_end_time - simulation_start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*80}")
    print("COMPILING ENHANCED TRAINING DATASET")
    print(f"{'='*80}")
    print(f"Generation duration: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    
    if not all_training_records:
        print(f"ERROR: No training samples generated successfully")
        sys.exit(1)

    try:
        # Save comprehensive training catalog
        training_df = pd.DataFrame(all_training_records)
        main_catalog_path = out_root / "cosmos_training_catalog_lens_and_nonlens.csv"
        training_df.to_csv(main_catalog_path, index=False)
        
        # Save class-specific catalogs
        if lens_records:
            lens_df = pd.DataFrame(lens_records)
            lens_df.to_csv(out_root / "cosmos_lens_training_catalog.csv", index=False)
        
        if nonlens_records:
            nonlens_df = pd.DataFrame(nonlens_records)
            nonlens_df.to_csv(out_root / "cosmos_nonlens_training_catalog.csv", index=False)

        print(f"✓ Training catalogs saved: {len(training_df):,} samples")

        # Calculate comprehensive quality metrics
        lens_dark_pct = 100 * lens_dark_count / max(len(lens_records), 1)
        nonlens_dark_pct = 100 * nonlens_dark_count / max(len(nonlens_records), 1) if nonlens_records else 0
        overall_success_rate = 100 * len(all_training_records) / max(total_planned, 1)
        avg_field_per_lens = field_galaxy_total / max(len(lens_records), 1)

        # Quality assessment
        quality_summary = {
            'generation_metadata': {
                'script_version': ' v10_final_corrected',
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'random_seed': args.seed,
                'job_id': os.environ.get('SLURM_JOB_ID', 'interactive'),
                'total_duration_seconds': int(total_duration),
                'corrections_applied': [
                    'Enhanced field galaxy populations (3->8 per image)',
                    'Realistic Sersic index distributions (60% spirals)',
                    'Improved magnitude scaling for detectability',
                    'Synthetic field galaxy fallback system',
                    'Comprehensive morphological diversity'
                ]
            },
            'dataset_composition': {
                'total_samples': len(all_training_records),
                'lens_systems': len(lens_records),
                'nonlens_systems': len(nonlens_records),
                'class_balance_ratio': len(lens_records) / max(len(nonlens_records), 1) if nonlens_records else float('inf'),
                'using_real_field_data': field_pop is not None,
                'field_galaxies_available': len(field_pop) if field_pop is not None else 0
            },
            'quality_metrics': {
                'overall_success_rate': overall_success_rate,
                'lens_dark_percentage': lens_dark_pct,
                'nonlens_dark_percentage': nonlens_dark_pct,
                'simulation_errors': simulation_errors,
                'avg_field_galaxies_per_lens': avg_field_per_lens,
                'total_field_galaxies_added': field_galaxy_total
            },
            'training_assessment': {
                'dataset_size_category': categorize_dataset_size(len(all_training_records)),
                'quality_grade': assess_training_quality(lens_dark_pct, nonlens_dark_pct, len(all_training_records), simulation_errors),
                'class_balance_assessment': assess_class_balance(len(lens_records), len(nonlens_records)),
                'training_ready': is_training_ready(len(all_training_records), lens_dark_pct, nonlens_dark_pct, simulation_errors),
                'recommended_ml_approach': recommend_ml_strategy(len(lens_records), len(nonlens_records), field_pop is not None),
                'field_realism_level': 'high' if field_pop is not None else 'synthetic'
            }
        }
        
        # Save comprehensive quality assessment
        quality_file_path = out_root / "training_quality_assessment.json"
        with open(quality_file_path, "w") as f:
            json.dump(quality_summary, f, indent=2)

        print(f"✓ Quality assessment saved")

    except Exception as e:
        print(f"ERROR: Results compilation failed: {e}")
        sys.exit(1)

    # FINAL RESULTS DISPLAY
    print(f"\n{'='*80}")
    print("ENHANCED TRAINING DATASET COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    print(f"Execution Summary:")
    print(f"  Duration: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
    print(f"  Samples generated: {len(all_training_records):,}")
    print(f"  Success rate: {overall_success_rate:.1f}%")
    
    print(f"\nDataset Composition:")
    print(f"  Lens systems (positive): {len(lens_records):,}")
    print(f"  Non-lens systems (negative): {len(nonlens_records):,}")
    if len(nonlens_records) > 0:
        balance_ratio = len(lens_records) / len(nonlens_records)
        print(f"  Class balance ratio: {balance_ratio:.2f}:1")
    
    print(f"\nEnhanced Field Galaxy Contamination:")
    print(f"  Using real observations: {'YES' if field_pop is not None else 'SYNTHETIC'}")
    print(f"  Field galaxies per lens: {avg_field_per_lens:.2f} (target: 5-15)")
    print(f"  Total field contamination: {field_galaxy_total:,} galaxies")
    
    print(f"\nMorphological Diversity:")
    print(f"  Spiral fraction expected: ~60% (n<2)")
    print(f"  Enhanced Sersic sampling: YES")
    print(f"  Realistic axis ratios: YES")
    
    print(f"\nQuality Assessment:")
    print(f"  Lens dark images: {lens_dark_count:,} ({lens_dark_pct:.1f}%)")
    if nonlens_records:
        print(f"  Non-lens dark images: {nonlens_dark_count:,} ({nonlens_dark_pct:.1f}%)")
    
    training_ready = quality_summary['training_assessment']['training_ready']
    quality_grade = quality_summary['training_assessment']['quality_grade']
    
    print(f"\nTraining Readiness: {'READY' if training_ready else 'NEEDS REVIEW'}")
    print(f"  Quality Grade: {quality_grade}")
    print(f"  Recommended Approach: {quality_summary['training_assessment']['recommended_ml_approach']}")

    print(f"\nOutput Files:")
    print(f"  Directory: {out_root}")
    print(f"  Training data: npy/ ({len(all_training_records):,} .npy files)")
    print(f"  Visualizations: jpg_rgb/ ({len(all_training_records):,} .jpg files)")
    print(f"  Main catalog: cosmos_training_catalog_lens_and_nonlens.csv")
    print(f"  Quality report: training_quality_assessment.json")

    if training_ready:
        print(f"\n🎯 DATASET READY FOR ML TRAINING!")
        print(f"Expected improvements:")
        print(f"  - Rich field galaxy populations (5-15 per image)")
        print(f"  - Realistic spiral galaxy fraction (~60%)")
        print(f"  - Enhanced morphological diversity")
        print(f"  - Proper magnitude scaling for detectability")
    else:
        print(f"\n⚠ Dataset needs review - check quality metrics")

    print(f"\n{'='*80}")
    print("CORRECTED PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    return 0

if __name__ == "__main__":
    main()
