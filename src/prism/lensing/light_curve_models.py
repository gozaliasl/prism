#!/usr/bin/env python3
"""
Enhanced light curve models using theoretical packages.

This module provides physically motivated light curve generation for:
- Quasars/AGN: Using celerite2 for DRW (Damped Random Walk) model
- Supernovae: Using sncosmo for realistic SN templates

Falls back to simplified models if packages are not available.
"""

import numpy as np
from typing import Dict, Optional
import warnings

# Try to import celerite2 for quasar/AGN DRW modeling
try:
    from celerite2 import terms, GaussianProcess
    CELERITE2_AVAILABLE = True
except ImportError:
    CELERITE2_AVAILABLE = False
    warnings.warn("celerite2 not available. Using simplified quasar/AGN light curves.")

# Try to import sncosmo for supernova templates
try:
    import sncosmo
    SNCOSMO_AVAILABLE = True
except ImportError:
    SNCOSMO_AVAILABLE = False
    warnings.warn("sncosmo not available. Using simplified supernova light curves.")


def generate_quasar_light_curve_drw(
    time_array: np.ndarray,
    base_magnitude: float,
    black_hole_mass: Optional[float] = None,  # log10(M_BH/M_sun)
    redshift: float = 2.0,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate quasar light curve using DRW (Damped Random Walk) model.
    
    This is the theoretically consistent model for quasar variability
    based on Kelly et al. (2009) and MacLeod et al. (2010).
    
    Parameters:
    -----------
    time_array : np.ndarray
        Time array in days
    base_magnitude : float
        Base magnitude of the quasar
    black_hole_mass : float, optional
        Black hole mass in log10(M_BH/M_sun). If None, uses typical value ~8.5
    redshift : float
        Source redshift (affects timescale)
    rng : np.random.Generator, optional
        Random number generator
    config : dict, optional
        Configuration dictionary with variability parameters
    
    Returns:
    --------
    np.ndarray
        Magnitude array as function of time
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if not CELERITE2_AVAILABLE:
        # Fallback to simplified model
        return _generate_quasar_light_curve_simple(time_array, base_magnitude, rng, config)
    
    # DRW parameters from MacLeod et al. (2010)
    # Timescale: τ ≈ 100 * (M_BH/10^8)^0.5 * (1+z)^0.5 days
    if black_hole_mass is None:
        # Typical quasar: log10(M_BH/M_sun) ≈ 8.5
        black_hole_mass = 8.5
    
    M_BH_8 = 10**(black_hole_mass - 8)  # Mass in units of 10^8 M_sun
    tau_days = 100 * np.sqrt(M_BH_8) * np.sqrt(1 + redshift)
    
    # Amplitude: σ ≈ 0.2 mag (typical for quasars, from observations)
    # Can be overridden by config
    if config:
        source_cfg = config.get('time_delays', {}).get('source_types', {}).get('quasar', {})
        sigma = source_cfg.get('variability_amplitude', 0.2)
    else:
        sigma = 0.2
    
    # Ensure timescale is reasonable (not too short)
    tau_days = max(tau_days, 10.0)  # Minimum 10 days
    
    # Create DRW kernel: S(ω) = S₀ / (1 + (ωτ)²)
    # For celerite2 RealTerm: a = S₀, c = 1/τ
    # Use log parameters: log_a = log(S₀), log_c = -log(τ)
    # Note: celerite2 RealTerm uses 'a' and 'c' directly, or 'log_sigma' and 'log_rho'
    try:
        # Try new API first (celerite2 >= 0.3.0)
        kernel = terms.RealTerm(
            log_sigma=np.log(sigma),
            log_rho=-np.log(tau_days)
        )
    except TypeError:
        # Fall back to old API (celerite2 < 0.3.0)
        kernel = terms.RealTerm(
            a=sigma**2,
            c=1.0/tau_days
        )
    
    # Generate GP
    gp = GaussianProcess(kernel, mean=0.0)
    gp.compute(time_array)
    
    # Sample light curve
    # Note: celerite2 0.3.2 sample() doesn't accept random_state/rng
    # It uses numpy's global random state, so we set the seed temporarily
    if rng is not None:
        # Get seed from rng if possible, otherwise use default
        try:
            seed = rng.integers(0, 2**31)
        except:
            seed = 42
        np.random.seed(seed)
    
    magnitude_variation = gp.sample()
    magnitude = base_magnitude + magnitude_variation
    
    return magnitude


def generate_agn_light_curve_drw(
    time_array: np.ndarray,
    base_magnitude: float,
    black_hole_mass: Optional[float] = None,
    redshift: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate AGN light curve using DRW model.
    
    Similar to quasar but typically with:
    - Lower mass black holes (log10(M_BH/M_sun) ≈ 7-8)
    - Shorter timescales
    - Similar variability amplitude
    
    Parameters:
    -----------
    time_array : np.ndarray
        Time array in days
    base_magnitude : float
        Base magnitude
    black_hole_mass : float, optional
        Black hole mass in log10(M_BH/M_sun). Default ~7.5 for AGN
    redshift : float
        Source redshift
    rng : np.random.Generator, optional
        Random number generator
    config : dict, optional
        Configuration dictionary
    
    Returns:
    --------
    np.ndarray
        Magnitude array
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if not CELERITE2_AVAILABLE:
        # Fallback to simplified model
        return _generate_agn_light_curve_simple(time_array, base_magnitude, rng, config)
    
    # AGN typically have lower mass BHs
    if black_hole_mass is None:
        black_hole_mass = 7.5
    
    # Use same DRW model as quasar
    return generate_quasar_light_curve_drw(
        time_array, base_magnitude, black_hole_mass, redshift, rng, config
    )


def generate_supernova_light_curve(
    time_array: np.ndarray,
    base_magnitude: float,
    sn_type: str = 'Ia',
    redshift: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate supernova light curve using sncosmo templates.
    
    Parameters:
    -----------
    time_array : np.ndarray
        Time array in days
    base_magnitude : float
        Peak magnitude (will be normalized)
    sn_type : str
        Type of supernova: 'Ia', 'Ib', 'Ic', 'II-P', 'II-L'
    redshift : float
        Source redshift
    rng : np.random.Generator, optional
        Random number generator
    config : dict, optional
        Configuration dictionary
    
    Returns:
    --------
    np.ndarray
        Magnitude array
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if not SNCOSMO_AVAILABLE:
        # Fallback to simplified model
        return _generate_supernova_light_curve_simple(time_array, base_magnitude, rng, config)
    
    # Load appropriate template
    template_map = {
        'Ia': 'salt2',
        'Ib': 'hsiao',
        'Ic': 'hsiao',
        'II-P': 'nugent-sn2p',
        'II-L': 'nugent-sn2l'
    }
    
    template_name = template_map.get(sn_type, 'salt2')
    
    try:
        model = sncosmo.Model(source=template_name)
    except Exception:
        # Fallback to SALT2 if template not available
        model = sncosmo.Model(source='salt2')
    
    # Set redshift
    model.set(z=redshift)
    
    # Set peak time (at middle of time array)
    t0 = time_array[len(time_array) // 2]
    model.set(t0=t0)
    
    # Generate light curve in observer frame
    # At high redshift, use redder bandpasses that match the redshifted spectrum
    # Try multiple bandpasses in order of preference
    bandpasses_to_try = ['bessellb', 'sdssr', 'sdssi', 'bessellr', 'bessellv']
    magnitude = None
    
    for bp_name in bandpasses_to_try:
        try:
            flux = model.bandflux(bp_name, time_array, zp=25.0, zpsys='ab')
            if not np.all(np.isfinite(flux)) or np.any(flux <= 0):
                continue  # Non-physical flux (e.g. bandpass entirely outside template range)
            magnitude = -2.5 * np.log10(flux) + 25.0
            if not np.all(np.isfinite(magnitude)):
                continue
            break  # Success, use this bandpass
        except Exception:
            continue  # Try next bandpass
    
    if magnitude is None:
        # If all bandpasses fail (e.g., at very high redshift where template
        # is redshifted to IR but no IR bandpasses available), use simplified approach
        # This happens when z > ~3.5 where rest-frame optical is redshifted to IR
        return _generate_supernova_light_curve_simple(time_array, base_magnitude, rng, config)
    
    # Normalize to base magnitude (peak magnitude)
    magnitude = magnitude - np.min(magnitude) + base_magnitude
    
    # Note: SALT2 templates are inherently smooth because they're parameterized models
    # designed for fitting, not high-frequency variations. This is physically realistic
    # for SN Ia light curves, which are smooth functions of time. Real observations
    # would have measurement noise, but the underlying light curve is smooth.
    
    return magnitude


# Fallback simplified models
def _generate_quasar_light_curve_simple(
    time_array: np.ndarray,
    base_magnitude: float,
    rng: np.random.Generator,
    config: Optional[Dict] = None
) -> np.ndarray:
    """Simplified quasar light curve (fallback)."""
    if config:
        source_cfg = config.get('time_delays', {}).get('source_types', {}).get('quasar', {})
        amplitude = source_cfg.get('variability_amplitude', 0.8)
        timescale = source_cfg.get('variability_timescale', 50.0)
    else:
        amplitude = 0.8
        timescale = 50.0
    
    phase = rng.uniform(0, 2 * np.pi)
    magnitude = base_magnitude + amplitude * np.sin(
        2 * np.pi * time_array / timescale + phase
    )
    magnitude += rng.normal(0, 0.1, len(time_array))
    return magnitude


def _generate_agn_light_curve_simple(
    time_array: np.ndarray,
    base_magnitude: float,
    rng: np.random.Generator,
    config: Optional[Dict] = None
) -> np.ndarray:
    """Simplified AGN light curve (fallback)."""
    if config:
        source_cfg = config.get('time_delays', {}).get('source_types', {}).get('agn', {})
        amplitude = source_cfg.get('variability_amplitude', 1.2)
        timescale = source_cfg.get('variability_timescale', 40.0)
    else:
        amplitude = 1.2
        timescale = 40.0
    
    phase = rng.uniform(0, 2 * np.pi)
    magnitude = base_magnitude + amplitude * np.sin(
        2 * np.pi * time_array / (timescale * 0.5) + phase
    )
    magnitude += rng.normal(0, 0.15, len(time_array))
    return magnitude


def _generate_supernova_light_curve_simple(
    time_array: np.ndarray,
    base_magnitude: float,
    rng: np.random.Generator,
    config: Optional[Dict] = None
) -> np.ndarray:
    """Simplified supernova light curve (fallback)."""
    if config:
        source_cfg = config.get('time_delays', {}).get('source_types', {}).get('supernova', {})
        amplitude = source_cfg.get('variability_amplitude', 2.0)
        timescale = source_cfg.get('variability_timescale', 30.0)
    else:
        amplitude = 2.0
        timescale = 30.0
    
    # Supernova: starts faint, rises to peak, then declines
    # Realistic timescales: Type Ia rise ~15-20 days, decline ~100-200 days
    rise_time = timescale  # Rise time (days) - typically 15-20 days for Type Ia
    decline_time = timescale * 5.0  # Decline time (days) - typically 100-200 days for Type Ia
    
    # Peak occurs after rise_time (realistic: peak at ~15-20 days after explosion)
    t_peak = rise_time  # Peak at rise_time days (realistic for Type Ia)
    
    magnitude = np.zeros_like(time_array)
    for i, t in enumerate(time_array):
        if t < t_peak:
            # Rise phase: magnitude decreases (gets brighter) as we approach peak
            # At t=0: magnitude = base_magnitude + amplitude (faintest, explosion)
            # At t = t_peak: magnitude = base_magnitude (peak brightness)
            # Use linear interpolation: magnitude = base + amp * (1 - t/t_peak)
            # This gives smooth rise from faint to bright
            magnitude[i] = base_magnitude + amplitude * (1 - t / t_peak)
        else:
            # Decline phase: magnitude increases (gets fainter) after peak
            # At t = t_peak: magnitude = base_magnitude (peak brightness)
            # At t >> t_peak: magnitude = base_magnitude + amplitude (faintest)
            dt_from_peak = t - t_peak
            magnitude[i] = base_magnitude + amplitude * (1 - np.exp(-dt_from_peak / decline_time))
    
    return magnitude

