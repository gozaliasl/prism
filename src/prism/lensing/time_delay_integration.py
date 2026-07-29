#!/usr/bin/env python3
"""
Time Delay Integration for Main Simulation Pipeline

This module integrates time delay functionality into the main JWST lens simulator,
generating multiple epochs for variable sources (quasars, supernovae, AGN).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from prism.core.constants import C_LIGHT_KM_S, C_LIGHT_MPC_PER_DAY, ARCSEC_TO_RAD

# Try to import time delay simulator
try:
    from prism.lensing.time_delay_simulation import TimeDelaySimulator
    TIME_DELAY_MODULE_AVAILABLE = True
except ImportError:
    TIME_DELAY_MODULE_AVAILABLE = False
    print("Warning: time_delay_simulation module not available")

# Try to import lenstronomy
try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.TimeDelays.time_delays import TimeDelays
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    LENSTRONOMY_AVAILABLE = False

# Try to import enhanced light curve models
try:
    from prism.lensing.light_curve_models import (
        generate_quasar_light_curve_drw,
        generate_agn_light_curve_drw,
        generate_supernova_light_curve
    )
    ENHANCED_LIGHT_CURVES_AVAILABLE = True
except ImportError:
    ENHANCED_LIGHT_CURVES_AVAILABLE = False
    print("Warning: Enhanced light curve models not available. Using simplified models.")


def should_generate_time_delays(config: Dict, rng: np.random.Generator) -> bool:
    """
    Determine if a lens system should have time delays based on config.
    
    Args:
        config: Configuration dictionary
        rng: Random number generator
    
    Returns:
        bool: True if time delays should be generated
    """
    time_delay_cfg = config.get('time_delays', {})
    if not time_delay_cfg.get('enabled', False):
        return False
    
    fraction = time_delay_cfg.get('fraction_variable_sources', 0.15)
    return rng.random() < fraction


def select_variable_source_type(config: Dict, rng: np.random.Generator) -> str:
    """
    Select variable source type based on config fractions.
    
    Args:
        config: Configuration dictionary
        rng: Random number generator
    
    Returns:
        str: Source type ('quasar', 'supernova', 'agn')
    """
    source_types = config.get('time_delays', {}).get('source_types', {})
    
    types = []
    fractions = []
    for stype, params in source_types.items():
        types.append(stype)
        fractions.append(params.get('fraction', 0.33))
    
    # Normalize fractions
    fractions = np.array(fractions)
    fractions = fractions / fractions.sum()
    
    return rng.choice(types, p=fractions)


def calculate_time_delays_simplified(
    lens_model_list: List[str],
    kwargs_lens: List[Dict],
    source_x: float,
    source_y: float,
    lens_redshift: float,
    source_redshift: float,
    einstein_radius: float,
    rng: np.random.Generator
) -> Dict:
    """
    Calculate time delays using physical Fermat potential equation.
    Uses lenstronomy when available, otherwise uses improved physical approximation.
    
    The time delay between images i and j follows the Fermat potential:
    Δt_ij = (1+z_l)/c * D_l*D_s/D_ls * [0.5*(θ_i - β)² - 0.5*(θ_j - β)² - ψ(θ_i) + ψ(θ_j)]
    
    Args:
        lens_model_list: List of lens model names
        kwargs_lens: Lens model parameters
        source_x: Source x position (arcsec)
        source_y: Source y position (arcsec)
        lens_redshift: Lens redshift
        source_redshift: Source redshift
        einstein_radius: Einstein radius (arcsec)
        rng: Random number generator
    
    Returns:
        Dictionary with time delays and image positions
    """
    # Try to use lenstronomy for accurate calculation
    if LENSTRONOMY_AVAILABLE:
        try:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            
            # Use TimeDelaySimulator for proper calculation
            if TIME_DELAY_MODULE_AVAILABLE:
                simulator = TimeDelaySimulator(cosmology=cosmo)
                result = simulator.calculate_time_delays(
                    lens_model_list=lens_model_list,
                    kwargs_lens=kwargs_lens,
                    source_x=source_x,
                    source_y=source_y,
                    lens_redshift=lens_redshift,
                    source_redshift=source_redshift
                )
                return result
            else:
                # Use lenstronomy directly
                from lenstronomy.LensModel.lens_model import LensModel
                from lenstronomy.TimeDelays.time_delays import TimeDelays
                from lenstronomy.Cosmo.lens_cosmo import LensCosmo
                
                lens_model = LensModel(lens_model_list=lens_model_list)
                lens_cosmo = LensCosmo(z_lens=lens_redshift, z_source=source_redshift, cosmo=cosmo)
                
                # Solve lens equation to find image positions
                beta_x, beta_y = source_x, source_y
                x_image, y_image = lens_model.ray_shooting(beta_x, beta_y, kwargs_lens)
                
                # For multiple images, we need to solve the lens equation properly
                # For now, use a grid search to find images
                from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
                solver = LensEquationSolver(lens_model)
                image_positions = solver.image_position_from_source(
                    sourcePos_x=beta_x,
                    sourcePos_y=beta_y,
                    kwargs_lens=kwargs_lens,
                    min_distance=0.05,
                    search_window=einstein_radius * 3,
                    precision_limit=1e-5,
                    num_iter_max=100
                )
                
                if len(image_positions[0]) == 0:
                    # Fallback: create images around Einstein ring
                    n_images = rng.integers(2, 4)
                    image_positions = []
                    for i in range(n_images):
                        angle = 2 * np.pi * i / n_images
                        radius = einstein_radius * rng.uniform(0.8, 1.2)
                        image_positions.append((radius * np.cos(angle), radius * np.sin(angle)))
                else:
                    image_positions = list(zip(image_positions[0], image_positions[1]))
                
                # Calculate time delays using Fermat potential
                time_delays_calc = TimeDelays(lens_model, lens_cosmo)
                time_delay_list = []
                magnifications = []
                
                for x_img, y_img in image_positions:
                    dt = time_delays_calc.time_delay(
                        x_image=x_img,
                        y_image=y_img,
                        kwargs_lens=kwargs_lens,
                        source_x=source_x,
                        source_y=source_y
                    )
                    time_delay_list.append(dt)
                    mag = lens_model.magnification(x_img, y_img, kwargs_lens)
                    magnifications.append(mag)
                
                time_delay_days = np.array(time_delay_list) / 86400.0  # Convert to days
                time_delay_days -= time_delay_days.min()  # Relative to first image
                
                return {
                    'time_delays': time_delay_days,
                    'image_positions': image_positions,
                    'magnifications': np.array(magnifications),
                    'lens_redshift': lens_redshift,
                    'source_redshift': source_redshift,
                    'time_delay_method': 'lenstronomy_ray_traced'
                }
        except Exception as e:
            print(f"[TIME_DELAY] lenstronomy calculation failed: {e}, using improved fallback")
            # Fall through to improved physical approximation
    
    # Improved physical approximation using Fermat potential structure
    # When lenstronomy is unavailable, use the proper equation structure
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    # Calculate angular diameter distances
    D_l = cosmo.angular_diameter_distance(lens_redshift).value  # Mpc
    D_s = cosmo.angular_diameter_distance(source_redshift).value  # Mpc
    D_ls = cosmo.angular_diameter_distance_z1z2(lens_redshift, source_redshift).value  # Mpc
    
    einstein_radius_rad = einstein_radius * ARCSEC_TO_RAD
    
    # Generate image positions around Einstein ring (simplified)
    n_images = rng.integers(2, 5)
    image_positions = []
    for i in range(n_images):
        angle = 2 * np.pi * i / n_images + rng.uniform(-0.2, 0.2)
        radius = einstein_radius * rng.uniform(0.7, 1.1)
        x_img = radius * np.cos(angle)
        y_img = radius * np.sin(angle)
        image_positions.append((x_img, y_img))
    
    # Calculate time delays using Fermat potential structure
    # Δt = (1+z_l)/c * D_l*D_s/D_ls * [0.5*(θ - β)² - ψ(θ)]
    # For SIE lens: ψ(θ) ≈ θ_E * |θ| (simplified)
    source_pos = np.array([source_x, source_y])
    time_delays = []
    magnifications = []
    
    for x_img, y_img in image_positions:
        img_pos = np.array([x_img, y_img])
        
        # Geometric term: 0.5 * (θ - β)²
        diff = img_pos - source_pos
        geometric_term = 0.5 * np.sum(diff**2) * (ARCSEC_TO_RAD**2)

        # Potential term: -ψ(θ) ≈ -θ_E * |θ| for SIE (simplified)
        img_sep = np.sqrt(x_img**2 + y_img**2) * ARCSEC_TO_RAD
        potential_term = -einstein_radius_rad * img_sep

        # Fermat potential difference
        fermat_potential = geometric_term + potential_term

        # Time delay: Δt = (1+z_l)/c * D_l*D_s/D_ls * Fermat_potential
        dt_days = (1 + lens_redshift) / C_LIGHT_MPC_PER_DAY * (D_l * D_s / D_ls) * fermat_potential

        time_delays.append(dt_days)

        # Simplified magnification (inverse of image separation)
        mag = einstein_radius / max(img_sep / ARCSEC_TO_RAD, 0.1)
        magnifications.append(mag)
    
    time_delays = np.array(time_delays)
    time_delays -= time_delays.min()  # Relative to first image
    
    return {
        'time_delays': time_delays,
        'image_positions': image_positions,
        'magnifications': np.array(magnifications),
        'lens_redshift': lens_redshift,
        'source_redshift': source_redshift,
        # Image positions/magnifications from random placement, NOT from the lens
        # equation.  Rows produced via this path must be excluded from science
        # catalogs or clearly flagged as approximate.
        'time_delay_method': 'fallback_fermat_approx'
    }


def generate_epoch_times(
    config: Dict,
    rng: np.random.Generator,
    light_curve: Optional[np.ndarray] = None,
    time_array: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate epoch times for time-series observations.
    
    Optimizes epoch selection to maximize source brightness/shape variation
    by selecting epochs at key points in the light curve (peak, minimum, and
    intermediate points that maximize the range).
    
    Args:
        config: Configuration dictionary
        rng: Random number generator
        light_curve: Optional pre-computed light curve (magnitude array)
        time_array: Optional time array corresponding to light_curve
    
    Returns:
        Array of epoch times in days
    """
    epochs_cfg = config.get('time_delays', {}).get('epochs', {})
    min_epochs = epochs_cfg.get('min_epochs', 4)
    max_epochs = epochs_cfg.get('max_epochs', 4)
    time_range = epochs_cfg.get('time_range_days', 200)
    random_epochs = epochs_cfg.get('random_epochs', False)
    optimize_epochs = epochs_cfg.get('optimize_epochs', True)  # New option to optimize
    
    # Fixed to 4 epochs for maximum visibility
    n_epochs = max_epochs  # Use max_epochs (should be 4)
    
    if random_epochs:
        # Random epochs within time range
        epoch_times = rng.uniform(0, time_range, n_epochs)
        epoch_times = np.sort(epoch_times)
    elif optimize_epochs and light_curve is not None and time_array is not None:
        # Optimize epoch selection based on light curve to maximize variation
        # Strategy: Select epochs at peak brightness, minimum brightness, and
        # two intermediate points that maximize the magnitude range
        
        mag_min = light_curve.min()  # Brightest (lowest magnitude)
        mag_max = light_curve.max()  # Faintest (highest magnitude)
        mag_range = mag_max - mag_min
        
        if mag_range > 0.1:  # Only optimize if there's significant variation
            # Target magnitudes: peak, 1/3 from peak, 2/3 from peak, minimum
            target_mags = [
                mag_min,  # Epoch 0: Maximum brightness (peak)
                mag_min + 0.33 * mag_range,  # Epoch 1: 1/3 from peak
                mag_min + 0.67 * mag_range,  # Epoch 2: 2/3 from peak
                mag_max,  # Epoch 3: Minimum brightness (faintest)
            ]
            
            # Find times where light curve is closest to each target magnitude
            selected_epoch_times = []
            for target_mag in target_mags:
                idx = np.argmin(np.abs(light_curve - target_mag))
                selected_epoch_times.append(time_array[idx])
            
            epoch_times = np.array(selected_epoch_times)
            # Sort by time to ensure chronological order
            epoch_times = np.sort(epoch_times)
        else:
            # If variation is too small, fall back to uniform spacing
            epoch_times = np.linspace(0, time_range, n_epochs)
    else:
        # Uniform spacing across time range (fallback)
        epoch_times = np.linspace(0, time_range, n_epochs)
    
    return epoch_times


def generate_light_curve_for_source(
    time_array: np.ndarray,
    source_type: str,
    config: Dict,
    rng: np.random.Generator,
    base_magnitude: float = 20.0,
    redshift: float = 2.0,
    black_hole_mass: float = None
) -> np.ndarray:
    """
    Generate light curve for variable source.
    
    Uses enhanced theoretical models (celerite2 for quasars/AGN, sncosmo for SNe)
    when available, falls back to simplified models otherwise.
    
    Args:
        time_array: Time array in days
        source_type: Source type ('quasar', 'supernova', 'agn')
        config: Configuration dictionary
        rng: Random number generator
        base_magnitude: Base magnitude of source
        redshift: Source redshift (for enhanced models)
        black_hole_mass: Black hole mass in log10(M_BH/M_sun) (for enhanced models)
    
    Returns:
        Array of magnitudes as function of time
    """
    # Check if enhanced models should be used
    source_cfg = config.get('time_delays', {}).get('source_types', {}).get(source_type, {})
    use_enhanced = source_cfg.get('use_drw', False) if source_type in ['quasar', 'agn'] else source_cfg.get('use_sncosmo', False)
    
    # Use enhanced models if available and requested
    if ENHANCED_LIGHT_CURVES_AVAILABLE and use_enhanced:
        if source_type == 'quasar':
            # Get black hole mass from config if not provided
            if black_hole_mass is None:
                black_hole_mass = source_cfg.get('black_hole_mass', None)
            return generate_quasar_light_curve_drw(
                time_array, base_magnitude, black_hole_mass, redshift, rng, config
            )
        
        elif source_type == 'agn':
            # Get black hole mass from config if not provided
            if black_hole_mass is None:
                black_hole_mass = source_cfg.get('black_hole_mass', None)
            return generate_agn_light_curve_drw(
                time_array, base_magnitude, black_hole_mass, redshift, rng, config
            )
        
        elif source_type == 'supernova':
            sn_type = source_cfg.get('sn_type', 'Ia')
            return generate_supernova_light_curve(
                time_array, base_magnitude, sn_type, redshift, rng, config
            )
    
    # Fallback to simplified models
    amplitude = source_cfg.get('variability_amplitude', 0.8)
    timescale = source_cfg.get('variability_timescale', 50.0)
    
    if source_type == 'quasar':
        # DRW-like variability (simplified)
        phase = rng.uniform(0, 2 * np.pi)
        magnitude = base_magnitude + amplitude * np.sin(
            2 * np.pi * time_array / timescale + phase
        )
        magnitude += rng.normal(0, 0.1, len(time_array))
    
    elif source_type == 'supernova':
        # Rise and decline (simplified)
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
    
    else:  # agn
        # Sinusoidal with shorter timescale (simplified)
        phase = rng.uniform(0, 2 * np.pi)
        magnitude = base_magnitude + amplitude * np.sin(
            2 * np.pi * time_array / (timescale * 0.5) + phase
        )
        magnitude += rng.normal(0, 0.15, len(time_array))
    
    return magnitude


def apply_time_delay_to_source_magnitude(
    base_magnitude: float,
    light_curve: np.ndarray,
    time_array: np.ndarray,
    time_delays: np.ndarray,
    observation_time: float
) -> np.ndarray:
    """
    Apply time delays to source magnitudes for each image.
    
    Args:
        base_magnitude: Base source magnitude
        light_curve: Light curve array
        time_array: Time array for light curve
        time_delays: Time delays for each image (days)
        observation_time: Observation time (days)
    
    Returns:
        Array of magnitudes for each image at observation time
    """
    image_magnitudes = []
    
    for dt in time_delays:
        # Source time for this image
        source_time = observation_time - dt
        
        # Interpolate light curve
        if source_time < time_array.min():
            mag = light_curve[0]
        elif source_time > time_array.max():
            mag = light_curve[-1]
        else:
            mag = np.interp(source_time, time_array, light_curve)
        
        image_magnitudes.append(mag)
    
    return np.array(image_magnitudes)


def create_time_delay_metadata(
    lens_id: int,
    source_type: str,
    time_delays_result: Dict,
    epoch_times: np.ndarray,
    light_curve: np.ndarray,
    time_array: np.ndarray
) -> Dict:
    """
    Create metadata dictionary for time delay system.
    
    Args:
        lens_id: Lens system ID
        source_type: Variable source type
        time_delays_result: Time delay calculation results
        epoch_times: Array of epoch times
        light_curve: Light curve array
        time_array: Time array for light curve
    
    Returns:
        Metadata dictionary
    """
    # Handle both numpy arrays and lists
    def to_list(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        return list(x) if isinstance(x, (list, tuple)) else [x]
    
    return {
        'lens_id': lens_id,
        'has_time_delays': True,
        'source_type': source_type,
        'n_images': len(time_delays_result['image_positions']),
        'time_delays_days': to_list(time_delays_result['time_delays']),
        'magnifications': to_list(time_delays_result['magnifications']),
        'n_epochs': len(epoch_times),
        'epoch_times_days': to_list(epoch_times),
        'lens_redshift': time_delays_result.get('lens_redshift', np.nan),
        'source_redshift': time_delays_result.get('source_redshift', np.nan)
    }

