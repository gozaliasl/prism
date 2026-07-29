"""
Utility functions for parameter handling and fallback strategies.

Handles:
1. Numpy RandomState compatibility (integers vs randint)
2. Missing filter parameters (Sersic n, morphological params)
3. Parameter filling with nearby filter or statistical values
"""

import numpy as np
import warnings


def safe_random_integers(rng, low, high):
    """
    Safely generate random integers handling both old and new numpy RandomState APIs.
    
    Parameters
    ----------
    rng : np.random.RandomState or np.random.Generator
        Random number generator
    low : int
        Lower bound (inclusive)
    high : int  
        Upper bound (exclusive)
        
    Returns
    -------
    int
        Random integer in [low, high)
    """
    try:
        # Try new numpy API (1.19+)
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high))
        # Fall back to old API
        elif hasattr(rng, "randint"):
            return int(rng.randint(low, high))
        else:
            # Last resort: use uniform and convert
            return int(rng.uniform(low, high))
    except AttributeError as e:
        warnings.warn(f"RNG compatibility issue: {e}. Using uniform fallback.", UserWarning)
        return int(rng.uniform(low, high))


def fill_missing_parameter(param_dict, target_param, filter_name, 
                          nearby_filters=None, all_filters=None):
    """
    Fill missing parameter using fallback strategy:
    1. Use nearby filter if available
    2. Use mean/median of other filters
    3. Return None if no fallback available
    
    Parameters
    ----------
    param_dict : dict
        Dictionary with parameters, keys like 'F115W_sersic_n', 'F150W_sersic_n', etc.
    target_param : str
        Parameter name (e.g., 'sersic_n', 'axis_ratio')
    filter_name : str
        Filter name (e.g., 'F444W')
    nearby_filters : list, optional
        List of filters to check in order (e.g., ['F277W', 'F150W', 'F115W'])
    all_filters : list, optional
        All available filters to consider for mean/median fallback
        
    Returns
    -------
    float or None
        Filled parameter value or None if unable to fill
    """
    # First try nearby filters
    if nearby_filters:
        for nearby_filter in nearby_filters:
            key = f"{nearby_filter}_{target_param}"
            if key in param_dict and param_dict[key] is not None:
                value = param_dict[key]
                warnings.warn(
                    f"Parameter {filter_name}_{target_param} was missing. "
                    f"Filled from nearby {nearby_filter}: {value:.3f}",
                    UserWarning
                )
                return value
    
    # Then try mean/median of all available filters
    if all_filters:
        available_values = []
        for f in all_filters:
            key = f"{f}_{target_param}"
            if key in param_dict and param_dict[key] is not None:
                try:
                    available_values.append(float(param_dict[key]))
                except (ValueError, TypeError):
                    continue
        
        if available_values:
            mean_value = np.mean(available_values)
            warnings.warn(
                f"Parameter {filter_name}_{target_param} was missing. "
                f"Filled with mean of {len(available_values)} filters: {mean_value:.3f}",
                UserWarning
            )
            return mean_value
    
    return None


def ensure_filter_parameters(param_dict, filter_names=None, 
                            parameters=None, default_ranges=None):
    """
    Ensure all expected filter parameters are present. Fill missing ones with fallback strategy.
    
    Parameters
    ----------
    param_dict : dict
        Dictionary with filter-specific parameters
    filter_names : list, optional
        Filter names to process. Defaults to ['F115W', 'F150W', 'F277W', 'F444W']
    parameters : list, optional
        Parameter names to check. Defaults to ['sersic_n', 'axis_ratio', 'position_angle']
    default_ranges : dict, optional
        Fallback ranges for random value generation {param_name: (min, max, ...)}
        
    Returns
    -------
    dict
        Updated param_dict with filled missing values
    """
    if filter_names is None:
        filter_names = ['F115W', 'F150W', 'F277W', 'F444W']
    
    if parameters is None:
        parameters = ['sersic_n', 'axis_ratio', 'position_angle']
    
    if default_ranges is None:
        default_ranges = {
            'sersic_n': (0.5, 4.0),
            'axis_ratio': (0.4, 1.0),
            'position_angle': (0, 360)
        }
    
    # Filter ordering for nearby fallback
    filter_order = ['F115W', 'F150W', 'F277W', 'F444W']
    
    for param in parameters:
        for i, filter_name in enumerate(filter_names):
            key = f"{filter_name}_{param}"
            
            # Skip if already present and valid
            if key in param_dict and param_dict[key] is not None:
                try:
                    float(param_dict[key])
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Try to fill using nearby filters
            nearby = [filter_order[j] for j in range(len(filter_order)) 
                     if filter_order[j] in filter_names and j != i]
            
            filled_value = fill_missing_parameter(
                param_dict, param, filter_name,
                nearby_filters=nearby,
                all_filters=filter_names
            )
            
            if filled_value is not None:
                param_dict[key] = filled_value
            elif param in default_ranges:
                # Use default range
                min_val, max_val = default_ranges[param][:2]
                if param == 'position_angle':
                    default_val = np.random.uniform(min_val, max_val)
                elif param == 'sersic_n':
                    default_val = np.random.uniform(min_val, max_val)
                else:
                    default_val = np.random.uniform(min_val, max_val)
                
                warnings.warn(
                    f"Parameter {key} was missing and no nearby filters available. "
                    f"Using default range [{min_val}, {max_val}]: {default_val:.3f}",
                    UserWarning
                )
                param_dict[key] = default_val
    
    return param_dict


def validate_parameter_dict(param_dict, expected_keys=None):
    """
    Validate that parameter dictionary has expected keys and valid values.
    
    Parameters
    ----------
    param_dict : dict
        Dictionary to validate
    expected_keys : list, optional
        Expected keys to check
        
    Returns
    -------
    tuple
        (is_valid: bool, missing_keys: list, invalid_keys: list)
    """
    missing_keys = []
    invalid_keys = []
    
    if expected_keys:
        for key in expected_keys:
            if key not in param_dict:
                missing_keys.append(key)
            elif param_dict[key] is None:
                invalid_keys.append(f"{key}=None")
            else:
                try:
                    float(param_dict[key])
                except (ValueError, TypeError):
                    invalid_keys.append(f"{key}={param_dict[key]} (not numeric)")
    
    is_valid = len(missing_keys) == 0 and len(invalid_keys) == 0
    
    return is_valid, missing_keys, invalid_keys
