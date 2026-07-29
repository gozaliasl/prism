#!/usr/bin/env python3
"""
Time Delay Image Modification Module

This module modifies lensed images to show different brightnesses for different
lensed images based on their time delays. When a variable source brightens at
time T, this change appears in different images at different times due to
differences in light travel time and gravitational potential.

For example, if image A has delay 0 days and image B has delay 15 days:
- At observation time T: Image A shows source at time T, Image B shows source at time T-15
- At observation time T+15: Image A shows source at time T+15, Image B shows source at time T

This creates the realistic time delay effect where brightness changes propagate
through the lensing geometry.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import lenstronomy for image position detection
try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    LENSTRONOMY_AVAILABLE = False


def apply_per_image_time_delay_brightness(
    image: np.ndarray,
    time_delays_result: Dict,
    image_magnitudes: np.ndarray,
    lens_model_list: List[str],
    kwargs_lens: List[Dict],
    source_x: float,
    source_y: float,
    pixel_scale: float = 0.03,
    numpix: int = 300
) -> np.ndarray:
    """
    Apply per-image brightness differences to a lensed image based on time delays.
    
    This function modifies the brightness of different lensed image regions
    to reflect the time-delayed source brightness. Each lensed image shows
    the source at a different phase of its light curve.
    
    Args:
        image: Input lensed image (numpix x numpix array)
        time_delays_result: Dictionary with 'image_positions' and 'time_delays'
        image_magnitudes: Array of source magnitudes for each image at observation time
        lens_model_list: List of lens model names (e.g., ['SIE', 'SHEAR'])
        kwargs_lens: Lens model parameters
        source_x: Source x position (arcsec)
        source_y: Source y position (arcsec)
        pixel_scale: Pixel scale in arcsec/pixel (default: 0.03 for JWST)
        numpix: Image size in pixels (default: 300)
    
    Returns:
        Modified image with per-image brightness differences applied
    """
    if not LENSTRONOMY_AVAILABLE:
        # Fallback: return original image if lenstronomy not available
        return image
    
    try:
        # Get image positions from time delay calculation
        image_positions = time_delays_result.get('image_positions', [])
        if len(image_positions) == 0:
            return image
        
        # Convert image positions from arcsec to pixels
        center_pix = numpix // 2
        image_pixel_positions = []
        for x_img, y_img in image_positions:
            # Convert arcsec to pixels (assuming image center is at (0, 0) in arcsec)
            x_pix = center_pix + x_img / pixel_scale
            y_pix = center_pix + y_img / pixel_scale
            image_pixel_positions.append((x_pix, y_pix))
        
        # Create modified image
        modified_image = image.copy()
        
        # Calculate brightness factors for each image
        # Magnitude difference: brighter = lower magnitude
        base_magnitude = np.min(image_magnitudes)  # Brightest image
        brightness_factors = []
        for mag in image_magnitudes:
            # Convert magnitude difference to flux ratio
            # flux_ratio = 10^(-0.4 * (mag - base_mag))
            flux_ratio = 10 ** (-0.4 * (mag - base_magnitude))
            brightness_factors.append(flux_ratio)
        
        # Apply brightness modifications to each image region
        # We'll use a Gaussian kernel around each image position
        y, x = np.ogrid[:numpix, :numpix]
        
        for i, ((x_pix, y_pix), brightness_factor) in enumerate(zip(image_pixel_positions, brightness_factors)):
            # Create a mask for this image region
            # Use a radius based on Einstein radius (approximate)
            # Typical lensed image size is ~0.5-2 arcsec = ~17-67 pixels
            image_radius_pix = 30  # Approximate radius in pixels
            
            # Distance from image center
            dist_sq = (x - x_pix)**2 + (y - y_pix)**2
            mask = dist_sq <= image_radius_pix**2
            
            # Apply brightness factor to this region
            # Use smooth transition to avoid sharp edges
            if np.any(mask):
                # Smooth weighting: Gaussian falloff
                weight = np.exp(-dist_sq / (2 * (image_radius_pix * 0.5)**2))
                weight = np.clip(weight, 0, 1)
                
                # Apply brightness modification
                # Only modify the source region (bright regions, not the lens galaxy)
                # We'll identify source regions as pixels above a threshold
                source_threshold = np.percentile(image[mask], 50)  # Median brightness in region
                source_mask = (image > source_threshold) & mask
                
                if np.any(source_mask):
                    # Apply brightness factor to source pixels
                    modified_image[source_mask] *= brightness_factor
                    
                    # Smooth transition at edges
                    transition_mask = mask & ~source_mask
                    if np.any(transition_mask):
                        transition_weight = weight[transition_mask]
                        modified_image[transition_mask] *= (1.0 + (brightness_factor - 1.0) * transition_weight)
        
        return modified_image
        
    except Exception as e:
        print(f"[WARNING] Failed to apply per-image time delay brightness: {e}")
        return image


def create_time_delay_annotated_image(
    image: np.ndarray,
    time_delays_result: Dict,
    image_magnitudes: np.ndarray,
    observation_time: float,
    numpix: int = 300,
    pixel_scale: float = 0.03
) -> Tuple[np.ndarray, Dict]:
    """
    Create an annotated image showing time delay information.
    
    Args:
        image: Input lensed image
        time_delays_result: Dictionary with time delay information
        image_magnitudes: Array of source magnitudes for each image
        observation_time: Observation time in days
        numpix: Image size in pixels
        pixel_scale: Pixel scale in arcsec/pixel
    
    Returns:
        Tuple of (annotated_image, annotation_info)
    """
    # For now, return the original image with metadata
    # In a full implementation, this would add text annotations, markers, etc.
    annotation_info = {
        'n_images': len(time_delays_result.get('image_positions', [])),
        'time_delays': time_delays_result.get('time_delays', []),
        'image_magnitudes': image_magnitudes.tolist(),
        'observation_time': observation_time
    }
    
    return image, annotation_info

