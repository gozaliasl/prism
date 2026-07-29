#!/usr/bin/env python3
"""
Intelligent Lensed Image Detection Module

This module detects lensed source images in simulated lens systems using:
1. Source detection algorithms (SExtractor-like, photutils)
2. Geometric constraints (Einstein radius, symmetry)
3. Computer vision techniques (edge detection, arc finding)
4. Optional ML-based arc detection

Combines lenstronomy predictions (as priors) with actual image detection
for robust identification of lensed images.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import photutils for source detection (handle API changes)
PHOTUTILS_AVAILABLE = False
try:
    # photutils >= 1.9/2.0 moved everything into segmentation
    from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
    from photutils.background import Background2D, MedianBackground
    PHOTUTILS_AVAILABLE = True
except ImportError:
    try:
        # Some intermediate releases exposed detect_sources at the package root
        from photutils import detect_sources, deblend_sources
        from photutils.segmentation import SourceCatalog
        from photutils.background import Background2D, MedianBackground
        PHOTUTILS_AVAILABLE = True
    except ImportError:
        try:
            # Very old layout kept detect_sources in photutils.detection
            from photutils.detection import detect_sources
            from photutils.segmentation import deblend_sources, SourceCatalog
            from photutils.background import Background2D, MedianBackground
            PHOTUTILS_AVAILABLE = True
        except ImportError:
            PHOTUTILS_AVAILABLE = False
            print("[WARNING] photutils not available - using basic source detection")

# Try to import scikit-image for advanced image processing
try:
    from skimage import feature, filters, measure, morphology
    from skimage.segmentation import watershed
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARNING] scikit-image not available - using basic methods")

# Try to import lenstronomy for prior predictions
try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    LENSTRONOMY_AVAILABLE = False


def detect_sources_sextractor_like(
    image: np.ndarray,
    threshold: float = None,
    npixels: int = 10,  # Increased from 5 to reduce noise
    deblend: bool = True
) -> List[Dict]:
    """
    Detect sources using SExtractor-like algorithm (photutils).
    
    Args:
        image: 2D image array
        threshold: Detection threshold (if None, uses background estimation)
        npixels: Minimum number of connected pixels for a source
        deblend: Whether to deblend overlapping sources
    
    Returns:
        List of source dictionaries with positions, fluxes, and properties
    """
    if not PHOTUTILS_AVAILABLE:
        return detect_sources_basic(image, threshold, npixels)
    
    # Estimate background if threshold not provided
    if threshold is None:
        try:
            bkg_estimator = MedianBackground()
            bkg = Background2D(image, (50, 50), filter_size=(3, 3),
                             bkg_estimator=bkg_estimator)
            # Use higher threshold (3-sigma instead of 2-sigma) to reduce false positives
            threshold = bkg.background + 3.0 * bkg.background_rms
        except:
            # Fallback: use percentile-based threshold (more conservative)
            threshold = np.percentile(image, 98)  # Increased from 95 to 98
    
    # Detect sources
    segm = detect_sources(image, threshold, npixels=npixels)
    
    if segm is None or segm.nlabels == 0:
        return []
    
    # Deblend if requested
    if deblend:
        try:
            segm = deblend_sources(image, segm, npixels=npixels,
                                 nlevels=32, contrast=0.001, mode='exponential')
        except:
            pass  # Continue without deblending if it fails
    
    # Extract source properties
    cat = SourceCatalog(image, segm)
    sources = []
    
    for obj in cat:
        # Get centroid (weighted by flux)
        y_centroid, x_centroid = obj.centroid
        
        # Get flux and other properties
        flux = obj.segment_flux
        area = obj.area.value
        
        # Get bounding box
        bbox = obj.bbox
        
        sources.append({
            'x': float(x_centroid),
            'y': float(y_centroid),
            'flux': float(flux),
            'area': int(area),
            'bbox': bbox,
            'label': int(obj.label)
        })
    
    return sources


def detect_sources_basic(
    image: np.ndarray,
    threshold: float = None,
    npixels: int = 5
) -> List[Dict]:
    """
    Basic source detection using thresholding and connected components.
    
    Args:
        image: 2D image array
        threshold: Detection threshold
        npixels: Minimum number of connected pixels
    
    Returns:
        List of source dictionaries
    """
    from scipy import ndimage
    
    # Estimate threshold if not provided
    if threshold is None:
        threshold = np.percentile(image, 95)
    
    # Create binary mask
    mask = image > threshold
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    sources = []
    for label_id in range(1, num_features + 1):
        # Get pixels for this source
        y_coords, x_coords = np.where(labeled == label_id)
        
        if len(y_coords) < npixels:
            continue
        
        # Calculate weighted centroid
        values = image[y_coords, x_coords]
        total_flux = np.sum(values)
        
        if total_flux <= 0:
            continue
        
        x_centroid = np.sum(x_coords * values) / total_flux
        y_centroid = np.sum(y_coords * values) / total_flux
        
        # Bounding box
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        sources.append({
            'x': float(x_centroid),
            'y': float(y_centroid),
            'flux': float(total_flux),
            'area': len(y_coords),
            'bbox': (y_min, y_max, x_min, x_max),
            'label': label_id
        })
    
    return sources


def detect_arcs_geometric(
    image: np.ndarray,
    lens_center: Tuple[float, float],
    einstein_radius_pix: float,
    pixel_scale: float = 0.03
) -> List[Dict]:
    """
    Detect arcs using geometric constraints (elongation, curvature, position).
    
    Args:
        image: 2D image array
        lens_center: (x, y) position of lens center in pixels
        einstein_radius_pix: Einstein radius in pixels
        pixel_scale: Pixel scale in arcsec/pixel
    
    Returns:
        List of detected arc candidates
    """
    if not SKIMAGE_AVAILABLE:
        return []
    
    # Edge detection to find arc-like structures
    edges = feature.canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    
    # Find contours
    contours = measure.find_contours(edges, 0.5)
    
    arc_candidates = []
    center_x, center_y = lens_center
    
    for contour in contours:
        if len(contour) < 10:  # Too short to be an arc
            continue
        
        # Calculate distance from lens center
        distances = np.sqrt((contour[:, 1] - center_x)**2 + (contour[:, 0] - center_y)**2)
        mean_distance = np.mean(distances)
        
        # Check if around Einstein radius (within factor of 2)
        if mean_distance < einstein_radius_pix * 0.5 or mean_distance > einstein_radius_pix * 2.5:
            continue
        
        # Calculate elongation (arcs are elongated)
        x_range = contour[:, 1].max() - contour[:, 1].min()
        y_range = contour[:, 0].max() - contour[:, 0].min()
        elongation = max(x_range, y_range) / min(x_range, y_range) if min(x_range, y_range) > 0 else 1.0
        
        if elongation < 1.5:  # Not elongated enough
            continue
        
        # Calculate curvature (arcs are curved)
        # Simple curvature estimate: angle between segments
        angles = []
        for i in range(1, len(contour) - 1):
            v1 = contour[i] - contour[i-1]
            v2 = contour[i+1] - contour[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
        
        mean_curvature = np.mean(angles) if angles else np.pi
        
        # Arcs have moderate curvature (not straight, not too curved)
        if mean_curvature < 0.5 or mean_curvature > 2.5:
            continue
        
        # Get centroid of arc
        arc_x = np.mean(contour[:, 1])
        arc_y = np.mean(contour[:, 0])
        
        arc_candidates.append({
            'x': float(arc_x),
            'y': float(arc_y),
            'elongation': float(elongation),
            'curvature': float(mean_curvature),
            'distance_from_lens': float(mean_distance),
            'length': float(len(contour))
        })
    
    return arc_candidates


def identify_lensed_images_hybrid(
    image: np.ndarray,
    lens_center: Tuple[float, float],
    einstein_radius_pix: float,
    lenstronomy_predictions: Optional[List[Tuple[float, float]]] = None,
    pixel_scale: float = 0.03,
    numpix: int = 300,
    use_source_detection: bool = True,
    use_arc_detection: bool = True,
    use_geometric_constraints: bool = True
) -> List[Dict]:
    """
    Hybrid approach to identify lensed images combining multiple methods.
    
    Strategy:
    1. Use lenstronomy predictions as priors (if available)
    2. Detect sources using SExtractor-like algorithm
    3. Detect arcs using geometric constraints
    4. Filter candidates based on:
       - Distance from lens center (around Einstein radius)
       - Flux (bright enough to be lensed source)
       - Not the lens galaxy itself
       - Match to lenstronomy predictions (if available)
    
    Args:
        image: 2D image array
        lens_center: (x, y) position of lens center in pixels
        einstein_radius_pix: Einstein radius in pixels
        lenstronomy_predictions: Optional list of (x, y) predicted positions in arcsec
        pixel_scale: Pixel scale in arcsec/pixel
        numpix: Image size in pixels
        use_source_detection: Use SExtractor-like source detection
        use_arc_detection: Use geometric arc detection
        use_geometric_constraints: Apply geometric filtering
    
    Returns:
        List of identified lensed image candidates with positions and properties
    """
    center_x, center_y = lens_center
    candidates = []
    
    # Convert lenstronomy predictions to pixels if provided
    lenstronomy_pix = []
    if lenstronomy_predictions:
        center_pix = numpix // 2
        for x_arcsec, y_arcsec in lenstronomy_predictions:
            x_pix = center_pix + x_arcsec / pixel_scale
            y_pix = center_pix + y_arcsec / pixel_scale
            lenstronomy_pix.append((x_pix, y_pix))
    
    # Method 1: Source detection (SExtractor-like)
    if use_source_detection:
        sources = detect_sources_sextractor_like(image, npixels=10)  # More conservative
        
        # Calculate flux statistics for filtering
        if sources:
            fluxes = [s['flux'] for s in sources]
            max_flux = max(fluxes)
            median_flux = np.median(fluxes)
        else:
            max_flux = 1.0
            median_flux = 1.0
        
        for source in sources:
            x, y = source['x'], source['y']
            
            # Calculate distance from lens center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Filter: must be around Einstein radius (not too close, not too far)
            if use_geometric_constraints:
                # Exclude lens galaxy region (too close to center) - MORE AGGRESSIVE
                if dist < einstein_radius_pix * 0.7:  # Increased from 0.5 to 0.7
                    continue
                # Exclude very distant sources (likely field galaxies) - MORE AGGRESSIVE
                if dist > einstein_radius_pix * 2.0:  # Reduced from 2.5 to 2.0
                    continue
                
                # Additional filter: flux should be reasonable
                flux_ratio = source['flux'] / max_flux if max_flux > 0 else 1.0
                
                # Exclude very bright sources (likely lens galaxy or bright field galaxies) - MORE AGGRESSIVE
                if flux_ratio > 0.6:  # Reduced from 0.8 to 0.6
                    continue
                
                # Exclude very faint sources (likely noise) - MORE AGGRESSIVE
                if flux_ratio < 0.1:  # Increased from 0.05 to 0.1
                    continue
                
                # Additional: exclude sources that are too large (likely lens galaxy or large field galaxies)
                # Lensed images are typically compact
                if source.get('area', 0) > 500:  # Too many pixels
                    continue
            
            # Check if matches lenstronomy prediction (if available)
            matches_prediction = False
            if lenstronomy_pix:
                for pred_x, pred_y in lenstronomy_pix:
                    pred_dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
                    if pred_dist < einstein_radius_pix * 0.5:  # Within half Einstein radius
                        matches_prediction = True
                        break
            
            candidates.append({
                'x': x,
                'y': y,
                'x_arcsec': (x - numpix // 2) * pixel_scale,
                'y_arcsec': (y - numpix // 2) * pixel_scale,
                'distance_from_lens_pix': dist,
                'distance_from_lens_arcsec': dist * pixel_scale,
                'flux': source['flux'],
                'area': source['area'],
                'method': 'source_detection',
                'matches_lenstronomy': matches_prediction,
                'confidence': 0.7 if matches_prediction else 0.5
            })
    
    # Method 2: Arc detection (geometric) - MORE CONSERVATIVE
    if use_arc_detection and SKIMAGE_AVAILABLE:
        arcs = detect_arcs_geometric(image, lens_center, einstein_radius_pix, pixel_scale)
        
        for arc in arcs:
            x, y = arc['x'], arc['y']
            dist = arc['distance_from_lens']
            
            # Additional geometric filtering for arcs
            if use_geometric_constraints:
                # Must be in the right distance range
                if dist < einstein_radius_pix * 0.7 or dist > einstein_radius_pix * 2.0:
                    continue
                # Must be significantly elongated (real arcs are very elongated)
                if arc['elongation'] < 2.0:  # Increased from 1.5 to 2.0
                    continue
            
            # Check if matches lenstronomy prediction
            matches_prediction = False
            if lenstronomy_pix:
                for pred_x, pred_y in lenstronomy_pix:
                    pred_dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
                    if pred_dist < einstein_radius_pix * 0.4:  # Tighter matching
                        matches_prediction = True
                        break
            
            candidates.append({
                'x': x,
                'y': y,
                'x_arcsec': (x - numpix // 2) * pixel_scale,
                'y_arcsec': (y - numpix // 2) * pixel_scale,
                'distance_from_lens_pix': dist,
                'distance_from_lens_arcsec': dist * pixel_scale,
                'elongation': arc['elongation'],
                'curvature': arc['curvature'],
                'method': 'arc_detection',
                'matches_lenstronomy': matches_prediction,
                'confidence': 0.85 if matches_prediction else 0.65  # Slightly higher confidence
            })
    
    # Method 3: Use lenstronomy predictions directly (ONLY if no detection found nearby)
    # This is a fallback - only use if we have very few candidates
    if lenstronomy_pix and len(candidates) < 2:
        for pred_x, pred_y in lenstronomy_pix:
            # Check if we already have a candidate near this position
            has_nearby = False
            for cand in candidates:
                dist = np.sqrt((cand['x'] - pred_x)**2 + (cand['y'] - pred_y)**2)
                if dist < einstein_radius_pix * 0.4:  # Tighter matching
                    has_nearby = True
                    break
            
            if not has_nearby:
                # Check if prediction is in reasonable range
                dist = np.sqrt((pred_x - center_x)**2 + (pred_y - center_y)**2)
                if dist < einstein_radius_pix * 0.7 or dist > einstein_radius_pix * 2.0:
                    continue  # Skip if prediction is in excluded region
                
                # Use prediction as candidate (lower confidence)
                candidates.append({
                    'x': pred_x,
                    'y': pred_y,
                    'x_arcsec': (pred_x - numpix // 2) * pixel_scale,
                    'y_arcsec': (pred_y - numpix // 2) * pixel_scale,
                    'distance_from_lens_pix': dist,
                    'distance_from_lens_arcsec': dist * pixel_scale,
                    'method': 'lenstronomy_prediction',
                    'matches_lenstronomy': True,
                    'confidence': 0.3  # Even lower confidence since not detected
                })
    
    # Sort by confidence and distance from lens
    candidates.sort(key=lambda c: (-c['confidence'], c['distance_from_lens_pix']))
    
    # Remove duplicates (candidates very close to each other)
    unique_candidates = []
    for cand in candidates:
        is_duplicate = False
        for existing in unique_candidates:
            dist = np.sqrt((cand['x'] - existing['x'])**2 + (cand['y'] - existing['y'])**2)
            if dist < einstein_radius_pix * 0.2:  # Within 20% of Einstein radius (tighter)
                is_duplicate = True
                # Keep the one with higher confidence
                if cand['confidence'] > existing['confidence']:
                    unique_candidates.remove(existing)
                    unique_candidates.append(cand)
                break
        
        if not is_duplicate:
            unique_candidates.append(cand)
    
    # Final filtering: limit to top N candidates and ensure reasonable distribution
    # Sort by confidence and distance from lens
    unique_candidates.sort(key=lambda c: (-c['confidence'], c['distance_from_lens_pix']))
    
    # Limit to reasonable number (max 4 candidates, typically 2-4 images expected)
    max_candidates = min(4, len(unique_candidates))  # Reduced from 6 to 4
    unique_candidates = unique_candidates[:max_candidates]
    
    # Additional aggressive filtering: remove candidates that are clearly the lens galaxy
    filtered_candidates = []
    for cand in unique_candidates:
        # Skip if too close to center (likely lens galaxy)
        if cand['distance_from_lens_pix'] < einstein_radius_pix * 0.7:
            continue
        
        # Skip if very bright and close (definitely lens galaxy)
        if (cand['distance_from_lens_pix'] < einstein_radius_pix * 0.9 and 
            cand.get('flux', 0) > np.percentile([c.get('flux', 0) for c in unique_candidates if c.get('flux', 0) > 0], 70) if unique_candidates else False):
            continue
        
        # Skip if confidence is too low (likely false positive)
        if cand['confidence'] < 0.4:
            continue
        
        filtered_candidates.append(cand)
    
    # If we filtered out too many, keep at least the top 2
    if len(filtered_candidates) < 2 and len(unique_candidates) >= 2:
        # Keep top 2 by confidence, but still apply distance filter
        top_candidates = sorted(unique_candidates, key=lambda c: -c['confidence'])[:2]
        filtered_candidates = [c for c in top_candidates if c['distance_from_lens_pix'] >= einstein_radius_pix * 0.7]
    
    return filtered_candidates if filtered_candidates else unique_candidates[:min(2, len(unique_candidates))]


def detect_lensed_images_from_catalog(
    image_path: Path,
    lens_id: int,
    time_delay_catalog: Optional[pd.DataFrame] = None,
    lens_center: Optional[Tuple[float, float]] = None,
    pixel_scale: float = 0.03,
    numpix: int = 300
) -> List[Dict]:
    """
    Detect lensed images from a single image file using catalog information.
    
    Args:
        image_path: Path to image file
        lens_id: Lens system ID
        time_delay_catalog: Optional DataFrame with lens information
        lens_center: Optional (x, y) lens center position (default: image center)
        pixel_scale: Pixel scale in arcsec/pixel
        numpix: Image size in pixels
    
    Returns:
        List of detected lensed image candidates
    """
    from PIL import Image
    
    # Load image
    img = Image.open(image_path)
    if img.mode == 'RGB':
        # Convert to grayscale (use average or luminance)
        img_array = np.array(img.convert('L')) / 255.0
    else:
        img_array = np.array(img) / 255.0
    
    # Get lens center (default to image center)
    if lens_center is None:
        lens_center = (numpix // 2, numpix // 2)
    
    # Get Einstein radius from catalog if available
    einstein_radius_pix = None
    lenstronomy_predictions = None
    
    if time_delay_catalog is not None:
        row = time_delay_catalog[time_delay_catalog['lens_id'] == lens_id]
        if len(row) > 0:
            theta_E = float(row.iloc[0].get('theta_E', 1.0))
            einstein_radius_pix = theta_E / pixel_scale
            
            # Get image positions if available
            if 'image_positions_arcsec' in row.columns:
                try:
                    import ast
                    pos_str = row.iloc[0]['image_positions_arcsec']
                    if pd.notna(pos_str):
                        lenstronomy_predictions = ast.literal_eval(str(pos_str))
                except:
                    pass
    
    # Default Einstein radius if not in catalog
    if einstein_radius_pix is None:
        einstein_radius_pix = 1.0 / pixel_scale  # ~1 arcsec default
    
    # Detect lensed images
    candidates = identify_lensed_images_hybrid(
        img_array,
        lens_center,
        einstein_radius_pix,
        lenstronomy_predictions,
        pixel_scale,
        numpix
    )
    
    return candidates
