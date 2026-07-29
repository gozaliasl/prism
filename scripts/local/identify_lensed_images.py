#!/usr/bin/env python3
"""
Unified Lensed Image Detection and Annotation Script

This script provides multiple detection methods for identifying lensed images:
1. Catalog-based: Uses actual positions from time_delay_catalog.csv
2. Hybrid detection: Combines SExtractor-like, geometric arc detection, and lenstronomy
3. Visual detection: Finds elongated, curved structures around Einstein radius
4. ML-enhanced: Uses trained models (if available) with photometry

All methods are consolidated in this single script with a unified interface.
"""

import numpy as np
import pandas as pd
import ast
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy and skimage for shape detection
try:
    from scipy.ndimage import binary_erosion
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Try to import segmentation model
try:
    from prism.ml.segmentation_model import detect_lensed_sources_with_model
    SEGMENTATION_MODEL_AVAILABLE = True
except ImportError:
    SEGMENTATION_MODEL_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import detection functions
try:
    from prism.lensing.detect_lensed_images import (
        identify_lensed_images_hybrid,
        detect_lensed_images_from_catalog
    )
    HYBRID_DETECTION_AVAILABLE = True
except ImportError:
    HYBRID_DETECTION_AVAILABLE = False
    print("[WARNING] Hybrid detection not available")

# Try ML detection
try:
    from prism.ml.ml_arc_detector import detect_lensed_images_ml_enhanced, extract_photometry
    ML_DETECTION_AVAILABLE = True
except ImportError:
    ML_DETECTION_AVAILABLE = False

# Try lenstronomy
try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    from astropy.cosmology import FlatLambdaCDM
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    LENSTRONOMY_AVAILABLE = False


class LensedImageDetector:
    """
    Unified detector for lensed images with multiple methods.
    """
    
    def __init__(self, method='auto'):
        """
        Initialize detector.
        
        Args:
            method: Detection method ('catalog', 'hybrid', 'visual', 'ml', 'auto')
        """
        self.method = method
    
    def detect_from_catalog(
        self,
        image_path: Path,
        lens_id: int,
        catalog: pd.DataFrame,
        pixel_scale: float = 0.03,
        numpix: int = 300
    ) -> list:
        """
        Detect using actual positions from catalog (most accurate if available).
        
        Returns:
            List of detected images with positions
        """
        row = catalog[catalog['lens_id'] == lens_id]
        if len(row) == 0:
            return []
        
        row = row.iloc[0]
        image_positions = []
        
        # Get actual positions from catalog
        if 'image_positions_arcsec' in row and pd.notna(row['image_positions_arcsec']):
            try:
                positions = ast.literal_eval(str(row['image_positions_arcsec']))
                if isinstance(positions, list) and len(positions) > 0:
                    if isinstance(positions[0], (list, tuple)):
                        image_positions = [(float(x), float(y)) for x, y in positions]
            except:
                pass
        
        # Convert to pixel coordinates and validate
        center_pix = numpix // 2
        detected = []
        
        # Load image to validate positions
        img = Image.open(image_path)
        if img.mode == 'RGB':
            img_array = np.array(img.convert('L')) / 255.0
        else:
            img_array = np.array(img) / 255.0
        
        h, w = img_array.shape
        
        for i, (x_arcsec, y_arcsec) in enumerate(image_positions):
            x_pix = center_pix + x_arcsec / pixel_scale
            y_pix = center_pix + y_arcsec / pixel_scale
            
            # Validate position is within image
            if 0 <= x_pix < w and 0 <= y_pix < h:
                # Check if source exists at this position
                y_int, x_int = int(y_pix), int(x_pix)
                if 0 <= y_int < h and 0 <= x_int < w:
                    flux = img_array[y_int, x_int]
                    threshold = np.percentile(img_array, 85)
                    
                    if flux > threshold:
                        detected.append({
                            'x': x_pix,
                            'y': y_pix,
                            'x_arcsec': x_arcsec,
                            'y_arcsec': y_arcsec,
                            'flux': float(flux),
                            'confidence': 0.95,  # High confidence from catalog
                            'method': 'catalog'
                        })
        
        return detected
    
    def detect_hybrid(
        self,
        image_path: Path,
        lens_id: int,
        catalog: pd.DataFrame,
        pixel_scale: float = 0.03,
        numpix: int = 300
    ) -> list:
        """Detect using hybrid method (SExtractor + geometric + lenstronomy)."""
        if not HYBRID_DETECTION_AVAILABLE:
            return []
        
        try:
            candidates = detect_lensed_images_from_catalog(
                image_path, lens_id, catalog, pixel_scale=pixel_scale, numpix=numpix
            )
            return candidates
        except Exception as e:
            print(f"[WARNING] Hybrid detection failed: {e}")
            return []
    
    def detect_visual(
        self,
        image_path: Path,
        lens_id: int,
        catalog: pd.DataFrame,
        pixel_scale: float = 0.03
    ) -> list:
        """
        Detect by finding elongated, curved structures around Einstein radius.
        """
        # Load image
        img = Image.open(image_path)
        if img.mode == 'RGB':
            img_array = np.array(img.convert('L')) / 255.0
        else:
            img_array = np.array(img) / 255.0
        
        h, w = img_array.shape
        
        # Get lens parameters
        row = catalog[catalog['lens_id'] == lens_id]
        if len(row) == 0:
            return []
        
        row = row.iloc[0]
        theta_E = float(row.get('theta_E', 1.0))
        einstein_radius_pix = theta_E / pixel_scale
        
        # Find lens center (brightest source in center region)
        if w > h * 2:  # Panoramic
            lens_center = (w // 6, h // 2)
        else:
            lens_center = (w // 2, h // 2)
        
        center_x, center_y = lens_center
        
        # Use hybrid detection which includes visual methods
        if HYBRID_DETECTION_AVAILABLE:
            return self.detect_hybrid(image_path, lens_id, catalog, pixel_scale, max(h, w))
        else:
            return []
    
    def detect_ml(
        self,
        image_path: Path,
        lens_id: int,
        catalog: pd.DataFrame,
        model_path: Path = None,
        pixel_scale: float = 0.03,
        numpix: int = 300
    ) -> list:
        """Detect using ML model (if available)."""
        if not ML_DETECTION_AVAILABLE:
            return []
        
        try:
            # Load image
            img = Image.open(image_path)
            if img.mode == 'RGB':
                img_array = np.array(img.convert('L')) / 255.0
            else:
                img_array = np.array(img) / 255.0
            
            h, w = img_array.shape
            lens_center = (w // 2, h // 2)
            
            # Get Einstein radius
            row = catalog[catalog['lens_id'] == lens_id]
            if len(row) == 0:
                return []
            
            einstein_radius_pix = float(row.iloc[0].get('theta_E', 1.0)) / pixel_scale
            
            # ML detection
            candidates = detect_lensed_images_ml_enhanced(
                img_array, lens_center, einstein_radius_pix,
                model_path=model_path, use_ml=(model_path is not None),
                use_traditional=True, pixel_scale=pixel_scale, numpix=max(h, w)
            )
            
            return candidates
        except Exception as e:
            print(f"[WARNING] ML detection failed: {e}")
            return []
    
    def detect_segmentation(
        self,
        image_path: Path,
        lens_id: int,
        catalog: pd.DataFrame,
        model_path: Path,
        pixel_scale: float = 0.03
    ) -> list:
        """Detect using trained U-Net segmentation model (most accurate)."""
        if not SEGMENTATION_MODEL_AVAILABLE:
            return []
        
        try:
            # Load image
            img = Image.open(image_path)
            if img.mode == 'RGB':
                img_array = np.array(img.convert('L')) / 255.0
            else:
                img_array = np.array(img.convert('L')) / 255.0
            
            # Detect with model
            mask, detected_sources = detect_lensed_sources_with_model(
                img_array, model_path, patch_size=128, stride=64, threshold=0.5
            )
            
            # Convert to standard format
            candidates = []
            for source in detected_sources:
                candidates.append({
                    'x': source['x'],
                    'y': source['y'],
                    'x_arcsec': (source['x'] - img_array.shape[1] // 2) * pixel_scale,
                    'y_arcsec': (source['y'] - img_array.shape[0] // 2) * pixel_scale,
                    'confidence': source['confidence'],
                    'area': source['area'],
                    'method': 'segmentation_unet',
                    'mask': mask  # Store full mask for annotation
                })
            
            return candidates
        except Exception as e:
            print(f"[WARNING] Segmentation detection failed: {e}")
            return []
    
    def detect(
        self,
        image_path: Path,
        lens_id: int,
        catalog: pd.DataFrame,
        model_path: Path = None,
        pixel_scale: float = 0.03,
        numpix: int = 300
    ) -> list:
        """
        Detect lensed images using the specified method (or auto-select).
        
        Returns:
            List of detected images
        """
        if self.method == 'segmentation' or self.method == 'unet':
            if model_path and model_path.exists():
                return self.detect_segmentation(image_path, lens_id, catalog, model_path, pixel_scale)
            else:
                print("[WARNING] Segmentation model path not provided or not found")
                return []
        elif self.method == 'catalog':
            return self.detect_from_catalog(image_path, lens_id, catalog, pixel_scale, numpix)
        elif self.method == 'hybrid':
            return self.detect_hybrid(image_path, lens_id, catalog, pixel_scale, numpix)
        elif self.method == 'visual':
            return self.detect_visual(image_path, lens_id, catalog, pixel_scale)
        elif self.method == 'ml':
            return self.detect_ml(image_path, lens_id, catalog, model_path, pixel_scale, numpix)
        elif self.method == 'auto':
            # Try segmentation first if model available, then catalog, then hybrid
            if model_path and model_path.exists() and SEGMENTATION_MODEL_AVAILABLE:
                detected = self.detect_segmentation(image_path, lens_id, catalog, model_path, pixel_scale)
                if len(detected) > 0:
                    return detected
            
            detected = self.detect_from_catalog(image_path, lens_id, catalog, pixel_scale, numpix)
            if len(detected) > 0:
                return detected
            
            detected = self.detect_hybrid(image_path, lens_id, catalog, pixel_scale, numpix)
            if len(detected) > 0:
                return detected
            
            return self.detect_visual(image_path, lens_id, catalog, pixel_scale)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def get_source_shape(
    image_array: np.ndarray,
    x: float,
    y: float,
    lens_center: tuple = None,
    einstein_radius_pix: float = None,
    search_radius: float = 15.0
) -> list:
    """
    Get the actual shape/contour of a lensed source at the given position.
    Excludes the lens galaxy (central, bright source).
    
    Returns:
        List of (x, y) coordinates forming a polygon around the source
    """
    try:
        from photutils.segmentation import detect_sources, SourceCatalog
        from photutils.background import Background2D, MedianBackground
    except ImportError:
        return None
    
    try:
        h, w = image_array.shape
        
        # Get lens center if not provided (assume image center)
        if lens_center is None:
            if w > h * 2:  # Panoramic
                lens_center = (w // 6, h // 2)
            else:
                lens_center = (w // 2, h // 2)
        
        center_x, center_y = lens_center
        
        # Detect sources near the position
        bkg_estimator = MedianBackground()
        bkg = Background2D(image_array, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        threshold = bkg.background + 2.0 * bkg.background_rms
        
        segm = detect_sources(image_array, threshold, npixels=5)
        if segm is None or segm.nlabels == 0:
            return None
        
        # Find the source closest to the given position
        cat = SourceCatalog(image_array, segm)
        sources = cat.to_table()
        
        min_dist = float('inf')
        best_source = None
        
        # Find lens galaxy (brightest, most central source)
        lens_flux = 0
        lens_source = None
        for source in sources:
            # Convert astropy quantities to floats if needed
            try:
                sx = float(source['xcentroid'])
                sy = float(source['ycentroid'])
            except (TypeError, ValueError):
                sx, sy = source['xcentroid'], source['ycentroid']
            
            dist_from_center = np.sqrt((sx - center_x)**2 + (sy - center_y)**2)
            try:
                flux = float(source['segment_flux'])
            except (TypeError, ValueError):
                flux = source['segment_flux']
            try:
                area = float(source.get('area', 0))
            except (TypeError, ValueError):
                area = source.get('area', 0)
            
            # Lens galaxy is bright, large, and close to center
            # Use stricter criteria: very close to center AND very bright
            max_dist = 25.0 if einstein_radius_pix is None else min(25.0, float(einstein_radius_pix) * 0.5)
            area_val = float(area)
            if dist_from_center < max_dist and flux > lens_flux and area_val > 100.0:
                lens_flux = flux
                lens_source = source
        
        # Find the source closest to the given position (excluding lens)
        for source in sources:
            # Convert astropy quantities to floats if needed
            try:
                sx = float(source['xcentroid'])
                sy = float(source['ycentroid'])
            except (TypeError, ValueError):
                sx, sy = source['xcentroid'], source['ycentroid']
            
            dist = np.sqrt((sx - x)**2 + (sy - y)**2)
            dist_from_center = np.sqrt((sx - center_x)**2 + (sy - center_y)**2)
            
            # STRICT FILTERING: Only accept sources that are lensed images
            
            # 1. Skip if too close to lens center (definitely lens galaxy)
            if einstein_radius_pix:
                einstein_radius_val = float(einstein_radius_pix)
                if dist_from_center < einstein_radius_val * 0.8:  # Increased threshold
                    continue
                # Must be around Einstein radius (not too far either = field galaxy)
                if dist_from_center > einstein_radius_val * 2.0:  # Reduced threshold
                    continue
            else:
                # Fallback: must be at least 25 pixels from center
                if dist_from_center < 25:
                    continue
            
            # 2. Skip if this is the lens galaxy itself
            if lens_source and source['label'] == lens_source['label']:
                continue
            
            # 3. Skip if too bright (likely lens galaxy or bright field galaxy)
            # Lensed images are typically fainter than the lens
            if lens_source:
                try:
                    source_flux = float(source['segment_flux'])
                except (TypeError, ValueError):
                    source_flux = source['segment_flux']
                if source_flux > float(lens_flux) * 0.4:  # Even more strict: must be < 40% of lens flux
                    continue
            
            # 4. Skip if too large (lensed images are typically compact/elongated, not huge)
            try:
                area = float(source.get('area', 0))
            except (TypeError, ValueError):
                area = source.get('area', 0)
                try:
                    area = float(area.value) if hasattr(area, 'value') else float(area)
                except:
                    area = 0
            if area > 1000:  # Too many pixels, likely field galaxy or lens
                continue
            
            # 5. Must be close to the target position (within search radius)
            if dist < min_dist and dist < search_radius:
                min_dist = dist
                best_source = source
        
        if best_source is None:
            return None
        
        # FINAL VALIDATION: Ensure it's definitely a lensed image, not lens galaxy
        
        # Convert to floats
        try:
            best_x = float(best_source['xcentroid'])
            best_y = float(best_source['ycentroid'])
        except (TypeError, ValueError):
            best_x, best_y = best_source['xcentroid'], best_source['ycentroid']
        
        best_dist_from_center = np.sqrt((best_x - center_x)**2 + (best_y - center_y)**2)
        
        # Must be far enough from center (STRICT: only lensed images, not lens)
        if einstein_radius_pix:
            einstein_radius_val = float(einstein_radius_pix)
            if best_dist_from_center < einstein_radius_val * 0.8:  # Increased from 0.7 to 0.8
                return None  # Too close to center
            # Also check: must be within reasonable range (not too far = field galaxy)
            if best_dist_from_center > einstein_radius_val * 2.0:  # Reduced from 2.5 to 2.0
                return None  # Too far, likely field galaxy
        else:
            if best_dist_from_center < 25:  # Increased from 20 to 25
                return None  # Too close to center
        
        # Must not be the lens galaxy
        if lens_source:
            try:
                lens_x = float(lens_source['xcentroid'])
                lens_y = float(lens_source['ycentroid'])
            except (TypeError, ValueError):
                lens_x, lens_y = lens_source['xcentroid'], lens_source['ycentroid']
            
            lens_dist_from_center = np.sqrt((lens_x - center_x)**2 + (lens_y - center_y)**2)
            
            # If best source is closer to center than lens, skip
            if best_dist_from_center < lens_dist_from_center * 1.2:  # Increased from 1.1
                return None
            
            # If best source is too bright compared to lens, skip
            try:
                best_flux = float(best_source['segment_flux'])
            except (TypeError, ValueError):
                best_flux = best_source['segment_flux']
            if best_flux > float(lens_flux) * 0.4:  # More strict: must be < 40% of lens flux
                return None
        
        # Get the segmentation region for this source
        label_id = int(best_source['label'])  # Ensure it's a Python int
        mask = segm.data == label_id
        
        # Get contour points
        if not SCIPY_AVAILABLE or not SKIMAGE_AVAILABLE:
            return None
        
        # Smooth the mask slightly
        mask_smooth = binary_erosion(mask, iterations=1)
        
        # Find contours
        contours = measure.find_contours(mask_smooth.astype(float), 0.5)
        
        if len(contours) == 0:
            return None
        
        # Use the largest contour
        largest_contour = max(contours, key=len)
        
        # Convert to (x, y) coordinates (contours are in (row, col) format)
        # Ensure all values are Python floats (handle any quantity types)
        polygon = []
        for row, col in largest_contour:
            try:
                x_val = float(col)
                y_val = float(row)
            except (TypeError, ValueError):
                # Handle astropy quantities or other types
                x_val = float(col.value) if hasattr(col, 'value') else float(col)
                y_val = float(row.value) if hasattr(row, 'value') else float(row)
            polygon.append((x_val, y_val))
        
        return polygon
    except Exception as e:
        print(f"[WARNING] Could not extract source shape: {e}")
        return None


def create_annotated_image(
    image_path: Path,
    detected_images: list,
    time_delays: list = None,
    output_path: Path = None,
    pixel_scale: float = 0.03,
    numpix: int = 300,
    catalog: pd.DataFrame = None,
    lens_id: int = None
):
    """
    Create annotated image showing detected lensed images with shapes following source structure.
    Excludes the lens galaxy from annotations.
    
    Args:
        image_path: Path to input image
        detected_images: List of detected images (dicts with x, y, confidence, etc.)
        time_delays: Optional list of time delays for labels
        output_path: Path to save annotated image
        pixel_scale: Pixel scale
        numpix: Image size
        catalog: Optional catalog for getting Einstein radius
        lens_id: Optional lens ID for getting Einstein radius
    """
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Load as array for shape detection
    if img.mode == 'RGB':
        img_array = np.array(img.convert('L')) / 255.0
    else:
        img_array = np.array(img.convert('L')) / 255.0
    
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Get image dimensions
    h, w = annotated.size[1], annotated.size[0]
    
    # Determine lens center
    if w > h * 2:  # Panoramic
        lens_center = (w // 6, h // 2)
    else:
        lens_center = (w // 2, h // 2)
    
    # Get Einstein radius if available
    einstein_radius_pix = None
    if catalog is not None and lens_id is not None:
        row = catalog[catalog['lens_id'] == lens_id]
        if len(row) > 0:
            theta_E = float(row.iloc[0].get('theta_E', 1.0))
            einstein_radius_pix = theta_E / pixel_scale
    
    # Colors
    colors = [
        (255, 100, 100, 255),    # Red
        (100, 255, 100, 255),    # Green
        (100, 100, 255, 255),    # Blue
        (255, 255, 100, 255),    # Yellow
        (255, 100, 255, 255),    # Magenta
        (100, 255, 255, 255),    # Cyan
    ]
    
    # Font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font = ImageFont.load_default()
    
    # Annotate each detected image
    for i, candidate in enumerate(detected_images):
        x, y = candidate['x'], candidate['y']
        confidence = candidate.get('confidence', 0.5)
        method = candidate.get('method', 'unknown')
        flux = candidate.get('flux', None)
        
        # Color by confidence
        if confidence >= 0.8:
            color = colors[0]  # Red - high confidence
        elif confidence >= 0.6:
            color = colors[2]  # Blue - medium confidence
        else:
            color = colors[3]  # Yellow - low confidence
        
        color_rgb = color[:3]
        
        # Check if we have a segmentation mask (from U-Net model)
        if 'mask' in candidate and candidate['mask'] is not None:
            # Use the segmentation mask to get the exact shape
            mask = candidate['mask']
            y_int, x_int = int(y), int(x)
            
            # Find the connected component containing this pixel
            if SKIMAGE_AVAILABLE:
                from skimage import measure
                labeled = measure.label(mask)
                if 0 <= y_int < mask.shape[0] and 0 <= x_int < mask.shape[1]:
                    label_id = labeled[y_int, x_int]
                    if label_id > 0:
                        # Get contour of this region
                        region_mask = labeled == label_id
                        contours = measure.find_contours(region_mask.astype(float), 0.5)
                        if len(contours) > 0:
                            largest_contour = max(contours, key=len)
                            polygon = [(float(col), float(row)) for row, col in largest_contour]
                        else:
                            polygon = None
                    else:
                        polygon = None
                else:
                    polygon = None
            else:
                polygon = None
        else:
            # Get actual source shape (excluding lens galaxy)
            polygon = get_source_shape(
                img_array, x, y, 
                lens_center=lens_center,
                einstein_radius_pix=einstein_radius_pix
            )
        
        if polygon and len(polygon) > 2:
            # Draw polygon following source structure
            # White outline (thicker, outer)
            draw.polygon(polygon, outline=(255, 255, 255, 255), width=4)
            # Colored outline (inner)
            draw.polygon(polygon, outline=color_rgb, width=2)
            
            # Get bounding box for label placement
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
        else:
            # Fallback to circle if shape detection fails
            radius = int(8 + confidence * 7)
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, outline=(255, 255, 255, 255), width=3)
            draw.ellipse([x - radius + 2, y - radius + 2, x + radius - 2, y + radius - 2],
                         outline=color_rgb, width=2)
            x_center, y_center = x, y
            x_max = x + radius
            y_min = y - radius
        
        # Small crosshair at centroid
        cross_size = 5
        draw.line([x - cross_size, y, x + cross_size, y], fill=(255, 255, 255, 255), width=2)
        draw.line([x, y - cross_size, x, y + cross_size], fill=(255, 255, 255, 255), width=2)
        
        # Label
        label_parts = [f"#{i+1}", f"{confidence:.2f}"]
        if time_delays and i < len(time_delays):
            label_parts.append(f"{time_delays[i]:.1f}d")
        if flux is not None:
            label_parts.append(f"F={flux:.1e}")
        label = "\n".join(label_parts)
        
        # Place label near the source (to the right or left)
        label_x = x_max + 8 if polygon else x_center + 15
        label_y = y_center - 20
        if label_x > w - 100:
            label_x = (x_min - 100) if polygon else (x_center - 100)
        if label_y < 10:
            label_y = (y_max + 8) if polygon else (y_center + 15)
        
        # Text shadow
        draw.text((label_x + 1, label_y + 1), label, fill=(0, 0, 0, 255), font=font)
        draw.text((label_x, label_y), label, fill=color_rgb, font=font)
    
    # Save
    if output_path:
        annotated.save(output_path)
        print(f"✅ Saved annotated image: {output_path}")


def parse_time_delays(delays_str):
    """Parse time delays from string representation."""
    try:
        return ast.literal_eval(delays_str)
    except:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Unified lensed image detection and annotation"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Simulation output directory'
    )
    parser.add_argument(
        '--lens-id',
        type=int,
        help='Specific lens ID (if not provided, processes all)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=0,
        help='Epoch to process'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='auto',
        choices=['catalog', 'hybrid', 'visual', 'ml', 'segmentation', 'unet', 'auto'],
        help='Detection method (default: auto). segmentation/unet uses trained U-Net model.'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to ML model (for ml method)'
    )
    parser.add_argument(
        '--output-dir-annotated',
        type=str,
        default=None,
        help='Directory for annotated images (default: output_dir/annotated)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    # Read catalog - try current directory first, then parent if in subdirectory
    catalog_path = output_dir / 'time_delay_catalog.csv'
    if not catalog_path.exists():
        # Check if we're in a subdirectory (like annotated_ml, annotated, etc.)
        parent_dir = output_dir.parent
        parent_catalog = parent_dir / 'time_delay_catalog.csv'
        if parent_catalog.exists():
            print(f"⚠️  Note: Found catalog in parent directory, using: {parent_dir}")
            output_dir = parent_dir
            catalog_path = parent_catalog
        else:
            print(f"❌ Error: Time delay catalog not found in:")
            print(f"   - {catalog_path}")
            print(f"   - {parent_catalog}")
            print(f"\n💡 Tip: Use the main simulation output directory (not subdirectories)")
            sys.exit(1)
    
    catalog = pd.read_csv(catalog_path)
    
    # Filter by lens ID
    if args.lens_id is not None:
        catalog = catalog[catalog['lens_id'] == args.lens_id]
        if len(catalog) == 0:
            print(f"❌ Error: Lens ID {args.lens_id} not found")
            sys.exit(1)
    
    # Image directory
    jpg_dir = output_dir / 'jpg_rgb'
    if not jpg_dir.exists():
        print(f"❌ Error: RGB images directory not found: {jpg_dir}")
        sys.exit(1)
    
    # Output directory
    if args.output_dir_annotated:
        annotated_dir = Path(args.output_dir_annotated)
    else:
        annotated_dir = output_dir / 'annotated'
    annotated_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    detector = LensedImageDetector(method=args.method)
    model_path = Path(args.model_path) if args.model_path else None
    
    print(f"📊 Processing {len(catalog)} lens system(s) with method: {args.method}")
    
    processed = 0
    for idx, row in catalog.iterrows():
        lens_id = int(row['lens_id'])
        
        # Find image file
        image_file = jpg_dir / f"cosmos_lens_{lens_id:06d}_epoch{args.epoch:02d}.jpg"
        if not image_file.exists():
            image_file = jpg_dir / f"cosmos_lens_{lens_id:03d}_epoch{args.epoch:02d}.jpg"
        
        if not image_file.exists():
            print(f"⚠️  Skipping lens {lens_id}: image not found")
            continue
        
        # Detect
        try:
            detected = detector.detect(
                image_file, lens_id, catalog, model_path=model_path
            )
        except Exception as e:
            print(f"⚠️  Error detecting lens {lens_id}: {e}")
            continue
        
        if len(detected) == 0:
            print(f"⚠️  Lens {lens_id}: No images detected")
            continue
        
        print(f"  Lens {lens_id}: Detected {len(detected)} images")
        
        # Get time delays
        time_delays = parse_time_delays(row.get('time_delays_days', '[]'))
        
        # Create annotated image
        output_file = annotated_dir / f"cosmos_lens_{lens_id:06d}_epoch{args.epoch:02d}_annotated.png"
        create_annotated_image(
            image_file, detected, time_delays, output_file,
            catalog=catalog, lens_id=lens_id
        )
        
        processed += 1
    
    print(f"\n✅ Processed {processed} lens system(s)")
    print(f"📁 Annotated images saved to: {annotated_dir}")


if __name__ == '__main__':
    main()
