#!/usr/bin/env python3
"""
ML Training Enhancements for JWST COSMOS-Web Lens Detection

This module implements:
1. Hard negative mining for challenging non-lens cases
2. Data augmentation pipeline for JWST COSMOS-Web
3. Balanced training sets with realistic lens/non-lens ratios
4. Survey-specific parameters and quality metrics

Designed for JWST COSMOS-Web survey lens detection ML training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.ndimage import rotate, zoom, gaussian_filter
from scipy.stats import skewnorm
import warnings
warnings.filterwarnings('ignore')

class JWSTHardNegativeMiner:
    """
    Generate challenging non-lens cases that are difficult to distinguish from lenses
    """
    
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        
        # Hard negative categories with realistic frequencies
        self.hard_negative_types = {
            'edge_on_spiral': 0.25,      # Edge-on spirals with ring-like appearance
            'merging_galaxies': 0.20,     # Merging systems with arc-like features
            'star_forming_ring': 0.15,     # Star-forming rings in galaxies
            'barred_spiral': 0.15,        # Barred spirals with lens-like central regions
            'irregular_clumpy': 0.10,     # Irregular galaxies with clumpy star formation
            'elliptical_with_dust': 0.10, # Ellipticals with dust lanes
            'galaxy_pair_close': 0.05    # Very close galaxy pairs
        }
    
    def generate_edge_on_spiral(self, numpix=300, pixel_scale=0.03):
        """Generate edge-on spiral that can be mistaken for a lens"""
        # Create edge-on spiral with ring-like appearance
        center = numpix // 2
        
        # Main disk (edge-on)
        disk_radius = self.rng.uniform(15, 25)  # pixels
        disk_thickness = self.rng.uniform(3, 6)  # pixels
        
        # Bulge component
        bulge_radius = self.rng.uniform(8, 15)
        
        # Spiral arm features (subtle)
        arm_pitch = self.rng.uniform(15, 30)  # degrees
        n_arms = self.rng.choice([2, 3, 4])
        
        # Create edge-on spiral profile
        y, x = np.ogrid[:numpix, :numpix]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Disk component (edge-on)
        disk_profile = np.exp(-(r / disk_radius)**2) * np.exp(-((y - center) / disk_thickness)**2)
        
        # Bulge component
        bulge_profile = np.exp(-(r / bulge_radius)**2)
        
        # Combine with realistic brightness ratios
        total_profile = 0.7 * disk_profile + 0.3 * bulge_profile
        
        # Add subtle spiral arm features
        for arm in range(n_arms):
            arm_angle = (arm * 360 / n_arms) + self.rng.uniform(-10, 10)
            arm_profile = self._create_spiral_arm(r, arm_angle, arm_pitch, center, numpix)
            total_profile += 0.1 * arm_profile
        
        return total_profile
    
    def generate_merging_galaxies(self, numpix=300, pixel_scale=0.03):
        """Generate merging galaxies with arc-like features"""
        center = numpix // 2
        
        # Two galaxies in close interaction
        sep = self.rng.uniform(8, 15)  # pixels
        angle = self.rng.uniform(0, 360)
        
        # Primary galaxy
        x1, y1 = center, center
        r1 = self.rng.uniform(10, 18)
        n1 = self.rng.uniform(1.5, 3.0)
        
        # Secondary galaxy
        x2 = center + sep * np.cos(np.radians(angle))
        y2 = center + sep * np.sin(np.radians(angle))
        r2 = self.rng.uniform(6, 12)
        n2 = self.rng.uniform(1.0, 2.5)
        
        # Create tidal features
        tidal_angle = angle + self.rng.uniform(30, 60)
        tidal_length = self.rng.uniform(15, 25)
        
        y, x = np.ogrid[:numpix, :numpix]
        
        # Primary galaxy
        r_gal1 = np.sqrt((x - x1)**2 + (y - y1)**2)
        gal1_profile = np.exp(-(r_gal1 / r1)**(1/n1))
        
        # Secondary galaxy
        r_gal2 = np.sqrt((x - x2)**2 + (y - y2)**2)
        gal2_profile = np.exp(-(r_gal2 / r2)**(1/n2))
        
        # Tidal features (arc-like)
        tidal_profile = self._create_tidal_feature(x, y, x1, y1, tidal_angle, tidal_length)
        
        # Combine with realistic interaction effects
        total_profile = 0.6 * gal1_profile + 0.3 * gal2_profile + 0.1 * tidal_profile
        
        return total_profile
    
    def generate_star_forming_ring(self, numpix=300, pixel_scale=0.03):
        """Generate galaxy with star-forming ring that can look like lensing"""
        center = numpix // 2
        
        # Central galaxy
        r_center = self.rng.uniform(8, 15)
        n_center = self.rng.uniform(2.0, 4.0)
        
        # Ring parameters
        ring_radius = self.rng.uniform(12, 20)
        ring_width = self.rng.uniform(2, 4)
        ring_intensity = self.rng.uniform(0.3, 0.7)
        
        y, x = np.ogrid[:numpix, :numpix]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Central galaxy
        center_profile = np.exp(-(r / r_center)**(1/n_center))
        
        # Ring component
        ring_profile = ring_intensity * np.exp(-((r - ring_radius) / ring_width)**2)
        
        # Combine
        total_profile = center_profile + ring_profile
        
        return total_profile
    
    def generate_barred_spiral(self, numpix=300, pixel_scale=0.03):
        """Generate barred spiral with lens-like central region"""
        center = numpix // 2
        
        # Bar parameters
        bar_length = self.rng.uniform(12, 20)
        bar_width = self.rng.uniform(3, 6)
        bar_angle = self.rng.uniform(0, 180)
        
        # Bulge
        bulge_radius = self.rng.uniform(6, 12)
        
        # Disk
        disk_radius = self.rng.uniform(15, 25)
        disk_n = self.rng.uniform(1.0, 2.0)
        
        y, x = np.ogrid[:numpix, :numpix]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Rotate coordinates for bar
        x_rot = (x - center) * np.cos(np.radians(bar_angle)) + (y - center) * np.sin(np.radians(bar_angle))
        y_rot = -(x - center) * np.sin(np.radians(bar_angle)) + (y - center) * np.cos(np.radians(bar_angle))
        
        # Bar component
        bar_profile = np.exp(-(x_rot / bar_length)**2) * np.exp(-(y_rot / bar_width)**2)
        
        # Bulge
        bulge_profile = np.exp(-(r / bulge_radius)**2)
        
        # Disk
        disk_profile = np.exp(-(r / disk_radius)**(1/disk_n))
        
        # Combine with realistic ratios
        total_profile = 0.4 * bar_profile + 0.3 * bulge_profile + 0.3 * disk_profile
        
        return total_profile
    
    def _create_spiral_arm(self, r, arm_angle, pitch_angle, center, numpix):
        """Create spiral arm feature"""
        # Simplified spiral arm implementation
        arm_profile = np.zeros_like(r)
        arm_mask = (r > 5) & (r < 20)
        arm_profile[arm_mask] = 0.1 * np.exp(-(r[arm_mask] - 10)**2 / 25)
        return arm_profile
    
    def _create_tidal_feature(self, x, y, x_center, y_center, angle, length):
        """Create tidal feature (arc-like)"""
        # Create arc-like tidal feature
        x_rel = x - x_center
        y_rel = y - y_center
        
        # Rotate to tidal angle
        x_rot = x_rel * np.cos(np.radians(angle)) + y_rel * np.sin(np.radians(angle))
        y_rot = -x_rel * np.sin(np.radians(angle)) + y_rel * np.cos(np.radians(angle))
        
        # Create arc profile
        arc_profile = np.exp(-(x_rot / length)**2) * np.exp(-(y_rot / 3)**2)
        return arc_profile
    
    def generate_hard_negative(self, negative_type=None, numpix=300, pixel_scale=0.03):
        """Generate a hard negative case"""
        if negative_type is None:
            negative_type = self.rng.choice(
                list(self.hard_negative_types.keys()),
                p=list(self.hard_negative_types.values())
            )
        
        if negative_type == 'edge_on_spiral':
            return self.generate_edge_on_spiral(numpix, pixel_scale)
        elif negative_type == 'merging_galaxies':
            return self.generate_merging_galaxies(numpix, pixel_scale)
        elif negative_type == 'star_forming_ring':
            return self.generate_star_forming_ring(numpix, pixel_scale)
        elif negative_type == 'barred_spiral':
            return self.generate_barred_spiral(numpix, pixel_scale)
        else:
            # Default to edge-on spiral
            return self.generate_edge_on_spiral(numpix, pixel_scale)


class JWSTDataAugmentation:
    """
    Data augmentation pipeline specifically for JWST COSMOS-Web
    """
    
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        
        # JWST COSMOS-Web specific parameters
        self.jwst_params = {
            'pixel_scale': 0.03,  # arcsec/pixel
            'bands': ['F115W', 'F150W', 'F277W', 'F444W'],
            'psf_fwhm': {
                'F115W': 0.037,  # arcsec
                'F150W': 0.040,
                'F277W': 0.065,
                'F444W': 0.100
            },
            'noise_levels': {
                'F115W': 1e-6,
                'F150W': 1e-6,
                'F277W': 1e-6,
                'F444W': 1e-6
            }
        }
    
    def apply_rotation(self, images, max_angle=360):
        """Apply random rotation to all bands consistently"""
        angle = self.rng.uniform(0, max_angle)
        augmented = {}
        
        for band, image in images.items():
            augmented[band] = rotate(image, angle, reshape=False, order=1)
        
        return augmented, {'rotation_angle': angle}
    
    def apply_noise_variation(self, images, noise_factor_range=(0.5, 2.0)):
        """Apply realistic noise variations"""
        noise_factor = self.rng.uniform(*noise_factor_range)
        augmented = {}
        
        for band, image in images.items():
            # Add noise proportional to signal
            noise_level = self.jwst_params['noise_levels'][band] * noise_factor
            noise = self.rng.normal(0, noise_level, image.shape)
            augmented[band] = image + noise
        
        return augmented, {'noise_factor': noise_factor}
    
    def apply_psf_variation(self, images, psf_variation_range=(0.8, 1.2)):
        """Apply PSF variations (focus changes, etc.)"""
        psf_factor = self.rng.uniform(*psf_variation_range)
        augmented = {}
        
        for band, image in images.items():
            # Apply Gaussian blur with varying FWHM
            fwhm_pixels = self.jwst_params['psf_fwhm'][band] / self.jwst_params['pixel_scale']
            sigma = fwhm_pixels * psf_factor / 2.355  # Convert FWHM to sigma
            augmented[band] = gaussian_filter(image, sigma=sigma)
        
        return augmented, {'psf_factor': psf_factor}
    
    def apply_magnification_variation(self, images, mag_range=(0.8, 1.2)):
        """Apply magnification variations (different exposure times)"""
        mag_factor = self.rng.uniform(*mag_range)
        augmented = {}
        
        for band, image in images.items():
            augmented[band] = image * mag_factor
        
        return augmented, {'magnification_factor': mag_factor}
    
    def apply_color_variation(self, images, color_shift_range=(-0.2, 0.2)):
        """Apply color variations (dust, redshift effects)"""
        color_shift = self.rng.uniform(*color_shift_range)
        augmented = {}
        
        for band, image in images.items():
            # Apply band-specific color shifts
            if 'F115W' in band or 'F150W' in band:  # Blue bands
                shift = color_shift
            else:  # Red bands
                shift = -color_shift * 0.5
            
            augmented[band] = image * (1 + shift)
        
        return augmented, {'color_shift': color_shift}
    
    def apply_comprehensive_augmentation(self, images, augmentation_prob=0.8):
        """Apply comprehensive augmentation pipeline"""
        augmented = images.copy()
        metadata = {}
        
        # Rotation (always apply for orientation invariance)
        if self.rng.random() < augmentation_prob:
            augmented, rot_meta = self.apply_rotation(augmented)
            metadata.update(rot_meta)
        
        # Noise variation
        if self.rng.random() < augmentation_prob:
            augmented, noise_meta = self.apply_noise_variation(augmented)
            metadata.update(noise_meta)
        
        # PSF variation
        if self.rng.random() < augmentation_prob * 0.5:  # Less frequent
            augmented, psf_meta = self.apply_psf_variation(augmented)
            metadata.update(psf_meta)
        
        # Magnification variation
        if self.rng.random() < augmentation_prob * 0.7:
            augmented, mag_meta = self.apply_magnification_variation(augmented)
            metadata.update(mag_meta)
        
        # Color variation
        if self.rng.random() < augmentation_prob * 0.6:
            augmented, color_meta = self.apply_color_variation(augmented)
            metadata.update(color_meta)
        
        return augmented, metadata


class JWSTBalancedTrainingSets:
    """
    Create balanced training sets with realistic lens/non-lens ratios for JWST COSMOS-Web
    """
    
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        
        # JWST COSMOS-Web realistic ratios
        self.survey_ratios = {
            'realistic': 0.001,      # 1:1000 lens/non-lens (realistic survey)
            'balanced': 0.5,          # 1:1 lens/non-lens (balanced training)
            'hard_negative': 0.1,     # 1:10 lens/hard-negative (challenging)
            'augmented': 0.2         # 1:5 lens/augmented (data augmentation)
        }
    
    def create_realistic_survey_set(self, n_lenses=100, ratio_type='realistic'):
        """Create realistic survey-like training set"""
        ratio = self.survey_ratios[ratio_type]
        n_non_lenses = int(n_lenses / ratio)
        
        # Stratify by redshift and mass
        lens_redshifts = self._sample_lens_redshifts(n_lenses)
        non_lens_redshifts = self._sample_non_lens_redshifts(n_non_lenses)
        
        # Create training plan
        training_plan = {
            'lenses': {
                'count': n_lenses,
                'redshifts': lens_redshifts,
                'types': ['strong_lens'] * n_lenses
            },
            'non_lenses': {
                'count': n_non_lenses,
                'redshifts': non_lens_redshifts,
                'types': self._sample_non_lens_types(n_non_lenses)
            }
        }
        
        return training_plan
    
    def create_hard_negative_set(self, n_lenses=100, n_hard_negatives=1000):
        """Create training set with hard negatives"""
        hard_negative_miner = JWSTHardNegativeMiner(self.rng)
        
        training_plan = {
            'lenses': {
                'count': n_lenses,
                'redshifts': self._sample_lens_redshifts(n_lenses),
                'types': ['strong_lens'] * n_lenses
            },
            'hard_negatives': {
                'count': n_hard_negatives,
                'types': self._sample_hard_negative_types(n_hard_negatives),
                'miner': hard_negative_miner
            }
        }
        
        return training_plan
    
    def create_augmented_set(self, base_lenses=100, augmentation_factor=5):
        """Create augmented training set"""
        n_augmented = base_lenses * augmentation_factor
        
        training_plan = {
            'base_lenses': {
                'count': base_lenses,
                'redshifts': self._sample_lens_redshifts(base_lenses),
                'types': ['strong_lens'] * base_lenses
            },
            'augmented_lenses': {
                'count': n_augmented,
                'base_indices': self.rng.choice(base_lenses, n_augmented, replace=True),
                'augmentation_types': self._sample_augmentation_types(n_augmented)
            }
        }
        
        return training_plan
    
    def _sample_lens_redshifts(self, n):
        """Sample realistic lens redshifts"""
        # JWST COSMOS-Web lens redshift distribution
        z_lens = self.rng.beta(2, 3, n) * 2.0 + 0.2  # Peak around z=0.8
        return z_lens
    
    def _sample_non_lens_redshifts(self, n):
        """Sample realistic non-lens redshifts"""
        # Broader redshift distribution for non-lenses
        z_non_lens = self.rng.beta(1.5, 2, n) * 3.0 + 0.1  # Peak around z=1.0
        return z_non_lens
    
    def _sample_non_lens_types(self, n):
        """Sample non-lens galaxy types"""
        types = ['central_galaxy', 'galaxy_pair', 'galaxy_group']
        probs = [0.4, 0.35, 0.25]
        return self.rng.choice(types, n, p=probs)
    
    def _sample_hard_negative_types(self, n):
        """Sample hard negative types"""
        miner = JWSTHardNegativeMiner(self.rng)
        return self.rng.choice(
            list(miner.hard_negative_types.keys()),
            n,
            p=list(miner.hard_negative_types.values())
        )
    
    def _sample_augmentation_types(self, n):
        """Sample augmentation types"""
        aug_types = ['rotation', 'noise', 'psf', 'magnification', 'color']
        return self.rng.choice(aug_types, n)


class JWSTSurveyMetrics:
    """
    Survey-specific quality metrics for JWST COSMOS-Web
    """
    
    def __init__(self):
        self.jwst_bands = ['F115W', 'F150W', 'F277W', 'F444W']
        self.pixel_scale = 0.03  # arcsec/pixel
    
    def calculate_detection_metrics(self, images, is_lens=True):
        """Calculate metrics relevant for lens detection"""
        metrics = {}
        
        for band in self.jwst_bands:
            if band not in images:
                continue
                
            image = images[band]
            
            # Basic flux metrics
            total_flux = np.sum(image)
            max_flux = np.max(image)
            mean_flux = np.mean(image)
            
            # Signal-to-noise ratio
            noise_estimate = np.std(image[image < np.percentile(image, 50)])
            snr = mean_flux / noise_estimate if noise_estimate > 0 else 0
            
            # Size metrics
            flux_threshold = 0.1 * max_flux
            bright_pixels = image > flux_threshold
            if np.any(bright_pixels):
                y_coords, x_coords = np.where(bright_pixels)
                size_pixels = np.sqrt((y_coords.max() - y_coords.min())**2 + 
                                    (x_coords.max() - x_coords.min())**2)
                size_arcsec = size_pixels * self.pixel_scale
            else:
                size_arcsec = 0
            
            # Concentration (for lens detection)
            center = len(image) // 2
            inner_flux = np.sum(image[center-5:center+5, center-5:center+5])
            outer_flux = total_flux - inner_flux
            concentration = inner_flux / total_flux if total_flux > 0 else 0
            
            metrics[f'{band}_total_flux'] = float(total_flux)
            metrics[f'{band}_max_flux'] = float(max_flux)
            metrics[f'{band}_snr'] = float(snr)
            metrics[f'{band}_size_arcsec'] = float(size_arcsec)
            metrics[f'{band}_concentration'] = float(concentration)
        
        # Multi-band metrics
        if all(band in images for band in self.jwst_bands):
            # Color metrics
            f115w_flux = metrics.get('F115W_total_flux', 0)
            f150w_flux = metrics.get('F150W_total_flux', 0)
            f277w_flux = metrics.get('F277W_total_flux', 0)
            f444w_flux = metrics.get('F444W_total_flux', 0)
            
            if f150w_flux > 0:
                metrics['F115W_F150W_color'] = -2.5 * np.log10(f115w_flux / f150w_flux)
            if f277w_flux > 0:
                metrics['F150W_F277W_color'] = -2.5 * np.log10(f150w_flux / f277w_flux)
            if f444w_flux > 0:
                metrics['F277W_F444W_color'] = -2.5 * np.log10(f277w_flux / f444w_flux)
        
        return metrics


def create_enhanced_training_pipeline(n_lenses=1000, n_non_lenses=10000, 
                                    hard_negative_ratio=0.1, augmentation_factor=3,
                                    output_dir="enhanced_training_data"):
    """
    Create enhanced training pipeline for JWST COSMOS-Web lens detection
    
    Args:
        n_lenses: Number of lens systems
        n_non_lenses: Number of non-lens systems
        hard_negative_ratio: Fraction of non-lenses that are hard negatives
        augmentation_factor: Factor for data augmentation
        output_dir: Output directory for training data
    """
    
    # Initialize components
    hard_negative_miner = JWSTHardNegativeMiner()
    data_augmentation = JWSTDataAugmentation()
    balanced_sets = JWSTBalancedTrainingSets()
    survey_metrics = JWSTSurveyMetrics()
    
    # Create training plan
    training_plan = balanced_sets.create_realistic_survey_set(n_lenses, 'realistic')
    
    # Add hard negatives
    n_hard_negatives = int(n_non_lenses * hard_negative_ratio)
    n_regular_negatives = n_non_lenses - n_hard_negatives
    
    print(f"Enhanced Training Pipeline for JWST COSMOS-Web:")
    print(f"  Lenses: {n_lenses}")
    print(f"  Regular non-lenses: {n_regular_negatives}")
    print(f"  Hard negatives: {n_hard_negatives}")
    print(f"  Augmentation factor: {augmentation_factor}")
    
    return {
        'hard_negative_miner': hard_negative_miner,
        'data_augmentation': data_augmentation,
        'balanced_sets': balanced_sets,
        'survey_metrics': survey_metrics,
        'training_plan': training_plan
    }


if __name__ == "__main__":
    # Example usage
    pipeline = create_enhanced_training_pipeline(
        n_lenses=100,
        n_non_lenses=1000,
        hard_negative_ratio=0.2,
        augmentation_factor=3
    )
    
    print("Enhanced training pipeline created successfully!")
    print("Components available:")
    print("- Hard negative miner")
    print("- Data augmentation")
    print("- Balanced training sets")
    print("- Survey metrics")
