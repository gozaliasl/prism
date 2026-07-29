#!/usr/bin/env python3
"""
Advanced Lens Features for JWST COSMOS-Web Lens Detection

Production-active class
-----------------------
RealisticMassProfiles  — NFW/SIE profile construction and binary-lens
                         parameterization; imported by jwst_lens_simulator.py.

Quarantined (unused in production)
-----------------------------------
MultiSourceLensingSystem, SurveySpecificNoiseModels, QualityAssessmentMetrics,
create_advanced_lens_system, create_summary_statistics,
add_jwst_diffraction_spikes, add_jwst_artifacts_realistic.

These classes/functions are NOT imported anywhere in the production pipeline.
Several contain physically incorrect formulas (wrong ρ_crit, missing D_ls/D_s
factors) that have been fixed in the production paths above.  They are
preserved at the bottom of this file for historical reference only and should
not be used for science-grade outputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import lenstronomy for advanced lensing
try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Profiles.nfw import NFW
    from lenstronomy.LensModel.Profiles.sis import SIS
    from lenstronomy.LensModel.Profiles.sersic import Sersic
    LENSTRONOMY_AVAILABLE = True
except ImportError:
    LENSTRONOMY_AVAILABLE = False
    print("Warning: lenstronomy not available, using simplified models")

class MultiSourceLensingSystem:
    """
    Multi-source lensing system implementation for realistic lens detection
    DEPRECATED — not imported in the production pipeline; see quarantine notice
    below RealisticMassProfiles.
    """
    
    def __init__(self, rng=None):
        # Handle both old RandomState and new Generator APIs
        if rng is None:
            self.rng = np.random.default_rng(42)
        elif isinstance(rng, np.random.RandomState):
            # Convert old RandomState to new Generator
            seed = int(rng.randint(0, 2**31))
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        
        # Multi-source lensing parameters
        self.multi_source_params = {
            'max_sources': 5,  # Maximum number of sources per lens
            'source_redshift_range': (1.0, 6.0),  # Source redshift range
            'source_separation_range': (0.5, 3.0),  # Arcsec
            'source_magnitude_range': (20.0, 26.0),  # Source magnitude range
            'source_size_range': (0.05, 0.3),  # Source size range (arcsec)
            'source_type_distribution': {
                'star_forming': 0.6,  # Star-forming galaxies
                'passive': 0.3,       # Passive galaxies
                'quasar': 0.1        # Quasars
            }
        }
    
    def generate_multi_source_system(self, lens_redshift, lens_mass_log10, 
                                   einstein_radius, numpix=300, pixel_scale=0.03):
        """
        Generate multi-source lensing system
        
        Args:
            lens_redshift: Lens redshift
            lens_mass_log10: Lens mass (log10 M_sun)
            einstein_radius: Einstein radius (arcsec)
            numpix: Image size in pixels
            pixel_scale: Arcsec per pixel
        
        Returns:
            dict: Multi-source lensing system parameters
        """
        
        # Determine number of sources (1-5, weighted toward fewer sources)
        n_sources = self.rng.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        sources = []
        for i in range(n_sources):
            source = self._generate_single_source(
                lens_redshift, lens_mass_log10, einstein_radius, 
                numpix, pixel_scale, source_id=i
            )
            sources.append(source)
        
        # Calculate lensing properties
        lensing_properties = self._calculate_lensing_properties(
            sources, lens_redshift, einstein_radius
        )
        
        return {
            'n_sources': n_sources,
            'sources': sources,
            'lensing_properties': lensing_properties,
            'system_complexity': self._assess_system_complexity(sources)
        }
    
    def _generate_single_source(self, lens_redshift, lens_mass_log10, einstein_radius,
                               numpix, pixel_scale, source_id):
        """Generate a single source with realistic properties"""
        
        # Source redshift (must be higher than lens)
        source_redshift = self.rng.uniform(
            lens_redshift + 0.3, 
            self.multi_source_params['source_redshift_range'][1]
        )
        
        # Source type
        source_type = self.rng.choice(
            list(self.multi_source_params['source_type_distribution'].keys()),
            p=list(self.multi_source_params['source_type_distribution'].values())
        )
        
        # Source position (within Einstein radius)
        max_sep = einstein_radius * 0.8  # Stay within Einstein radius
        sep = self.rng.uniform(0.1, max_sep)
        angle = self.rng.uniform(0, 2 * np.pi)
        
        source_x = sep * np.cos(angle)
        source_y = sep * np.sin(angle)
        
        # Source properties
        source_magnitude = self.rng.uniform(*self.multi_source_params['source_magnitude_range'])
        source_size = self.rng.uniform(*self.multi_source_params['source_size_range'])
        
        # Source morphology based on type
        if source_type == 'star_forming':
            n_sersic = self.rng.uniform(0.8, 2.0)  # Disk-like
            axis_ratio = self.rng.uniform(0.3, 0.8)  # More elongated
        elif source_type == 'passive':
            n_sersic = self.rng.uniform(2.0, 4.0)  # Bulge-like
            axis_ratio = self.rng.uniform(0.6, 1.0)  # More round
        else:  # quasar
            n_sersic = self.rng.uniform(1.0, 2.0)  # Intermediate
            axis_ratio = self.rng.uniform(0.4, 0.9)  # Variable
        
        # Calculate magnification (simplified)
        magnification = self._calculate_magnification(
            source_x, source_y, einstein_radius
        )
        
        return {
            'source_id': source_id,
            'redshift': source_redshift,
            'type': source_type,
            'position': (source_x, source_y),
            'magnitude': source_magnitude,
            'size': source_size,
            'n_sersic': n_sersic,
            'axis_ratio': axis_ratio,
            'magnification': magnification,
            'lensed_magnitude': source_magnitude - 2.5 * np.log10(magnification)
        }
    
    def _calculate_magnification(self, source_x, source_y, einstein_radius):
        """Calculate magnification for source position"""
        # Simplified magnification calculation
        r = np.sqrt(source_x**2 + source_y**2)
        
        if r < einstein_radius * 0.1:
            # Very close to center - high magnification
            return self.rng.uniform(5.0, 20.0)
        elif r < einstein_radius * 0.5:
            # Within Einstein radius - moderate magnification
            return self.rng.uniform(2.0, 8.0)
        else:
            # Outside Einstein radius - low magnification
            return self.rng.uniform(1.0, 3.0)
    
    def _calculate_lensing_properties(self, sources, lens_redshift, einstein_radius):
        """Calculate overall lensing properties"""
        
        total_magnification = sum(s['magnification'] for s in sources)
        avg_magnification = total_magnification / len(sources)
        
        # Calculate image separations
        image_separations = []
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                sep = np.sqrt(
                    (source1['position'][0] - source2['position'][0])**2 +
                    (source1['position'][1] - source2['position'][1])**2
                )
                image_separations.append(sep)
        
        return {
            'total_magnification': total_magnification,
            'average_magnification': avg_magnification,
            'max_magnification': max(s['magnification'] for s in sources),
            'image_separations': image_separations,
            'max_separation': max(image_separations) if image_separations else 0,
            'lensing_efficiency': self._calculate_lensing_efficiency(sources, einstein_radius)
        }
    
    def _calculate_lensing_efficiency(self, sources, einstein_radius):
        """Calculate lensing efficiency (fraction of sources within Einstein radius)"""
        sources_within_einstein = sum(
            1 for s in sources 
            if np.sqrt(s['position'][0]**2 + s['position'][1]**2) < einstein_radius
        )
        return sources_within_einstein / len(sources)
    
    def _assess_system_complexity(self, sources):
        """Assess system complexity for ML training"""
        n_sources = len(sources)
        total_magnification = sum(s['magnification'] for s in sources)
        
        if n_sources == 1:
            complexity = 'simple'
        elif n_sources <= 3 and total_magnification < 10:
            complexity = 'moderate'
        else:
            complexity = 'complex'
        
        return {
            'complexity_level': complexity,
            'n_sources': n_sources,
            'total_magnification': total_magnification,
            'difficulty_score': n_sources * np.log10(total_magnification + 1)
        }


class RealisticMassProfiles:
    """
    Realistic mass profiles including NFW, substructure, and environmental effects
    """
    
    def __init__(self, rng=None, config=None):
        # Handle both old RandomState and new Generator APIs
        if rng is None:
            self.rng = np.random.default_rng(42)
        elif isinstance(rng, np.random.RandomState):
            # Convert old RandomState to new Generator
            seed = int(rng.randint(0, 2**31))
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        
        self.config = config or {}
        
        # Mass profile parameters
        self.mass_profile_params = {
            'nfw_concentration_range': (3.0, 15.0),  # NFW concentration parameter
            'substructure_mass_fraction': 0.1,  # 10% of mass in substructure
            'substructure_count_range': (5, 20),  # Number of substructures
            'environmental_shear_range': (0.01, 0.1),  # External shear
            'environmental_convergence_range': (0.0, 0.05)  # External convergence
        }
    
    def generate_nfw_profile(self, lens_mass_log10, lens_redshift, 
                            concentration=None, virial_radius=None):
        """
        Generate NFW mass profile
        
        Args:
            lens_mass_log10: Lens mass (log10 M_sun)
            lens_redshift: Lens redshift
            concentration: NFW concentration (if None, will be sampled)
            virial_radius: Virial radius (if None, will be calculated)
        
        Returns:
            dict: NFW profile parameters
        """
        
        # Convert mass to M_sun
        mass_sun = 10**lens_mass_log10
        
        # Sample concentration if not provided
        if concentration is None:
            concentration = self.rng.uniform(*self.mass_profile_params['nfw_concentration_range'])
        
        # Calculate virial radius (simplified)
        if virial_radius is None:
            # Rough estimate: R_vir ~ (M / (4π/3 * 200 * ρ_crit))^(1/3)
            # Using critical density at redshift
            from astropy.cosmology import FlatLambdaCDM
            _cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            rho_crit = _cosmo.critical_density(lens_redshift).to('Msun/Mpc^3').value
            virial_radius = (3 * mass_sun / (4 * np.pi * 200 * rho_crit))**(1/3)
        
        # Calculate scale radius
        scale_radius = virial_radius / concentration
        
        # Calculate Einstein radius (simplified)
        einstein_radius = self._calculate_nfw_einstein_radius(
            mass_sun, scale_radius, concentration, lens_redshift
        )
        
        return {
            'profile_type': 'NFW',
            'mass_sun': mass_sun,
            'concentration': concentration,
            'virial_radius': virial_radius,
            'scale_radius': scale_radius,
            'einstein_radius': einstein_radius,
            'redshift': lens_redshift
        }
    
    def generate_substructure(self, main_mass_log10, lens_redshift, 
                            n_substructures=None, total_mass_fraction=None):
        """
        Generate substructure around main lens
        
        Args:
            main_mass_log10: Main lens mass (log10 M_sun)
            lens_redshift: Lens redshift
            n_substructures: Number of substructures (if None, will be sampled)
            total_mass_fraction: Total mass fraction in substructure
        
        Returns:
            list: List of substructure parameters
        """
        
        if n_substructures is None:
            n_substructures = self.rng.integers(*self.mass_profile_params['substructure_count_range'])
        
        if total_mass_fraction is None:
            total_mass_fraction = self.mass_profile_params['substructure_mass_fraction']
        
        main_mass = 10**main_mass_log10
        total_substructure_mass = main_mass * total_mass_fraction
        
        substructures = []
        for i in range(n_substructures):
            # Mass of individual substructure (power law distribution)
            mass_fraction = self.rng.power(2.0)  # Power law with index 2
            substructure_mass = total_substructure_mass * mass_fraction
            
            # Position (within virial radius)
            virial_radius = self._estimate_virial_radius(main_mass, lens_redshift)
            r = self.rng.uniform(0.1 * virial_radius, 0.8 * virial_radius)
            theta = self.rng.uniform(0, 2 * np.pi)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Substructure properties
            substructure = {
                'substructure_id': i,
                'mass_sun': substructure_mass,
                'position': (x, y),
                'redshift': lens_redshift + self.rng.uniform(-0.1, 0.1),  # Small redshift difference
                'profile_type': 'SIS',  # Simplified isothermal sphere
                'einstein_radius': self._calculate_sis_einstein_radius(
                    substructure_mass, lens_redshift
                )
            }
            substructures.append(substructure)
        
        return substructures
    
    def generate_environmental_effects(self, lens_redshift, lens_mass_log10):
        """
        Generate environmental effects (shear, convergence)
        
        Args:
            lens_redshift: Lens redshift
            lens_mass_log10: Lens mass (log10 M_sun)
        
        Returns:
            dict: Environmental effects parameters
        """
        
        # External shear (from large-scale structure)
        shear_magnitude = self.rng.uniform(*self.mass_profile_params['environmental_shear_range'])
        shear_angle = self.rng.uniform(0, 2 * np.pi)
        
        # External convergence (from line-of-sight structure)
        convergence = self.rng.uniform(*self.mass_profile_params['environmental_convergence_range'])
        
        # Redshift-dependent effects
        redshift_factor = (1 + lens_redshift)**2  # Stronger effects at higher redshift
        
        return {
            'external_shear': {
                'magnitude': shear_magnitude * redshift_factor,
                'angle': shear_angle
            },
            'external_convergence': convergence * redshift_factor,
            'redshift_factor': redshift_factor
        }
    
    def _calculate_nfw_einstein_radius(self, mass_sun, scale_radius, concentration, redshift):
        """Calculate NFW Einstein radius (simplified)"""
        # Simplified calculation - in reality this requires numerical integration
        # For now, use a scaling relation
        einstein_radius = 0.5 * scale_radius * np.log(1 + concentration) / concentration
        return einstein_radius
    
    def _calculate_sis_einstein_radius(self, mass_sun, redshift):
        """Calculate SIS Einstein radius"""
        # Simplified SIS Einstein radius calculation
        # R_E = 4π * (σ_v/c)² * D_A(z)
        # Using scaling relation: σ_v ∝ M^(1/4)
        sigma_v = 200 * (mass_sun / 1e12)**0.25  # km/s
        einstein_radius = 4 * np.pi * (sigma_v / 3e5)**2  # Simplified
        return einstein_radius
    
    def _estimate_virial_radius(self, mass_sun, redshift):
        """Estimate virial radius for given mass and redshift"""
        # R_vir = (3M / (4π * 200 * ρ_crit))^(1/3)
        from astropy.cosmology import FlatLambdaCDM
        _cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        rho_crit = _cosmo.critical_density(redshift).to('Msun/Mpc^3').value
        virial_radius = (3 * mass_sun / (4 * np.pi * 200 * rho_crit))**(1/3)
        return virial_radius
    
    def generate_binary_lens_system(self, primary_mass_log10, primary_redshift,
                                   primary_theta_E, mass_profile_type='SIE',
                                   source_redshift=2.0):
        """
        Generate binary lens system with two comparable-mass deflectors
        
        Args:
            primary_mass_log10: Primary lens mass (log10 M_sun)
            primary_redshift: Primary lens redshift
            primary_theta_E: Primary Einstein radius (arcsec)
            mass_profile_type: 'SIE' or 'NFW'
        
        Returns:
            dict: Binary lens system parameters
        """
        binary_cfg = self.config.get('binary_lenses', {})
        
        # Sample mass ratio
        mass_ratio_min = binary_cfg.get('mass_ratio', {}).get('min', 0.3)
        mass_ratio_max = binary_cfg.get('mass_ratio', {}).get('max', 1.0)
        mass_ratio = self.rng.uniform(mass_ratio_min, mass_ratio_max)
        
        secondary_mass_log10 = primary_mass_log10 + np.log10(mass_ratio)
        
        # Calculate secondary Einstein radius (scales as M^0.5)
        secondary_theta_E = primary_theta_E * np.sqrt(mass_ratio)
        
        # Sample separation
        sep_min = binary_cfg.get('separation', {}).get('min_in_einstein_radii', 0.5)
        sep_max = binary_cfg.get('separation', {}).get('max_in_einstein_radii', 2.0)
        separation_factor = self.rng.uniform(sep_min, sep_max)
        separation_arcsec = (primary_theta_E + secondary_theta_E) * separation_factor
        
        # Random position angle
        position_angle = self.rng.uniform(0, 2 * np.pi)
        
        # Position components (primary at origin, secondary offset)
        x1, y1 = 0.0, 0.0
        x2 = separation_arcsec * np.cos(position_angle)
        y2 = separation_arcsec * np.sin(position_angle)
        
        # Sample redshifts
        same_z_frac = binary_cfg.get('redshift_config', {}).get('same_redshift_fraction', 0.8)
        if self.rng.random() < same_z_frac:
            # True physical pair at same redshift
            secondary_redshift = primary_redshift
        else:
            # Line-of-sight projection with small redshift difference
            max_delta_z = binary_cfg.get('redshift_config', {}).get('max_delta_z', 0.3)
            delta_z = self.rng.uniform(-max_delta_z, max_delta_z)
            secondary_redshift = max(0.1, primary_redshift + delta_z)
        
        # Generate ellipticity parameters
        correlated_ellipticities = binary_cfg.get('orientation', {}).get('correlated_ellipticities', True)
        
        # Primary ellipticity
        e1_primary = self.rng.normal(0, 0.2)
        e2_primary = self.rng.normal(0, 0.2)
        
        if correlated_ellipticities and abs(secondary_redshift - primary_redshift) < 0.1:
            # Correlated for physical pairs
            e1_secondary = e1_primary + self.rng.normal(0, 0.1)
            e2_secondary = e2_primary + self.rng.normal(0, 0.1)
        else:
            # Independent for line-of-sight pairs
            e1_secondary = self.rng.normal(0, 0.2)
            e2_secondary = self.rng.normal(0, 0.2)
        
        # Build lens model parameters
        if mass_profile_type.upper() == 'SIE':
            lens_model_list = ['SIE', 'SIE', 'SHEAR', 'CONVERGENCE']

            kwargs_lens_1 = {
                'theta_E': float(primary_theta_E),
                'center_x': float(x1),
                'center_y': float(y1),
                'e1': float(e1_primary),
                'e2': float(e2_primary)
            }

            kwargs_lens_2 = {
                'theta_E': float(secondary_theta_E),
                'center_x': float(x2),
                'center_y': float(y2),
                'e1': float(e1_secondary),
                'e2': float(e2_secondary)
            }

            # External shear + convergence from environment
            env_effects = self.generate_environmental_effects(primary_redshift, primary_mass_log10)
            _shear_mag = env_effects['external_shear']['magnitude']
            _shear_ang = env_effects['external_shear']['angle']
            kappa_ext = float(env_effects['external_convergence'])
            kwargs_shear = {
                'gamma1': float(_shear_mag * np.cos(2 * _shear_ang)),
                'gamma2': float(_shear_mag * np.sin(2 * _shear_ang))
            }
            kwargs_convergence = {'kappa': kappa_ext}

            kwargs_lens = [kwargs_lens_1, kwargs_lens_2, kwargs_shear, kwargs_convergence]
            
        elif mass_profile_type.upper() == 'NFW':
            lens_model_list = ['NFW_ELLIPSE', 'NFW_ELLIPSE', 'SHEAR', 'CONVERGENCE']

            # Derive physical (M, c) -> lensing (Rs, alpha_Rs) via lenstronomy's
            # LensCosmo.nfw_physical2angle, exactly as done for subhalos and the
            # NFW+NFW binary path in jwst_lens_simulator.py -- replaces the
            # previous ad hoc 'alpha_Rs = theta_E' rescaling.
            from astropy.cosmology import FlatLambdaCDM
            from lenstronomy.Cosmo.lens_cosmo import LensCosmo
            _cosmo_bin = FlatLambdaCDM(H0=70, Om0=0.3)
            _lens_cosmo_1 = LensCosmo(z_lens=float(primary_redshift), z_source=float(source_redshift), cosmo=_cosmo_bin)
            _lens_cosmo_2 = LensCosmo(z_lens=float(secondary_redshift), z_source=float(source_redshift), cosmo=_cosmo_bin)

            nfw_1 = self.generate_nfw_profile(primary_mass_log10, primary_redshift)
            nfw_2 = self.generate_nfw_profile(secondary_mass_log10, secondary_redshift)

            Rs_1, alpha_Rs_1 = _lens_cosmo_1.nfw_physical2angle(M=float(nfw_1['mass_sun']), c=float(nfw_1['concentration']))
            Rs_2, alpha_Rs_2 = _lens_cosmo_2.nfw_physical2angle(M=float(nfw_2['mass_sun']), c=float(nfw_2['concentration']))

            kwargs_lens_1 = {
                'Rs': float(Rs_1),
                'alpha_Rs': float(alpha_Rs_1),
                'center_x': float(x1),
                'center_y': float(y1),
                'e1': float(e1_primary),
                'e2': float(e2_primary)
            }

            kwargs_lens_2 = {
                'Rs': float(Rs_2),
                'alpha_Rs': float(alpha_Rs_2),
                'center_x': float(x2),
                'center_y': float(y2),
                'e1': float(e1_secondary),
                'e2': float(e2_secondary)
            }
            
            # External shear + convergence
            env_effects = self.generate_environmental_effects(primary_redshift, primary_mass_log10)
            _shear_mag = env_effects['external_shear']['magnitude']
            _shear_ang = env_effects['external_shear']['angle']
            kappa_ext = float(env_effects['external_convergence'])
            kwargs_shear = {
                'gamma1': float(_shear_mag * np.cos(2 * _shear_ang)),
                'gamma2': float(_shear_mag * np.sin(2 * _shear_ang))
            }
            kwargs_convergence = {'kappa': kappa_ext}

            kwargs_lens = [kwargs_lens_1, kwargs_lens_2, kwargs_shear, kwargs_convergence]
        else:
            raise ValueError(f"Unknown mass_profile_type: {mass_profile_type}")
        
        return {
            'is_binary': True,
            'mass_profile_type': mass_profile_type,
            'lens_model_list': lens_model_list,
            'kwargs_lens': kwargs_lens,
            'primary': {
                'mass_log10': primary_mass_log10,
                'redshift': primary_redshift,
                'theta_E': primary_theta_E,
                'position': (x1, y1),
                'e1': e1_primary,
                'e2': e2_primary
            },
            'secondary': {
                'mass_log10': secondary_mass_log10,
                'redshift': secondary_redshift,
                'theta_E': secondary_theta_E,
                'position': (x2, y2),
                'e1': e1_secondary,
                'e2': e2_secondary
            },
            'mass_ratio': mass_ratio,
            'separation_arcsec': separation_arcsec,
            'position_angle_rad': position_angle,
            'same_redshift': abs(secondary_redshift - primary_redshift) < 0.01,
            'kappa_ext': kappa_ext
        }


# =============================================================================
# QUARANTINE ZONE — NOT USED IN PRODUCTION
# All code below this marker is unused in the live pipeline (jwst_lens_simulator.py
# imports only RealisticMassProfiles above).  Several routines contain known
# physically-incorrect formulas.  Do NOT use for science-grade outputs.
# =============================================================================


class SurveySpecificNoiseModels:
    """
    Survey-specific noise models for JWST COSMOS-Web
    """
    
    def __init__(self, rng=None):
        # Handle both old RandomState and new Generator APIs
        if rng is None:
            self.rng = np.random.default_rng(42)
        elif isinstance(rng, np.random.RandomState):
            # Convert old RandomState to new Generator
            seed = int(rng.randint(0, 2**31))
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        
        # JWST COSMOS-Web noise parameters
        self.jwst_noise_params = {
            'F115W': {
                'background_noise': 1e-6,  # Background noise level
                'read_noise': 0.5e-6,    # Read noise
                'dark_current': 0.1e-6,  # Dark current
                'psf_fwhm': 0.037,       # PSF FWHM (arcsec)
                'pixel_scale': 0.03      # Pixel scale (arcsec/pixel)
            },
            'F150W': {
                'background_noise': 1e-6,
                'read_noise': 0.5e-6,
                'dark_current': 0.1e-6,
                'psf_fwhm': 0.040,
                'pixel_scale': 0.03
            },
            'F277W': {
                'background_noise': 1e-6,
                'read_noise': 0.5e-6,
                'dark_current': 0.1e-6,
                'psf_fwhm': 0.065,
                'pixel_scale': 0.03
            },
            'F444W': {
                'background_noise': 1e-6,
                'read_noise': 0.5e-6,
                'dark_current': 0.1e-6,
                'psf_fwhm': 0.100,
                'pixel_scale': 0.03
            }
        }
    
    def generate_survey_noise(self, images, exposure_time=1000, 
                            background_level=None, add_cosmic_rays=True):
        """
        Generate survey-specific noise for JWST COSMOS-Web
        
        Args:
            images: Dict of band -> image arrays
            exposure_time: Exposure time in seconds
            background_level: Background level (if None, will be sampled)
            add_cosmic_rays: Whether to add cosmic ray hits
        
        Returns:
            dict: Noisy images with metadata
        """
        
        noisy_images = {}
        noise_metadata = {}
        
        for band, image in images.items():
            if band not in self.jwst_noise_params:
                noisy_images[band] = image
                continue
            
            band_params = self.jwst_noise_params[band]
            
            # Generate noise components
            noise_components = self._generate_noise_components(
                image.shape, band_params, exposure_time, background_level
            )
            
            # Add cosmic rays if requested
            if add_cosmic_rays:
                cosmic_ray_noise = self._generate_cosmic_rays(image.shape, exposure_time)
                noise_components['cosmic_rays'] = cosmic_ray_noise
            
            # Combine noise
            total_noise = sum(noise_components.values())
            noisy_image = image + total_noise
            
            noisy_images[band] = noisy_image
            noise_metadata[band] = {
                'noise_components': noise_components,
                'total_noise_rms': np.std(total_noise),
                'signal_to_noise': np.mean(image) / np.std(total_noise) if np.std(total_noise) > 0 else 0
            }
        
        return noisy_images, noise_metadata
    
    def _generate_noise_components(self, image_shape, band_params, exposure_time, background_level):
        """Generate individual noise components"""
        
        noise_components = {}
        
        # Background noise (Poisson)
        if background_level is None:
            background_level = band_params['background_noise'] * exposure_time
        
        background_noise = self.rng.poisson(background_level, image_shape) - background_level
        noise_components['background'] = background_noise.astype(float)
        
        # Read noise (Gaussian)
        read_noise = self.rng.normal(0, band_params['read_noise'], image_shape)
        noise_components['read_noise'] = read_noise
        
        # Dark current (Poisson)
        dark_current = self.rng.poisson(band_params['dark_current'] * exposure_time, image_shape)
        noise_components['dark_current'] = dark_current.astype(float)
        
        # Systematic noise (correlated)
        systematic_noise = self._generate_systematic_noise(image_shape)
        noise_components['systematic'] = systematic_noise
        
        return noise_components
    
    def _generate_cosmic_rays(self, image_shape, exposure_time):
        """Generate cosmic ray hits"""
        # Cosmic ray rate (hits per pixel per second)
        cosmic_ray_rate = 1e-4  # Simplified rate
        
        # Number of cosmic ray hits
        n_hits = self.rng.poisson(cosmic_ray_rate * image_shape[0] * image_shape[1] * exposure_time)
        
        cosmic_ray_image = np.zeros(image_shape)
        
        for _ in range(n_hits):
            # Random position
            y, x = self.rng.integers(0, image_shape[0]), self.rng.integers(0, image_shape[1])
            
            # Cosmic ray intensity (exponential distribution)
            intensity = self.rng.exponential(1e-5)
            
            # Add cosmic ray (localized)
            cosmic_ray_image[y, x] += intensity
        
        return cosmic_ray_image
    
    def _generate_systematic_noise(self, image_shape):
        """Generate systematic noise (correlated)"""
        # Large-scale systematic variations
        x = np.linspace(0, 1, image_shape[1])
        y = np.linspace(0, 1, image_shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Multiple systematic patterns
        systematic_noise = (
            0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) +
            0.05 * np.sin(4 * np.pi * X) +
            0.03 * np.cos(6 * np.pi * Y)
        )
        
        return systematic_noise * 1e-6  # Scale to appropriate level


class QualityAssessmentMetrics:
    """
    Quality assessment metrics for lens detection training
    """
    
    def __init__(self, rng=None):
        # Handle both old RandomState and new Generator APIs
        if rng is None:
            self.rng = np.random.default_rng(42)
        elif isinstance(rng, np.random.RandomState):
            # Convert old RandomState to new Generator
            seed = int(rng.randint(0, 2**31))
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        
        # Quality assessment parameters
        self.quality_params = {
            'detection_thresholds': {
                'signal_to_noise': 3.0,      # Minimum S/N for detection
                'magnification': 1.5,        # Minimum magnification
                'image_separation': 0.1,    # Minimum image separation (arcsec)
                'total_flux': 1e-6          # Minimum total flux
            },
            'quality_grades': {
                'excellent': 0.9,    # 90%+ quality
                'good': 0.8,         # 80-90% quality
                'fair': 0.6,         # 60-80% quality
                'poor': 0.4          # <60% quality
            }
        }
    
    def assess_lens_quality(self, lens_system, images, noise_metadata=None):
        """
        Assess quality of lens system for ML training
        
        Args:
            lens_system: Lens system parameters
            images: Dict of band -> image arrays
            noise_metadata: Noise metadata from noise model
        
        Returns:
            dict: Quality assessment results
        """
        
        quality_metrics = {}
        
        # Basic detection metrics
        detection_metrics = self._calculate_detection_metrics(images, noise_metadata)
        quality_metrics['detection'] = detection_metrics
        
        # Lensing-specific metrics
        lensing_metrics = self._calculate_lensing_metrics(lens_system, images)
        quality_metrics['lensing'] = lensing_metrics
        
        # Multi-wavelength consistency
        consistency_metrics = self._calculate_consistency_metrics(images)
        quality_metrics['consistency'] = consistency_metrics
        
        # Overall quality grade
        quality_grade = self._calculate_quality_grade(quality_metrics)
        quality_metrics['quality_grade'] = quality_grade
        
        # Training suitability
        training_suitability = self._assess_training_suitability(quality_metrics)
        quality_metrics['training_suitability'] = training_suitability
        
        return quality_metrics
    
    def _calculate_detection_metrics(self, images, noise_metadata):
        """Calculate detection-related metrics"""
        
        detection_metrics = {}
        
        for band, image in images.items():
            # Signal-to-noise ratio
            if noise_metadata and band in noise_metadata:
                snr = noise_metadata[band]['signal_to_noise']
            else:
                # Estimate S/N from image
                signal = np.mean(image)
                noise = np.std(image)
                snr = signal / noise if noise > 0 else 0
            
            # Flux metrics
            total_flux = np.sum(image)
            max_flux = np.max(image)
            mean_flux = np.mean(image)
            
            # Size metrics
            flux_threshold = 0.1 * max_flux
            bright_pixels = image > flux_threshold
            if np.any(bright_pixels):
                y_coords, x_coords = np.where(bright_pixels)
                size_pixels = np.sqrt((y_coords.max() - y_coords.min())**2 + 
                                    (x_coords.max() - x_coords.min())**2)
            else:
                size_pixels = 0
            
            detection_metrics[band] = {
                'signal_to_noise': snr,
                'total_flux': total_flux,
                'max_flux': max_flux,
                'mean_flux': mean_flux,
                'size_pixels': size_pixels,
                'detectable': snr > self.quality_params['detection_thresholds']['signal_to_noise']
            }
        
        return detection_metrics
    
    def _calculate_lensing_metrics(self, lens_system, images):
        """Calculate lensing-specific metrics"""
        
        lensing_metrics = {}
        
        # Einstein radius
        einstein_radius = lens_system.get('einstein_radius', 1.0)
        
        # Magnification (if available)
        magnification = lens_system.get('magnification', 1.0)
        
        # Image separation (if available)
        image_separation = lens_system.get('image_separation', 0.0)
        
        # Lensing efficiency
        lensing_efficiency = self._calculate_lensing_efficiency(
            einstein_radius, magnification, image_separation
        )
        
        lensing_metrics = {
            'einstein_radius': einstein_radius,
            'magnification': magnification,
            'image_separation': image_separation,
            'lensing_efficiency': lensing_efficiency,
            'lensing_quality': self._assess_lensing_quality(
                einstein_radius, magnification, image_separation
            )
        }
        
        return lensing_metrics
    
    def _calculate_consistency_metrics(self, images):
        """Calculate multi-wavelength consistency metrics"""
        
        if len(images) < 2:
            return {'consistency_score': 1.0, 'wavelength_coverage': 1.0}
        
        # Calculate flux ratios between bands
        bands = list(images.keys())
        flux_ratios = {}
        
        for i, band1 in enumerate(bands):
            for band2 in bands[i+1:]:
                flux1 = np.sum(images[band1])
                flux2 = np.sum(images[band2])
                
                if flux2 > 0:
                    ratio = flux1 / flux2
                    flux_ratios[f'{band1}_{band2}'] = ratio
        
        # Calculate consistency score
        if flux_ratios:
            ratio_values = list(flux_ratios.values())
            consistency_score = 1.0 - np.std(ratio_values) / np.mean(ratio_values)
            consistency_score = max(0.0, min(1.0, consistency_score))
        else:
            consistency_score = 1.0
        
        return {
            'consistency_score': consistency_score,
            'flux_ratios': flux_ratios,
            'wavelength_coverage': len(images) / 4.0  # 4 JWST bands
        }
    
    def _calculate_lensing_efficiency(self, einstein_radius, magnification, image_separation):
        """Calculate lensing efficiency"""
        
        # Efficiency based on Einstein radius (larger = more efficient)
        radius_efficiency = min(1.0, einstein_radius / 2.0)
        
        # Efficiency based on magnification (higher = more efficient)
        mag_efficiency = min(1.0, magnification / 5.0)
        
        # Efficiency based on image separation (larger = more efficient)
        sep_efficiency = min(1.0, image_separation / 1.0)
        
        # Combined efficiency
        efficiency = (radius_efficiency + mag_efficiency + sep_efficiency) / 3.0
        
        return efficiency
    
    def _assess_lensing_quality(self, einstein_radius, magnification, image_separation):
        """Assess lensing quality"""
        
        quality_score = 0.0
        
        # Einstein radius quality
        if einstein_radius > 0.5:
            quality_score += 0.3
        elif einstein_radius > 0.2:
            quality_score += 0.2
        
        # Magnification quality
        if magnification > 3.0:
            quality_score += 0.4
        elif magnification > 1.5:
            quality_score += 0.2
        
        # Image separation quality
        if image_separation > 0.5:
            quality_score += 0.3
        elif image_separation > 0.2:
            quality_score += 0.2
        
        return quality_score
    
    def _calculate_quality_grade(self, quality_metrics):
        """Calculate overall quality grade"""
        
        # Extract key metrics
        detection_scores = []
        for band_metrics in quality_metrics['detection'].values():
            detection_scores.append(band_metrics['signal_to_noise'])
        
        avg_snr = np.mean(detection_scores) if detection_scores else 0
        consistency_score = quality_metrics['consistency']['consistency_score']
        lensing_quality = quality_metrics['lensing']['lensing_quality']
        
        # Combined quality score
        quality_score = (avg_snr / 10.0 + consistency_score + lensing_quality) / 3.0
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Assign grade
        if quality_score >= 0.9:
            return 'excellent'
        elif quality_score >= 0.8:
            return 'good'
        elif quality_score >= 0.6:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_training_suitability(self, quality_metrics):
        """Assess suitability for ML training"""
        
        quality_grade = quality_metrics['quality_grade']
        detection_metrics = quality_metrics['detection']
        
        # Check if any band is detectable
        any_detectable = any(
            band_metrics['detectable'] for band_metrics in detection_metrics.values()
        )
        
        # Suitability based on quality grade
        if quality_grade in ['excellent', 'good']:
            suitability = 'high'
        elif quality_grade == 'fair':
            suitability = 'medium'
        else:
            suitability = 'low'
        
        # Additional checks
        if not any_detectable:
            suitability = 'low'
        
        return {
            'suitability_level': suitability,
            'any_detectable': any_detectable,
            'quality_grade': quality_grade,
            'recommended_for_training': suitability in ['high', 'medium']
        }


def create_advanced_lens_system(n_lenses=100, output_dir="advanced_lens_data"):
    """
    Create advanced lens system with all features
    
    Args:
        n_lenses: Number of lens systems to generate
        output_dir: Output directory for data
    
    Returns:
        dict: Advanced lens system configuration
    """
    
    print("🚀 Creating Advanced Lens System for JWST COSMOS-Web")
    print(f"   Lenses: {n_lenses}")
    print(f"   Output: {output_dir}")
    
    # Initialize components
    multi_source = MultiSourceLensingSystem()
    mass_profiles = RealisticMassProfiles()
    noise_models = SurveySpecificNoiseModels()
    quality_metrics = QualityAssessmentMetrics()
    
    # Generate advanced lens systems
    advanced_systems = []
    
    for i in range(n_lenses):
        # Generate lens parameters
        lens_redshift = np.random.uniform(0.2, 2.0)
        lens_mass_log10 = np.random.normal(11.2, 0.3)
        einstein_radius = np.random.lognormal(np.log(0.8), 0.4)
        
        # Generate multi-source system
        multi_source_system = multi_source.generate_multi_source_system(
            lens_redshift, lens_mass_log10, einstein_radius
        )
        
        # Generate realistic mass profile
        nfw_profile = mass_profiles.generate_nfw_profile(
            lens_mass_log10, lens_redshift
        )
        
        # Generate substructure
        substructures = mass_profiles.generate_substructure(
            lens_mass_log10, lens_redshift
        )
        
        # Generate environmental effects
        environmental_effects = mass_profiles.generate_environmental_effects(
            lens_redshift, lens_mass_log10
        )
        
        # Create dummy images for demonstration
        dummy_images = {
            'F115W': np.random.exponential(1e-6, (100, 100)),
            'F150W': np.random.exponential(1e-6, (100, 100)),
            'F277W': np.random.exponential(1e-6, (100, 100)),
            'F444W': np.random.exponential(1e-6, (100, 100))
        }
        
        # Generate survey noise
        noisy_images, noise_metadata = noise_models.generate_survey_noise(dummy_images)
        
        # Assess quality
        lens_system = {
            'lens_id': f"advanced_lens_{i:06d}",
            'redshift': lens_redshift,
            'mass_log10': lens_mass_log10,
            'einstein_radius': einstein_radius,
            'magnification': multi_source_system['lensing_properties']['average_magnification']
        }
        
        quality_assessment = quality_metrics.assess_lens_quality(
            lens_system, noisy_images, noise_metadata
        )
        
        # Combine all components
        advanced_system = {
            'lens_id': f"advanced_lens_{i:06d}",
            'basic_parameters': lens_system,
            'multi_source_system': multi_source_system,
            'mass_profile': nfw_profile,
            'substructures': substructures,
            'environmental_effects': environmental_effects,
            'noise_metadata': noise_metadata,
            'quality_assessment': quality_assessment
        }
        
        advanced_systems.append(advanced_system)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save advanced systems
    systems_path = output_path / "advanced_lens_systems.json"
    with open(systems_path, 'w') as f:
        json.dump(advanced_systems, f, indent=2)
    
    # Create summary statistics
    summary_stats = create_summary_statistics(advanced_systems)
    summary_path = output_path / "advanced_lens_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"✅ Advanced lens system created: {systems_path}")
    print(f"✅ Summary statistics: {summary_path}")
    
    return {
        'advanced_systems': advanced_systems,
        'summary_statistics': summary_stats,
        'output_directory': output_dir
    }


def create_summary_statistics(advanced_systems):
    """Create summary statistics for advanced lens systems"""
    
    # Extract statistics
    n_systems = len(advanced_systems)
    redshifts = [s['basic_parameters']['redshift'] for s in advanced_systems]
    masses = [s['basic_parameters']['mass_log10'] for s in advanced_systems]
    einstein_radii = [s['basic_parameters']['einstein_radius'] for s in advanced_systems]
    
    # Multi-source statistics
    n_sources_list = [s['multi_source_system']['n_sources'] for s in advanced_systems]
    magnifications = [s['multi_source_system']['lensing_properties']['average_magnification'] for s in advanced_systems]
    
    # Quality statistics
    quality_grades = [s['quality_assessment']['quality_grade'] for s in advanced_systems]
    training_suitability = [s['quality_assessment']['training_suitability']['suitability_level'] for s in advanced_systems]
    
    # Create summary
    summary = {
        'dataset_overview': {
            'total_systems': n_systems,
            'redshift_range': [min(redshifts), max(redshifts)],
            'redshift_mean': np.mean(redshifts),
            'mass_range': [min(masses), max(masses)],
            'mass_mean': np.mean(masses),
            'einstein_radius_range': [min(einstein_radii), max(einstein_radii)],
            'einstein_radius_mean': np.mean(einstein_radii)
        },
        'multi_source_statistics': {
            'n_sources_range': [min(n_sources_list), max(n_sources_list)],
            'n_sources_mean': np.mean(n_sources_list),
            'magnification_range': [min(magnifications), max(magnifications)],
            'magnification_mean': np.mean(magnifications)
        },
        'quality_statistics': {
            'quality_grade_distribution': {
                grade: quality_grades.count(grade) for grade in set(quality_grades)
            },
            'training_suitability_distribution': {
                level: training_suitability.count(level) for level in set(training_suitability)
            }
        }
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Creating Advanced Lens System for JWST COSMOS-Web...")
    
    advanced_config = create_advanced_lens_system(
        n_lenses=50,
        output_dir="advanced_lens_demo"
    )
    
    print("✅ Advanced lens system created successfully!")
    print("Features implemented:")
    print("  ✅ Multi-source lensing systems")
    print("  ✅ Realistic mass profiles (NFW, substructure)")
    print("  ✅ Survey-specific noise models")
    print("  ✅ Quality assessment metrics")
    print("  ✅ Environmental effects")
    print("  ✅ Multi-wavelength consistency")


def add_jwst_diffraction_spikes(image, bright_positions, spike_configs=None, rng=None):
    """
    Add realistic JWST 4-point diffraction spikes for bright stars.
    
    JWST's rectangular primary mirror creates characteristic 4-spike diffraction
    patterns for bright objects. This function adds these patterns to simulate
    realistic JWST observations.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (single band)
    bright_positions : list of tuple
        List of (y, x) pixel positions of bright stars/objects
    spike_configs : dict, optional
        Configuration for spikes:
        - 'spike_length': How far spikes extend (pixels), default 30
        - 'brightness_scale': Fraction of central brightness for spike, default 0.6 (60%)
        - 'profile': 'linear' (default) or 'exponential' decay along spike
    rng : np.random.RandomState or Generator
        Random number generator
        
    Returns:
    --------
    np.ndarray
        Image with diffraction spikes added
        
    Example:
    --------
    >>> image = np.random.randn(300, 300)
    >>> bright_stars = [(50, 75), (200, 150)]  # Two bright stars
    >>> image_with_spikes = add_jwst_diffraction_spikes(
    ...     image, bright_stars, 
    ...     spike_configs={'spike_length': 50, 'brightness_scale': 0.7}
    ... )
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    if not bright_positions or len(bright_positions) == 0:
        return image
    
    enhanced = np.array(image, copy=True, dtype=np.float32)
    
    # Default config
    if spike_configs is None:
        spike_configs = {}
    spike_length = spike_configs.get('spike_length', 30)  # Longer spikes by default
    brightness_scale = spike_configs.get('brightness_scale', 1.5)  # 150% of star brightness - MUCH brighter for visibility after normalization
    profile = spike_configs.get('profile', 'linear')
    
    # Process each bright star
    for y_center, x_center in bright_positions:
        y_center = int(y_center)
        x_center = int(x_center)
        
        # Ensure star position is within bounds
        if not (0 <= y_center < enhanced.shape[0] and 0 <= x_center < enhanced.shape[1]):
            continue
        
        # Get central brightness (use image value at star center)
        central_brightness = abs(enhanced[y_center, x_center])
        if central_brightness < 1e-6:
            central_brightness = np.std(enhanced)  # Fallback to std if center is very dim
        
        # Six directions: 4 CARDINAL (strong) + 2 DIAGONAL (weak)
        # Cardinal directions - full strength
        cardinal_directions = [
            (-1, 0),  # North (up)
            (1, 0),   # South (down)
            (0, 1),   # East (right)
            (0, -1),  # West (left)
        ]
        
        # Diagonal directions - weaker (50% intensity)
        diagonal_directions = [
            (-1, 1),   # Northeast
            (1, -1),   # Southwest
        ]
        
        # Process cardinal spikes (full strength)
        for dy, dx in cardinal_directions:
            # Draw spike in this direction
            for dist in range(1, spike_length + 1):
                ny = y_center + dy * dist
                nx = x_center + dx * dist
                
                # Check bounds
                if not (0 <= ny < enhanced.shape[0] and 0 <= nx < enhanced.shape[1]):
                    break
                
                # Calculate intensity (decreases with distance)
                if profile == 'exponential':
                    # Exponential falloff: more realistic for diffraction
                    normalized_dist = dist / spike_length
                    intensity_factor = np.exp(-2 * normalized_dist)  # Slightly longer decay
                else:
                    # Linear falloff: simple triangular profile
                    intensity_factor = max(0, 1.0 - dist / spike_length)
                
                spike_intensity = central_brightness * brightness_scale * intensity_factor
                enhanced[ny, nx] += spike_intensity
        
        # Process diagonal spikes (50% strength)
        for dy, dx in diagonal_directions:
            diag_spike_length = int(spike_length * 0.7)  # Slightly shorter diagonals
            weaker_brightness_scale = brightness_scale * 0.5  # 50% of cardinal strength
            
            # Draw spike in this direction
            for dist in range(1, diag_spike_length + 1):
                ny = y_center + dy * dist
                nx = x_center + dx * dist
                
                # Check bounds
                if not (0 <= ny < enhanced.shape[0] and 0 <= nx < enhanced.shape[1]):
                    break
                
                # Calculate intensity (decreases with distance)
                if profile == 'exponential':
                    normalized_dist = dist / diag_spike_length
                    intensity_factor = np.exp(-2 * normalized_dist)
                else:
                    intensity_factor = max(0, 1.0 - dist / diag_spike_length)
                
                spike_intensity = central_brightness * weaker_brightness_scale * intensity_factor
                enhanced[ny, nx] += spike_intensity
    
    return enhanced.astype(np.float32)


def add_jwst_artifacts_realistic(images, artifact_level='moderate', numpix=300, 
                                rng=None, add_spikes=False):
    """
    Add realistic JWST detector artifacts including optional diffraction spikes.
    
    Parameters:
    -----------
    images : dict
        Dictionary of images by band
    artifact_level : str
        'low', 'moderate', or 'high'
    numpix : int
        Image pixel size
    rng : np.random.RandomState or Generator
        Random number generator
    add_spikes : bool
        If True, add diffraction spikes for bright field stars
        
    Returns:
    --------
    dict
        Images with artifacts and optional spikes added
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    enhanced_images = {}
    
    for band, image in images.items():
        enhanced = np.array(image, copy=True)
        
        # Existing artifact code (cosmic rays, electronic noise)
        # [Would call existing artifact function here]
        
        # Optionally add diffraction spikes for bright objects
        if add_spikes:
            # Identify bright field stars (anything significantly above noise)
            threshold = np.percentile(enhanced[enhanced > 0], 90)  # Top 10%
            bright_mask = enhanced > threshold
            
            # Find positions of bright objects
            bright_y, bright_x = np.where(bright_mask)
            bright_positions = list(zip(bright_y, bright_x))
            
            if len(bright_positions) > 0:
                # Add spikes
                spike_configs = {
                    'spike_length': max(8, int(10 * numpix / 300)),  # Scale with image size
                    'brightness_scale': 0.3,
                    'profile': 'exponential'
                }
                enhanced = add_jwst_diffraction_spikes(
                    enhanced, bright_positions, spike_configs, rng
                )
        
        enhanced_images[band] = enhanced.astype(np.float32)
    
    return enhanced_images
