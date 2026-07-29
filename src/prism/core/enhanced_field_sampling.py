#!/usr/bin/env python3
"""
Enhanced Field Galaxy Sampling using Learned Environment Properties

This module integrates the learned environment properties to generate
realistic field galaxy populations around massive galaxies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from prism.ml.environment_learning import EnvironmentLearner
import warnings
warnings.filterwarnings('ignore')
from astropy.cosmology import FlatLambdaCDM

class EnhancedFieldSampler:
    """
    Enhanced field galaxy sampling using learned environment properties
    """
    
    def __init__(self, model_dir="models", data_dir="data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.learner = None
        self.field_catalog = None
        self.trained = False
        # Cosmology for angular size conversion
        self._cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        # Cache for CWMGs mass–size bins
        self._cwmgs_ms_bins = None

    def _angular_diameter_distance(self, z):
        return self._cosmo.angular_diameter_distance(z).value  # Mpc

    def _rng_integers(self, rng, low, high):
        """Compatibility helper for RandomState and Generator integer sampling.

        Returns an integer in [low, high) to match NumPy's modern `integers` API.
        """
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high))
        return int(rng.randint(low, high))

    def _load_cwmgs_mass_size_bins(self):
        """Load CWMGs (massive galaxies) redshift-binned mass–size relation.
        Expects CSV at analysis/mass_size_relations/massive_galaxy_redshift_binned_relations.csv
        with columns: z_min, z_max, alpha, beta.
        """
        if self._cwmgs_ms_bins is not None:
            return self._cwmgs_ms_bins
        # Prefer data folder at runtime
        data_path = Path(__file__).resolve().parents[1] / "data/massive_galaxy_redshift_binned_relations.csv"
        csv_path = data_path if data_path.exists() else Path(__file__).resolve().parents[1] / "analysis/mass_size_relations/massive_galaxy_redshift_binned_relations.csv"
        if not csv_path.exists():
            self._cwmgs_ms_bins = []
            return self._cwmgs_ms_bins
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
                # Typical ranges: |alpha| < 1.0, |beta| < 10
                slope, intercept = a, b
                if not (abs(slope) < 1.0 and abs(intercept) < 10.0):
                    if abs(intercept) < 1.0 and abs(slope) < 10.0:
                        slope, intercept = intercept, slope
                    else:
                        continue
                bins.append({"z_min": zmin, "z_max": zmax, "slope": slope, "intercept": intercept})
            self._cwmgs_ms_bins = bins
        except Exception:
            self._cwmgs_ms_bins = []
        return self._cwmgs_ms_bins

    def _cwmgs_re_kpc(self, mass_log10, z, rng):
        """Compute Re (kpc) from CWMGs mass–size relation with scatter.
        Falls back to a conservative small-size prior if bins missing.
        """
        bins = self._load_cwmgs_mass_size_bins()
        if bins:
            sel = None
            for b in bins:
                if b["z_min"] <= z < b["z_max"]:
                    sel = b
                    break
            if sel is None:
                sel = sorted(bins, key=lambda b: abs(0.5 * (b["z_min"] + b["z_max"]) - z))[0]
            log_re = sel["slope"] * mass_log10 + sel["intercept"]
        else:
            # Very mild global prior if file missing
            log_re = 0.2 * mass_log10 - 2.0
        log_re += rng.normal(0, 0.2)
        re_kpc = 10 ** log_re
        return float(np.clip(re_kpc, 0.2, 30.0))
        
    def load_models(self):
        """Load pre-trained environment models"""
        try:
            self.learner = EnvironmentLearner(self.data_dir)
            self.learner.load_models(self.model_dir)
            self.trained = True
            print("Environment models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load environment models: {e}")
            print("Falling back to default environment sampling")
            self.trained = False
    
    def load_field_catalog(self):
        """Load field galaxy catalog for sampling"""
        try:
            self.field_catalog = pd.read_csv(self.data_dir / "merged_lens_field_catalog.csv")
            print(f"Loaded {len(self.field_catalog)} field galaxies")
        except Exception as e:
            print(f"Warning: Could not load field catalog: {e}")
            self.field_catalog = None
    
    def predict_environment_properties(self, redshift, mass_log10):
        """
        Predict environment properties for a galaxy
        
        Args:
            redshift: Galaxy redshift
            mass_log10: Log10 stellar mass in M_sun
            
        Returns:
            dict: Environment properties
        """
        if self.trained and self.learner:
            return self.learner.predict_environment(redshift, mass_log10)
        else:
            # Fallback to simple rules
            if mass_log10 >= 11.5:
                return {
                    'env_type': 'group',
                    'n_galaxies': np.random.randint(3, 6),
                    'max_radius_arcsec': 3.0
                }
            elif mass_log10 >= 11.0:
                return {
                    'env_type': 'galaxy_pair',
                    'n_galaxies': np.random.randint(1, 4),
                    'max_radius_arcsec': 2.5
                }
            else:
                return {
                    'env_type': 'isolated_field',
                    'n_galaxies': np.random.randint(0, 3),
                    'max_radius_arcsec': 2.0
                }
    
    def sample_field_galaxies_enhanced(self, central_redshift, central_mass_log10, 
                                     n_max=5, rng=None, numpix=300, 
                                     pixel_scale=0.03, avoid_center_arcsec=0.6,
                                     psf_data=None, lens_radius=None, lens_id=None):
        """
        Enhanced field galaxy sampling using learned environment properties
        
        Args:
            central_redshift: Central galaxy redshift
            central_mass_log10: Central galaxy log10 mass
            n_max: Exact target number of field galaxies (caller-drawn)
            rng: Random number generator
            numpix: Image size in pixels
            pixel_scale: Arcsec per pixel
            avoid_center_arcsec: Minimum separation from center
            psf_data: PSF data for convolution
            lens_radius: Lens radius for size constraints
            lens_id: Lens ID for lens-specific sampling
            
        Returns:
            list: Field galaxy parameters with lenstronomy keys
                  (center_x, center_y, R_sersic, n_sersic, e1, e2, …)
        """
        if rng is None:
            rng = np.random.RandomState(42)
        
        # Predict environment properties (morphology / clustering context).
        # The *count* is supplied by the caller via n_max — already drawn from
        # COSMOS surface density × FOV × environment richness
        # (field_galaxy_count_target). Re-drawing with hardcoded absolute
        # caps (poisson 1.2 / max 2) under-populated Euclid 96px cutouts.
        env_props = self.predict_environment_properties(central_redshift, central_mass_log10)
        env_props.setdefault('max_radius_arcsec', 0.45 * numpix * pixel_scale)
        env_props.setdefault('env_type', 'isolated_field')

        n_field = max(0, int(n_max))
        if n_field == 0:
            return []
        
        # Prefer catalog rows for structural/photometric realism, but always
        # convert to the standard lenstronomy field-galaxy schema.
        raw = []
        if self.field_catalog is not None:
            raw = self._sample_from_catalog(
                n_field, lens_id, env_props, central_redshift, avoid_center_arcsec,
                rng, numpix, pixel_scale, lens_radius, psf_data)

        if len(raw) < n_field:
            # Top up with synthetics if catalog under-delivers
            need = n_field - len(raw)
            raw.extend(self._sample_synthetic(
                need, env_props, central_redshift, rng, numpix, pixel_scale))

        return self._normalize_to_lenstronomy_schema(
            raw[:n_field], rng, numpix, pixel_scale, avoid_center_arcsec,
            lens_radius, env_props, central_redshift)

    def _normalize_to_lenstronomy_schema(self, galaxies, rng, numpix, pixel_scale,
                                         avoid_center_arcsec, lens_radius, env_props,
                                         central_redshift):
        """Ensure every field galaxy has center_x/y, R_sersic, n_sersic, e1/e2."""
        half = 0.5 * numpix * pixel_scale
        max_r = float(env_props.get('max_radius_arcsec', 0.9 * half))
        min_r = max(0.3, float(avoid_center_arcsec))
        out = []
        for gal in galaxies:
            if not isinstance(gal, dict):
                continue
            # Already normalized (synthetic path)
            if all(k in gal for k in ('center_x', 'center_y', 'R_sersic', 'n_sersic', 'e1', 'e2')):
                out.append(gal)
                continue

            # Place randomly in the FOV annulus (catalog sep is JWST-scale)
            radius = rng.uniform(min_r, max(min_r + 0.1, min(max_r, half * 0.95)))
            angle = rng.uniform(0, 2 * np.pi)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Structural params from common COSMOS-Web / merged-catalog columns
            re = gal.get('R_sersic', gal.get('rearc_f150w', gal.get('rearc_f115w', np.nan)))
            try:
                re = float(re)
            except (TypeError, ValueError):
                re = np.nan
            if not np.isfinite(re) or re <= 0:
                re = float(rng.uniform(0.05, 0.6))
            if lens_radius is not None:
                re = min(re, 0.6 * float(lens_radius))
            # Floor at ~1.5 detector pixels so Euclid 0.1″/pix field members
            # are resolved galaxies, not single-pixel "beads".
            min_re = max(0.12, 1.5 * float(pixel_scale))
            re = float(np.clip(re, min_re, 1.2))

            n_ser = gal.get('n_sersic', gal.get('nsersic_f150w', gal.get('nsersic_f115w', np.nan)))
            try:
                n_ser = float(n_ser)
            except (TypeError, ValueError):
                n_ser = np.nan
            if not np.isfinite(n_ser):
                n_ser = float(rng.uniform(0.5, 4.0))
            n_ser = float(np.clip(n_ser, 0.3, 6.0))

            q = gal.get('axis_ratio', gal.get('qratio_f150w', gal.get('qratio_f115w', np.nan)))
            try:
                q = float(q)
            except (TypeError, ValueError):
                q = np.nan
            if not np.isfinite(q):
                q = float(rng.uniform(0.4, 1.0))
            q = float(np.clip(q, 0.2, 1.0))

            pa = gal.get('position_angle', gal.get('phi_f150w', gal.get('phi_f115w', rng.uniform(-180, 180))))
            try:
                pa = float(pa)
            except (TypeError, ValueError):
                pa = float(rng.uniform(-180, 180))
            e = (1.0 - q) / (1.0 + q)
            e1 = e * np.cos(2.0 * np.radians(pa))
            e2 = e * np.sin(2.0 * np.radians(pa))

            mag = gal.get('magnitude', gal.get('mag_f150w', gal.get('mag_f115w', np.nan)))
            try:
                mag = float(mag)
            except (TypeError, ValueError):
                mag = np.nan
            if not np.isfinite(mag):
                mag = float(rng.uniform(20.0, 26.0))

            z = gal.get('redshift', gal.get('field_redshift', gal.get('LP_zfinal', np.nan)))
            try:
                z = float(z)
            except (TypeError, ValueError):
                z = np.nan
            if not np.isfinite(z):
                if rng.random() < 0.5:
                    z = float(rng.uniform(0.1, max(0.15, central_redshift - 0.1)))
                else:
                    z = float(rng.uniform(central_redshift + 0.1, central_redshift + 1.5))

            out.append({
                'center_x': float(x),
                'center_y': float(y),
                'R_sersic': re,
                'n_sersic': n_ser,
                'e1': float(np.clip(e1, -0.7, 0.7)),
                'e2': float(np.clip(e2, -0.7, 0.7)),
                'axis_ratio': q,
                'position_angle': pa,
                'magnitude': float(mag),
                'redshift': float(z),
                'field_redshift': float(z),
                'env_type': env_props.get('env_type', 'isolated_field'),
                'is_synthetic': False,
                'from_catalog': True,
            })
        return out
    
    def _sample_from_catalog(self, n_field, lens_id, env_props, central_redshift,
                           avoid_center_arcsec, rng, numpix, pixel_scale, 
                           lens_radius, psf_data):
        """Sample field galaxies from real catalog"""
        
        # Filter for lens-specific galaxies if available
        if lens_id in self.field_catalog['lens_id'].values:
            lens_galaxies = self.field_catalog[self.field_catalog['lens_id'] == lens_id].copy()
        else:
            # Use general field population
            lens_galaxies = self.field_catalog.copy()
        
        # Apply environment-based radius constraint
        max_radius = env_props['max_radius_arcsec']
        if 'sep_arcsec' in lens_galaxies.columns:
            # Filter by separation
            radius_mask = (lens_galaxies['sep_arcsec'] <= max_radius) & \
                         (lens_galaxies['sep_arcsec'] > avoid_center_arcsec)
            lens_galaxies = lens_galaxies[radius_mask]
        
        # Filter by redshift (foreground/background)
        if 'LP_zfinal' in lens_galaxies.columns:
            # Sample mix of foreground and background galaxies
            foreground_mask = lens_galaxies['LP_zfinal'] < central_redshift
            background_mask = lens_galaxies['LP_zfinal'] > central_redshift
            
            # Determine mix based on environment
            if env_props['env_type'] == 'group':
                # Groups have more foreground galaxies
                fg_ratio = 0.7
            elif env_props['env_type'] == 'galaxy_pair':
                fg_ratio = 0.5
            else:
                fg_ratio = 0.3
            
            n_foreground = int(n_field * fg_ratio)
            n_background = n_field - n_foreground
            
            field_galaxies = []
            
            # Sample foreground galaxies
            if n_foreground > 0 and foreground_mask.sum() > 0:
                fg_galaxies = lens_galaxies[foreground_mask]
                n_fg_sample = min(n_foreground, len(fg_galaxies))
                fg_indices = rng.choice(len(fg_galaxies), size=n_fg_sample, replace=False)
                field_galaxies.extend(fg_galaxies.iloc[fg_indices].to_dict('records'))
            
            # Sample background galaxies
            if n_background > 0 and background_mask.sum() > 0:
                bg_galaxies = lens_galaxies[background_mask]
                n_bg_sample = min(n_background, len(bg_galaxies))
                bg_indices = rng.choice(len(bg_galaxies), size=n_bg_sample, replace=False)
                field_galaxies.extend(bg_galaxies.iloc[bg_indices].to_dict('records'))
            
            return field_galaxies[:n_field]
        
        # Fallback to random sampling
        if len(lens_galaxies) > 0:
            n_sample = min(n_field, len(lens_galaxies))
            indices = rng.choice(len(lens_galaxies), size=n_sample, replace=False)
            return lens_galaxies.iloc[indices].to_dict('records')
        
        return []
    
    def _sample_synthetic(self, n_field, env_props, central_redshift, rng, numpix, pixel_scale):
        """Generate synthetic field galaxies"""
        field_galaxies = []
        
        for i in range(n_field):
            # Generate position within environment radius
            max_radius = env_props['max_radius_arcsec']
            radius = rng.uniform(0.5, max_radius)
            angle = rng.uniform(0, 2 * np.pi)
            
            # center_x/center_y must be in arcseconds for lenstronomy
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Generate redshift (foreground or background)
            if rng.random() < 0.5:  # 50% foreground
                z = rng.uniform(0.1, central_redshift - 0.1)
            else:  # 50% background
                z = rng.uniform(central_redshift + 0.1, central_redshift + 1.0)
            
            # Generate galaxy properties
            # Estimate a plausible effective radius using CWMGs relation
            re_kpc = self._cwmgs_re_kpc(mass_log10=rng.uniform(9.5, 11.5), z=z, rng=rng)
            # Convert to arcsec
            da_mpc = self._angular_diameter_distance(z)
            theta_eff_arcsec = (re_kpc / 1000.0) / da_mpc * 206265.0

            axis_ratio = rng.uniform(0.5, 1.0)
            ellipticity = (1.0 - axis_ratio) / (1.0 + axis_ratio)
            e1 = ellipticity * np.cos(2.0 * angle)
            e2 = ellipticity * np.sin(2.0 * angle)

            galaxy = {
                'center_x': x,
                'center_y': y,
                'R_sersic': float(np.clip(theta_eff_arcsec, 0.03, 1.2)),
                'n_sersic': rng.uniform(0.5, 4.0),
                'e1': float(e1),
                'e2': float(e2),
                'axis_ratio': float(axis_ratio),
                'position_angle': float(np.degrees(angle)),
                'magnitude': rng.uniform(20, 26),
                'redshift': z,
                'env_type': env_props['env_type'],
                'is_synthetic': True
            }
            
            field_galaxies.append(galaxy)
        
        return field_galaxies
    
    def get_environment_statistics(self):
        """Get statistics about learned environments"""
        if not self.trained:
            return "Models not trained yet"
        
        # This would require access to the training data
        # For now, return basic info
        return {
            'trained': self.trained,
            'model_dir': str(self.model_dir),
            'field_catalog_loaded': self.field_catalog is not None
        }

def main():
    """Test the enhanced field sampler"""
    sampler = EnhancedFieldSampler()
    
    # Load models and data
    sampler.load_models()
    sampler.load_field_catalog()
    
    # Test sampling
    print("Testing enhanced field galaxy sampling...")
    
    test_cases = [
        (0.5, 11.0, "Low-z massive galaxy"),
        (1.0, 11.5, "Mid-z massive galaxy"), 
        (2.0, 10.5, "High-z massive galaxy")
    ]
    
    for z, mass, description in test_cases:
        print(f"\n{description} (z={z}, log10(M)={mass}):")
        
        # Predict environment
        env_props = sampler.predict_environment_properties(z, mass)
        print(f"  Environment: {env_props}")
        
        # Sample field galaxies
        field_galaxies = sampler.sample_field_galaxies_enhanced(
            z, mass, n_max=5, rng=np.random.RandomState(42)
        )
        print(f"  Sampled {len(field_galaxies)} field galaxies")
        
        if field_galaxies:
            redshifts = [g.get('redshift', 'N/A') for g in field_galaxies]
            print(f"  Field galaxy redshifts: {redshifts}")

    @staticmethod
    def _rng_integers(rng, low, high):
        """Compatibility helper for RandomState vs Generator APIs."""
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
            # Fallback for old numpy or edge case
            import warnings
            warnings.warn(f"RNG compatibility issue: {e}. Using uniform fallback.", UserWarning)
            return int(rng.uniform(low, high))

if __name__ == "__main__":
    main()
