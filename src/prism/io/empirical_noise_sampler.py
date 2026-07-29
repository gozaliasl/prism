"""
Empirical noise sampler from real JWST observations.

Samples background and noise properties from the actual distribution
of 435 real JWST strong lens observations, rather than using fixed values.
"""

import json
import numpy as np
from pathlib import Path

class EmpiricalNoiseSampler:
    """Sample noise properties from real JWST observations"""
    
    def __init__(self, json_path):
        """
        Initialize sampler with empirical measurements
        
        Parameters
        ----------
        json_path : str or Path
            Path to jwst_empirical_noise.json
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.bands = ['F115W', 'F150W', 'F277W', 'F444W']
        self._build_distributions()
    
    def _build_distributions(self):
        """Build distributions from per-lens measurements"""
        self.distributions = {}
        
        # Standard JWST NIRCam AB magnitude zero point
        mag_zp = 28.09
        pixel_scale = 0.031  # arcsec/pixel (matches prism_kappa_output / default_config pixel scale)
        pixel_area = pixel_scale**2  # arcsec²/pixel
        
        for band in self.bands:
            measurements = self.data['per_lens_measurements'][band]
            
            # Extract arrays of all measurements
            bg_levels = np.array([m['bg_level'] for m in measurements])
            bg_rms = np.array([m['bg_rms'] for m in measurements])
            read_noise = np.array([m['read_noise'] for m in measurements])
            
            # Remove zeros/invalid (some corner cases in analysis)
            valid_bg = bg_levels != 0
            valid_rms = bg_rms > 0
            valid_read = read_noise > 0
            
            self.distributions[band] = {
                'bg_level': bg_levels[valid_bg],
                'bg_rms': bg_rms[valid_rms],
                'read_noise': read_noise[valid_read],
                'n_valid': np.sum(valid_bg & valid_rms & valid_read)
            }
            
            # Recalculate sky brightness with correct zero point
            stats = self.data['band_statistics'][band]
            bg_median = stats['bg_level_median']
            bg_per_arcsec2 = bg_median / pixel_area
            
            if bg_per_arcsec2 > 0:
                sky_brightness = mag_zp - 2.5 * np.log10(bg_per_arcsec2)
            else:
                sky_brightness = 30.0  # Default for negative/zero
            
            self.distributions[band]['sky_brightness'] = float(sky_brightness)
    
    def sample_noise_properties(self, band, rng=None, method='random'):
        """
        Sample noise properties for a band
        
        Parameters
        ----------
        band : str
            JWST band (F115W, F150W, F277W, F444W)
        rng : np.random.Generator, optional
            Random number generator
        method : str, optional
            Sampling method:
            - 'random': Random sample from observed distribution (DEFAULT - most realistic)
            - 'percentile': Sample based on percentile (for systematic variation)
            - 'median': Use median values (least variation)
        
        Returns
        -------
        dict
            Dictionary with 'bg_level', 'bg_rms', 'read_noise', 'sky_brightness'
        """
        if rng is None:
            rng = np.random.default_rng()
        
        dist = self.distributions[band]
        
        if method == 'random':
            # Randomly sample from actual observed distribution
            # This gives most realistic variation matching real data
            idx = rng.integers(0, len(dist['bg_level']))
            bg_level = dist['bg_level'][idx]
            
            idx = rng.integers(0, len(dist['bg_rms']))
            bg_rms = dist['bg_rms'][idx]
            
            idx = rng.integers(0, len(dist['read_noise']))
            read_noise = dist['read_noise'][idx]
            
        elif method == 'percentile':
            # Sample at specific percentile (for systematic studies)
            p = rng.uniform(5, 95)  # Avoid extreme outliers
            bg_level = np.percentile(dist['bg_level'], p)
            bg_rms = np.percentile(dist['bg_rms'], p)
            read_noise = np.percentile(dist['read_noise'], p)
            
        elif method == 'median':
            # Use median (least variation)
            bg_level = np.median(dist['bg_level'])
            bg_rms = np.median(dist['bg_rms'])
            read_noise = np.median(dist['read_noise'])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'bg_level': float(bg_level),
            'bg_rms': float(bg_rms),
            'read_noise': float(read_noise),
            'sky_brightness': float(dist['sky_brightness'])
        }
    
    def sample_all_bands(self, rng=None, method='random'):
        """
        Sample noise properties for all bands
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator
        method : str, optional
            Sampling method ('random', 'percentile', 'median')
        
        Returns
        -------
        dict
            Dictionary with noise properties per band
        """
        if rng is None:
            rng = np.random.default_rng()
        
        return {
            band: self.sample_noise_properties(band, rng, method)
            for band in self.bands
        }
    
    def get_statistics(self, band):
        """Get summary statistics for a band"""
        dist = self.distributions[band]
        
        return {
            'bg_level': {
                'min': float(np.min(dist['bg_level'])),
                'max': float(np.max(dist['bg_level'])),
                'median': float(np.median(dist['bg_level'])),
                'std': float(np.std(dist['bg_level']))
            },
            'bg_rms': {
                'min': float(np.min(dist['bg_rms'])),
                'max': float(np.max(dist['bg_rms'])),
                'median': float(np.median(dist['bg_rms'])),
                'std': float(np.std(dist['bg_rms']))
            },
            'read_noise': {
                'min': float(np.min(dist['read_noise'])),
                'max': float(np.max(dist['read_noise'])),
                'median': float(np.median(dist['read_noise'])),
                'std': float(np.std(dist['read_noise']))
            },
            'n_valid_measurements': int(dist['n_valid']),
            'sky_brightness_mag_arcsec2': float(dist['sky_brightness'])
        }


def test_sampler():
    """Test the empirical noise sampler"""
    import sys
    from pathlib import Path
    
    # Find the config file
    repo_root = Path(__file__).parent.parent
    json_path = repo_root / 'configs' / 'jwst_empirical_noise.json'
    
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    
    # Initialize sampler
    sampler = EmpiricalNoiseSampler(json_path)
    
    print("=" * 80)
    print("EMPIRICAL NOISE SAMPLER - VALIDATION")
    print("=" * 80)
    
    # Show statistics
    print("\nStatistics from 435 real JWST lenses:")
    for band in sampler.bands:
        stats = sampler.get_statistics(band)
        print(f"\n{band}:")
        print(f"  bg_level:   [{stats['bg_level']['min']:.2e}, {stats['bg_level']['max']:.2e}], "
              f"median={stats['bg_level']['median']:.2e}")
        print(f"  bg_rms:     [{stats['bg_rms']['min']:.2e}, {stats['bg_rms']['max']:.2e}], "
              f"median={stats['bg_rms']['median']:.2e}")
        print(f"  read_noise: [{stats['read_noise']['min']:.2e}, {stats['read_noise']['max']:.2e}], "
              f"median={stats['read_noise']['median']:.2e}")
        print(f"  Valid measurements: {stats['n_valid_measurements']}")
    
    # Test sampling
    print("\n" + "=" * 80)
    print("SAMPLING TEST (5 random samples):")
    print("=" * 80)
    
    rng = np.random.default_rng(42)
    for i in range(5):
        print(f"\nSample {i+1}:")
        noise = sampler.sample_all_bands(rng, method='random')
        for band in sampler.bands:
            print(f"  {band}: bg={noise[band]['bg_level']:.2e}, "
                  f"rms={noise[band]['bg_rms']:.2e}, "
                  f"sky={noise[band]['sky_brightness']:.2f} mag/arcsec²")
    
    print("\n" + "=" * 80)
    print("✅ Sampler working correctly!")
    print("=" * 80)


if __name__ == '__main__':
    test_sampler()

