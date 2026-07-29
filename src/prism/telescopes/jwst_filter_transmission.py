"""
JWST Filter Transmission Curves and Photometry System

Implements realistic JWST NIRCam filter transmission curves with:
- Accurate wavelength-dependent transmission functions
- Multi-band photometry from SED convolution
- Filter-specific noise characteristics
- K-corrections accounting for filter shape
"""

import numpy as np
from scipy import interpolate
from typing import Dict, Tuple, Optional, List


class JWSTFilterSystem:
    """
    Realistic JWST NIRCam filter transmission system.
    Based on official STScI filter definitions.
    """
    
    # JWST NIRCam filter definitions
    # Format: {band: (center_wavelength_um, fwhm_um, profile_type)}
    FILTER_DEFINITIONS = {
        'F090W': (0.90, 0.156, 'gaussian'),      # Short wavelength
        'F115W': (1.15, 0.214, 'gaussian'),      # Default band 1
        'F150W': (1.50, 0.266, 'gaussian'),      # Default band 2
        'F200W': (2.00, 0.404, 'gaussian'),      # Medium wavelength
        'F277W': (2.77, 0.660, 'gaussian'),      # Default band 3
        'F356W': (3.56, 0.806, 'gaussian'),      # Long-medium
        'F444W': (4.44, 1.024, 'gaussian'),      # Default band 4 (reddest)
    }
    
    # Filter-specific noise characteristics (based on STScI ETC data)
    # Format: {band: {'zeropoint': AB mag, 'effective_exposure_time': seconds,
    #                  'background_per_sec': e-/s/pix, 'read_noise': e-}}
    FILTER_NOISE_PROPERTIES = {
        'F090W': {
            'zeropoint': 29.94,
            'background': 0.15,
            'read_noise': 8.5,
            'saturation_electrons': 180000,
            'pixel_sensitivity': 0.13,  # e-/ADU
            'excess_noise_factor': 1.0,
        },
        'F115W': {
            'zeropoint': 29.94,
            'background': 0.12,
            'read_noise': 8.5,
            'saturation_electrons': 180000,
            'pixel_sensitivity': 0.13,
            'excess_noise_factor': 1.0,
        },
        'F150W': {
            'zeropoint': 30.00,
            'background': 0.15,
            'read_noise': 8.5,
            'saturation_electrons': 180000,
            'pixel_sensitivity': 0.13,
            'excess_noise_factor': 1.0,
        },
        'F200W': {
            'zeropoint': 30.02,
            'background': 0.22,
            'read_noise': 8.8,
            'saturation_electrons': 180000,
            'pixel_sensitivity': 0.13,
            'excess_noise_factor': 1.05,  # Slightly higher noise
        },
        'F277W': {
            'zeropoint': 29.37,
            'background': 0.75,
            'read_noise': 11.0,
            'saturation_electrons': 180000,
            'pixel_sensitivity': 0.13,
            'excess_noise_factor': 1.15,  # Elevated thermal background
        },
        'F356W': {
            'zeropoint': 28.71,
            'background': 1.50,
            'read_noise': 11.0,
            'saturation_electrons': 180000,
            'pixel_sensitivity': 0.13,
            'excess_noise_factor': 1.30,  # Strong thermal background
        },
        'F444W': {
            'zeropoint': 27.87,
            'background': 2.80,
            'read_noise': 11.0,
            'saturation_electrons': 180000,
            'pixel_sensitivity': 0.13,
            'excess_noise_factor': 1.50,  # Very strong thermal background
        },
    }
    
    def __init__(self):
        """Initialize filter system with transmission curves."""
        self.transmission_curves = {}
        self._build_transmission_curves()
    
    def _build_transmission_curves(self):
        """Build realistic transmission curves for all JWST filters."""
        for band, (center, fwhm, profile) in self.FILTER_DEFINITIONS.items():
            if profile == 'gaussian':
                self.transmission_curves[band] = self._gaussian_transmission(center, fwhm)
            else:
                self.transmission_curves[band] = self._gaussian_transmission(center, fwhm)
    
    @staticmethod
    def _gaussian_transmission(center_um: float, fwhm_um: float, 
                               wavelength_range: Tuple[float, float] = None,
                               n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a realistic Gaussian-profile filter transmission curve.
        
        Parameters:
        -----------
        center_um : float
            Filter center wavelength in microns
        fwhm_um : float
            Filter full width at half maximum in microns
        wavelength_range : tuple
            (min, max) wavelengths in microns. Default is center ± 3*sigma
        n_points : int
            Number of wavelength samples
        
        Returns:
        --------
        wavelengths, transmission : tuple of arrays
            Wavelengths (μm) and transmission (0-1)
        """
        sigma = fwhm_um / (2 * np.sqrt(2 * np.log(2)))
        
        if wavelength_range is None:
            wavelength_range = (center_um - 4*sigma, center_um + 4*sigma)
        
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
        
        # Gaussian profile with peak transmission ~0.85 (realistic for coated filters)
        transmission = 0.85 * np.exp(-0.5 * ((wavelengths - center_um) / sigma) ** 2)
        
        return wavelengths, transmission
    
    def get_transmission(self, band: str, wavelengths: np.ndarray) -> np.ndarray:
        """
        Get transmission for a band at specified wavelengths.
        
        Parameters:
        -----------
        band : str
            Filter name (e.g., 'F150W')
        wavelengths : np.ndarray
            Wavelengths in microns
        
        Returns:
        --------
        transmission : np.ndarray
            Transmission at each wavelength (0-1)
        """
        if band not in self.transmission_curves:
            raise ValueError(f"Unknown filter: {band}")
        
        wave_ref, trans_ref = self.transmission_curves[band]
        
        # Interpolate transmission to requested wavelengths
        f_interp = interpolate.interp1d(
            wave_ref, trans_ref,
            kind='cubic', bounds_error=False, fill_value=0.0
        )
        
        return f_interp(wavelengths)
    
    def convolve_sed_to_magnitude(self, wavelengths: np.ndarray, 
                                   sed_flux: np.ndarray,
                                   band: str,
                                   redshift: float = 0.0) -> float:
        """
        Convert SED spectrum to magnitude in a specific JWST band.
        
        Performs proper SED-filter convolution:
        mag = -2.5 * log10(∫ SED(λ) * T(λ) dλ / ∫ T(λ) * Fν_ref(λ) dλ)
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Rest-frame wavelengths (μm)
        sed_flux : np.ndarray
            SED flux (arbitrary units, e.g., f_ν)
        band : str
            Filter name
        redshift : float
            Galaxy redshift for K-correction
        
        Returns:
        --------
        magnitude : float
            AB magnitude in specified band
        """
        if band not in self.FILTER_NOISE_PROPERTIES:
            raise ValueError(f"Unknown filter: {band}")

        # NOTE (Phase 2 investigation, see
        # analysis/sim_obs_comparison/reports/phase1_real_vs_sim_comparison.md):
        # physically, the observed-frame filter should be redshifted into the
        # rest frame before convolving with the rest-frame SED (T(lambda_rest
        # * (1+z))), and _calculate_kcorrection below is presently a no-op
        # (its two terms cancel exactly for all z/bands). Attempted this fix
        # and it made colors worse given the current SED/extinction
        # calibration -- left as rest-frame (no shift) pending SED
        # recalibration (Phase 3).
        transmission = self.get_transmission(band, wavelengths)

        # Flux-weighted average through filter, normalized by the filter's
        # own bandpass integral (∫T dλ). Without this normalization, bands
        # with wider transmission curves (e.g. F444W, FWHM~1.0um) integrate
        # to systematically larger "numerator" values than narrow bands
        # (e.g. F115W, FWHM~0.21um) purely from bandpass width -- not from
        # the SED's actual flux -- which was producing colors ~1.4 mag too
        # red across all SED types (color_F115W_F444W ~2.0-3.4 vs real ~0.5).
        transmission_integral = np.trapz(transmission, wavelengths)
        numerator = np.trapz(sed_flux * transmission, wavelengths) / transmission_integral

        if numerator <= 0:
            return 99.0  # Non-detection

        # Zero-point flux (AB system reference)
        zp = self.FILTER_NOISE_PROPERTIES[band]['zeropoint']

        # Magnitude calculation
        magnitude = zp - 2.5 * np.log10(numerator)
        
        # Apply K-correction for redshift
        if redshift > 0:
            k_corr = self._calculate_kcorrection(wavelengths, sed_flux, band, redshift)
            magnitude += k_corr
        
        return float(magnitude)
    
    @staticmethod
    def _calculate_kcorrection(wavelengths: np.ndarray,
                               sed_flux: np.ndarray,
                               band: str,
                               redshift: float) -> float:
        """
        Calculate K-correction for bandpass and redshift.
        
        K-correction accounts for the fact that at higher redshift,
        we observe longer rest-frame wavelengths through the same filter.
        
        K = 2.5 * log10((1+z)) + color_correction_term
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Rest-frame wavelengths
        sed_flux : np.ndarray
            SED flux
        band : str
            Filter band
        redshift : float
            Galaxy redshift
        
        Returns:
        --------
        k_corr : float
            K-correction in magnitudes
        """
        if redshift < 0.01:  # No correction for z ~ 0
            return 0.0
        
        # Luminosity distance dimming correction
        luminosity_distance_factor = 2.5 * np.log10(1 + redshift)
        
        # Wavelength shift: observed = rest * (1+z)
        # This affects which part of SED is sampled
        wavelength_shift_factor = -2.5 * np.log10(1 + redshift)
        
        # Total K-correction
        k_corr = luminosity_distance_factor + wavelength_shift_factor
        
        return k_corr
    
    def get_multiband_photometry(self, wavelengths: np.ndarray,
                                  sed_flux: np.ndarray,
                                  bands: List[str],
                                  redshift: float = 0.0) -> Dict[str, float]:
        """
        Calculate multi-band photometry from SED.
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Rest-frame wavelengths (μm)
        sed_flux : np.ndarray
            SED flux (arbitrary units)
        bands : list of str
            Filter names
        redshift : float
            Galaxy redshift
        
        Returns:
        --------
        photometry : dict
            {band_name: magnitude} for each band
        """
        photometry = {}
        for band in bands:
            mag = self.convolve_sed_to_magnitude(wavelengths, sed_flux, band, redshift)
            photometry[band] = mag
        
        return photometry
    
    def get_noise_properties(self, band: str) -> Dict[str, float]:
        """
        Get noise characteristics for a filter.
        
        Parameters:
        -----------
        band : str
            Filter name
        
        Returns:
        --------
        properties : dict
            Noise properties including zero-point, background, read noise
        """
        if band not in self.FILTER_NOISE_PROPERTIES:
            raise ValueError(f"Unknown filter: {band}")
        
        return self.FILTER_NOISE_PROPERTIES[band].copy()
    
    def calculate_color_from_transmission(self, wavelengths: np.ndarray,
                                          sed_flux: np.ndarray,
                                          band1: str, band2: str,
                                          redshift: float = 0.0) -> float:
        """
        Calculate color (magnitude difference) from SED and filter transmission.
        
        Color = mag(band1) - mag(band2)
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Rest-frame wavelengths (μm)
        sed_flux : np.ndarray
            SED flux
        band1, band2 : str
            Filter names
        redshift : float
            Galaxy redshift
        
        Returns:
        --------
        color : float
            Magnitude difference in ABs
        """
        mag1 = self.convolve_sed_to_magnitude(wavelengths, sed_flux, band1, redshift)
        mag2 = self.convolve_sed_to_magnitude(wavelengths, sed_flux, band2, redshift)
        
        return mag1 - mag2
    
    def estimate_background_limited_magnitude(self, band: str,
                                              exposure_time: float = 10000.0,
                                              num_pixels: float = 1.0) -> float:
        """
        Estimate background-limited magnitude detection threshold.
        
        Parameters:
        -----------
        band : str
            Filter name
        exposure_time : float
            Total exposure time in seconds
        num_pixels : float
            Number of pixels in aperture
        
        Returns:
        --------
        limit_mag : float
            5-sigma detection limit
        """
        props = self.get_noise_properties(band)
        
        # Background noise in electrons
        background_electrons = (props['background'] * exposure_time * 
                               props['excess_noise_factor'] * np.sqrt(num_pixels))
        
        # Read noise contribution
        read_noise_electrons = props['read_noise'] * np.sqrt(num_pixels)
        
        # Total noise
        total_noise = np.sqrt(background_electrons**2 + read_noise_electrons**2)
        
        # 5-sigma detection
        sigma5_electrons = 5 * total_noise
        
        # Convert electrons to magnitude (AB)
        zp = props['zeropoint']
        mag_limit = zp - 2.5 * np.log10(sigma5_electrons)
        
        return mag_limit


# Create global filter system instance
JWST_FILTERS_SYSTEM = JWSTFilterSystem()


def get_filter_transmission_curve(band: str, 
                                   wavelength_range: Tuple[float, float] = None,
                                   n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get transmission curve for a JWST filter.
    
    Parameters:
    -----------
    band : str
        Filter name (e.g., 'F150W')
    wavelength_range : tuple
        (min, max) wavelengths in microns
    n_points : int
        Number of wavelength samples
    
    Returns:
    --------
    wavelengths, transmission : tuple of arrays
    """
    return JWST_FILTERS_SYSTEM.transmission_curves[band]


def convolve_sed_to_magnitude(wavelengths: np.ndarray,
                              sed_flux: np.ndarray,
                              band: str,
                              redshift: float = 0.0) -> float:
    """Convert SED to magnitude through filter convolution."""
    return JWST_FILTERS_SYSTEM.convolve_sed_to_magnitude(
        wavelengths, sed_flux, band, redshift
    )


def get_filter_noise_properties(band: str) -> Dict[str, float]:
    """Get noise characteristics for a filter."""
    return JWST_FILTERS_SYSTEM.get_noise_properties(band)
