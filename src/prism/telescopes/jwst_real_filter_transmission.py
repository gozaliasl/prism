"""
Enhanced JWST NIRCam Filter Transmission System with Real STScI Throughput Curves

This module loads actual JWST NIRCam system throughput curves from STScI data
and provides accurate photometric calculations based on real filter transmission.

Data source: JWST NIRCam Throughput Curves (May 2024)
- Location: data/nircam_throughputs/mean_throughputs/
- Format: ASCII text with wavelength (microns) and throughput columns
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import trapezoid
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings


class RealJWSTFilterSystem:
    """
    JWST NIRCam filter transmission system using real STScI throughput curves.
    
    Loads actual filter transmission data from ASCII files and provides:
    - Accurate wavelength-dependent transmission
    - Real system throughput (optics + detector + filters)
    - Multi-band photometry calculations
    - K-corrections with real filter profiles
    """
    
    # Default data path (relative to project root)
    DEFAULT_DATA_PATH = Path("data/nircam_throughputs/mean_throughputs")
    
    # NIRCam filters with standard wavelengths (for reference)
    FILTER_WAVELENGTHS = {
        'F070W': 0.704,   'F090W': 0.902,   'F115W': 1.154,  'F140M': 1.404,
        'F150W': 1.501,   'F150W2': 1.501,  'F162M': 1.621,  'F164N': 1.644,
        'F182M': 1.821,   'F187N': 1.874,   'F200W': 2.002,  'F210M': 2.104,
        'F212N': 2.119,   'F250M': 2.503,   'F277W': 2.776,  'F300M': 2.996,
        'F322W2': 3.235,  'F323N': 3.237,   'F335M': 3.355,  'F356W': 3.568,
        'F360M': 3.621,   'F405N': 4.054,   'F410M': 4.102,  'F430M': 4.302,
        'F444W': 4.421,   'F460M': 4.624,   'F466N': 4.659,  'F470N': 4.707,
        'F480M': 4.801,
    }
    
    # Filter categories
    SHORT_WAVELENGTH_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2']
    MEDIUM_WAVELENGTH_FILTERS = ['F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
    LONG_WAVELENGTH_FILTERS = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 
                                'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 'F444W',
                                'F460M', 'F466N', 'F470N', 'F480M']
    
    def __init__(self, data_path: Optional[Path] = None, verbose: bool = True):
        """
        Initialize filter system with real throughput curves.
        
        Parameters:
        -----------
        data_path : Path or str
            Path to directory containing throughput data files
        verbose : bool
            Print loading information
        """
        self.verbose = verbose
        self.transmission_curves = {}
        self.available_filters = []
        
        # Determine the data path - prefer absolute path relative to workspace root
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Construct path relative to this file's location
            current_dir = Path(__file__).parent  # src/
            workspace_root = current_dir.parent   # project root
            self.data_path = workspace_root / self.DEFAULT_DATA_PATH
        
        if not self.data_path.exists():
            warnings.warn(f"Data path not found: {self.data_path}")
            return
        
        self._load_all_filters()
    
    def _load_all_filters(self):
        """Load all available filter throughput curves."""
        loaded_count = 0
        failed_filters = []
        
        for filter_name in sorted(self.FILTER_WAVELENGTHS.keys()):
            try:
                success = self._load_filter(filter_name)
                if success:
                    loaded_count += 1
                    self.available_filters.append(filter_name)
                else:
                    failed_filters.append(filter_name)
            except Exception as e:
                failed_filters.append(f"{filter_name} ({str(e)})")
        
        if self.verbose:
            print(f"✓ Loaded {loaded_count} real JWST NIRCam filter curves")
            if failed_filters:
                print(f"  ⚠ Could not load: {', '.join(failed_filters[:5])}")
                if len(failed_filters) > 5:
                    print(f"    ... and {len(failed_filters)-5} more")
    
    def _load_filter(self, filter_name: str) -> bool:
        """
        Load throughput curve for a single filter.
        
        Parameters:
        -----------
        filter_name : str
            Filter name (e.g., 'F150W')
        
        Returns:
        --------
        bool
            True if successfully loaded, False otherwise
        """
        # Construct filename pattern
        filename = self.data_path / f"{filter_name}_May2024_mean_system_throughput.txt"
        
        if not filename.exists():
            return False
        
        try:
            # Load ASCII data: wavelength (microns) and throughput
            data = np.loadtxt(filename, skiprows=1, dtype=float)
            
            if data.shape[1] != 2:
                return False
            
            wavelengths = data[:, 0]  # microns
            throughput = data[:, 1]   # transmission (0-1)
            
            # Store the interpolation function
            self.transmission_curves[filter_name] = (wavelengths, throughput)
            
            return True
            
        except Exception as e:
            if self.verbose and filter_name in ['F150W', 'F277W', 'F444W']:  # Print only for key filters
                print(f"  Warning loading {filter_name}: {e}")
            return False
    
    def get_transmission(self, band: str, wavelengths: np.ndarray) -> np.ndarray:
        """
        Get real transmission for a filter at specified wavelengths.
        
        Parameters:
        -----------
        band : str
            Filter name (e.g., 'F150W')
        wavelengths : np.ndarray
            Wavelengths in microns where transmission is needed
        
        Returns:
        --------
        transmission : np.ndarray
            Transmission at each wavelength (0-1), interpolated from real curves
        """
        if band not in self.transmission_curves:
            raise ValueError(f"Filter {band} not available. Available: {self.available_filters}")
        
        wave_ref, trans_ref = self.transmission_curves[band]
        
        # Create interpolation function from real data
        # Use linear interpolation to preserve real curve features
        f_interp = interpolate.interp1d(
            wave_ref, trans_ref,
            kind='linear',  # Linear interpolation preserves curve shape
            bounds_error=False,
            fill_value=0.0
        )
        
        return f_interp(wavelengths)
    
    def get_filter_properties(self, band: str) -> Dict:
        """
        Get properties of a filter from its transmission curve.
        
        Parameters:
        -----------
        band : str
            Filter name
        
        Returns:
        --------
        properties : dict
            Contains center wavelength, FWHM, peak transmission, bandpass width
        """
        if band not in self.transmission_curves:
            raise ValueError(f"Filter {band} not available")
        
        wavelengths, transmission = self.transmission_curves[band]
        
        # Find peak transmission
        peak_idx = np.argmax(transmission)
        peak_transmission = transmission[peak_idx]
        peak_wavelength = wavelengths[peak_idx]
        
        # Find wavelength range at 50% peak transmission (FWHM)
        half_max = peak_transmission * 0.5
        above_half = wavelengths[transmission >= half_max]
        
        if len(above_half) > 0:
            fwhm = above_half[-1] - above_half[0]
            blue_edge = above_half[0]
            red_edge = above_half[-1]
        else:
            fwhm = 0
            blue_edge = wavelengths[0]
            red_edge = wavelengths[-1]
        
        # Effective wavelength (flux-weighted mean)
        effective_wavelength = trapezoid(wavelengths * transmission, wavelengths) / trapezoid(transmission, wavelengths)
        
        # Bandpass width (from 1% transmission)
        bandpass_01pct = wavelengths[transmission >= 0.01 * peak_transmission]
        if len(bandpass_01pct) > 0:
            bandpass_width = bandpass_01pct[-1] - bandpass_01pct[0]
        else:
            bandpass_width = fwhm
        
        return {
            'filter': band,
            'center_wavelength_um': self.FILTER_WAVELENGTHS.get(band, peak_wavelength),
            'peak_wavelength_um': peak_wavelength,
            'effective_wavelength_um': effective_wavelength,
            'peak_transmission': peak_transmission,
            'fwhm_um': fwhm,
            'blue_edge_um': blue_edge,
            'red_edge_um': red_edge,
            'bandpass_width_um': bandpass_width,
            'wavelength_range': (wavelengths[0], wavelengths[-1]),
        }
    
    def convolve_sed_to_magnitude(self, wavelengths: np.ndarray,
                                   sed_flux: np.ndarray,
                                   band: str,
                                   zeropoint: Optional[float] = None,
                                   redshift: float = 0.0) -> float:
        """
        Convert SED to magnitude using real filter transmission.
        
        Performs proper SED-filter convolution with real throughput curves:
        mag = -2.5 * log10(∫ SED(λ) * T(λ) dλ / reference_flux)
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Rest-frame wavelengths (μm)
        sed_flux : np.ndarray
            SED flux (arbitrary units)
        band : str
            Filter name
        zeropoint : float
            AB magnitude zeropoint (if None, uses value from filter properties)
        redshift : float
            Galaxy redshift for K-correction
        
        Returns:
        --------
        magnitude : float
            Magnitude in specified band
        """
        if band not in self.transmission_curves:
            raise ValueError(f"Filter {band} not available")

        # `wavelengths`/`sed_flux` describe the galaxy's REST-FRAME SED. To get
        # the magnitude actually observed through a JWST filter, the SED must
        # first be redshifted into the observed frame (lambda_obs = lambda_rest
        # * (1+z)) -- this is what shifts spectral features (4000A break,
        # Balmer break, dust bump, PAH lines, etc.) into/out of each filter
        # bandpass and is the actual physical origin of observed galaxy colour
        # as a function of redshift. A flat post-hoc K-correction (identical
        # for every band) cannot reproduce this -- it cancels out of any
        # colour and leaves every galaxy with the same (rest-frame) SED shape
        # regardless of z.
        wave_obs = wavelengths * (1.0 + redshift)
        transmission = self.get_transmission(band, wave_obs)

        # Flux-weighted average through filter (integrate in the observed frame;
        # bandpass-averaged flux is what the detector measures)
        numerator = trapezoid(sed_flux * transmission, wave_obs)

        if numerator <= 0 or np.isnan(numerator):
            return 99.0  # Non-detection

        # Use provided zeropoint or default (AB system)
        if zeropoint is None:
            zeropoint = 0.0  # Relative magnitudes if no zeropoint specified

        # Magnitude calculation (relative to zeropoint)
        magnitude = zeropoint - 2.5 * np.log10(numerator)

        # Dimming from cosmological surface-brightness/flux-density scaling
        # (applies equally to all bands -- affects overall normalisation, not colour)
        if redshift > 0.01:
            magnitude += -2.5 * np.log10(1.0 / (1.0 + redshift))

        return float(magnitude)
    
    def get_multiband_photometry(self, wavelengths: np.ndarray,
                                  sed_flux: np.ndarray,
                                  bands: List[str],
                                  zeropoints: Optional[Dict[str, float]] = None,
                                  redshift: float = 0.0) -> Dict[str, float]:
        """
        Calculate multi-band photometry from SED using real filters.
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Rest-frame wavelengths (μm)
        sed_flux : np.ndarray
            SED flux (arbitrary units)
        bands : list of str
            Filter names to calculate
        zeropoints : dict
            {band: zeropoint} for each filter (if None, uses relative mags)
        redshift : float
            Galaxy redshift
        
        Returns:
        --------
        photometry : dict
            {band_name: magnitude} for each band
        """
        if zeropoints is None:
            zeropoints = {b: 0.0 for b in bands}
        
        photometry = {}
        for band in bands:
            if band not in self.available_filters:
                warnings.warn(f"Filter {band} not available, skipping")
                continue
            
            zp = zeropoints.get(band, 0.0)
            mag = self.convolve_sed_to_magnitude(
                wavelengths, sed_flux, band, zeropoint=zp, redshift=redshift
            )
            photometry[band] = mag
        
        return photometry
    
    def calculate_color_from_transmission(self, wavelengths: np.ndarray,
                                          sed_flux: np.ndarray,
                                          band1: str, band2: str,
                                          redshift: float = 0.0) -> float:
        """
        Calculate color (magnitude difference) using real filter transmission.
        
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
        mag1 = self.convolve_sed_to_magnitude(wavelengths, sed_flux, band1, redshift=redshift)
        mag2 = self.convolve_sed_to_magnitude(wavelengths, sed_flux, band2, redshift=redshift)
        
        return mag1 - mag2
    
    def get_filter_categories(self) -> Dict[str, List[str]]:
        """
        Get filters organized by wavelength category.
        
        Returns:
        --------
        categories : dict
            'short': F070W-F150W2
            'medium': F162M-F212N
            'long': F250M-F480M
        """
        return {
            'short': [f for f in self.SHORT_WAVELENGTH_FILTERS if f in self.available_filters],
            'medium': [f for f in self.MEDIUM_WAVELENGTH_FILTERS if f in self.available_filters],
            'long': [f for f in self.LONG_WAVELENGTH_FILTERS if f in self.available_filters],
        }
    
    def list_available_filters(self) -> Dict[str, Dict]:
        """
        List all available filters with their properties.
        
        Returns:
        --------
        filters_info : dict
            {filter_name: properties} for each available filter
        """
        filters_info = {}
        for band in self.available_filters:
            filters_info[band] = self.get_filter_properties(band)
        return filters_info


# Global instance - loads on import
REAL_JWST_FILTERS = RealJWSTFilterSystem(verbose=True)


def load_real_filters(data_path: Optional[str] = None) -> RealJWSTFilterSystem:
    """
    Create and return a real filter system instance.
    
    Parameters:
    -----------
    data_path : str
        Path to NIRCam throughput data directory
    
    Returns:
    --------
    filter_system : RealJWSTFilterSystem
    """
    return RealJWSTFilterSystem(data_path=data_path, verbose=True)


def get_real_transmission(band: str, wavelengths: np.ndarray) -> np.ndarray:
    """Get real transmission curve for a filter."""
    return REAL_JWST_FILTERS.get_transmission(band, wavelengths)


def get_filter_properties_real(band: str) -> Dict:
    """Get properties from real transmission curve."""
    return REAL_JWST_FILTERS.get_filter_properties(band)
