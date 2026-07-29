"""
Multi-telescope filter transmission system.
Loads real throughput curves for JWST NIRCam, Roman WFI, and Subaru Suprime-Cam.
"""

import numpy as np
from pathlib import Path
from scipy.integrate import trapezoid

class MultiTelescopeFilterSystem:
    """
    Unified filter system supporting multiple telescopes with real transmission curves.
    """
    
    def __init__(self, workspace_root=None):
        """
        Initialize multi-telescope filter system.
        
        Parameters
        ----------
        workspace_root : Path or str, optional
            Root directory of the workspace (for resolving relative paths)
        """
        if workspace_root is None:
            # Try to find workspace root
            current = Path(__file__).resolve().parent
            while current != current.parent:
                if (current / 'data').exists():
                    workspace_root = current
                    break
                current = current.parent
            else:
                workspace_root = Path(__file__).resolve().parent.parent
        
        self.workspace_root = Path(workspace_root)
        self.data_dir = self.workspace_root / 'data'
        
        # Define filter directories
        self.filter_dirs = {
            'jwst': self.data_dir / 'nircam_throughputs',
            'roman': self.data_dir / 'roman_filters',
            'ground_based': self.data_dir / 'subaru_suprime_filters',
            'subaru': self.data_dir / 'subaru_suprime_filters',
            'euclid': self.data_dir / 'euclid_filters',
            'lsst': self.data_dir / 'lsst_filters',
        }
        # Telescopes whose filter files have wavelengths in Angstroms (not microns)
        self._angstrom_telescopes = {'roman', 'ground_based', 'subaru', 'lsst'}
        # Specific filter files in nanometres (NISP passband tables)
        self._nanometre_filters = {
            'EUCLID_Y', 'EUCLID_J', 'EUCLID_H',   # NISP passband files are in nm
        }
        # VIS file is in Angstroms; NISP files are in nm — both handled below
        
        # Filter name mappings (uppercase names to file patterns)
        self.filter_files = {
            # JWST NIRCam (29 filters)
            'F070W': 'jwst/nircam_F070W_CLEAR_05242024_ModAB_mean.txt',
            'F090W': 'jwst/nircam_F090W_CLEAR_05242024_ModAB_mean.txt',
            'F115W': 'jwst/nircam_F115W_CLEAR_05242024_ModAB_mean.txt',
            'F140M': 'jwst/nircam_F140M_CLEAR_05242024_ModAB_mean.txt',
            'F150W': 'jwst/nircam_F150W_CLEAR_05242024_ModAB_mean.txt',
            'F162M': 'jwst/nircam_F162M_CLEAR_05242024_ModAB_mean.txt',
            'F182M': 'jwst/nircam_F182M_CLEAR_05242024_ModAB_mean.txt',
            'F200W': 'jwst/nircam_F200W_CLEAR_05242024_ModAB_mean.txt',
            'F210M': 'jwst/nircam_F210M_CLEAR_05242024_ModAB_mean.txt',
            'F250M': 'jwst/nircam_F250M_CLEAR_05242024_ModAB_mean.txt',
            'F277W': 'jwst/nircam_F277W_CLEAR_05242024_ModAB_mean.txt',
            'F300M': 'jwst/nircam_F300M_CLEAR_05242024_ModAB_mean.txt',
            'F322W2': 'jwst/nircam_F322W2_CLEAR_05242024_ModAB_mean.txt',
            'F323N': 'jwst/nircam_F323N_CLEAR_05242024_ModAB_mean.txt',
            'F335M': 'jwst/nircam_F335M_CLEAR_05242024_ModAB_mean.txt',
            'F356W': 'jwst/nircam_F356W_CLEAR_05242024_ModAB_mean.txt',
            'F360M': 'jwst/nircam_F360M_CLEAR_05242024_ModAB_mean.txt',
            'F405N': 'jwst/nircam_F405N_CLEAR_05242024_ModAB_mean.txt',
            'F410M': 'jwst/nircam_F410M_CLEAR_05242024_ModAB_mean.txt',
            'F430M': 'jwst/nircam_F430M_CLEAR_05242024_ModAB_mean.txt',
            'F444W': 'jwst/nircam_F444W_CLEAR_05242024_ModAB_mean.txt',
            'F460M': 'jwst/nircam_F460M_CLEAR_05242024_ModAB_mean.txt',
            'F466N': 'jwst/nircam_F466N_CLEAR_05242024_ModAB_mean.txt',
            'F470N': 'jwst/nircam_F470N_CLEAR_05242024_ModAB_mean.txt',
            'F480M': 'jwst/nircam_F480M_CLEAR_05242024_ModAB_mean.txt',
            
            # Roman WFI (8 filters)
            'ROMAN_F062': 'roman/Roman_WFI.F062.dat',
            'ROMAN_F087': 'roman/Roman_WFI.F087.dat',
            'ROMAN_F106': 'roman/Roman_WFI.F106.dat',
            'ROMAN_F129': 'roman/Roman_WFI.F129.dat',
            'ROMAN_F146': 'roman/Roman_WFI.F146.dat',
            'ROMAN_F158': 'roman/Roman_WFI.F158.dat',
            'ROMAN_F184': 'roman/Roman_WFI.F184.dat',
            'ROMAN_F213': 'roman/Roman_WFI.F213.dat',
            
            # Subaru Suprime-Cam (7 filters)
            'SUBARU_B': 'ground_based/Subaru_Suprime.B_filter.dat',
            'SUBARU_V': 'ground_based/Subaru_Suprime.V_filter.dat',
            'SUBARU_G': 'ground_based/Subaru_Suprime.g_filter.dat',
            'SUBARU_R': 'ground_based/Subaru_Suprime.r_filter.dat',
            'SUBARU_I': 'ground_based/Subaru_Suprime.i_filter.dat',
            'SUBARU_Z': 'ground_based/Subaru_Suprime.z_filter.dat',
            'SUBARU_Y': 'ground_based/Subaru_Suprime.Y_filter.dat',
            
            # Euclid (4 filters: VIS + 3 NISP)
            'EUCLID_VIS': 'euclid/Euclid_VIS.vis.dat',
            'EUCLID_Y': 'euclid/NISP-PHOTO-PASSBANDS-V1-Y_throughput.dat',
            'EUCLID_J': 'euclid/NISP-PHOTO-PASSBANDS-V1-J_throughput.dat',
            'EUCLID_H': 'euclid/NISP-PHOTO-PASSBANDS-V1-H_throughput.dat',

            # LSST / Rubin Observatory (6 filters; Ivezić et al. 2019)
            'LSST_U': 'lsst/LSST_U.dat',
            'LSST_G': 'lsst/LSST_G.dat',
            'LSST_R': 'lsst/LSST_R.dat',
            'LSST_I': 'lsst/LSST_I.dat',
            'LSST_Z': 'lsst/LSST_Z.dat',
            'LSST_Y': 'lsst/LSST_Y.dat',
        }
        
        # Central wavelengths (microns) for each filter
        self.filter_wavelengths = {
            # JWST NIRCam
            'F070W': 0.704, 'F090W': 0.901, 'F115W': 1.154, 'F140M': 1.404,
            'F150W': 1.501, 'F162M': 1.626, 'F182M': 1.845, 'F200W': 1.990,
            'F210M': 2.093, 'F250M': 2.503, 'F277W': 2.786, 'F300M': 2.996,
            'F322W2': 3.247, 'F323N': 3.237, 'F335M': 3.365, 'F356W': 3.563,
            'F360M': 3.621, 'F405N': 4.055, 'F410M': 4.082, 'F430M': 4.280,
            'F444W': 4.421, 'F460M': 4.624, 'F466N': 4.654, 'F470N': 4.707,
            'F480M': 4.834,
            
            # Roman WFI (Angstroms in file → convert to microns)
            'ROMAN_F062': 0.62, 'ROMAN_F087': 0.87, 'ROMAN_F106': 1.06,
            'ROMAN_F129': 1.29, 'ROMAN_F146': 1.46, 'ROMAN_F158': 1.58,
            'ROMAN_F184': 1.84, 'ROMAN_F213': 2.13,
            
            # Subaru Suprime-Cam (Angstroms in file → convert to microns)
            'SUBARU_B': 0.445, 'SUBARU_V': 0.551, 'SUBARU_G': 0.477,
            'SUBARU_R': 0.623, 'SUBARU_I': 0.764, 'SUBARU_Z': 0.907,
            'SUBARU_Y': 0.999,
            
            # Euclid (VIS optical + NISP NIR)
            'EUCLID_VIS': 0.700, 'EUCLID_Y': 1.020, 'EUCLID_J': 1.250, 'EUCLID_H': 1.650,

            # LSST / Rubin (Ivezić et al. 2019)
            'LSST_U': 0.367, 'LSST_G': 0.482, 'LSST_R': 0.622,
            'LSST_I': 0.754, 'LSST_Z': 0.869, 'LSST_Y': 0.971,
        }
        
        # Cache for loaded transmission curves
        self._transmission_cache = {}
        
    def load_filter_transmission(self, filter_name):
        """
        Load transmission curve for a specific filter.
        
        Parameters
        ----------
        filter_name : str
            Filter name (e.g., 'F150W', 'ROMAN_F106', 'SUBARU_G')
        
        Returns
        -------
        wavelength : np.ndarray
            Wavelength in microns
        transmission : np.ndarray
            Transmission (0-1 or 0-100 depending on file format)
        """
        filter_upper = filter_name.upper()
        
        if filter_upper in self._transmission_cache:
            return self._transmission_cache[filter_upper]
        
        if filter_upper not in self.filter_files:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        # Get file path
        rel_path = self.filter_files[filter_upper]
        telescope = rel_path.split('/')[0]
        filename = rel_path.split('/')[1]
        
        file_path = self.filter_dirs[telescope] / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Filter file not found: {file_path}")
        
        # Load data
        data = np.loadtxt(file_path)
        wavelength_file = data[:, 0]
        transmission = data[:, 1]
        
        # Convert wavelength to microns
        if filter_upper in self._nanometre_filters:
            wavelength_microns = wavelength_file / 1000.0   # nm → μm
        elif telescope in self._angstrom_telescopes:
            wavelength_microns = wavelength_file / 10000.0  # Å → μm
        elif telescope == 'euclid':
            # VIS file is in Angstroms
            wavelength_microns = wavelength_file / 10000.0
        else:
            # JWST files are already in microns
            wavelength_microns = wavelength_file
        
        # Normalize transmission to 0-1 if needed (some files have percentages)
        if np.max(transmission) > 10:
            transmission = transmission / 100.0
        
        self._transmission_cache[filter_upper] = (wavelength_microns, transmission)
        return wavelength_microns, transmission
    
    def get_filter_properties(self, filter_name):
        """
        Get filter properties including central wavelength and effective bandwidth.
        
        Parameters
        ----------
        filter_name : str
            Filter name
        
        Returns
        -------
        dict : Filter properties (lambda_eff, bandwidth, etc.)
        """
        filter_upper = filter_name.upper()
        
        # Load transmission curve
        wavelength, transmission = self.load_filter_transmission(filter_upper)
        
        # Calculate effective wavelength (transmission-weighted)
        lambda_eff = trapezoid(wavelength * transmission, wavelength) / trapezoid(transmission, wavelength)
        
        # Calculate effective bandwidth (FWHM approximation)
        half_max = np.max(transmission) / 2.0
        above_half = transmission >= half_max
        if np.any(above_half):
            bandwidth = wavelength[above_half][-1] - wavelength[above_half][0]
        else:
            bandwidth = 0.1 * lambda_eff  # Fallback
        
        return {
            'lambda_eff': lambda_eff,
            'lambda_pivot': lambda_eff,
            'bandwidth': bandwidth,
            'wavelength': wavelength,
            'transmission': transmission
        }
    
    def convolve_sed_to_magnitude(self, filter_name, sed_wavelength, sed_flux):
        """
        Convolve SED with filter transmission to compute magnitude.
        
        Parameters
        ----------
        filter_name : str
            Filter name
        sed_wavelength : np.ndarray
            SED wavelength in microns
        sed_flux : np.ndarray
            SED flux (arbitrary units, will be normalized)
        
        Returns
        -------
        float : Flux through filter (integrated)
        """
        # Load filter transmission
        filter_wave, filter_trans = self.load_filter_transmission(filter_name)
        
        # Interpolate SED onto filter wavelength grid
        sed_interp = np.interp(filter_wave, sed_wavelength, sed_flux, left=0, right=0)
        
        # Compute flux through filter
        flux_through_filter = trapezoid(sed_interp * filter_trans * filter_wave, filter_wave)
        normalization = trapezoid(filter_trans * filter_wave, filter_wave)
        
        return flux_through_filter / normalization if normalization > 0 else 0.0


# Global instance
MULTI_TELESCOPE_FILTERS = None

def get_multi_telescope_filters(workspace_root=None):
    """Get or create global multi-telescope filter system."""
    global MULTI_TELESCOPE_FILTERS
    if MULTI_TELESCOPE_FILTERS is None:
        MULTI_TELESCOPE_FILTERS = MultiTelescopeFilterSystem(workspace_root)
    return MULTI_TELESCOPE_FILTERS
