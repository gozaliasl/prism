"""
detector_chain.py — Physically realistic detector/camera effect pipeline

Implements the full signal chain:
  Photons → PSF → IPC → charge diffusion → brighter-fatter
           → non-linearity → dark current → read noise + 1/f
           → gain/ADC → PRNU/flat field → persistence → pipeline output

Supported telescopes:
  - JWST NIRCam (HgCdTe H2RG, ~0.031 arcsec/pix SW, ~0.063 LW)
  - Roman WFI   (HgCdTe H4RG, ~0.11 arcsec/pix)
  - Euclid VIS  (silicon CCD, ~0.1 arcsec/pix)
  - Subaru HSC  (silicon CCD, ~0.168 arcsec/pix)
  - LSST/Rubin  (silicon CCD, ~0.2 arcsec/pix)

References:
  Donlon et al. 2018 (JWST IPC); Plazas et al. 2018 (brighter-fatter);
  Rauscher et al. 2017 (JWST detector characterisation);
  Mandelbaum et al. 2015 (HSC); Ivezić et al. 2019 (LSST);
  Euclid Collaboration 2022; Akeson et al. 2019 (Roman).
"""

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Detector parameter tables
# ---------------------------------------------------------------------------

# JWST NIRCam: Rauscher et al. 2017, NIRCam ETC calibration
JWST_PARAMS = {
    # IPC coupling fraction to each of 4 cardinal neighbours (H2RG measured)
    'ipc_alpha': 0.0173,          # ~1.73 % per neighbour (SW channel)
    # Charge diffusion sigma in pixels (sub-pixel broadening during integration)
    'charge_diff_sigma': 0.25,    # pixels, conservative SW value
    # Brighter-fatter coefficient  Δr²/flux  [pix² per e-]
    'bfe_alpha': 4.0e-7,
    # Non-linearity polynomial coefficients  (response relative to linear)
    # DN_true = c0*DN + c1*DN² + c2*DN³   (c0≈1, c1,c2 from ground cal)
    'nl_coeffs': [1.0, -1.2e-6, 5.0e-13],
    # Full-well / saturation in electrons
    'full_well': 83_000,          # H2RG SW measured median
    # Gain in e-/ADU
    'gain': 2.05,                 # NIRCam SW measured
    # ADC bit depth
    'adc_bits': 16,
    # Dark current  e-/s  (cryogenic, very low for HgCdTe)
    'dark_current': 0.0022,
    # Read noise per frame  e-  (correlated double sampling)
    'read_noise_cds': 13.0,
    # 1/f noise: row-correlated amplitude (e-)
    'one_f_amplitude': 0.6,
    # 1/f noise: column-correlated amplitude (e-)
    'one_f_col_amplitude': 0.2,
    # PRNU (pixel response non-uniformity) rms fraction
    'prnu_rms': 0.015,
    # Persistence: fraction of full-well trapped per pixel
    'persistence_fraction': 2.0e-4,
    # Persistence decay time constant (seconds)
    'persistence_tau': 500.0,
    # Per-band gain correction factors (LW slightly different)
    'band_gain_correction': {
        'F090W': 1.00, 'F115W': 1.00, 'F150W': 1.00, 'F200W': 1.00,
        'F277W': 0.97, 'F300M': 0.97, 'F356W': 0.97, 'F360M': 0.97,
        'F444W': 0.95, 'F470N': 0.95,
    },
    'pixel_scale': 0.031,
}

# Roman WFI: Akeson et al. 2019; Roman Science Requirements Document
ROMAN_PARAMS = {
    'ipc_alpha': 0.020,           # H4RG, slightly higher than H2RG
    'charge_diff_sigma': 0.20,
    'bfe_alpha': 3.5e-7,
    'nl_coeffs': [1.0, -1.0e-6, 4.0e-13],
    'full_well': 100_000,
    'gain': 1.458,
    'adc_bits': 16,
    'dark_current': 0.010,
    'read_noise_cds': 12.0,
    'one_f_amplitude': 0.5,
    'one_f_col_amplitude': 0.15,
    'prnu_rms': 0.012,
    'persistence_fraction': 1.5e-4,
    'persistence_tau': 600.0,
    'band_gain_correction': {},
    'pixel_scale': 0.11,
}

# Euclid VIS: CCD273 detector (silicon, e2v); Euclid Collaboration 2022
EUCLID_PARAMS = {
    'ipc_alpha': 0.0,             # CCDs have no IPC (charge well isolation)
    'charge_diff_sigma': 0.30,    # slightly larger for thick CCD
    'bfe_alpha': 2.5e-7,          # brighter-fatter in silicon
    'nl_coeffs': [1.0, -2.0e-7, 0.0],   # CCDs more linear than HgCdTe
    'full_well': 175_000,
    'gain': 3.1,
    'adc_bits': 16,
    'dark_current': 0.001,        # CCD at 153 K: very low
    'read_noise_cds': 4.5,        # e- rms, CCD much lower than HgCdTe
    'one_f_amplitude': 0.1,       # CCD: negligible 1/f
    'one_f_col_amplitude': 0.05,
    'prnu_rms': 0.005,
    'persistence_fraction': 0.0,  # CCDs have no persistence
    'persistence_tau': 1.0,
    'band_gain_correction': {},
    'pixel_scale': 0.10,
}

# Subaru HSC: Miyazaki et al. 2018 (HAMAMATSU fully-depleted CCD)
SUBARU_HSC_PARAMS = {
    'ipc_alpha': 0.0,
    'charge_diff_sigma': 0.45,    # thick fully-depleted CCD → more diffusion
    'bfe_alpha': 5.0e-7,          # strong BFE in thick silicon
    'nl_coeffs': [1.0, -3.0e-7, 0.0],
    'full_well': 150_000,
    'gain': 3.0,
    'adc_bits': 16,
    'dark_current': 0.0023,
    'read_noise_cds': 4.5,
    'one_f_amplitude': 0.08,
    'one_f_col_amplitude': 0.04,
    'prnu_rms': 0.007,
    'persistence_fraction': 0.0,
    'persistence_tau': 1.0,
    'band_gain_correction': {},
    'pixel_scale': 0.168,
}

# LSST/Rubin Observatory: Ivezić et al. 2019; LSST SRD
# ITL/e2v CCDs, 189 sensors, 3.2 Gpix focal plane
LSST_PARAMS = {
    'ipc_alpha': 0.0,
    'charge_diff_sigma': 0.40,    # fully-depleted high-resistivity CCD
    'bfe_alpha': 6.0e-7,          # strong BFE (100 μm thick CCD)
    'nl_coeffs': [1.0, -4.0e-7, 0.0],
    'full_well': 100_000,
    'gain': 1.7,
    'adc_bits': 18,               # 18-bit ADC
    'dark_current': 0.002,
    'read_noise_cds': 9.0,        # e-  (target spec)
    'one_f_amplitude': 0.15,
    'one_f_col_amplitude': 0.06,
    'prnu_rms': 0.008,
    'persistence_fraction': 0.0,
    'persistence_tau': 1.0,
    # LSST 30 s visits; sky noise dominates so BFE/persistence matter less
    'band_gain_correction': {},
    'pixel_scale': 0.200,
    # LSST-specific: sky brightness per band (AB mag/arcsec²)
    'sky_brightness': {
        'u': 22.96, 'g': 22.26, 'r': 21.20, 'i': 20.48, 'z': 19.60, 'y': 18.61
    },
    # Nominal single-visit 5σ depth (AB mag)
    'single_visit_depth': {
        'u': 23.9, 'g': 25.0, 'r': 24.7, 'i': 24.0, 'z': 23.3, 'y': 22.1
    },
}

TELESCOPE_PARAMS = {
    'jwst':    JWST_PARAMS,
    'roman':   ROMAN_PARAMS,
    'euclid':  EUCLID_PARAMS,
    'subaru':  SUBARU_HSC_PARAMS,
    'lsst':    LSST_PARAMS,
}


# ---------------------------------------------------------------------------
# Helper: build IPC convolution kernel
# ---------------------------------------------------------------------------

def _ipc_kernel(alpha: float) -> np.ndarray:
    """
    3×3 IPC kernel.
    Central pixel retains (1 - 4*alpha) of its charge;
    each cardinal neighbour receives alpha.
    """
    k = np.zeros((3, 3), dtype=np.float64)
    k[1, 1] = 1.0 - 4.0 * alpha
    k[0, 1] = k[2, 1] = k[1, 0] = k[1, 2] = alpha
    return k


def _charge_diff_kernel(sigma: float) -> np.ndarray:
    """
    Gaussian charge diffusion kernel (3×3, normalised).
    """
    sz = 7
    x = np.arange(sz) - sz // 2
    xx, yy = np.meshgrid(x, x)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= k.sum()
    return k


# ---------------------------------------------------------------------------
# Core detector chain class
# ---------------------------------------------------------------------------

class DetectorChain:
    """
    Applies the full physical detector signal chain to a noiseless flux image.

    Usage
    -----
    chain = DetectorChain('jwst', band='F150W', rng=rng, exposure_time=1028.0)
    output_image = chain.apply(flux_image_e_per_s)

    The input ``flux_image`` must be in **electrons per second** (e-/s).
    The output is in **electrons per second** (e-/s), matching the lenstronomy
    convention so it slots in directly after sim.image_model_class().image().

    Parameters
    ----------
    telescope : str
        One of 'jwst', 'roman', 'euclid', 'subaru', 'lsst'
    band : str
        Filter name (e.g. 'F150W'). Used for band-specific gain correction.
    rng : np.random.Generator
    exposure_time : float
        Exposure time in seconds (used to convert flux → electrons).
    numpix : int
        Image side length in pixels (for PRNU map caching).
    persistence_map : np.ndarray or None
        Pre-accumulated persistence charge map in electrons from previous
        exposures. If None, no persistence is added.
    seed_prnu : int or None
        Fixed seed for the PRNU map so the same detector tile is used
        across all images in a run (physically correct).
    enabled : dict
        Override which effects are on/off, e.g.
        ``{'ipc': False, 'persistence': False}``.
    """

    _PRNU_CACHE: dict = {}      # class-level cache keyed by (telescope, numpix, seed)

    def __init__(self, telescope: str = 'jwst', band: str = 'F150W',
                 rng: np.random.Generator = None,
                 exposure_time: float = 1028.0,
                 numpix: int = 300,
                 persistence_map: np.ndarray = None,
                 seed_prnu: int = 0,
                 enabled: dict = None):

        if telescope not in TELESCOPE_PARAMS:
            raise ValueError(f"Unknown telescope '{telescope}'. "
                             f"Choose from: {list(TELESCOPE_PARAMS)}")

        self.telescope = telescope
        self.band      = band.upper()
        self.rng       = rng or np.random.default_rng(42)
        self.t_exp     = float(exposure_time)
        self.numpix    = int(numpix)
        self.params    = TELESCOPE_PARAMS[telescope]
        self.persistence_map = persistence_map  # e-

        # Which effects are active
        defaults = dict(
            ipc=True, charge_diffusion=True, brighter_fatter=True,
            nonlinearity=True, dark_current=True,
            poisson_shot_noise=True,
            read_noise=True, one_f_noise=True,
            gain_adc=True, prnu=True, persistence=True,
            saturation=True,
        )
        if enabled:
            defaults.update(enabled)
        self.enabled = defaults

        # Pre-build kernels
        alpha = self.params['ipc_alpha']
        self._ipc_kernel = _ipc_kernel(alpha) if alpha > 0 else None

        sigma_cd = self.params['charge_diff_sigma']
        self._cd_kernel = _charge_diff_kernel(sigma_cd) if sigma_cd > 0 else None

        # PRNU map (fixed for this detector tile / seed)
        prnu_key = (telescope, numpix, seed_prnu)
        if prnu_key not in DetectorChain._PRNU_CACHE:
            prng = np.random.default_rng(seed_prnu + hash(telescope) % (2**31))
            prnu = prng.normal(1.0, self.params['prnu_rms'],
                               (numpix, numpix)).astype(np.float32)
            prnu = np.clip(prnu, 0.5, 1.5)
            DetectorChain._PRNU_CACHE[prnu_key] = prnu
        self._prnu_map = DetectorChain._PRNU_CACHE[prnu_key]

        # Per-band gain correction
        bg_corr = self.params.get('band_gain_correction', {})
        self._gain_corr = float(bg_corr.get(self.band, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, flux_image: np.ndarray) -> np.ndarray:
        """
        Run the complete detector chain on ``flux_image`` [e-/s].

        Returns the processed image in e-/s (same units as input).
        """
        p = self.params
        im = np.array(flux_image, dtype=np.float64)

        # 1. Convert to total electrons accumulated during exposure
        im_e = im * self.t_exp

        # 2. IPC — charge coupling between adjacent pixels (HgCdTe only)
        if self.enabled['ipc'] and self._ipc_kernel is not None:
            im_e = self._apply_ipc(im_e)

        # 3. Charge diffusion — sub-pixel Gaussian broadening
        if self.enabled['charge_diffusion'] and self._cd_kernel is not None:
            im_e = self._apply_charge_diffusion(im_e)

        # 4. Brighter-fatter effect — flux-dependent PSF broadening
        if self.enabled['brighter_fatter']:
            im_e = self._apply_brighter_fatter(im_e)

        # 5. Non-linearity — polynomial detector response curve
        if self.enabled['nonlinearity']:
            im_e = self._apply_nonlinearity(im_e)

        # 6. Dark current — thermal electrons accumulated during integration
        if self.enabled['dark_current']:
            im_e = self._apply_dark_current(im_e)

        # 7. Poisson shot noise on signal + background
        if self.enabled['poisson_shot_noise']:
            im_e = self._apply_poisson_noise(im_e)

        # 8. Read noise + structured 1/f noise
        if self.enabled['read_noise']:
            im_e = self._apply_read_noise(im_e)
        if self.enabled['one_f_noise']:
            im_e = self._apply_one_f_noise(im_e)

        # 9. Saturation clipping (before gain so it's in electrons)
        if self.enabled['saturation']:
            im_e = np.minimum(im_e, float(p['full_well']))

        # 10. PRNU / flat-field pixel-to-pixel sensitivity variations
        if self.enabled['prnu']:
            im_e = self._apply_prnu(im_e)

        # 11. Persistence from previous bright exposures
        if self.enabled['persistence'] and self.persistence_map is not None:
            im_e = self._apply_persistence(im_e)

        # 12. Gain / ADC quantization
        if self.enabled['gain_adc']:
            im_e = self._apply_gain_adc(im_e)

        # 13. Return to e-/s units (pipeline convention)
        out = im_e / self.t_exp
        return out.astype(np.float32)

    def make_persistence_map(self, flux_image: np.ndarray) -> np.ndarray:
        """
        Compute the persistence charge map left behind after this exposure.
        Call this after apply() and pass the result to the next epoch's chain.
        Returns: persistence charge [e-] to add to next exposure.
        """
        p = self.params
        if p['persistence_fraction'] == 0:
            return np.zeros_like(flux_image)
        im_e = np.array(flux_image, dtype=np.float64) * self.t_exp
        # Charge trapped is proportional to the fraction of full-well filled
        fill_fraction = np.clip(im_e / p['full_well'], 0, 1)
        trapped = fill_fraction * p['full_well'] * p['persistence_fraction']
        return trapped.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal effect methods
    # ------------------------------------------------------------------

    def _apply_ipc(self, im_e: np.ndarray) -> np.ndarray:
        return convolve(im_e, self._ipc_kernel, mode='reflect')

    def _apply_charge_diffusion(self, im_e: np.ndarray) -> np.ndarray:
        return fftconvolve(im_e, self._cd_kernel, mode='same')

    def _apply_brighter_fatter(self, im_e: np.ndarray) -> np.ndarray:
        """
        Brighter-fatter: bright pixels spread charge to neighbours.
        Implemented as a local, flux-dependent blurring kernel.
        The effective PSF variance scales as:  σ²_eff = σ²_PSF + α * N_e
        We approximate this as an additional Gaussian convolution whose
        sigma varies spatially with the local flux.
        """
        alpha = self.params['bfe_alpha']
        if alpha <= 0:
            return im_e
        # Extra sigma in pixels from BFE for each pixel
        extra_sigma = np.sqrt(np.maximum(alpha * im_e, 0.0))
        # Apply a mild spatial blur proportional to mean flux
        mean_extra = float(np.percentile(extra_sigma, 95))
        if mean_extra < 0.01:
            return im_e
        k = _charge_diff_kernel(np.clip(mean_extra, 0.05, 1.5))
        return fftconvolve(im_e, k, mode='same')

    def _apply_nonlinearity(self, im_e: np.ndarray) -> np.ndarray:
        """
        Apply polynomial non-linearity correction (forward model).
        c = [c0, c1, c2]:  measured = c0*true + c1*true² + c2*true³
        We apply the forward model (converting true electrons to measured counts).
        """
        c = self.params['nl_coeffs']
        fw = float(self.params['full_well'])
        x = np.clip(im_e / fw, 0, 1)           # normalise to [0,1]
        nl_frac = c[0] + c[1] * fw * x + c[2] * (fw**2) * x**2
        return im_e * nl_frac

    def _apply_dark_current(self, im_e: np.ndarray) -> np.ndarray:
        dc_total = self.params['dark_current'] * self.t_exp
        dark = self.rng.poisson(dc_total, im_e.shape).astype(np.float64)
        return im_e + dark

    def _apply_poisson_noise(self, im_e: np.ndarray) -> np.ndarray:
        """True Poisson shot noise on the signal (replaces Gaussian approx)."""
        positive = np.maximum(im_e, 0.0)
        # For large counts, Poisson ≈ Normal; only use rng.poisson for <1e6
        noisy = np.where(
            positive < 1e6,
            self.rng.poisson(positive).astype(np.float64),
            positive + self.rng.normal(0, np.sqrt(positive + 1e-12))
        )
        return noisy

    def _apply_read_noise(self, im_e: np.ndarray) -> np.ndarray:
        rn = self.params['read_noise_cds']
        return im_e + self.rng.normal(0.0, rn, im_e.shape)

    def _apply_one_f_noise(self, im_e: np.ndarray) -> np.ndarray:
        """
        Structured 1/f noise:
          - Row-correlated component (horizontal banding) — dominant in HgCdTe
          - Column-correlated component (vertical banding) — amplifier crosstalk
        Each row/column gets the same 1/f noise realisation (correlated pattern).
        """
        n = self.numpix

        # ---- row-correlated (horizontal stripes) ----
        amp_row = self.params['one_f_amplitude']
        if amp_row > 0:
            freqs = np.fft.rfftfreq(n)
            freqs[0] = freqs[1]          # avoid division by zero at DC
            power = np.sqrt(1.0 / freqs)
            power[0] = 0.0              # no DC offset
            coeffs_r = self.rng.normal(0, 1, len(power)) + \
                       1j * self.rng.normal(0, 1, len(power))
            row_pattern = np.fft.irfft(coeffs_r * power, n=n).real
            row_pattern *= amp_row / (row_pattern.std() + 1e-12)
            im_e = im_e + row_pattern[np.newaxis, :]   # broadcast over rows

        # ---- column-correlated (vertical stripes) ----
        amp_col = self.params['one_f_col_amplitude']
        if amp_col > 0:
            freqs = np.fft.rfftfreq(n)
            freqs[0] = freqs[1]
            power = np.sqrt(1.0 / freqs)
            power[0] = 0.0
            coeffs_c = self.rng.normal(0, 1, len(power)) + \
                       1j * self.rng.normal(0, 1, len(power))
            col_pattern = np.fft.irfft(coeffs_c * power, n=n).real
            col_pattern *= amp_col / (col_pattern.std() + 1e-12)
            im_e = im_e + col_pattern[:, np.newaxis]   # broadcast over columns

        return im_e

    def _apply_prnu(self, im_e: np.ndarray) -> np.ndarray:
        """Pixel Response Non-Uniformity: per-pixel QE variation."""
        return im_e * self._prnu_map

    def _apply_persistence(self, im_e: np.ndarray) -> np.ndarray:
        """Add residual charge from a previous bright exposure."""
        p = self.params
        tau = p['persistence_tau']
        # Decay factor: how much of the trapped charge leaks into this exposure
        # For a single subsequent exposure starting at t=0:
        # charge_released ∝ (1 - exp(-t_exp/tau))
        decay = 1.0 - np.exp(-self.t_exp / tau) if tau > 0 else 1.0
        return im_e + self.persistence_map * decay

    def _apply_gain_adc(self, im_e: np.ndarray) -> np.ndarray:
        """Convert e- to ADU with gain, quantize (integer), convert back."""
        p = self.params
        gain = p['gain'] * self._gain_corr
        adc_max = 2**p['adc_bits'] - 1
        adu = np.clip(np.round(im_e / gain), 0, adc_max).astype(np.int32)
        return adu.astype(np.float64) * gain    # back to e- (quantized)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_detector_chain(telescope: str, band: str,
                        rng: np.random.Generator,
                        exposure_time: float,
                        numpix: int = 300,
                        persistence_map: np.ndarray = None,
                        seed_prnu: int = 0,
                        enabled: dict = None) -> DetectorChain:
    """
    Factory function. Returns a ready-to-use DetectorChain.

    Parameters
    ----------
    telescope : str
        'jwst' | 'roman' | 'euclid' | 'subaru' | 'lsst'
    band : str
        Filter name.
    rng : np.random.Generator
    exposure_time : float
        Seconds.
    numpix : int
        Image size in pixels.
    persistence_map : np.ndarray, optional
        Persistence electrons from previous exposure.
    seed_prnu : int
        Seed for the fixed PRNU map (same seed → same detector tile).
    enabled : dict, optional
        Override specific effects. E.g. ``{'persistence': False}``.

    Returns
    -------
    DetectorChain
    """
    telescope = telescope.lower()
    return DetectorChain(
        telescope=telescope, band=band, rng=rng,
        exposure_time=exposure_time, numpix=numpix,
        persistence_map=persistence_map,
        seed_prnu=seed_prnu, enabled=enabled
    )


# ---------------------------------------------------------------------------
# Per-telescope parameter introspection helper
# ---------------------------------------------------------------------------

def describe_telescope(telescope: str) -> str:
    """Print a formatted summary of detector parameters for a telescope."""
    if telescope not in TELESCOPE_PARAMS:
        return f"Unknown telescope '{telescope}'"
    p = TELESCOPE_PARAMS[telescope]
    lines = [f"\n{'='*60}", f"  Detector parameters: {telescope.upper()}", f"{'='*60}"]
    for k, v in p.items():
        if k == 'band_gain_correction':
            continue
        lines.append(f"  {k:<30s} {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    numpix = 64
    # Synthetic lens image: Gaussian profile
    x = np.arange(numpix) - numpix // 2
    xx, yy = np.meshgrid(x, x)
    flux = 0.05 * np.exp(-(xx**2 + yy**2) / (2 * 5**2))   # e-/s

    for tel in TELESCOPE_PARAMS:
        chain = make_detector_chain(tel, band='F150W', rng=rng,
                                    exposure_time=1028.0, numpix=numpix)
        out = chain.apply(flux)
        snr_peak = float(out.max() / np.std(out[out < out.max() * 0.01]))
        print(f"{tel:8s}  out max={out.max():.4e}  noise rms={np.std(out):.4e}  "
              f"peak SNR~{snr_peak:.1f}")
    print("\nAll detector chains OK.")
