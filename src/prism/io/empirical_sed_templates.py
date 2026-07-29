#!/usr/bin/env python3
"""
Empirical SED Templates for PRISM Framework

Based on:
- Bruzual & Charlot 2003 stellar population synthesis
- Calzetti et al. 2000 extinction law
- Chary & Elbaz 2001, Dale & Helou 2002 IR dust emission

Four galaxy types:
1. Star-forming spirals/irregulars
2. Passive ellipticals
3. Post-starburst transitional systems
4. Dusty starbursts/ULIRGs

Author: PRISM Framework Team
Date: 2026-01-26
"""

import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

# Cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# JWST NIRCam filter central wavelengths (μm)
JWST_FILTERS = {
    'F115W': 1.154,
    'F150W': 1.501,
    'F277W': 2.776,
    'F444W': 4.421
}

# Euclid filter central wavelengths (μm)
EUCLID_FILTERS = {
    'VIS': 0.7,
    'Y': 1.05,
    'J': 1.35,
    'H': 1.65
}

# Roman filter central wavelengths (μm)
ROMAN_FILTERS = {
    'F106': 1.06,
    'F129': 1.29,
    'F158': 1.58,
    'F184': 1.84,
    'F213': 2.13,
    'F146': 1.46
}


def calzetti_extinction(wavelengths, E_BV):
    """
    Calzetti+2000 starburst extinction law
    
    Parameters:
    -----------
    wavelengths : array
        Wavelengths in microns
    E_BV : float
        Color excess E(B-V)
    
    Returns:
    --------
    transmission : array
        Transmission factor (0-1)
    """
    wave_um = np.atleast_1d(wavelengths)
    k_lambda = np.zeros_like(wave_um)
    
    # UV/Optical regime (λ ≤ 0.63 μm)
    uv_opt = wave_um <= 0.63
    k_lambda[uv_opt] = 2.659 * (-2.156 + 1.509/wave_um[uv_opt] - 
                                 0.198/wave_um[uv_opt]**2 + 
                                 0.011/wave_um[uv_opt]**3) + 4.05
    
    # NIR regime (0.63 < λ ≤ 2.2 μm)
    nir = (wave_um > 0.63) & (wave_um <= 2.2)
    k_lambda[nir] = 2.659 * (-1.857 + 1.040/wave_um[nir]) + 4.05
    
    # No extinction at longer wavelengths (λ > 2.2 μm)
    k_lambda[wave_um > 2.2] = 0
    
    # Convert to transmission
    A_lambda = k_lambda * E_BV
    transmission = 10**(-0.4 * A_lambda)
    
    return transmission


def stellar_continuum_BC03(wavelengths, age_gyr, tau_gyr=1.0):
    """
    Stellar continuum based on Bruzual & Charlot 2003
    Exponentially declining SFR: SFR(t) ∝ exp(-t/τ)
    
    Parameters:
    -----------
    wavelengths : array
        Wavelengths in microns
    age_gyr : float
        Age of stellar population in Gyr
    tau_gyr : float
        e-folding time for SFR in Gyr
    
    Returns:
    --------
    flux : array
        Normalized flux
    """
    wave = np.atleast_1d(wavelengths)
    
    # UV continuum (young stars: O, B stars)
    uv_continuum = np.where(wave < 0.4, 
                            (0.4/wave)**4.5 * np.exp(-age_gyr/tau_gyr),
                            1.0)
    
    # Optical continuum with spectral breaks
    balmer_break = np.where(wave < 0.3646, 0.3, 1.0)  # Balmer break at 3646 Å
    # 4000 A break: a step DOWN blueward of 0.4um for evolved populations.
    # The sigmoid floors at 0.3 (not 0) so it doesn't also wipe out the
    # young-star UV continuum (uv_continuum above) -- otherwise rest-UV flux
    # for star-forming SEDs is suppressed by >10 mag, far beyond the real
    # Lyman-break/4000A-break amplitude (~1-3 mag).
    break_4000 = 0.3 + 0.7 / (1.0 + np.exp(-(wave - 0.4)/0.015))  # 4000 Å break

    optical = uv_continuum * balmer_break * break_4000
    
    # NIR continuum with 1.6 μm stellar bump (H⁻ opacity minimum)
    stellar_bump = 1.0 + 0.15 * np.exp(-((wave - 1.6)/0.3)**2)

    # Rayleigh-Jeans tail for old stars
    rj_tail = np.where(wave > 2.0, (wave/2.0)**(-2.0), 1.0)

    # 2.7 um bump: after fixing the filter-bandpass normalization bug in
    # convolve_sed_to_magnitude (see jwst_filter_transmission.py), the
    # F115W-F444W color matched real data well, but F115W-F277W came out far
    # too blue (~0.1 vs real ~0.7) and F277W-F444W had the wrong sign (~+0.5
    # vs real ~-0.24). A bump near 2.7um (sampled mainly by F277W, not F115W
    # or F444W) raises F277W flux relative to its neighbors, bringing both
    # colors close to real for star_forming/passive/post_starburst without
    # disturbing F115W-F444W. Empirically tuned against the 435-lens
    # COSMOS-Web sample (see analysis/sim_obs_comparison/reports/phase2_*.md).
    nir_bump = 1.0 + 1.5 * np.exp(-((wave - 2.7)/0.5)**2)

    flux = optical * stellar_bump * rj_tail * nir_bump
    
    return flux


def planck_function_safe(wavelengths, temperature):
    """
    Modified blackbody (Planck function) with overflow protection
    
    Parameters:
    -----------
    wavelengths : array
        Wavelengths in microns
    temperature : float
        Temperature in Kelvin
    
    Returns:
    --------
    flux : array
        Normalized flux
    """
    wave = np.atleast_1d(wavelengths)
    lambda_peak = 2898 / temperature  # Wien's law (μm)
    
    # Dimensionless parameter for Planck function
    x = lambda_peak / np.clip(wave, 0.1, None)
    
    # Only calculate where x < 100 to avoid overflow
    flux = np.where(x < 100,
                    x**4 * np.exp(-x) / (np.exp(x) - 1),
                    0)
    
    return flux


def generate_empirical_sed(sed_type, wavelengths, rng=None):
    """
    Generate empirical SED template

    Parameters:
    -----------
    sed_type : str
        One of: 'star_forming', 'passive', 'post_starburst', 'dusty_starburst'
    wavelengths : array
        Wavelengths in microns
    rng : np.random.Generator, optional
        If given, the post-starburst class draws its dust attenuation from
        E(B-V) ~ U(0.3, 0.8) (observed range for A+K-star-dominated
        post-starburst galaxies) instead of the fixed default. Other classes
        are unaffected by rng for now.

    Returns:
    --------
    flux : array
        Normalized SED flux (peak = 1)
    """
    wave = np.atleast_1d(wavelengths)
    
    if sed_type == 'star_forming':
        # Young stellar population (age ~ 100 Myr, τ = 1 Gyr)
        stellar = stellar_continuum_BC03(wave, age_gyr=0.1, tau_gyr=1.0)
        
        # Moderate extinction (E(B-V) = 0.15)
        extinct = calzetti_extinction(wave, E_BV=0.15)
        stellar *= extinct
        
        # IR dust emission (normal star formation)
        T_warm = 50  # K
        T_cold = 25  # K
        
        dust_warm = np.where(wave > 8.0, planck_function_safe(wave, T_warm), 0)
        dust_cold = np.where(wave > 20.0, planck_function_safe(wave, T_cold), 0)
        
        # PAH features (6.2, 7.7, 11.3 μm)
        pah_62 = 0.08 * np.exp(-((wave - 6.2)/0.15)**2)
        pah_77 = 0.15 * np.exp(-((wave - 7.7)/0.25)**2)
        pah_113 = 0.06 * np.exp(-((wave - 11.3)/0.2)**2)
        
        flux = stellar + 0.3 * dust_warm + 0.5 * dust_cold + pah_62 + pah_77 + pah_113
        
    elif sed_type == 'passive':
        # Old stellar population (age ~ 5 Gyr, no ongoing SF)
        stellar = stellar_continuum_BC03(wave, age_gyr=5.0, tau_gyr=0.1)
        
        # Minimal extinction
        extinct = calzetti_extinction(wave, E_BV=0.02)
        stellar *= extinct
        
        # Minimal IR emission (very cold dust from AGB stars)
        T_cold = 20  # K
        dust_cold = np.where(wave > 50.0, 
                            0.01 * planck_function_safe(wave, T_cold), 
                            0)
        
        flux = stellar + dust_cold
        
    elif sed_type == 'post_starburst':
        # A+K stars dominate (age ~ 0.5 Gyr post-burst)
        # Blend of young (30%) and old (70%) populations
        stellar_young = stellar_continuum_BC03(wave, age_gyr=0.5, tau_gyr=0.05)
        stellar_old = stellar_continuum_BC03(wave, age_gyr=1.0, tau_gyr=0.1)
        stellar = 0.3 * stellar_young + 0.7 * stellar_old

        # Dust attenuation: observed post-starburst galaxies span a wide
        # range from nearly dust-free to heavily obscured. Sample per-galaxy
        # when an rng is supplied; otherwise fall back to the dust-poor
        # (A+K-star, minimal residual gas) default.
        if rng is not None:
            e_bv_post_starburst = float(rng.uniform(0.3, 0.8))
        else:
            e_bv_post_starburst = 0.08
        extinct = calzetti_extinction(wave, E_BV=e_bv_post_starburst)
        stellar *= extinct

        # Weak residual IR emission
        T_warm = 35  # K
        dust = np.where(wave > 30.0,
                       0.08 * planck_function_safe(wave, T_warm),
                       0)
        
        flux = stellar + dust
        
    else:  # dusty_starburst (ULIRG/SMG template)
        # Young stellar population heavily obscured
        stellar = stellar_continuum_BC03(wave, age_gyr=0.05, tau_gyr=0.5)
        
        # Strong extinction (E(B-V) = 0.6)
        extinct = calzetti_extinction(wave, E_BV=0.6)
        stellar *= extinct
        
        # Strong multi-temperature IR emission (starburst)
        T_hot = 300   # K - hot dust near star-forming regions
        T_warm = 80   # K - warm dust in molecular clouds
        T_cold = 35   # K - cold dust in outer regions
        
        dust_hot = np.where(wave > 3.0, planck_function_safe(wave, T_hot), 0)
        dust_warm = np.where(wave > 8.0, planck_function_safe(wave, T_warm), 0)
        dust_cold = np.where(wave > 20.0, planck_function_safe(wave, T_cold), 0)
        
        # Strong PAH emission
        pah_62 = 0.25 * np.exp(-((wave - 6.2)/0.15)**2)
        pah_77 = 0.45 * np.exp(-((wave - 7.7)/0.25)**2)
        pah_113 = 0.18 * np.exp(-((wave - 11.3)/0.2)**2)
        pah_125 = 0.10 * np.exp(-((wave - 12.7)/0.18)**2)
        
        flux = (stellar + 0.4 * dust_hot + 1.2 * dust_warm + 1.8 * dust_cold + 
                pah_62 + pah_77 + pah_113 + pah_125)
    
    # Normalize to peak = 1
    flux_max = np.nanmax(flux)
    if flux_max > 0:
        flux = flux / flux_max
    
    return flux


def calculate_k_correction_empirical(redshift, filter_wave, sed_type='star_forming'):
    """
    Calculate K-correction using empirical SED templates
    
    K-correction accounts for:
    - Rest-frame spectral features shifting through observed filter
    - Bandpass mismatch between rest-frame and observed
    
    Parameters:
    -----------
    redshift : float
        Galaxy redshift
    filter_wave : float
        Observed filter central wavelength (μm)
    sed_type : str
        SED type: 'star_forming', 'passive', 'post_starburst', 'dusty_starburst'
    
    Returns:
    --------
    k_corr : float
        K-correction in magnitudes
    """
    if redshift < 0.01:
        return 0.0
    
    # Generate SED template covering rest-frame wavelengths
    # Need to cover observed filter wavelength in rest-frame
    rest_wave_min = filter_wave / (1 + redshift) / 2.0
    rest_wave_max = filter_wave / (1 + redshift) * 2.0
    rest_wavelengths = np.linspace(rest_wave_min, rest_wave_max, 200)
    
    # Generate empirical SED
    sed_flux = generate_empirical_sed(sed_type, rest_wavelengths)
    
    # Interpolate SED
    sed_interp = interp1d(rest_wavelengths, sed_flux, 
                          kind='linear', fill_value='extrapolate')
    
    # Calculate flux at rest-frame wavelength corresponding to observed filter
    rest_wave_obs = filter_wave / (1 + redshift)
    rest_wave_ref = 1.5  # Reference wavelength (μm) - typical optical
    
    # Flux ratio in rest-frame
    try:
        f_obs = sed_interp(rest_wave_obs)
        f_ref = sed_interp(rest_wave_ref)
        
        # K-correction (simplified, ignoring filter bandpass for now)
        if f_obs > 0 and f_ref > 0:
            k_corr = -2.5 * np.log10(f_obs / f_ref)
        else:
            k_corr = 0.0
    except:
        k_corr = 0.0
    
    return k_corr


def select_sed_type_from_redshift(redshift, rng):
    """
    Select SED type based on redshift evolution
    
    Observational trends:
    - z < 1: More passive ellipticals
    - 1 < z < 3: Mixed population
    - z > 3: More star-forming/dusty starbursts
    
    Parameters:
    -----------
    redshift : float
        Galaxy redshift
    rng : np.random.Generator
        Random number generator
    
    Returns:
    --------
    sed_type : str
        Selected SED type
    """
    if redshift < 1.0:
        # Low-z: More passive
        probs = [0.3, 0.5, 0.1, 0.1]  # SF, passive, post-starburst, dusty
    elif redshift < 3.0:
        # Mid-z: Mixed
        probs = [0.5, 0.3, 0.15, 0.05]
    else:
        # High-z: More star-forming/dusty
        probs = [0.6, 0.1, 0.2, 0.1]
    
    sed_types = ['star_forming', 'passive', 'post_starburst', 'dusty_starburst']
    return rng.choice(sed_types, p=probs)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate wavelength grid
    wavelengths = np.linspace(0.3, 300.0, 1000)
    
    # Generate all four SED types
    sed_types = ['star_forming', 'passive', 'post_starburst', 'dusty_starburst']
    colors = ['cyan', 'red', 'orange', 'green']
    labels = ['Star-forming', 'Passive', 'Post-starburst', 'Dusty Starburst']
    
    plt.figure(figsize=(12, 6))
    for sed_type, color, label in zip(sed_types, colors, labels):
        flux = generate_empirical_sed(sed_type, wavelengths)
        plt.plot(wavelengths, flux, color=color, linewidth=2, label=label)
    
    plt.xscale('log')
    plt.xlim(0.3, 300)
    plt.ylim(0, 1.05)
    plt.xlabel('Rest-frame Wavelength (μm)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Flux', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.title('Empirical SED Templates (BC03 + Calzetti+2000 + Chary & Elbaz 2001)', 
              fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('empirical_sed_templates.png', dpi=150)
    plt.show()
    
    print("✓ Empirical SED templates generated successfully!")
    print("\nTemplate properties:")
    print("  Star-forming: Age ~100 Myr, E(B-V)=0.15, moderate FIR")
    print("  Passive: Age ~5 Gyr, E(B-V)=0.02, minimal IR, strong 1.6 μm bump")
    print("  Post-starburst: Mixed ages 0.5-1 Gyr, E(B-V)=0.08, weak dust")
    print("  Dusty starburst: Age ~50 Myr, E(B-V)=0.6, strong PAH + FIR peak")

