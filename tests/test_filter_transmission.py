#!/usr/bin/env python3
"""
Test and demonstrate the new JWST Filter Transmission System.

This script shows:
1. Filter transmission curves
2. SED convolution to magnitudes
3. Color calculations from transmission
4. Filter-specific noise properties
5. Detection limit estimates
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from prism.telescopes.jwst_filter_transmission import (
    JWST_FILTERS_SYSTEM,
    JWSTFilterSystem
)

def test_filter_transmission():
    """Test 1: Inspect filter transmission curves."""
    print("=" * 70)
    print("TEST 1: Filter Transmission Curves")
    print("=" * 70)
    
    filters = JWST_FILTERS_SYSTEM.FILTER_DEFINITIONS
    
    for band, (center, fwhm, profile_type) in filters.items():
        wavelengths, transmission = JWST_FILTERS_SYSTEM.transmission_curves[band]
        peak_transmission = transmission.max()
        transmission_integral = np.trapz(transmission, wavelengths)
        
        print(f"\n{band}:")
        print(f"  Center:           {center:.2f} μm")
        print(f"  FWHM:             {fwhm:.3f} μm")
        print(f"  Peak Transmission: {peak_transmission:.3f}")
        print(f"  Integral (width):  {transmission_integral:.3f} μm")


def test_noise_properties():
    """Test 2: Filter-specific noise characteristics."""
    print("\n" + "=" * 70)
    print("TEST 2: Filter-Specific Noise Properties")
    print("=" * 70)
    
    print(f"\n{'Band':<8} {'ZP (AB)':<10} {'BG (e⁻/s)':<12} {'Read Noise (e⁻)':<16} {'Excess':<8}")
    print("-" * 60)
    
    for band in sorted(JWST_FILTERS_SYSTEM.FILTER_NOISE_PROPERTIES.keys()):
        props = JWST_FILTERS_SYSTEM.get_noise_properties(band)
        
        zp = props['zeropoint']
        bg = props['background']
        rn = props['read_noise']
        ex = props['excess_noise_factor']
        
        print(f"{band:<8} {zp:<10.2f} {bg:<12.2f} {rn:<16.1f} {ex:<8.2f}")


def test_sed_convolution():
    """Test 3: SED convolution to magnitudes."""
    print("\n" + "=" * 70)
    print("TEST 3: SED Convolution to Multi-Band Magnitudes")
    print("=" * 70)
    
    # Create simple power-law SED (representing star-forming galaxy)
    wavelengths = np.linspace(0.3, 5.0, 500)
    
    # Power-law SED: f_lambda ~ lambda^alpha (alpha = -1 for typical SFG)
    sed_blue = wavelengths ** (-1.0)  # Blue, star-forming
    sed_red = wavelengths ** (-0.3)   # Red, passive
    
    rng = np.random.default_rng(42)
    
    print("\n--- Blue (Star-Forming) Galaxy SED at z=0 ---")
    photometry_blue_z0 = JWST_FILTERS_SYSTEM.get_multiband_photometry(
        wavelengths, sed_blue,
        bands=['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W'],
        redshift=0.0
    )
    
    print("Magnitudes (arbitrary scale):")
    for band, mag in photometry_blue_z0.items():
        print(f"  {band}: {mag:6.2f}")
    
    print("\n--- Red (Passive) Galaxy SED at z=0.5 ---")
    photometry_red_z05 = JWST_FILTERS_SYSTEM.get_multiband_photometry(
        wavelengths, sed_red,
        bands=['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W'],
        redshift=0.5
    )
    
    print("Magnitudes (arbitrary scale):")
    for band, mag in photometry_red_z05.items():
        print(f"  {band}: {mag:6.2f}")


def test_color_calculations():
    """Test 4: Color calculations from transmission."""
    print("\n" + "=" * 70)
    print("TEST 4: Color Calculations from Filter Transmission")
    print("=" * 70)
    
    wavelengths = np.linspace(0.3, 5.0, 500)
    
    # Create galaxy SED templates
    sed_blue = wavelengths ** (-1.0)  # Star-forming (blue)
    sed_red = wavelengths ** (-0.3)   # Passive (red)
    
    print("\n--- Colors for Star-Forming Galaxy (Blue) ---")
    colors_blue = {
        'F115W-F444W': JWST_FILTERS_SYSTEM.calculate_color_from_transmission(
            wavelengths, sed_blue, 'F115W', 'F444W', redshift=0.0
        ),
        'F150W-F277W': JWST_FILTERS_SYSTEM.calculate_color_from_transmission(
            wavelengths, sed_blue, 'F150W', 'F277W', redshift=0.0
        ),
        'F090W-F200W': JWST_FILTERS_SYSTEM.calculate_color_from_transmission(
            wavelengths, sed_blue, 'F090W', 'F200W', redshift=0.0
        ),
    }
    
    for color_name, color_val in colors_blue.items():
        print(f"  {color_name}: {color_val:7.3f} mag (blue)" if color_val < 0 else 
              f"  {color_name}: {color_val:7.3f} mag (red)")
    
    print("\n--- Colors for Passive Galaxy (Red) ---")
    colors_red = {
        'F115W-F444W': JWST_FILTERS_SYSTEM.calculate_color_from_transmission(
            wavelengths, sed_red, 'F115W', 'F444W', redshift=0.0
        ),
        'F150W-F277W': JWST_FILTERS_SYSTEM.calculate_color_from_transmission(
            wavelengths, sed_red, 'F150W', 'F277W', redshift=0.0
        ),
        'F090W-F200W': JWST_FILTERS_SYSTEM.calculate_color_from_transmission(
            wavelengths, sed_red, 'F090W', 'F200W', redshift=0.0
        ),
    }
    
    for color_name, color_val in colors_red.items():
        print(f"  {color_name}: {color_val:7.3f} mag (blue)" if color_val < 0 else 
              f"  {color_name}: {color_val:7.3f} mag (red)")


def test_detection_limits():
    """Test 5: Detection limit estimates."""
    print("\n" + "=" * 70)
    print("TEST 5: 5-Sigma Detection Limits vs. Thermal Background")
    print("=" * 70)
    
    print(f"\n{'Band':<8} {'Limit (AB)':<12} {'Background (e⁻/s)':<20} {'Regime':<20}")
    print("-" * 65)
    
    for band in sorted(JWST_FILTERS_SYSTEM.FILTER_NOISE_PROPERTIES.keys()):
        limit_mag = JWST_FILTERS_SYSTEM.estimate_background_limited_magnitude(
            band, exposure_time=10000.0, num_pixels=4.0
        )
        props = JWST_FILTERS_SYSTEM.get_noise_properties(band)
        bg = props['background']
        
        if bg < 0.3:
            regime = "Photon-limited"
        elif bg < 1.0:
            regime = "Moderately thermal"
        else:
            regime = "Thermally limited"
        
        print(f"{band:<8} {limit_mag:<12.2f} {bg:<20.2f} {regime:<20}")


def test_transmission_curves_properties():
    """Test 6: Transmission curve properties."""
    print("\n" + "=" * 70)
    print("TEST 6: Transmission Curve Properties")
    print("=" * 70)
    
    print(f"\n{'Band':<8} {'Peak (μm)':<12} {'Peak Trans.':<15} {'FWHM':<10} {'Relative Width':<15}")
    print("-" * 65)
    
    for band in sorted(JWST_FILTERS_SYSTEM.FILTER_DEFINITIONS.keys()):
        center, fwhm, _ = JWST_FILTERS_SYSTEM.FILTER_DEFINITIONS[band]
        wavelengths, transmission = JWST_FILTERS_SYSTEM.transmission_curves[band]
        
        peak_idx = np.argmax(transmission)
        peak_wave = wavelengths[peak_idx]
        peak_trans = transmission[peak_idx]
        relative_width = fwhm / center * 100
        
        print(f"{band:<8} {peak_wave:<12.3f} {peak_trans:<15.3f} {fwhm:<10.3f} {relative_width:<15.1f}%")


if __name__ == '__main__':
    print("\n")
    print("█" * 70)
    print("  JWST FILTER TRANSMISSION SYSTEM - TEST SUITE")
    print("█" * 70)
    
    test_filter_transmission()
    test_noise_properties()
    test_sed_convolution()
    test_color_calculations()
    test_detection_limits()
    test_transmission_curves_properties()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")
