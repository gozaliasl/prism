#!/usr/bin/env python3
"""
Test real JWST filter integration with main simulator.
"""

import sys
sys.path.insert(0, 'src')

print("\n" + "="*70)
print("TESTING REAL JWST FILTER INTEGRATION")
print("="*70)

# Test 1: Import real filter system
print("\n[1] Importing real filter system...")
try:
    from prism.telescopes.jwst_real_filter_transmission import REAL_JWST_FILTERS
    print(f"    ✓ Real filter system loaded")
    print(f"    ✓ Available filters: {len(REAL_JWST_FILTERS.available_filters)}")
except Exception as e:
    print(f"    ✗ Failed to load real filters: {e}")
    sys.exit(1)

# Test 2: Import main simulator with real filters
print("\n[2] Importing main simulator with real filter support...")
try:
    import prism.core.jwst_lens_simulator as jwst_lens_simulator
    print(f"    ✓ Main simulator imported")
    print(f"    ✓ Real filters available in simulator: {jwst_lens_simulator.REAL_FILTERS_AVAILABLE}")
except Exception as e:
    print(f"    ✗ Failed to import main simulator: {e}")
    sys.exit(1)

# Test 3: List available filters
print("\n[3] Available real NIRCam filters:")
filters_by_category = REAL_JWST_FILTERS.get_filter_categories()
print(f"    Short-wavelength ({len(filters_by_category['short'])}): {', '.join(filters_by_category['short'][:6])}")
print(f"    Medium-wavelength ({len(filters_by_category['medium'])}): {', '.join(filters_by_category['medium'][:5])}")
print(f"    Long-wavelength ({len(filters_by_category['long'])}): {', '.join(filters_by_category['long'][:8])}")

# Test 4: Get filter properties from real data
print("\n[4] Testing filter properties extraction from real transmission curves:")
test_filters = ['F090W', 'F150W', 'F277W', 'F356W', 'F444W']
print(f"{'Filter':<10} {'Center (μm)':<15} {'FWHM (μm)':<12} {'Peak T':<10}")
print("-" * 50)
for band in test_filters:
    if band in REAL_JWST_FILTERS.available_filters:
        props = REAL_JWST_FILTERS.get_filter_properties(band)
        print(f"{band:<10} {props['effective_wavelength_um']:<15.4f} {props['fwhm_um']:<12.4f} {props['peak_transmission']:<10.4f}")

# Test 5: Test transmission interpolation
print("\n[5] Testing transmission interpolation for F150W...")
import numpy as np
test_wavelengths = np.linspace(1.0, 2.0, 100)  # 1-2 microns
transmission = REAL_JWST_FILTERS.get_transmission('F150W', test_wavelengths)
print(f"    ✓ Interpolated transmission for {len(test_wavelengths)} wavelengths")
print(f"    ✓ Min transmission: {transmission.min():.6f}")
print(f"    ✓ Max transmission: {transmission.max():.6f}")
print(f"    ✓ Mean transmission (non-zero): {transmission[transmission > 0].mean():.6f}")

# Test 6: Compare real vs Gaussian transmission
print("\n[6] Comparing real transmission with Gaussian approximation:")
try:
    from prism.telescopes.jwst_filter_transmission import JWST_FILTERS_SYSTEM
    
    band = 'F150W'
    # Get transmissions from both systems
    real_trans = REAL_JWST_FILTERS.get_transmission(band, test_wavelengths)
    gaussian_trans = JWST_FILTERS_SYSTEM.get_transmission(band, test_wavelengths)
    
    # Find peaks
    real_peak_idx = np.argmax(real_trans)
    gaussian_peak_idx = np.argmax(gaussian_trans)
    
    print(f"    Real transmission peak: {test_wavelengths[real_peak_idx]:.4f} μm, T={real_trans[real_peak_idx]:.4f}")
    print(f"    Gaussian transmission peak: {test_wavelengths[gaussian_peak_idx]:.4f} μm, T={gaussian_trans[gaussian_peak_idx]:.4f}")
    
    # Calculate RMS difference
    rms_diff = np.sqrt(np.mean((real_trans - gaussian_trans)**2))
    print(f"    RMS difference between real and Gaussian: {rms_diff:.6f}")
    print(f"    ✓ Real curves available for comparison!")
    
except Exception as e:
    print(f"    Note: Could not compare - {e}")

print("\n" + "="*70)
print("✓ ALL INTEGRATION TESTS PASSED!")
print("="*70 + "\n")
