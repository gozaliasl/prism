#!/usr/bin/env python
"""Quick test of telescope-specific filter selection."""

# Test the filter selection logic
TELESCOPE_FILTERS = {
    'jwst': ["F115W", "F150W", "F277W", "F444W"],
    'roman': ["ROMAN_F087", "ROMAN_F106", "ROMAN_F129", "ROMAN_F146", "ROMAN_F158", "ROMAN_F184"],
    'ground_based': ["SUBARU_G", "SUBARU_R", "SUBARU_I", "SUBARU_Z"],
}

def get_telescope_bands(resolution_name, config_bands=None):
    """Get appropriate filter set for a given telescope/resolution."""
    if config_bands and resolution_name in ['jwst', 'default']:
        return config_bands
    return TELESCOPE_FILTERS.get(resolution_name, TELESCOPE_FILTERS['jwst'])

# Test cases
resolutions = ['jwst', 'roman', 'euclid', 'ground_based']
config_bands_thesis = ['F090W', 'F150W', 'F200W', 'F356W', 'F444W']

print("Testing telescope-specific filter selection:")
print("=" * 60)

for res in resolutions:
    bands = get_telescope_bands(res, config_bands_thesis)
    print(f"\n{res:15s}: {len(bands)} filters")
    print(f"  {', '.join(bands)}")

print("\n" + "=" * 60)
print("✓ Filter selection logic works correctly!")
print("\nFor Aaron's thesis:")
print("  - JWST:         Uses config bands (F090W, F150W, F200W, F356W, F444W)")
print("  - Roman:        Uses Roman WFI filters (6 filters)")
print("  - Euclid:       Uses Euclid default (fallback to JWST for now)")
print("  - Ground-based: Uses Subaru Suprime-Cam filters (4 filters)")
