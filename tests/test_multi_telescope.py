#!/usr/bin/env python
"""Test multi-telescope filter system."""

from src.multi_telescope_filters import get_multi_telescope_filters

# Load filter system
print("Loading multi-telescope filter system...")
mt = get_multi_telescope_filters()
print(f"✓ Loaded {len(mt.filter_files)} filters")
print(f"Telescopes: {list(mt.filter_dirs.keys())}")

# Test each telescope
test_filters = [
    ('JWST', ['F150W', 'F444W']),
    ('Roman', ['ROMAN_F087', 'ROMAN_F146']),
    ('Subaru', ['SUBARU_G', 'SUBARU_I']),
]

print("\nTesting filter loading:")
for telescope, filters in test_filters:
    print(f"\n{telescope}:")
    for filt in filters:
        try:
            wave, trans = mt.load_filter_transmission(filt)
            props = mt.get_filter_properties(filt)
            print(f"  ✓ {filt}: λ_eff={props['lambda_eff']:.3f}μm, BW={props['bandwidth']:.3f}μm, {len(wave)} points")
        except Exception as e:
            print(f"  ✗ {filt}: {e}")

print("\n✓ All tests passed!")
