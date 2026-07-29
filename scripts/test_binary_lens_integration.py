#!/usr/bin/env python3
"""
Test script to verify binary lens system integration into main pipeline
"""

import sys
import yaml
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prism.core.simulator import CONFIG
from prism.core.advanced_lens_features import RealisticMassProfiles

def test_binary_lens_config():
    """Test that binary lens configuration is properly loaded"""
    print("=" * 60)
    print("Testing Binary Lens Configuration")
    print("=" * 60)
    
    if 'binary_lenses' in CONFIG:
        binary_cfg = CONFIG['binary_lenses']
        print(f"✓ Binary lens config found")
        print(f"  Enabled: {binary_cfg.get('enabled', False)}")
        print(f"  Fraction: {binary_cfg.get('fraction', 0.0)}")
        print(f"  Profile types: {binary_cfg.get('mass_profile_types', {})}")
        print(f"  Mass ratio range: {binary_cfg.get('mass_ratio', {})}")
        print(f"  Separation range: {binary_cfg.get('separation', {})}")
    else:
        print("✗ Binary lens config not found in CONFIG")
        return False
    
    return True

def test_binary_lens_generation():
    """Test binary lens system generation"""
    print("\n" + "=" * 60)
    print("Testing Binary Lens Generation")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    lens_z = 0.6
    theta_E = 1.2
    lens_mass = 11.5
    
    # Test SIE+SIE system
    print("\n--- Testing SIE+SIE binary system ---")
    mass_profiles = RealisticMassProfiles(lens_z, rng, config=CONFIG)
    lens_model_list, kwargs_lens = mass_profiles.generate_binary_lens_system(
        theta_E, lens_mass_log10=lens_mass
    )
    
    print(f"Lens model list: {lens_model_list}")
    print(f"Number of components: {len(lens_model_list)}")
    print(f"Component 1: {lens_model_list[0]}")
    if len(lens_model_list) > 1:
        print(f"Component 2: {lens_model_list[1]}")
    
    print(f"\nComponent parameters:")
    for i, (model, kwargs) in enumerate(zip(lens_model_list, kwargs_lens)):
        print(f"  [{i}] {model}:")
        for key, val in kwargs.items():
            print(f"      {key}: {val}")
    
    # Verify structure
    if len(lens_model_list) >= 2:
        print("\n✓ Binary lens system generated successfully")
        return True
    else:
        print("\n✗ Binary lens system incomplete")
        return False

def test_pipeline_integration():
    """Test that binary lens logic is integrated into pipeline"""
    print("\n" + "=" * 60)
    print("Testing Pipeline Integration")
    print("=" * 60)
    
    # Check if import statement exists
    try:
        from prism.core.simulator import BINARY_LENS_AVAILABLE
        print(f"✓ BINARY_LENS_AVAILABLE flag: {BINARY_LENS_AVAILABLE}")
    except ImportError:
        print("✗ Binary lens import not found in main pipeline")
        return False
    
    # Verify RealisticMassProfiles can be imported
    try:
        from prism.core.advanced_lens_features import RealisticMassProfiles
        print("✓ RealisticMassProfiles import successful")
    except ImportError:
        print("✗ Cannot import RealisticMassProfiles")
        return False
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BINARY LENS INTEGRATION TEST")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Configuration", test_binary_lens_config()))
    results.append(("Generation", test_binary_lens_generation()))
    results.append(("Pipeline Integration", test_pipeline_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    sys.exit(0 if all_passed else 1)
