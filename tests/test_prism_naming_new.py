#!/usr/bin/env python3
"""
Test script for new PRISM naming convention:
PRISM_[lens|nonlens]_[TYPE_][epoch_]ID.jpg/npy
"""

from src.lens_system_classifier import LensSystemClassifier
from pathlib import Path


def test_prism_naming_format():
    """Test PRISM naming generation with various formats."""
    
    print("\n" + "="*100)
    print("PRISM NAMING CONVENTION TEST - NEW FORMAT")
    print("="*100)
    
    test_cases = [
        {
            'lens_id': 0,
            'system_class': 'single_field',
            'is_nonlens': False,
            'epoch_index': None,
            'expected': 'PRISM_lens_SF_000000.jpg'
        },
        {
            'lens_id': 42,
            'system_class': 'binary_sie_sie',
            'is_nonlens': False,
            'epoch_index': None,
            'expected': 'PRISM_lens_BSIE_000042.jpg'
        },
        {
            'lens_id': 123,
            'system_class': 'binary_nfw_nfw',
            'is_nonlens': False,
            'epoch_index': None,
            'expected': 'PRISM_lens_BNFW_000123.jpg'
        },
        {
            'lens_id': 456,
            'system_class': 'group',
            'is_nonlens': False,
            'epoch_index': None,
            'expected': 'PRISM_lens_GRP_000456.jpg'
        },
        {
            'lens_id': 789,
            'system_class': 'shear_only',
            'is_nonlens': False,
            'epoch_index': None,
            'expected': 'PRISM_lens_SHEAR_000789.jpg'
        },
        {
            'lens_id': 100,
            'system_class': None,
            'is_nonlens': True,
            'epoch_index': None,
            'expected': 'PRISM_nonlens_000100.jpg'
        },
        {
            'lens_id': 200,
            'system_class': 'binary_sie_sie',
            'is_nonlens': False,
            'epoch_index': 1,
            'expected': 'PRISM_lens_BSIE_epoch01_000200.jpg'
        },
        {
            'lens_id': 300,
            'system_class': None,
            'is_nonlens': True,
            'epoch_index': 3,
            'expected': 'PRISM_nonlens_epoch03_000300.jpg'
        },
        {
            'lens_id': 999999,
            'system_class': 'single_field',
            'is_nonlens': False,
            'epoch_index': None,
            'expected': 'PRISM_lens_SF_999999.jpg'
        },
    ]
    
    print("\n" + f"{'TEST CASE':<5} {'LENS ID':<10} {'TYPE':<15} {'EPOCH':<8} {'GENERATED':<50} {'EXPECTED':<50} {'PASS':<6}")
    print("-" * 148)
    
    all_pass = True
    for i, test in enumerate(test_cases, 1):
        lens_id = test['lens_id']
        system_class = test['system_class']
        is_nonlens = test['is_nonlens']
        epoch_index = test['epoch_index']
        expected = test['expected']
        
        # Generate using classifier function
        if is_nonlens:
            # For nonlens, we use a simpler generation
            epoch_str = f"epoch{epoch_index:02d}_" if epoch_index is not None else ""
            generated = f"PRISM_nonlens_{epoch_str}{int(lens_id):06d}.jpg"
        else:
            short_code = LensSystemClassifier.get_short_code(system_class)
            epoch_str = f"epoch{epoch_index:02d}_" if epoch_index is not None else ""
            generated = f"PRISM_lens_{short_code}_{epoch_str}{int(lens_id):06d}.jpg"
        
        passed = generated == expected
        all_pass = all_pass and passed
        
        type_str = "nonlens" if is_nonlens else (system_class or "unknown")
        epoch_str_display = f"epoch{epoch_index:02d}" if epoch_index is not None else "none"
        
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"{i:<5} {lens_id:<10} {type_str:<15} {epoch_str_display:<8} {generated:<50} {expected:<50} {status:<6}")
    
    print("-" * 148)
    print(f"\nResults: {'All tests passed! ✓' if all_pass else 'Some tests failed ✗'}")
    
    # Test class code mappings
    print("\n" + "="*100)
    print("LENS SYSTEM CLASS CODE REFERENCE")
    print("="*100)
    
    print(f"\n{'CLASS':<20} {'SHORT CODE':<15} {'DESCRIPTION':<55}")
    print("-" * 90)
    
    for class_name in ['single_field', 'group', 'binary_sie_sie', 'binary_nfw_nfw', 'shear_only']:
        short_code = LensSystemClassifier.get_short_code(class_name)
        desc = LensSystemClassifier.get_description(class_name)
        print(f"{class_name:<20} {short_code:<15} {desc:<55}")
    
    # Test file naming patterns
    print("\n" + "="*100)
    print("FILE NAMING PATTERN EXAMPLES")
    print("="*100)
    
    print("\nLENS FILES (jpg_rgb/ and npy/):")
    print("  PRISM_lens_SF_000000.jpg          - Single field, lens 0")
    print("  PRISM_lens_BSIE_000001.jpg        - Binary SIE+SIE, lens 1")
    print("  PRISM_lens_BNFW_000042.jpg        - Binary NFW+NFW, lens 42")
    print("  PRISM_lens_GRP_000123.jpg         - Group lens, lens 123")
    print("  PRISM_lens_SHEAR_000456.jpg       - Shear-only, lens 456")
    print("  PRISM_lens_BSIE_epoch01_000789.jpg - Binary with time delay epoch 1")
    print("  PRISM_lens_SF_000000_4bands.npy   - 4-band numpy array")
    
    print("\nNON-LENS FILES (jpg_rgb/ and npy/):")
    print("  PRISM_nonlens_000000.jpg          - Non-lens sample 0")
    print("  PRISM_nonlens_000100.jpg          - Non-lens sample 100")
    print("  PRISM_nonlens_epoch01_000042.jpg  - Non-lens with time delay epoch 1")
    print("  PRISM_nonlens_000000_4bands.npy   - 4-band numpy array")
    
    print("\nINTERMEDIATE FILES (jpg_rgb/intermediate_*/ and npy/intermediate_*/):")
    print("  intermediate_lens_only/PRISM_lens_SF_000000.jpg       - Lens component only")
    print("  intermediate_sources_only/PRISM_lens_SF_000000.jpg    - Source component only")
    print("  intermediate_lens_sources/PRISM_lens_SF_000000.jpg    - Lens + sources")
    print("  intermediate_field_only/PRISM_lens_SF_000000.jpg      - Field galaxies only")
    
    print("\nDIAGNOSTIC/METADATA FILES (diagnostics/ and unified_npz/):")
    print("  PRISM_lens_SF_000000_diag.json    - Diagnostic metadata")
    print("  PRISM_lens_SF_000000.npz          - Unified storage (if enabled)")
    print("  PRISM_nonlens_000000.npz          - Unified non-lens storage")
    
    # Test glob patterns
    print("\n" + "="*100)
    print("GLOB PATTERN EXAMPLES FOR FILE DISCOVERY")
    print("="*100)
    
    print("\nFind all lens images:")
    print("  jpg_rgb/PRISM_lens_*.jpg")
    
    print("\nFind all non-lens images:")
    print("  jpg_rgb/PRISM_nonlens_*.jpg")
    
    print("\nFind specific lens type (Binary SIE+SIE):")
    print("  jpg_rgb/PRISM_lens_BSIE_*.jpg")
    
    print("\nFind all binary lenses:")
    print("  jpg_rgb/PRISM_lens_B*.jpg")
    
    print("\nFind time-delay images:")
    print("  jpg_rgb/PRISM_*_epoch*.jpg")
    
    print("\nFind lens-only intermediate images:")
    print("  jpg_rgb/intermediate_lens_only/PRISM_lens_*.jpg")
    
    print("\n" + "="*100 + "\n")
    
    return all_pass


if __name__ == '__main__':
    success = test_prism_naming_format()
    exit(0 if success else 1)
