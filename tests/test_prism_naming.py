#!/usr/bin/env python3
"""Test PRISM naming convention"""

from src.lens_system_classifier import LensSystemClassifier, create_classified_filename

classifier = LensSystemClassifier()

print("=" * 80)
print("PRISM NAMING CONVENTION TEST")
print("=" * 80 + "\n")

# Test classifications and filenames
test_cases = [
    (0, 'single_field', ['SIE', 'SHEAR']),
    (42, 'binary_sie_sie', ['SIE', 'SIE', 'SHEAR']),
    (123, 'binary_nfw_nfw', ['NFW', 'NFW', 'SHEAR']),
    (456, 'group', ['SIE', 'EXTERNAL_SHEAR']),
    (789, 'shear_only', ['SHEAR']),
]

print(f"{'Lens ID':<10} {'System Class':<20} {'Short Code':<8} {'Filename':<50}")
print("-" * 90)

for lens_id, expected_class, model_list in test_cases:
    # Get classification
    classified = classifier.classify_lens(model_list)
    short_code = classifier.get_short_code(classified)
    
    # Generate filename
    filename = create_classified_filename(lens_id, classified)
    rgb_filename = create_classified_filename(lens_id, classified, extension='_rgb.jpg')
    
    print(f"{lens_id:<10} {classified:<20} {short_code:<8} {filename:<50}")
    print(f"{'':10} {'':20} {'':8} {rgb_filename:<50}")
    print()

print("\n" + "=" * 80)
print("CLASS CODE REFERENCE")
print("=" * 80 + "\n")

for class_name in ['single_field', 'group', 'binary_sie_sie', 'binary_nfw_nfw', 'shear_only']:
    code = classifier.get_short_code(class_name)
    desc = classifier.get_description(class_name)
    print(f"{code:<8} {class_name:<20} {desc}")

print("\n" + "=" * 80)
print("\nExample file search commands:")
print("-" * 80)
print("Find all binary pairs:")
print("  ls jpg_rgb/PRISM_*_B*.jpg")
print("\nFind all single field lenses:")
print("  ls jpg_rgb/PRISM_*_SF.jpg")
print("\nFind all NFW pairs:")
print("  ls jpg_rgb/PRISM_*_BNFW.jpg")
print("\nFind all SIE+SIE pairs:")
print("  ls jpg_rgb/PRISM_*_BSIE.jpg")
print("=" * 80)
