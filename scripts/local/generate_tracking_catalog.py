#!/usr/bin/env python
"""
Generate unified tracking catalog for all three pair galaxy treatments.
Creates easy-to-search CSV showing which treatment each lens belongs to.
"""

import sys
import csv
from pathlib import Path
import json

def generate_tracking_catalog(output_dir):
    """Create unified catalog tracking all treatments."""
    output_dir = Path(output_dir)
    
    treatments = {
        'sie_sie': 'Binary SIE (point mass, fast)',
        'nfw_nfw': 'Binary NFW (dark matter halo, realistic)',
        'shear_only': 'External shear only (environmental)'
    }
    
    output_catalog = output_dir / 'TREATMENT_TRACKING.csv'
    
    with open(output_catalog, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lens_id', 'treatment', 'model_description', 'image_directory',
            'image_count', 'catalog_file', 'notes'
        ])
        
        for treatment, description in treatments.items():
            treatment_dir = output_dir / treatment
            
            # Count images
            img_dir = treatment_dir / 'jpg_rgb'
            if img_dir.exists():
                n_images = len(list(img_dir.glob('*.jpg')))
            else:
                n_images = 0
            
            # Get catalog
            cat_path = treatment_dir / 'cosmos_lens_training_catalog.csv'
            
            # Read number of lenses
            n_lenses = 0
            if cat_path.exists():
                with open(cat_path) as cf:
                    n_lenses = sum(1 for _ in csv.DictReader(cf))
            
            writer.writerow([
                '',  # lens_id (header)
                treatment,
                description,
                str(img_dir),
                n_images,
                str(cat_path),
                f'{n_lenses} lenses generated'
            ])
    
    print(f"✓ Generated tracking catalog: {output_catalog}")
    
    # Create summary JSON
    summary = {
        'output_directory': str(output_dir),
        'treatments': {}
    }
    
    for treatment in treatments.keys():
        treatment_dir = output_dir / treatment
        img_dir = treatment_dir / 'jpg_rgb'
        cat_path = treatment_dir / 'cosmos_lens_training_catalog.csv'
        
        summary['treatments'][treatment] = {
            'directory': str(treatment_dir),
            'images': {
                'directory': str(img_dir),
                'count': len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
            },
            'catalog': str(cat_path),
            'description': treatments[treatment]
        }
    
    summary_path = output_dir / 'TREATMENT_SUMMARY.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Generated summary: {summary_path}")
    
    # Create README
    readme_path = output_dir / 'TREATMENT_COMPARISON_README.md'
    with open(readme_path, 'w') as f:
        f.write(f"""# Pair Galaxy Treatment Comparison
## Realistic Lens Simulations

Generated: {output_dir.name}

### Treatment Overview

#### 1. SIE+SIE Binary Lens
- **Model**: Two Singular Isothermal Ellipsoids
- **Physics**: Point-mass deflection for each pair galaxy
- **Speed**: Fastest (baseline 1.0×)
- **Realism**: Simple approximation
- **Key Feature**: Sharp, crisp lensing arcs from point masses
- **Use Case**: Fast iteration, training baseline

#### 2. NFW+NFW Binary Lens
- **Model**: Two Navarro-Frenk-White dark matter halos
- **Physics**: Extended mass profiles matching real dark matter
- **Speed**: Slow (2-3× baseline)
- **Realism**: High - matches observations ⭐⭐⭐⭐⭐
- **Key Feature**: Softer arcs, extended image profiles
- **Use Case**: Production runs, physical accuracy, paper results

#### 3. Shear-Only (Simplified)
- **Model**: Single lens + external environmental shear
- **Physics**: Pair galaxy creates only perturbative shear (no binary lensing)
- **Speed**: Very fast (0.8× baseline)
- **Realism**: Low - ignores pair as discrete mass
- **Key Feature**: Weak lensing signals, minimal arc structure
- **Use Case**: Robustness tests, environmental effects study

### How to Find Specific Lenses

1. **By Treatment**:
   ```bash
   ls outputs/pair_treatments_*/sie_sie/jpg_rgb/
   ls outputs/pair_treatments_*/nfw_nfw/jpg_rgb/
   ls outputs/pair_treatments_*/shear_only/jpg_rgb/
   ```

2. **By Catalog**:
   - Quick search: `TREATMENT_TRACKING.csv`
   - Detailed: Each treatment's `cosmos_lens_training_catalog.csv`

3. **Visual Comparison**:
   ```bash
   open outputs/pair_treatments_*/comparison_lens_*.png
   ```

### Key Differences to Observe

| Feature | SIE+SIE | NFW+NFW | Shear-Only |
|---------|---------|---------|-----------|
| Arc sharpness | Very sharp | Soft edges | Barely visible |
| Image multiplicity | 4 (ABCD) | 4 extended | 2 weak |
| Arc morphology | Angular cusps | Smooth | Subtle |
| Computation | Fast | Slow | Very fast |
| Physical realism | Low | High | Very low |

### Output Structure

```
pair_treatments_YYYYMMDD_HHMMSS/
├── sie_sie/
│   ├── jpg_rgb/              ← Individual treatment images
│   ├── npy/                  ← Numpy data arrays
│   └── cosmos_lens_training_catalog.csv
├── nfw_nfw/
│   ├── jpg_rgb/
│   ├── npy/
│   └── cosmos_lens_training_catalog.csv
├── shear_only/
│   ├── jpg_rgb/
│   ├── npy/
│   └── cosmos_lens_training_catalog.csv
├── comparison_lens_*.png     ← Side-by-side comparisons
├── TREATMENT_TRACKING.csv    ← Master tracking catalog
├── TREATMENT_SUMMARY.json    ← Summary metadata
└── TREATMENT_COMPARISON_README.md ← This file
```

### Statistics

""")
        
        for treatment in treatments.keys():
            treatment_dir = output_dir / treatment
            cat_path = treatment_dir / 'cosmos_lens_training_catalog.csv'
            img_dir = treatment_dir / 'jpg_rgb'
            
            n_lenses = 0
            if cat_path.exists():
                with open(cat_path) as cf:
                    n_lenses = sum(1 for _ in csv.DictReader(cf))
            
            n_images = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
            
            f.write(f"\n**{treatment.upper().replace('_', ' ')}**:\n")
            f.write(f"- Lenses: {n_lenses}\n")
            f.write(f"- Images: {n_images}\n")
            f.write(f"- Catalog: `{cat_path.name}`\n")
    
    print(f"✓ Generated documentation: {readme_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_tracking_catalog.py <output_root_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    generate_tracking_catalog(output_dir)
    print("\n✓ All catalogs generated successfully!")

if __name__ == '__main__':
    main()
