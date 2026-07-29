#!/usr/bin/env python3
"""
Analyze pair lens types in simulation output.
Identifies which lenses are binary pairs (SIE+SIE, NFW+NFW) vs single lenses.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_output_directory(output_dir):
    """
    Analyze the simulation output directory to identify pair types.
    """
    output_dir = Path(output_dir)
    
    print("=" * 80)
    print(f"PAIR LENS TYPE ANALYSIS")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)
    
    # Load main catalog
    catalog_path = output_dir / "cosmos_lens_training_catalog.csv"
    if not catalog_path.exists():
        print(f"[ERROR] Catalog not found: {catalog_path}")
        return
    
    df = pd.read_csv(catalog_path)
    print(f"\n[INFO] Loaded catalog with {len(df)} lenses\n")
    
    # Analyze by checking diagnostics files
    diag_dir = output_dir / "diagnostics"
    pair_stats = {
        'single_lens': [],
        'sie_sie': [],
        'nfw_nfw': [],
        'unknown': []
    }
    
    if diag_dir.exists():
        print(f"[INFO] Analyzing diagnostic files from: {diag_dir}\n")
        
        for diag_file in sorted(diag_dir.glob("*_diag.json")):
            try:
                with open(diag_file, 'r') as f:
                    diag = json.load(f)
                
                lens_id = diag.get('lens_id', -1)
                
                # Try to determine pair type from generation_info or catalog
                # Look in the catalog for this lens
                matching_rows = df[df['lens_id'] == lens_id]
                
                if len(matching_rows) > 0:
                    row = matching_rows.iloc[0]
                    
                    # Check if it's a pair by looking at specific catalog columns
                    # or check if there are multiple lens models
                    is_pair = False
                    pair_type = 'single_lens'
                    
                    # Heuristic: Check column names for pair indicators
                    if 'lens_type' in row and 'pair' in str(row['lens_type']).lower():
                        is_pair = True
                    elif 'binary' in str(row).lower():
                        is_pair = True
                    
                    # If we can't determine from catalog, mark as unknown
                    if not is_pair:
                        pair_type = 'unknown'
                    
                    pair_stats[pair_type].append(lens_id)
                    
            except Exception as e:
                print(f"[WARNING] Error reading {diag_file}: {e}")
    
    # Try alternative: Check if catalog has pair type information
    print("[INFO] Checking catalog columns...")
    print(f"Available columns: {df.columns.tolist()}\n")
    
    # Look for pair/binary related columns
    pair_columns = [col for col in df.columns if 'pair' in col.lower() or 'binary' in col.lower()]
    if pair_columns:
        print(f"[INFO] Found pair-related columns: {pair_columns}\n")
        
        # Re-analyze using catalog columns
        pair_stats = {'single_lens': [], 'sie_sie': [], 'nfw_nfw': [], 'unknown': []}
        
        for idx, row in df.iterrows():
            lens_id = row.get('lens_id', idx)
            
            # Check pair type columns
            pair_type = 'single_lens'
            
            for col in pair_columns:
                val = str(row[col]).lower()
                if 'sie' in val:
                    pair_type = 'sie_sie'
                    break
                elif 'nfw' in val:
                    pair_type = 'nfw_nfw'
                    break
                elif 'pair' in val or 'binary' in val:
                    pair_type = 'unknown'
                    break
            
            pair_stats[pair_type].append(lens_id)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("PAIR TYPE DISTRIBUTION")
    print("=" * 80 + "\n")
    
    total_lenses = len(df)
    
    for pair_type in ['single_lens', 'sie_sie', 'nfw_nfw', 'unknown']:
        count = len(pair_stats[pair_type])
        percentage = (count / total_lenses) * 100 if total_lenses > 0 else 0
        print(f"{pair_type:20s}: {count:4d} lenses ({percentage:6.2f}%)")
        
        if count > 0 and count <= 10:
            print(f"                     IDs: {pair_stats[pair_type]}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_pairs = len(pair_stats['sie_sie']) + len(pair_stats['nfw_nfw'])
    single_lenses = len(pair_stats['single_lens'])
    pair_fraction = (total_pairs / total_lenses) * 100 if total_lenses > 0 else 0
    
    print(f"\nTotal lenses: {total_lenses}")
    print(f"Single lenses: {single_lenses} ({(single_lenses/total_lenses)*100:.1f}%)")
    print(f"Binary pairs: {total_pairs} ({pair_fraction:.1f}%)")
    if total_pairs > 0:
        sie_frac = (len(pair_stats['sie_sie']) / total_pairs) * 100
        nfw_frac = (len(pair_stats['nfw_nfw']) / total_pairs) * 100
        print(f"  - SIE+SIE: {len(pair_stats['sie_sie'])} ({sie_frac:.1f}% of pairs)")
        print(f"  - NFW+NFW: {len(pair_stats['nfw_nfw'])} ({nfw_frac:.1f}% of pairs)")
    
    print(f"Unknown: {len(pair_stats['unknown'])} ({(len(pair_stats['unknown'])/total_lenses)*100:.1f}%)")
    
    # Try to infer from config
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80 + "\n")
    
    config_path = output_dir.parent / "configs" / "default_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            lines = f.readlines()
            in_binary = False
            for line in lines:
                if 'binary_lenses:' in line:
                    in_binary = True
                if in_binary:
                    if line.strip().startswith('#'):
                        continue
                    if 'fraction:' in line or 'sie_sie:' in line or 'nfw_nfw:' in line:
                        print(line.strip())
                    if in_binary and line.strip() == '' and 'fraction' in locals():
                        break
    
    print("\n" + "=" * 80)
    print("VISUALIZATION HINT")
    print("=" * 80)
    print(f"""
To visualize the different pair types side-by-side:

1. For SIE+SIE pairs: Smoother, more regular arc patterns
   - Einstein Radius profiles are smooth, isotropic
   - Arcs should be more symmetric

2. For NFW+NFW pairs: More complex, clumpy arc patterns
   - Cusped density profiles
   - May show multiple arc segments

3. Check intermediate images at:
   jpg_rgb/intermediate_lens_sources/ - Shows the lensed arcs clearly
   jpg_rgb/intermediate_sources_only/ - Shows arcs without lens light

Use the comparison script to view side-by-side:
  python scripts/compare_mass_profiles.py

Or view individual lens intermediate stages:
  open {output_dir}/jpg_rgb/intermediate_lens_sources/cosmos_lens_000000.jpg
""")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/outputs/custom_20260215_155541'
    
    analyze_output_directory(output_dir)
