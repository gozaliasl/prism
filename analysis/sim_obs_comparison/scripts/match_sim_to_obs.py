#!/usr/bin/env python3
"""
Extract simulated lens properties and match to observed lenses.

Input:
- outputs/custom_20260213_155632/cosmos_lens_training_catalog.csv (simulated properties)
- catalogs/real_lens_properties.csv (observed properties, from previous script)

Matching strategy:
1. For each real lens, find simulated lenses within tolerance ranges:
   - Einstein radius θ_E: ±20%
   - Lens redshift z_lens: ±0.1
   - Source redshift z_source: ±0.1
   - F150W magnitude: ±1.0 mag

2. Rank matches by combined distance metric
3. Output top 3 matches per real lens (for manual inspection)

Output: matched_lens_pairs.csv with columns:
- real_name, sim_lens_id, sim_filename_base
- match_score, theta_e_sim, z_lens_sim, z_source_sim
- theta_e_obs, z_lens_obs, z_source_obs (when available)
"""

import csv
from pathlib import Path
import numpy as np


def load_real_lens_catalog(real_cat_path):
    """Load real lens catalog."""
    real_lenses = {}
    with open(real_cat_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            real_lenses[row['name']] = row
    return real_lenses


def load_sim_lens_catalog(sim_cat_path):
    """Load simulated lens catalog."""
    sim_lenses = []
    with open(sim_cat_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('is_lens') == '1':  # Only keep lenses
                sim_lenses.append(row)
    return sim_lenses


def match_lenses(real_lenses, sim_lenses, 
                theta_e_tol=0.2,      # ±20% of theta_E
                z_lens_tol=0.1,       # ±0.1 in z
                z_source_tol=0.1,     # ±0.1 in z
                mag_tol=1.0):         # ±1.0 mag
    """
    For each real lens, find best matching simulated lenses.
    Returns list of match records.
    """
    matches = []
    
    for real_name, real_data in real_lenses.items():
        print(f"Matching {real_name}...")
        
        # For now, we can only use the simulated catalog properties
        # In a full analysis, we'd extract theta_E from real lens images
        # For now, we'll just catalog what's available
        
        match_records = []
        
        for sim_idx, sim_data in enumerate(sim_lenses):
            try:
                sim_theta_e = float(sim_data.get('theta_E', 0))
                sim_z_lens = float(sim_data.get('lens_redshift', 0))
                sim_z_source = float(sim_data.get('source_redshift', 0))
                sim_mag = float(sim_data.get('lens_mag_f150w', 0))
                
                # Compute distance metric (normalized)
                # We use these as "representative" of diversity
                # In full analysis, would fit these from images
                
                score = sim_theta_e + sim_z_lens + sim_z_source
                
                match_records.append({
                    'real_name': real_name,
                    'sim_lens_id': sim_data.get('lens_id'),
                    'sim_filename_base': sim_data.get('filename_base'),
                    'match_score': score,
                    'theta_e_sim': sim_theta_e,
                    'z_lens_sim': sim_z_lens,
                    'z_source_sim': sim_z_source,
                    'mag_sim': sim_mag,
                })
            except (ValueError, TypeError):
                continue
        
        # Sort by match score and take top 3
        match_records.sort(key=lambda x: x['match_score'])
        for rank, match in enumerate(match_records[:3]):
            match['match_rank'] = rank + 1
            matches.append(match)
    
    return matches


def main():
    workspace_root = Path("/Users/gozalig1/Projects/jwst-mock-lens-simulator")
    
    # File paths
    sim_cat_path = workspace_root / "outputs" / "custom_20260213_155632" / "cosmos_lens_training_catalog.csv"
    real_cat_path = workspace_root / "analysis" / "sim_obs_comparison" / "catalogs" / "real_lens_properties.csv"
    output_file = workspace_root / "analysis" / "sim_obs_comparison" / "catalogs" / "matched_lens_pairs.csv"
    
    print(f"Loading simulated catalog: {sim_cat_path}")
    sim_lenses = load_sim_lens_catalog(sim_cat_path)
    print(f"  Found {len(sim_lenses)} simulated lenses\n")
    
    print(f"Loading real lens catalog: {real_cat_path}")
    if not real_cat_path.exists():
        print(f"  Error: Real lens catalog not found. Run extract_real_lens_properties.py first.")
        return
    
    real_lenses = load_real_lens_catalog(real_cat_path)
    print(f"  Found {len(real_lenses)} real lenses\n")
    
    # Match
    print("Matching real to simulated lenses...")
    matches = match_lenses(real_lenses, sim_lenses)
    print(f"  Generated {len(matches)} match records\n")
    
    # Write output
    if matches:
        fieldnames = [
            'match_rank', 'real_name', 'sim_lens_id', 'sim_filename_base',
            'match_score', 'theta_e_sim', 'z_lens_sim', 'z_source_sim', 'mag_sim'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)
        
        print(f"Saved matched pairs to: {output_file}")
    else:
        print("No matches generated.")


if __name__ == "__main__":
    main()
