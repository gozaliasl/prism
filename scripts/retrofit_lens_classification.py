#!/usr/bin/env python3
"""
Retrofit existing simulation outputs with lens system classification.
Converts file names to PRISM_[lens|nonlens]_TYPE_[epoch_]ID format.
Adds lens_system_class column to catalog if needed.
"""

import pandas as pd
from pathlib import Path
import sys
import argparse
import re
from typing import Dict, List, Tuple

# Import the classifier
from src.lens_system_classifier import LensSystemClassifier


def generate_prism_filename(lens_id: int, system_class: str = None, is_nonlens: bool = False, 
                           epoch_index: int = None, extension: str = '.jpg') -> str:
    """
    Generate PRISM-formatted filename.
    
    Format: PRISM_[lens|nonlens]_[TYPE_][epoch_]ID.ext
    
    Examples:
        - PRISM_lens_SF_000000.jpg
        - PRISM_lens_BSIE_epoch01_000001.jpg
        - PRISM_nonlens_000002.jpg
    """
    sample_type = "nonlens" if is_nonlens else "lens"
    epoch_str = f"epoch{epoch_index:02d}_" if epoch_index is not None else ""
    
    if is_nonlens:
        return f"PRISM_nonlens_{epoch_str}{int(lens_id):06d}{extension}"
    else:
        short_code = LensSystemClassifier.get_short_code(system_class or 'single_field')
        return f"PRISM_lens_{short_code}_{epoch_str}{int(lens_id):06d}{extension}"


def parse_old_filename(filename: str) -> Dict:
    """
    Parse old naming conventions to extract lens_id and other info.
    Supports patterns like:
    - cosmos_lens_XXXXXX
    - cosmos_nonlens_XXXXXX
    - PRISM_XXXXXX_CC (old PRISM format)
    """
    name = Path(filename).stem  # Remove extension
    
    # Try old PRISM format: PRISM_XXXXXX_TYPE
    match = re.search(r'PRISM_(\d{6})_(\w+)', name)
    if match:
        return {
            'lens_id': int(match.group(1)),
            'old_format': 'prism_old',
            'is_nonlens': False
        }
    
    # Try cosmos format: cosmos_[lens|nonlens]_XXXXXX
    match = re.search(r'cosmos_(lens|nonlens)_(\d{6})', name)
    if match:
        return {
            'lens_id': int(match.group(2)),
            'old_format': 'cosmos',
            'is_nonlens': match.group(1) == 'nonlens'
        }
    
    # Try new PRISM format: PRISM_[lens|nonlens]_...
    match = re.search(r'PRISM_(lens|nonlens)_.*_(\d{6})', name)
    if match:
        return {
            'lens_id': int(match.group(2)),
            'old_format': 'prism_new',
            'is_nonlens': match.group(1) == 'nonlens'
        }
    
    return None


def classify_existing_output(output_dir: str, dry_run: bool = True, apply_renaming: bool = False) -> Dict:
    """
    Analyze and classify lenses in existing output directory.
    Converts to new PRISM_[lens|nonlens]_TYPE_[epoch_]ID naming format.
    
    Parameters
    ----------
    output_dir : str
        Path to simulation output directory
    dry_run : bool
        If True, only preview changes without modifying files
    apply_renaming : bool
        If True, also rename image files
        
    Returns
    -------
    dict
        Summary of classification and renames
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Find catalog files
    catalog_candidates = [
        output_dir / 'cosmos_lens_training_catalog.csv',
        output_dir / 'cosmos_training_catalog_lens_and_nonlens.csv',
    ]
    
    catalog_path = None
    for cand in catalog_candidates:
        if cand.exists():
            catalog_path = cand
            break
    
    if not catalog_path:
        raise FileNotFoundError(f"No catalog found in {output_dir}")
    
    print(f"\n{'='*90}")
    print(f"PRISM NAMING RETROFIT: CONVERT TO [LENS|NONLENS]_TYPE_[EPOCH_]ID FORMAT")
    print(f"{'='*90}")
    print(f"\nProcessing: {output_dir}")
    print(f"Catalog: {catalog_path.name}")
    print(f"Dry-run mode: {dry_run}")
    
    # Load catalog
    catalog_df = pd.read_csv(catalog_path)
    print(f"[✓] Loaded catalog with {len(catalog_df)} samples")
    
    # Add lens_system_class column if not present
    if 'lens_system_class' not in catalog_df.columns:
        print("\n[INFO] Adding lens_system_class column to catalog...")
        classifier = LensSystemClassifier()
        
        # Classify based on available data
        catalog_df['lens_system_class'] = catalog_df.apply(
            lambda row: classify_lens_heuristic(row, classifier),
            axis=1
        )
        
        # Add description column
        catalog_df['lens_system_class_description'] = catalog_df['lens_system_class'].apply(
            lambda x: LensSystemClassifier.get_description(x)
        )
        
        print("[✓] Classification complete")
    
    # Show distribution
    print(f"\n{'LENS SYSTEM CLASS DISTRIBUTION':^90}")
    print(f"{'-'*90}")
    
    if 'lens_system_class' in catalog_df.columns:
        class_dist = catalog_df['lens_system_class'].value_counts().sort_index()
        total = len(catalog_df)
        
        for class_name, count in class_dist.items():
            pct = (count / total) * 100
            desc = LensSystemClassifier.get_description(class_name)
            short_code = LensSystemClassifier.get_short_code(class_name)
            print(f"  {class_name:20s} ({short_code:5s}): {count:4d} samples ({pct:6.2f}%) - {desc}")
        
        print(f"{'-'*90}")
    
    # Save updated catalog
    if not dry_run:
        catalog_df.to_csv(catalog_path, index=False)
        print(f"\n[✓] Updated catalog saved to: {catalog_path}")
    else:
        print(f"\n[DRY-RUN] Would save updated catalog to: {catalog_path}")
    
    summary = {
        'output_dir': str(output_dir),
        'total_samples': len(catalog_df),
        'class_distribution': {},
        'catalog_updated': not dry_run,
        'files_renamed': 0,
        'rename_summary': None
    }
    
    # Optionally rename files to new PRISM format
    if apply_renaming:
        print(f"\n{'RENAMING IMAGE FILES TO NEW PRISM FORMAT':^90}")
        print(f"{'-'*90}")
        
        jpg_dir = output_dir / 'jpg_rgb'
        npy_dir = output_dir / 'npy'
        
        rename_ops = []
        errors = []
        
        # Process jpg files
        if jpg_dir.exists():
            print(f"\nProcessing JPG files in: {jpg_dir.name}/")
            for old_path in sorted(jpg_dir.glob('*.jpg')):
                try:
                    parsed = parse_old_filename(old_path.name)
                    if not parsed:
                        errors.append(f"Could not parse: {old_path.name}")
                        continue
                    
                    lens_id = parsed['lens_id']
                    is_nonlens = parsed['is_nonlens']
                    
                    # Get system class from catalog if it's a lens
                    system_class = 'single_field'
                    if not is_nonlens and 'lens_system_class' in catalog_df.columns:
                        matching_rows = catalog_df[catalog_df['lens_id'] == lens_id]
                        if not matching_rows.empty:
                            system_class = matching_rows.iloc[0]['lens_system_class']
                    
                    # Generate new filename
                    new_name = generate_prism_filename(lens_id, system_class, is_nonlens, extension='.jpg')
                    new_path = jpg_dir / new_name
                    
                    if old_path.name != new_name:
                        rename_ops.append({
                            'old': old_path.name,
                            'new': new_name,
                            'path': str(old_path),
                            'status': 'rename'
                        })
                        
                        if not dry_run:
                            old_path.rename(new_path)
                    else:
                        rename_ops.append({
                            'old': old_path.name,
                            'new': new_name,
                            'status': 'skip'
                        })
                        
                except Exception as e:
                    errors.append(f"{old_path.name}: {str(e)}")
        
        # Process npy files (similar logic)
        if npy_dir.exists():
            print(f"\nProcessing NPY files in: {npy_dir.name}/")
            for old_path in sorted(npy_dir.glob('*.npy')):
                try:
                    parsed = parse_old_filename(old_path.name)
                    if not parsed:
                        continue
                    
                    lens_id = parsed['lens_id']
                    is_nonlens = parsed['is_nonlens']
                    
                    # Get system class from catalog if it's a lens
                    system_class = 'single_field'
                    if not is_nonlens and 'lens_system_class' in catalog_df.columns:
                        matching_rows = catalog_df[catalog_df['lens_id'] == lens_id]
                        if not matching_rows.empty:
                            system_class = matching_rows.iloc[0]['lens_system_class']
                    
                    # Generate new filename
                    new_name = generate_prism_filename(lens_id, system_class, is_nonlens, extension='.npy')
                    new_path = npy_dir / new_name
                    
                    if old_path.name != new_name:
                        rename_ops.append({
                            'old': old_path.name,
                            'new': new_name,
                            'path': str(old_path),
                            'status': 'rename'
                        })
                        
                        if not dry_run:
                            old_path.rename(new_path)
                    else:
                        rename_ops.append({
                            'old': old_path.name,
                            'new': new_name,
                            'status': 'skip'
                        })
                        
                except Exception as e:
                    errors.append(f"{old_path.name}: {str(e)}")
        
        # Report results
        renamed_count = len([op for op in rename_ops if op['status'] == 'rename'])
        skipped_count = len([op for op in rename_ops if op['status'] == 'skip'])
        
        print(f"\nFile rename operations:")
        print(f"  Total files found: {len(rename_ops) + len(errors)}")
        print(f"  To rename: {renamed_count}")
        print(f"  Already correct: {skipped_count}")
        print(f"  Errors: {len(errors)}")
        
        if errors:
            print(f"\nErrors encountered (first 5):")
            for err in errors[:5]:
                print(f"    - {err}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
        
        # Show sample renames
        sample_renames = [op for op in rename_ops if op['status'] == 'rename'][:10]
        if sample_renames:
            print(f"\nSample file renames (first {len(sample_renames)}):")
            for op in sample_renames:
                print(f"  {op['old']:50s}")
                print(f"    → {op['new']:50s}")
        
        if not dry_run:
            print(f"\n[✓] File renaming complete: {renamed_count} files renamed")
        else:
            print(f"\n[DRY-RUN] Would rename {renamed_count} files")
        
        summary['files_renamed'] = renamed_count
        summary['rename_summary'] = {
            'renamed': renamed_count,
            'skipped': skipped_count,
            'errors': len(errors),
            'operations': rename_ops
        }
    
    # Show sample entries
    print(f"\n{'SAMPLE CLASSIFIED LENSES/NON-LENSES':^90}")
    print(f"{'-'*90}")
    print(f"{'ID':<6} {'Type':<10} {'System Class':<20} {'Description':<40}")
    print(f"{'-'*90}")
    
    for idx, row in catalog_df.head(15).iterrows():
        lens_id = int(row.get('lens_id', idx))
        sys_type = 'nonlens' if row.get('system_type') == 'nonlens' else 'lens'
        sys_class = row.get('lens_system_class', 'unknown')
        desc = str(row.get('lens_system_class_description', ''))[:37]
        if len(str(row.get('lens_system_class_description', ''))) > 40:
            desc += '...'
        print(f"{lens_id:<6} {sys_type:<10} {sys_class:<20} {desc:<40}")
    
    print(f"\n{'='*90}\n")
    
    return summary


def classify_lens_heuristic(row: pd.Series, classifier: LensSystemClassifier) -> str:
    """
    Heuristically classify a lens based on available catalog columns.
    Without specific lens_model_list data, defaults to single_field.
    """
    # Check for specific indicators if available
    if 'system_type' in row and row['system_type'] == 'nonlens':
        return 'single_field'  # Placeholder for nonlens
    
    # Default to single_field
    return 'single_field'


def main():
    """Command-line interface for PRISM naming retrofit."""
    parser = argparse.ArgumentParser(
        description='Retrofit simulation output to new PRISM_[lens|nonlens]_TYPE_[epoch_]ID naming format'
    )
    parser.add_argument(
        'output_dir',
        help='Path to simulation output directory'
    )
    parser.add_argument(
        '--apply-renaming',
        action='store_true',
        help='Rename image files to new PRISM format'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Preview changes without applying them (default)'
    )
    parser.add_argument(
        '--no-dry-run',
        dest='dry_run',
        action='store_false',
        help='Apply changes to files and catalog'
    )
    
    args = parser.parse_args()
    
    try:
        summary = classify_existing_output(
            args.output_dir,
            dry_run=args.dry_run,
            apply_renaming=args.apply_renaming
        )
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
