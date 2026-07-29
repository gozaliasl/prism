"""
Lens system classifier for categorizing lenses based on their properties.
Enables classification of: single_field, group, binary_sie_sie, binary_nfw_nfw, shear_only
"""

import numpy as np
import pandas as pd
from pathlib import Path


class LensSystemClassifier:
    """Classify lens systems based on their generation parameters."""
    
    # Classification categories
    CATEGORIES = {
        'single_field': 'Single field galaxy lens',
        'group': 'Group lens (multiple components)',
        'binary_sie_sie': 'Binary pair: SIE+SIE',
        'binary_nfw_nfw': 'Binary pair: NFW+NFW',
        'shear_only': 'Shear-only (external perturbation)',
    }
    
    # Short codes for filenames (simplified for user request)
    SHORT_CODES = {
        'single_field': 'SF',
        'group': 'GR',
        'binary_sie_sie': 'BR',
        'binary_nfw_nfw': 'BR',
        'shear_only': 'BR',
    }
    
    @staticmethod
    def classify_lens(lens_model_list, kwargs_lens=None, n_components=1):
        """
        Classify a lens based on its model configuration.
        
        Parameters
        ----------
        lens_model_list : list
            List of lens model names (e.g., ['SIE', 'SHEAR'], ['SIE', 'SIE', 'SHEAR'])
        kwargs_lens : list, optional
            Lens model parameters
        n_components : int
            Number of lens components used in generation
            
        Returns
        -------
        str
            One of the classification categories
        """
        if lens_model_list is None or len(lens_model_list) == 0:
            return 'single_field'
        
        # Check for shear-only (external perturbation without lens)
        if len(lens_model_list) == 1 and lens_model_list[0] in ['SHEAR', 'EXTERNAL_SHEAR']:
            return 'shear_only'
        
        # Check for shear-only binary signature (SIE + multiple SHEAR components)
        if 'SIE' in lens_model_list and lens_model_list.count('SHEAR') >= 2:
            return 'shear_only'
        
        # Count different model types
        model_types = {}
        for model in lens_model_list:
            model_types[model] = model_types.get(model, 0) + 1
        
        # Check for binary systems (exactly 2 lens components + shear)
        # Group systems have 3+ lens components
        non_shear_count = sum(count for model, count in model_types.items() if model not in ['SHEAR', 'EXTERNAL_SHEAR'])
        
        if non_shear_count >= 3:
            # 3+ lens components = group
            return 'group'
        elif non_shear_count == 2:
            # Exactly 2 lens components = binary
            if 'SIE' in model_types and model_types['SIE'] >= 2:
                return 'binary_sie_sie'
            elif 'NFW' in model_types and model_types['NFW'] >= 2:
                return 'binary_nfw_nfw'
            else:
                # Mixed types with 2 components = still binary
                return 'binary_sie_sie'  # Default to SIE type
        
        # Single lens component (+ shear for external perturbation)
        if len(model_types) == 1:
            return 'single_field'
        elif len(model_types) == 2 and 'SHEAR' in model_types:
            return 'single_field'
        
        # Group lens (multiple different components)
        return 'group'
    
    @staticmethod
    def get_short_code(category):
        """Get short filename code for a lens category."""
        return LensSystemClassifier.SHORT_CODES.get(category, 'UNK')
    
    @staticmethod
    def get_description(category):
        """Get human-readable description of a lens category."""
        return LensSystemClassifier.CATEGORIES.get(category, 'Unknown')


def update_catalog_with_lens_class(catalog_df, lens_model_info=None):
    """
    Add lens_system_class column to catalog DataFrame.
    
    Parameters
    ----------
    catalog_df : pd.DataFrame
        Catalog with lens information
    lens_model_info : dict, optional
        Mapping of lens_id to lens_model_list for more accurate classification
        
    Returns
    -------
    pd.DataFrame
        Catalog with added 'lens_system_class' column
    """
    classifier = LensSystemClassifier()
    
    # Initialize with default classification
    catalog_df['lens_system_class'] = 'single_field'
    
    # If we have specific model information, use it for classification
    if lens_model_info:
        for idx, row in catalog_df.iterrows():
            lens_id = row.get('lens_id')
            if lens_id in lens_model_info:
                models, kwargs = lens_model_info[lens_id]
                classification = classifier.classify_lens(models, kwargs)
                catalog_df.at[idx, 'lens_system_class'] = classification
    else:
        # Heuristic classification based on existing catalog columns
        for idx, row in catalog_df.iterrows():
            # Default is single_field for basic catalogs
            catalog_df.at[idx, 'lens_system_class'] = 'single_field'
    
    return catalog_df


def create_classified_filename(lens_id, system_class, prefix='PRISM', extension='.jpg', 
                               is_nonlens=False, epoch_index=None):
    """
    Create a filename with lens system classification.
    
    Format: PRISM_[lens|nonlens]_TYPE_[epoch_]ID.ext
    
    Examples:
        - PRISM_lens_SF_000000.jpg (single field lens)
        - PRISM_lens_BSIE_epoch_000001.jpg (binary SIE+SIE with time delay)
        - PRISM_nonlens_000002.jpg (non-lens sample)
    
    Parameters
    ----------
    lens_id : int
        Lens identifier
    system_class : str
        Lens system classification (only used for lens samples)
    prefix : str
        Base filename prefix (default: 'PRISM')
    extension : str
        File extension (default: '.jpg')
    is_nonlens : bool
        If True, filename is for non-lens sample (no type code)
    epoch_index : int or None
        Optional epoch number for time-delay systems
        
    Returns
    -------
    str
        Classified filename
    """
    sample_type = "nonlens" if is_nonlens else "lens"
    
    if is_nonlens:
        # Non-lens format: PRISM_nonlens_[epoch_]ID.ext
        if epoch_index is not None:
            return f"{prefix}_nonlens_epoch{epoch_index:02d}_{int(lens_id):06d}{extension}"
        else:
            return f"{prefix}_nonlens_{int(lens_id):06d}{extension}"
    else:
        # Lens format: PRISM_lens_TYPE_[epoch_]ID.ext
        short_code = LensSystemClassifier.get_short_code(system_class)
        if epoch_index is not None:
            return f"{prefix}_lens_{short_code}_epoch{epoch_index:02d}_{int(lens_id):06d}{extension}"
        else:
            return f"{prefix}_lens_{short_code}_{int(lens_id):06d}{extension}"


def rename_files_with_classification(output_dir, catalog_df, dry_run=True):
    """
    Rename image files to include lens system classification.
    
    Parameters
    ----------
    output_dir : Path or str
        Output directory containing jpg_rgb subdirectory
    catalog_df : pd.DataFrame
        Catalog with 'lens_system_class' column
    dry_run : bool
        If True, only print what would be done; don't actually rename
        
    Returns
    -------
    dict
        Summary of rename operations
    """
    output_dir = Path(output_dir)
    jpg_dir = output_dir / 'jpg_rgb'
    
    if not jpg_dir.exists():
        raise FileNotFoundError(f"jpg_rgb directory not found: {jpg_dir}")
    
    summary = {
        'total': 0,
        'renamed': 0,
        'errors': [],
        'skipped': 0,
        'operations': []
    }
    
    # Find all image files
    for img_file in sorted(jpg_dir.glob('PRISM_*.jpg')):
        filename = img_file.name
        summary['total'] += 1
        
        # Extract lens_id from filename (e.g., PRISM_000123.jpg or PRISM_000123_rgb.jpg)
        try:
            # Parse original filename
            base_name = filename.replace('_rgb.jpg', '').replace('.jpg', '')
            parts = base_name.split('_')
            lens_id_str = parts[1]  # PRISM_XXXXXX
            lens_id = int(lens_id_str)
            
            # Find matching catalog entry
            matching = catalog_df[catalog_df['lens_id'] == lens_id]
            if len(matching) == 0:
                summary['skipped'] += 1
                continue
            
            system_class = matching.iloc[0]['lens_system_class']
            short_code = LensSystemClassifier.get_short_code(system_class)
            
            # Create new filename
            if '_rgb.jpg' in filename:
                new_filename = f"PRISM_{int(lens_id):06d}_{short_code}_rgb.jpg"
            else:
                new_filename = f"PRISM_{int(lens_id):06d}_{short_code}.jpg"
            
            old_path = img_file
            new_path = jpg_dir / new_filename
            
            operation = {
                'old': filename,
                'new': new_filename,
                'system_class': system_class,
                'status': 'skipped' if old_path == new_path else 'rename'
            }
            
            if not dry_run and old_path != new_path:
                if new_path.exists():
                    summary['errors'].append(f"Target already exists: {new_filename}")
                else:
                    old_path.rename(new_path)
                    summary['renamed'] += 1
                    operation['status'] = 'renamed'
            elif old_path != new_path:
                summary['renamed'] += 1
            
            summary['operations'].append(operation)
            
        except Exception as e:
            summary['errors'].append(f"Error processing {filename}: {e}")
    
    return summary


def update_output_catalog(output_dir, catalog_df=None, add_descriptions=True):
    """
    Update catalog CSV file with lens_system_class column.
    
    Parameters
    ----------
    output_dir : Path or str
        Output directory containing cosmos_lens_training_catalog.csv
    catalog_df : pd.DataFrame, optional
        Pre-loaded catalog; if None, will load from disk
    add_descriptions : bool
        If True, also add a human-readable description
        
    Returns
    -------
    pd.DataFrame
        Updated catalog
    """
    output_dir = Path(output_dir)
    catalog_path = output_dir / 'cosmos_lens_training_catalog.csv'
    
    if catalog_path.exists() and catalog_df is None:
        catalog_df = pd.read_csv(catalog_path)
    elif catalog_df is None:
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    
    # Add classification column
    if 'lens_system_class' not in catalog_df.columns:
        catalog_df['lens_system_class'] = 'single_field'
    
    # Optionally add description
    if add_descriptions and 'lens_system_class_description' not in catalog_df.columns:
        classifier = LensSystemClassifier()
        catalog_df['lens_system_class_description'] = catalog_df['lens_system_class'].apply(
            lambda x: classifier.get_description(x)
        )
    
    return catalog_df


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        
        # Test the classification system
        print("Testing lens system classifier...")
        print(f"\nTest classifications:")
        print(f"  ['SIE', 'SHEAR'] -> {LensSystemClassifier.classify_lens(['SIE', 'SHEAR'])}")
        print(f"  ['SIE', 'SIE', 'SHEAR'] -> {LensSystemClassifier.classify_lens(['SIE', 'SIE', 'SHEAR'])}")
        print(f"  ['NFW', 'NFW', 'SHEAR'] -> {LensSystemClassifier.classify_lens(['NFW', 'NFW', 'SHEAR'])}")
        print(f"  ['SHEAR'] -> {LensSystemClassifier.classify_lens(['SHEAR'])}")
        
        # Try to update catalog if it exists
        try:
            catalog_df = update_output_catalog(output_dir)
            print(f"\n✓ Updated catalog with {len(catalog_df)} entries")
            
            # Show distribution
            print(f"\nLens system class distribution:")
            print(catalog_df['lens_system_class'].value_counts())
            
        except Exception as e:
            print(f"Could not update catalog: {e}")
