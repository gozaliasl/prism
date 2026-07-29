#!/usr/bin/env python3
"""
Enhanced Environment Model Training

This script can train environment models from either:
1. Merged CSV catalog (176 massive galaxies) - for quick testing
2. Full galaxy catalog (if available) - for production training
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_training_data(data_dir="../data", use_full_catalog=False):
    """Load training data from available sources"""
    print("Loading training data...")
    
    data_dir = Path(data_dir)
    
    if use_full_catalog:
        # Try to load from full galaxy catalog CSV
        print("Attempting to load from full galaxy catalog CSV...")
        try:
            galaxy_path = data_dir / "galaxy_catalog.csv"
            if galaxy_path.exists():
                galaxy_df = pd.read_csv(galaxy_path)
                print(f"Loaded {len(galaxy_df)} galaxies from full CSV catalog")
                
                # Filter for massive galaxies with good measurements
                massive_mask = (
                    (galaxy_df['LP_mass_med_PDF'] >= 10.0) &
                    (galaxy_df['LP_zfinal'] > 0) &
                    (galaxy_df['LP_zfinal'] < 10) &
                    (galaxy_df['LP_warn_fl'] == 0) &
                    (galaxy_df['LP_mass_med_PDF'] > 0)
                )
                
                massive_galaxies = galaxy_df[massive_mask].copy()
                print(f"Found {len(massive_galaxies)} massive galaxies in full catalog")
                
                if len(massive_galaxies) > 0:
                    print(f"✅ Loaded {len(massive_galaxies)} massive galaxies from full CSV catalog")
                    return massive_galaxies, galaxy_df
            else:
                print(f"Full catalog CSV not found: {galaxy_path}")
        except Exception as e:
            print(f"Could not load full catalog: {e}")
    
    # Fall back to merged catalog
    print("Using merged CSV catalog...")
    merged_path = data_dir / "merged_lens_field_catalog.csv"
    galaxy_df = pd.read_csv(merged_path)
    print(f"Loaded {len(galaxy_df)} galaxies from merged catalog")
    
    # The merged catalog is already organized by lens systems
    # Each lens has lens_id, lens_ra, lens_dec and surrounding galaxies
    print("Merged catalog is organized by lens systems:")
    print(f"  - Each lens has lens_id, lens_ra, lens_dec")
    print(f"  - Surrounding galaxies are grouped by lens")
    print(f"  - No mass cut needed - central lens is already identified")
    
    # Filter for good measurements only
    good_measurements_mask = (
        (galaxy_df['LP_zfinal'] > 0) &
        (galaxy_df['LP_zfinal'] < 10) &
        (galaxy_df['LP_warn_fl'] == 0) &
        (galaxy_df['LP_mass_med_PDF'] > 0)
    )
    
    all_galaxies = galaxy_df[good_measurements_mask].copy()
    print(f"Found {len(all_galaxies)} galaxies with good measurements")
    
    # Get unique lens systems
    unique_lenses = all_galaxies[['lens_id', 'lens_ra', 'lens_dec']].drop_duplicates()
    print(f"Found {len(unique_lenses)} unique lens systems")
    
    # Show sample lens systems
    print("Sample lens systems:")
    for idx, (_, lens) in enumerate(unique_lenses.head(3).iterrows()):
        lens_galaxies = all_galaxies[all_galaxies['lens_id'] == lens['lens_id']]
        print(f"  {lens['lens_id']}: {len(lens_galaxies)} surrounding galaxies")
    
    # Use all galaxies for training (lens-centric approach)
    training_galaxies = all_galaxies
    
    return training_galaxies, galaxy_df

def analyze_environment_enhanced(training_galaxies, all_galaxies_df, lens_ids, 
                                search_radius_arcsec=5.0, sample_size=None):
    """Analyze environments around lens systems (lens-centric approach)"""
    print(f"Analyzing lens environments within {search_radius_arcsec} arcsec...")
    
    # Get unique lens systems
    unique_lenses = training_galaxies[['lens_id', 'lens_ra', 'lens_dec']].drop_duplicates()
    print(f"Found {len(unique_lenses)} unique lens systems")
    
    # Determine sample size
    if sample_size is None:
        sample_size = min(len(unique_lenses), 200)  # Default sample size for lens systems
    
    if len(unique_lenses) > sample_size:
        print(f"Sampling {sample_size} lens systems for training...")
        # Sample lens systems
        sampled_lenses = unique_lenses.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(sampled_lenses)} lens systems")
    else:
        sampled_lenses = unique_lenses
        print(f"Using all {len(sampled_lenses)} lens systems for training")
    
    env_data = []
    
    print(f"Analyzing environments for {len(sampled_lenses)} lens systems...")
    
    for idx, (_, lens_info) in enumerate(sampled_lenses.iterrows()):
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(sampled_lenses)} lens systems...")
        
        lens_id = lens_info['lens_id']
        lens_ra = lens_info['lens_ra']
        lens_dec = lens_info['lens_dec']
        
        # Get all galaxies for this lens system
        lens_galaxies = training_galaxies[training_galaxies['lens_id'] == lens_id]
        
        if len(lens_galaxies) == 0:
            continue
        
        # Get central lens properties (first galaxy in the group)
        central_gal = lens_galaxies.iloc[0]
        central_z = central_gal['LP_zfinal']
        central_mass = central_gal['LP_mass_med_PDF']
        
        # Count surrounding galaxies (excluding the central lens)
        surrounding_galaxies = lens_galaxies.iloc[1:] if len(lens_galaxies) > 1 else pd.DataFrame()
        
        if len(surrounding_galaxies) == 0:
            continue
        
        # Classify environment based on surrounding galaxy count
        n_surrounding = len(surrounding_galaxies)
        if n_surrounding <= 2:
            env_type = 'isolated_field'
        elif n_surrounding <= 5:
            env_type = 'galaxy_pair'
        else:
            env_type = 'group'
        
        # Calculate environment properties
        surrounding_masses = surrounding_galaxies['LP_mass_med_PDF'].values
        surrounding_redshifts = surrounding_galaxies['LP_zfinal'].values
        
        # Calculate separations from lens center
        ra_diff = surrounding_galaxies['RA_DETEC'] - lens_ra
        dec_diff = surrounding_galaxies['DEC_DETEC'] - lens_dec
        separations_arcsec = np.sqrt(ra_diff**2 + dec_diff**2) * 3600
        
        # Classify foreground/background
        foreground_mask = surrounding_redshifts < central_z
        background_mask = surrounding_redshifts > central_z
        
        n_foreground = foreground_mask.sum()
        n_background = background_mask.sum()
        
        # Calculate properties
        if len(surrounding_galaxies) > 0:
            avg_mass_surrounding = np.mean(surrounding_masses)
            mass_std_surrounding = np.std(surrounding_masses)
            avg_separation = np.mean(separations_arcsec)
            max_separation = np.max(separations_arcsec)
        else:
            avg_mass_surrounding = 0
            mass_std_surrounding = 0
            avg_separation = 0
            max_separation = 0
        
        env_data.append({
            'lens_id': lens_id,
            'lens_ra': lens_ra,
            'lens_dec': lens_dec,
            'central_z': central_z,
            'central_mass': central_mass,
            'env_type': env_type,
            'n_surrounding': n_surrounding,
            'n_foreground': n_foreground,
            'n_background': n_background,
            'avg_mass_surrounding': avg_mass_surrounding,
            'mass_std_surrounding': mass_std_surrounding,
            'avg_separation': avg_separation,
            'max_separation': max_separation
        })
    
    print(f"Completed environment analysis for {len(env_data)} lens systems")
    return pd.DataFrame(env_data)

def analyze_environment_massive_galaxies(massive_galaxies, all_galaxies_df, lens_ids, 
                                       search_radius_arcsec=5.0, sample_size=None):
    """Analyze environments around massive galaxies (full catalog approach)"""
    print(f"Analyzing environments around massive galaxies within {search_radius_arcsec} arcsec...")
    
    # Determine sample size
    if sample_size is None:
        sample_size = min(len(massive_galaxies), 2000)  # Default sample size
    
    if len(massive_galaxies) > sample_size:
        print(f"Sampling {sample_size} massive galaxies for training...")
        # Sample massive galaxies
        sampled_galaxies = massive_galaxies.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(sampled_galaxies)} massive galaxies")
    else:
        sampled_galaxies = massive_galaxies
        print(f"Using all {len(sampled_galaxies)} massive galaxies for training")
    
    env_data = []
    
    print(f"Analyzing environments for {len(sampled_galaxies)} massive galaxies...")
    
    for idx, (_, central_gal) in enumerate(sampled_galaxies.iterrows()):
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(sampled_galaxies)} massive galaxies...")
        
        # Get central galaxy properties
        central_ra = central_gal['RA_DETEC']
        central_dec = central_gal['DEC_DETEC']
        central_z = central_gal['LP_zfinal']
        central_mass = central_gal['LP_mass_med_PDF']
        
        # Find surrounding galaxies within search radius
        ra_diff = all_galaxies_df['RA_DETEC'] - central_ra
        dec_diff = all_galaxies_df['DEC_DETEC'] - central_dec
        separation_arcsec = np.sqrt(ra_diff**2 + dec_diff**2) * 3600
        
        # Filter galaxies within search radius
        nearby_mask = (separation_arcsec <= search_radius_arcsec) & (separation_arcsec > 0)
        nearby_galaxies = all_galaxies_df[nearby_mask]
        
        if len(nearby_galaxies) == 0:
            continue
        
        # Classify environment
        n_nearby = len(nearby_galaxies)
        if n_nearby <= 2:
            env_type = 'isolated_field'
        elif n_nearby <= 5:
            env_type = 'galaxy_pair'
        else:
            env_type = 'group'
        
        # Calculate environment properties
        nearby_masses = nearby_galaxies['LP_mass_med_PDF'].values
        nearby_redshifts = nearby_galaxies['LP_zfinal'].values
        nearby_separations = separation_arcsec[nearby_mask].values
        
        # Classify foreground/background
        foreground_mask = nearby_redshifts < central_z
        background_mask = nearby_redshifts > central_z
        
        n_foreground = foreground_mask.sum()
        n_background = background_mask.sum()
        
        # Calculate properties
        if len(nearby_galaxies) > 0:
            avg_mass_nearby = np.mean(nearby_masses)
            mass_std_nearby = np.std(nearby_masses)
            avg_separation = np.mean(nearby_separations)
            max_separation = np.max(nearby_separations)
        else:
            avg_mass_nearby = 0
            mass_std_nearby = 0
            avg_separation = 0
            max_separation = 0
        
        env_data.append({
            'central_ra': central_ra,
            'central_dec': central_dec,
            'central_z': central_z,
            'central_mass': central_mass,
            'env_type': env_type,
            'n_nearby': n_nearby,
            'n_foreground': n_foreground,
            'n_background': n_background,
            'avg_mass_nearby': avg_mass_nearby,
            'mass_std_nearby': mass_std_nearby,
            'avg_separation': avg_separation,
            'max_separation': max_separation
        })
    
    print(f"Completed environment analysis for {len(env_data)} massive galaxies")
    return pd.DataFrame(env_data)

def train_models(env_data):
    """Train ML models"""
    print("Training environment models...")
    
    # Prepare features
    feature_cols = ['central_z', 'central_mass']
    X = env_data[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train environment type classifier
    y_env = env_data['env_type']
    env_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    env_classifier.fit(X_scaled, y_env)
    
    # Train galaxy count regressor (handle both column names)
    if 'n_surrounding' in env_data.columns:
        y_count = env_data['n_surrounding']
    else:
        y_count = env_data['n_nearby']
    count_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    count_regressor.fit(X_scaled, y_count)
    
    # Train radius regressor
    y_radius = env_data['max_separation']
    radius_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    radius_regressor.fit(X_scaled, y_radius)
    
    # Evaluate models
    print("\n=== MODEL EVALUATION ===")
    
    # Environment classification
    y_pred_env = env_classifier.predict(X_scaled)
    print("Environment Classification:")
    print(classification_report(y_env, y_pred_env))
    
    # Count regression
    y_pred_count = count_regressor.predict(X_scaled)
    print(f"Count Regression R²: {r2_score(y_count, y_pred_count):.3f}")
    
    # Radius regression
    y_pred_radius = radius_regressor.predict(X_scaled)
    print(f"Radius Regression R²: {r2_score(y_radius, y_pred_radius):.3f}")
    
    return scaler, env_classifier, count_regressor, radius_regressor

def save_models(scaler, env_classifier, count_regressor, radius_regressor, output_dir="../models"):
    """Save trained models"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(env_classifier, output_path / "env_classifier.pkl")
    joblib.dump(count_regressor, output_path / "count_regressor.pkl")
    joblib.dump(radius_regressor, output_path / "radius_regressor.pkl")
    
    print(f"Models saved to {output_path}")

def main():
    """Main training pipeline"""
    print("=== Enhanced COSMOS Environment Learning ===")
    print("Training ML models for 50,000+ mock systems")
    print("")
    
    # Check if we should use full catalog
    use_full_catalog = len(sys.argv) > 1 and sys.argv[1] == "--full-catalog"
    
    if use_full_catalog:
        print("🎯 Using full galaxy catalog CSV for maximum training data")
    else:
        print("🎯 Using merged catalog (lens-centric approach)")
        print("   Use --full-catalog flag to try full galaxy catalog CSV")
    
    print("")
    
    # Load training data
    training_galaxies, all_galaxies_df = load_training_data(use_full_catalog=use_full_catalog)
    
    # Load lens catalog for identification
    lens_df = pd.read_csv("../data/cosmos_web_lens_structural_properties.csv")
    lens_ids = set(lens_df['ASSOC_ID'].values)
    print(f"Found {len(lens_ids)} lens systems")
    
    # Show statistics based on catalog type
    if use_full_catalog:
        # Full catalog: massive galaxies approach
        print(f"Massive galaxies in full catalog: {len(training_galaxies)}")
        print(f"Mass range: {training_galaxies['LP_mass_med_PDF'].min():.2f} - {training_galaxies['LP_mass_med_PDF'].max():.2f}")
        print(f"Redshift range: {training_galaxies['LP_zfinal'].min():.2f} - {training_galaxies['LP_zfinal'].max():.2f}")
        
        # Show sample massive galaxies
        print("Sample massive galaxies:")
        for idx, (_, gal) in enumerate(training_galaxies.head(3).iterrows()):
            print(f"  Galaxy {idx+1}: log10(M)={gal['LP_mass_med_PDF']:.2f}, z={gal['LP_zfinal']:.2f}")
        
        # Analyze environments (massive galaxy approach)
        env_data = analyze_environment_massive_galaxies(training_galaxies, all_galaxies_df, lens_ids)
    else:
        # Merged catalog: lens-centric approach
        unique_lenses = training_galaxies[['lens_id', 'lens_ra', 'lens_dec']].drop_duplicates()
        print(f"Lens systems in merged catalog: {len(unique_lenses)}")
        
        # Show mass and redshift distributions
        print(f"Mass range: {training_galaxies['LP_mass_med_PDF'].min():.2f} - {training_galaxies['LP_mass_med_PDF'].max():.2f}")
        print(f"Redshift range: {training_galaxies['LP_zfinal'].min():.2f} - {training_galaxies['LP_zfinal'].max():.2f}")
        
        # Show sample lens systems
        print("Sample lens systems:")
        for idx, (_, lens) in enumerate(unique_lenses.head(3).iterrows()):
            lens_galaxies = training_galaxies[training_galaxies['lens_id'] == lens['lens_id']]
            central_mass = lens_galaxies.iloc[0]['LP_mass_med_PDF']
            central_z = lens_galaxies.iloc[0]['LP_zfinal']
            print(f"  {lens['lens_id']}: {len(lens_galaxies)} galaxies, log10(M)={central_mass:.2f}, z={central_z:.2f}")
        
        # Analyze environments (lens-centric approach)
        env_data = analyze_environment_enhanced(training_galaxies, all_galaxies_df, lens_ids)
    
    print(f"\nEnvironment Analysis Results:")
    print(f"Total environments analyzed: {len(env_data)}")
    print(f"Environment distribution:")
    print(env_data['env_type'].value_counts())
    print(f"Lens systems analyzed: {len(env_data)} (all are lens systems)")
    
    # Show statistics
    print(f"\nEnvironment Statistics:")
    if 'n_surrounding' in env_data.columns:
        print(f"Average surrounding galaxies per environment:")
        print(env_data.groupby('env_type')['n_surrounding'].agg(['mean', 'std', 'min', 'max']))
    else:
        print(f"Average nearby galaxies per environment:")
        print(env_data.groupby('env_type')['n_nearby'].agg(['mean', 'std', 'min', 'max']))
    
    print(f"\nForeground vs Background distribution:")
    print(env_data.groupby('env_type')[['n_foreground', 'n_background']].mean())
    
    # Train models
    scaler, env_classifier, count_regressor, radius_regressor = train_models(env_data)
    
    # Save models
    save_models(scaler, env_classifier, count_regressor, radius_regressor)
    
    # Test predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    test_cases = [
        (0.5, 11.0, "Low-z massive galaxy"),
        (1.0, 11.5, "Mid-z massive galaxy"),
        (2.0, 10.5, "High-z massive galaxy"),
        (0.8, 11.2, "Typical lens galaxy"),
    ]
    
    for z, mass, description in test_cases:
        X_test = np.array([[z, mass]])
        X_test_scaled = scaler.transform(X_test)
        
        env_type = env_classifier.predict(X_test_scaled)[0]
        n_galaxies = int(np.clip(count_regressor.predict(X_test_scaled)[0], 0, 10))
        max_radius = radius_regressor.predict(X_test_scaled)[0]
        
        print(f"{description} (z={z}, log10(M)={mass}):")
        print(f"  Environment: {env_type}")
        print(f"  Expected galaxies: {n_galaxies}")
        print(f"  Max radius: {max_radius:.2f} arcsec")
    
    print("\n=== Training Complete ===")
    print("Models saved to ../models/")
    print("Ready for 50,000+ mock system generation!")

if __name__ == "__main__":
    main()
