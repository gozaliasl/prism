#!/usr/bin/env python3
"""
Environment Learning Module for JWST Mock Lens Simulator

This module learns from real COSMOS observations to understand how massive galaxies
(lens and non-lens) are surrounded by foreground and background galaxies.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnvironmentLearner:
    """
    Learn environment properties from real COSMOS observations
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.env_classifier = None
        self.count_regressor = None
        self.radius_regressor = None
        self.trained = False
        
    def load_data(self):
        """Load and prepare COSMOS data"""
        print("Loading COSMOS data...")
        
        # Load the full galaxy catalog (FITS file)
        print("Loading full COSMOS galaxy catalog...")
        try:
            with fits.open(self.data_dir / "galaxy_catalog.fits") as hdul:
                # Assume the main data is in HDU 1 (index 1)
                galaxy_data = hdul[1].data
                galaxy_df = pd.DataFrame(galaxy_data)
                print(f"Loaded {len(galaxy_df)} galaxies from FITS catalog")
        except Exception as e:
            print(f"Could not load FITS catalog: {e}")
            print("Falling back to merged catalog...")
            galaxy_df = pd.read_csv(self.data_dir / "merged_lens_field_catalog.csv")
            print(f"Loaded {len(galaxy_df)} galaxies from merged catalog")
        
        # Load lens catalog for identification
        lens_df = pd.read_csv(self.data_dir / "cosmos_web_lens_structural_properties.csv")
        lens_ids = set(lens_df['ASSOC_ID'].values)
        print(f"Found {len(lens_ids)} lens systems")
        
        # Filter for massive galaxies with good measurements
        print("Filtering for massive galaxies (LP_mass_med_PDF >= 10)...")
        massive_mask = (
            (galaxy_df['LP_mass_med_PDF'] >= 10.0) &  # Massive galaxies (log10 M_sun)
            (galaxy_df['LP_zfinal'] > 0) & 
            (galaxy_df['LP_zfinal'] < 10) & 
            (galaxy_df['LP_warn_fl'] == 0) &  # Good redshift quality
            (galaxy_df['LP_mass_med_PDF'] > 0)  # Valid mass
        )
        
        massive_galaxies = galaxy_df[massive_mask].copy()
        print(f"Found {len(massive_galaxies)} massive galaxies with good measurements")
        
        # Mark lens vs non-lens
        if 'lens_id' in massive_galaxies.columns:
            massive_galaxies['is_lens'] = massive_galaxies['lens_id'].isin(lens_ids)
        else:
            # If no lens_id column, assume all are non-lens for now
            massive_galaxies['is_lens'] = False
        
        print(f"Lens galaxies: {massive_galaxies['is_lens'].sum()}")
        print(f"Non-lens galaxies: {(~massive_galaxies['is_lens']).sum()}")
        
        # Show mass and redshift distributions
        print(f"Mass range: {massive_galaxies['LP_mass_med_PDF'].min():.2f} - {massive_galaxies['LP_mass_med_PDF'].max():.2f}")
        print(f"Redshift range: {massive_galaxies['LP_zfinal'].min():.2f} - {massive_galaxies['LP_zfinal'].max():.2f}")
        
        return massive_galaxies, lens_ids
    
    def analyze_environment(self, massive_galaxies, lens_ids, search_radius_arcsec=4.5, sample_size=5000):
        """
        Analyze environment around massive galaxies within search radius
        Sample a subset for training to make it computationally feasible
        """
        print(f"Analyzing environments within {search_radius_arcsec} arcsec...")
        print(f"Sampling {sample_size} massive galaxies for training...")
        
        # Sample a subset of massive galaxies for training
        if len(massive_galaxies) > sample_size:
            # Stratified sampling to ensure we get both lens and non-lens
            lens_galaxies = massive_galaxies[massive_galaxies['is_lens']]
            non_lens_galaxies = massive_galaxies[~massive_galaxies['is_lens']]
            
            # Sample proportionally
            n_lens_sample = min(len(lens_galaxies), sample_size // 4)  # 25% lens
            n_non_lens_sample = min(len(non_lens_galaxies), sample_size - n_lens_sample)
            
            lens_sample = lens_galaxies.sample(n=n_lens_sample, random_state=42)
            non_lens_sample = non_lens_galaxies.sample(n=n_non_lens_sample, random_state=42)
            
            training_galaxies = pd.concat([lens_sample, non_lens_sample])
            print(f"Sampled {len(training_galaxies)} galaxies: {len(lens_sample)} lens, {len(non_lens_sample)} non-lens")
        else:
            training_galaxies = massive_galaxies
            print(f"Using all {len(training_galaxies)} massive galaxies")
        
        env_data = []
        
        # Load all galaxies for environment analysis
        print("Loading all galaxies for environment analysis...")
        try:
            with fits.open(self.data_dir / "galaxy_catalog.fits") as hdul:
                all_galaxy_data = hdul[1].data
                all_galaxies_df = pd.DataFrame(all_galaxy_data)
                print(f"Loaded {len(all_galaxies_df)} total galaxies for environment analysis")
        except Exception as e:
            print(f"Could not load full catalog: {e}")
            all_galaxies_df = massive_galaxies  # Fallback to massive galaxies only
        
        print(f"Analyzing environments for {len(training_galaxies)} massive galaxies...")
        
        for idx, (_, central_gal) in enumerate(training_galaxies.iterrows()):
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(training_galaxies)} galaxies...")
            
            # Get central galaxy properties
            central_ra = central_gal['RA_DETEC']
            central_dec = central_gal['DEC_DETEC']
            central_z = central_gal['LP_zfinal']
            central_mass = central_gal['LP_mass_med_PDF']
            is_lens = central_gal['is_lens']
            
            # Find surrounding galaxies within search radius
            # Calculate angular separation (simplified)
            ra_diff = all_galaxies_df['RA_DETEC'] - central_ra
            dec_diff = all_galaxies_df['DEC_DETEC'] - central_dec
            # Convert to arcsec (rough approximation)
            separation_arcsec = np.sqrt(ra_diff**2 + dec_diff**2) * 3600
            
            # Filter galaxies within search radius
            nearby_mask = (separation_arcsec <= search_radius_arcsec) & (separation_arcsec > 0)
            nearby_galaxies = all_galaxies_df[nearby_mask]
            
            if len(nearby_galaxies) == 0:
                continue
                
            # Classify environment based on nearby galaxy count and properties
            n_nearby = len(nearby_galaxies)
            
            # Determine environment type
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
            
            # Classify galaxies as foreground/background
            foreground_mask = nearby_redshifts < central_z
            background_mask = nearby_redshifts > central_z
            
            n_foreground = foreground_mask.sum()
            n_background = background_mask.sum()
            
            # Calculate mass-weighted properties
            if len(nearby_galaxies) > 0:
                avg_mass_nearby = np.mean(nearby_masses)
                mass_std_nearby = np.std(nearby_masses)
                avg_separation = np.mean(nearby_separations)
            else:
                avg_mass_nearby = 0
                mass_std_nearby = 0
                avg_separation = 0
            
            env_data.append({
                'central_ra': central_ra,
                'central_dec': central_dec,
                'central_z': central_z,
                'central_mass': central_mass,
                'is_lens': is_lens,
                'env_type': env_type,
                'n_nearby': n_nearby,
                'n_foreground': n_foreground,
                'n_background': n_background,
                'avg_mass_nearby': avg_mass_nearby,
                'mass_std_nearby': mass_std_nearby,
                'avg_separation': avg_separation,
                'max_separation': np.max(nearby_separations) if len(nearby_separations) > 0 else 0
            })
        
        print(f"Completed environment analysis for {len(env_data)} galaxies")
        return pd.DataFrame(env_data)
    
    def train_models(self, env_data):
        """Train ML models to predict environment properties"""
        print("Training environment models...")
        
        # Prepare features
        feature_cols = ['central_z', 'central_mass']
        X = env_data[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train environment type classifier
        y_env = env_data['env_type']
        self.env_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.env_classifier.fit(X_scaled, y_env)
        
        # Train galaxy count regressor
        y_count = env_data['n_nearby']
        self.count_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.count_regressor.fit(X_scaled, y_count)
        
        # Train radius regressor (for halo size)
        y_radius = env_data['max_separation']
        self.radius_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.radius_regressor.fit(X_scaled, y_radius)
        
        self.trained = True
        
        # Evaluate models
        print("\n=== MODEL EVALUATION ===")
        
        # Environment classification
        y_pred_env = self.env_classifier.predict(X_scaled)
        print("Environment Classification:")
        print(classification_report(y_env, y_pred_env))
        
        # Count regression
        y_pred_count = self.count_regressor.predict(X_scaled)
        print(f"Count Regression R²: {r2_score(y_count, y_pred_count):.3f}")
        
        # Radius regression
        y_pred_radius = self.radius_regressor.predict(X_scaled)
        print(f"Radius Regression R²: {r2_score(y_radius, y_pred_radius):.3f}")
        
        return True
    
    def predict_environment(self, redshift, mass_log10):
        """Predict environment properties for a given galaxy"""
        if not self.trained:
            raise ValueError("Models not trained yet!")
        
        X = np.array([[redshift, mass_log10]])
        X_scaled = self.scaler.transform(X)
        
        env_type = self.env_classifier.predict(X_scaled)[0]
        n_galaxies = int(np.clip(self.count_regressor.predict(X_scaled)[0], 0, 10))
        max_radius = self.radius_regressor.predict(X_scaled)[0]
        
        return {
            'env_type': env_type,
            'n_galaxies': n_galaxies,
            'max_radius_arcsec': max_radius
        }
    
    def save_models(self, output_dir="models"):
        """Save trained models"""
        if not self.trained:
            raise ValueError("Models not trained yet!")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        joblib.dump(self.env_classifier, output_path / "env_classifier.pkl")
        joblib.dump(self.count_regressor, output_path / "count_regressor.pkl")
        joblib.dump(self.radius_regressor, output_path / "radius_regressor.pkl")
        
        print(f"Models saved to {output_path}")
    
    def load_models(self, model_dir="models"):
        """Load pre-trained models"""
        model_path = Path(model_dir)
        
        self.scaler = joblib.load(model_path / "scaler.pkl")
        self.env_classifier = joblib.load(model_path / "env_classifier.pkl")
        self.count_regressor = joblib.load(model_path / "count_regressor.pkl")
        self.radius_regressor = joblib.load(model_path / "radius_regressor.pkl")
        
        self.trained = True
        print(f"Models loaded from {model_path}")

def main():
    """Main training pipeline"""
    learner = EnvironmentLearner()
    
    # Load and analyze data
    massive_galaxies, lens_ids = learner.load_data()
    env_data = learner.analyze_environment(massive_galaxies, lens_ids)
    
    print(f"\nEnvironment Analysis Results:")
    print(f"Total environments analyzed: {len(env_data)}")
    print(f"Environment distribution:")
    print(env_data['env_type'].value_counts())
    print(f"Lens vs non-lens distribution:")
    print(env_data['is_lens'].value_counts())
    
    # Train models
    learner.train_models(env_data)
    
    # Save models
    learner.save_models()
    
    # Test predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    test_cases = [
        (0.5, 11.0),   # Low-z massive galaxy
        (1.0, 11.5),   # Mid-z massive galaxy
        (2.0, 10.5),   # High-z massive galaxy
    ]
    
    for z, mass in test_cases:
        pred = learner.predict_environment(z, mass)
        print(f"z={z}, log10(M)={mass}: {pred}")

if __name__ == "__main__":
    main()
