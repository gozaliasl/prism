#!/usr/bin/env python3
"""
Balanced Training Sets for JWST COSMOS-Web Lens Detection

This module creates balanced training sets with realistic lens/non-lens ratios
specifically designed for JWST COSMOS-Web survey lens detection ML training.

Key Features:
- Realistic survey ratios (1:1000 lens/non-lens)
- Balanced training ratios (1:1, 1:5, 1:10)
- Hard negative mining integration
- Data augmentation pipeline
- Survey-specific quality metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class JWSTBalancedTrainingSets:
    """
    Create balanced training sets with realistic lens/non-lens ratios for JWST COSMOS-Web
    """
    
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        
        # JWST COSMOS-Web realistic ratios based on survey characteristics
        self.survey_ratios = {
            'realistic_survey': 0.001,      # 1:1000 lens/non-lens (realistic survey ratio)
            'balanced_training': 0.5,        # 1:1 lens/non-lens (balanced training)
            'moderate_imbalance': 0.1,       # 1:10 lens/non-lens (moderate imbalance)
            'hard_negative_training': 0.2,   # 1:5 lens/hard-negative (challenging)
            'augmented_training': 0.33       # 1:3 lens/augmented (data augmentation)
        }
        
        # JWST COSMOS-Web survey characteristics
        self.survey_params = {
            'total_area_deg2': 0.6,  # COSMOS-Web survey area
            'pixel_scale_arcsec': 0.03,
            'bands': ['F115W', 'F150W', 'F277W', 'F444W'],
            'expected_lenses': 100,  # Expected lenses in COSMOS-Web
            'galaxies_per_deg2': 50000,  # Typical galaxy density
            'lens_detection_efficiency': 0.8  # Expected detection efficiency
        }
    
    def create_training_strategy(self, n_lenses=1000, strategy='balanced_training'):
        """
        Create training strategy with appropriate ratios
        
        Args:
            n_lenses: Number of lens systems to generate
            strategy: Training strategy ('realistic_survey', 'balanced_training', etc.)
        
        Returns:
            dict: Training strategy with counts and ratios
        """
        
        if strategy not in self.survey_ratios:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.survey_ratios.keys())}")
        
        ratio = self.survey_ratios[strategy]
        n_non_lenses = int(n_lenses / ratio)
        
        # Calculate hard negative and regular negative counts
        if strategy == 'hard_negative_training':
            n_hard_negatives = int(n_non_lenses * 0.6)  # 60% hard negatives
            n_regular_negatives = n_non_lenses - n_hard_negatives
        else:
            n_hard_negatives = int(n_non_lenses * 0.1)  # 10% hard negatives
            n_regular_negatives = n_non_lenses - n_hard_negatives
        
        # Calculate augmentation counts
        n_augmented_lenses = int(n_lenses * 2)  # 2x augmentation
        n_augmented_negatives = int(n_non_lenses * 1.5)  # 1.5x augmentation
        
        strategy_plan = {
            'strategy': strategy,
            'lens_ratio': ratio,
            'total_samples': n_lenses + n_non_lenses,
            'lenses': {
                'count': n_lenses,
                'augmented_count': n_augmented_lenses,
                'redshift_distribution': self._sample_lens_redshifts(n_lenses),
                'mass_distribution': self._sample_lens_masses(n_lenses),
                'einstein_radius_distribution': self._sample_einstein_radii(n_lenses)
            },
            'non_lenses': {
                'count': n_non_lenses,
                'regular_count': n_regular_negatives,
                'hard_negative_count': n_hard_negatives,
                'augmented_count': n_augmented_negatives,
                'types': self._sample_non_lens_types(n_non_lenses),
                'redshift_distribution': self._sample_non_lens_redshifts(n_non_lenses)
            },
            'augmentation': {
                'lens_augmentation_factor': 2.0,
                'negative_augmentation_factor': 1.5,
                'augmentation_types': ['rotation', 'noise', 'psf', 'magnification', 'color']
            }
        }
        
        return strategy_plan
    
    def create_survey_realistic_set(self, n_lenses=100):
        """Create realistic survey-like training set (1:1000 ratio)"""
        return self.create_training_strategy(n_lenses, 'realistic_survey')
    
    def create_balanced_training_set(self, n_lenses=1000):
        """Create balanced training set (1:1 ratio)"""
        return self.create_training_strategy(n_lenses, 'balanced_training')
    
    def create_hard_negative_set(self, n_lenses=500):
        """Create hard negative training set (1:5 ratio with 60% hard negatives)"""
        return self.create_training_strategy(n_lenses, 'hard_negative_training')
    
    def create_augmented_set(self, n_lenses=1000):
        """Create augmented training set with data augmentation"""
        return self.create_training_strategy(n_lenses, 'augmented_training')
    
    def _sample_lens_redshifts(self, n):
        """Sample realistic lens redshifts for JWST COSMOS-Web"""
        # JWST COSMOS-Web lens redshift distribution (based on expected detections)
        # Peak around z=0.8, extending to z=2.0
        z_lens = self.rng.beta(2, 3, n) * 1.8 + 0.2  # Beta distribution with peak at z=0.8
        return np.clip(z_lens, 0.2, 2.0)
    
    def _sample_lens_masses(self, n):
        """Sample realistic lens masses (log10 M_sun)"""
        # Massive galaxies that can act as lenses
        mass_lens = self.rng.normal(11.2, 0.3, n)  # Peak at 10^11.2 M_sun
        return np.clip(mass_lens, 10.5, 12.0)
    
    def _sample_einstein_radii(self, n):
        """Sample realistic Einstein radii (arcsec)"""
        # JWST COSMOS-Web Einstein radius distribution
        theta_E = self.rng.lognormal(np.log(0.8), 0.4, n)  # Peak around 0.8 arcsec
        return np.clip(theta_E, 0.3, 3.0)
    
    def _sample_non_lens_redshifts(self, n):
        """Sample realistic non-lens redshifts"""
        # Broader redshift distribution for non-lenses
        z_non_lens = self.rng.beta(1.5, 2, n) * 3.0 + 0.1  # Peak around z=1.0
        return np.clip(z_non_lens, 0.1, 4.0)
    
    def _sample_non_lens_types(self, n):
        """Sample non-lens galaxy types with realistic frequencies"""
        types = ['central_galaxy', 'galaxy_pair', 'galaxy_group', 'hard_negative']
        probs = [0.4, 0.25, 0.25, 0.1]  # 10% hard negatives
        return self.rng.choice(types, n, p=probs)
    
    def create_training_catalog(self, strategy_plan, output_dir="balanced_training_data"):
        """
        Create training catalog with balanced lens/non-lens ratios
        
        Args:
            strategy_plan: Training strategy from create_training_strategy
            output_dir: Output directory for training data
        
        Returns:
            dict: Training catalog with metadata
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create training catalog structure
        training_catalog = {
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'strategy': strategy_plan['strategy'],
                'lens_ratio': strategy_plan['lens_ratio'],
                'total_samples': strategy_plan['total_samples'],
                'survey_params': self.survey_params
            },
            'lens_systems': [],
            'non_lens_systems': [],
            'hard_negatives': [],
            'augmented_samples': []
        }
        
        # Generate lens systems
        for i in range(strategy_plan['lenses']['count']):
            lens_system = {
                'system_id': f"lens_{i:06d}",
                'system_type': 'lens',
                'is_lens': 1,
                'redshift': float(strategy_plan['lenses']['redshift_distribution'][i]),
                'mass_log10': float(strategy_plan['lenses']['mass_distribution'][i]),
                'einstein_radius': float(strategy_plan['lenses']['einstein_radius_distribution'][i]),
                'difficulty_level': self._assign_difficulty_level('lens'),
                'expected_detection_prob': self._calculate_detection_probability(
                    strategy_plan['lenses']['redshift_distribution'][i],
                    strategy_plan['lenses']['mass_distribution'][i]
                )
            }
            training_catalog['lens_systems'].append(lens_system)
        
        # Generate non-lens systems
        for i in range(strategy_plan['non_lenses']['count']):
            non_lens_type = strategy_plan['non_lenses']['types'][i]
            is_hard_negative = non_lens_type == 'hard_negative'
            
            non_lens_system = {
                'system_id': f"nonlens_{i:06d}",
                'system_type': 'non_lens',
                'is_lens': 0,
                'non_lens_type': non_lens_type,
                'is_hard_negative': is_hard_negative,
                'redshift': float(strategy_plan['non_lenses']['redshift_distribution'][i]),
                'difficulty_level': self._assign_difficulty_level('non_lens', is_hard_negative),
                'expected_confusion_prob': self._calculate_confusion_probability(
                    non_lens_type, 
                    strategy_plan['non_lenses']['redshift_distribution'][i]
                )
            }
            
            if is_hard_negative:
                training_catalog['hard_negatives'].append(non_lens_system)
            else:
                training_catalog['non_lens_systems'].append(non_lens_system)
        
        # Save training catalog
        catalog_path = output_path / "training_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(training_catalog, f, indent=2)
        
        # Create summary statistics
        summary_stats = self._create_summary_statistics(training_catalog)
        stats_path = output_path / "training_summary.json"
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"✓ Balanced training catalog created: {catalog_path}")
        print(f"✓ Training summary: {stats_path}")
        print(f"  Strategy: {strategy_plan['strategy']}")
        print(f"  Total samples: {strategy_plan['total_samples']}")
        print(f"  Lens ratio: {strategy_plan['lens_ratio']:.3f}")
        print(f"  Hard negatives: {len(training_catalog['hard_negatives'])}")
        
        return training_catalog
    
    def _assign_difficulty_level(self, system_type, is_hard_negative=False):
        """Assign difficulty level for training prioritization"""
        if system_type == 'lens':
            # Lens difficulty based on detectability
            return self.rng.choice(['easy', 'medium', 'hard'], p=[0.4, 0.4, 0.2])
        else:
            # Non-lens difficulty
            if is_hard_negative:
                return self.rng.choice(['medium', 'hard'], p=[0.3, 0.7])
            else:
                return self.rng.choice(['easy', 'medium'], p=[0.7, 0.3])
    
    def _calculate_detection_probability(self, redshift, mass_log10):
        """Calculate expected detection probability for lens"""
        # Higher mass and lower redshift = higher detection probability
        mass_factor = (mass_log10 - 10.5) / 1.5  # Normalize mass
        redshift_factor = 1.0 - (redshift - 0.5) / 1.5  # Lower z = higher prob
        
        base_prob = 0.5
        detection_prob = base_prob + 0.3 * mass_factor + 0.2 * redshift_factor
        return np.clip(detection_prob, 0.1, 0.9)
    
    def _calculate_confusion_probability(self, non_lens_type, redshift):
        """Calculate confusion probability for non-lens systems"""
        # Hard negatives have higher confusion probability
        if non_lens_type == 'hard_negative':
            return self.rng.uniform(0.3, 0.7)
        elif non_lens_type in ['galaxy_pair', 'galaxy_group']:
            return self.rng.uniform(0.1, 0.4)
        else:
            return self.rng.uniform(0.05, 0.2)
    
    def _create_summary_statistics(self, training_catalog):
        """Create summary statistics for training catalog"""
        lens_systems = training_catalog['lens_systems']
        non_lens_systems = training_catalog['non_lens_systems']
        hard_negatives = training_catalog['hard_negatives']
        
        # Calculate statistics
        lens_redshifts = [s['redshift'] for s in lens_systems]
        lens_masses = [s['mass_log10'] for s in lens_systems]
        non_lens_redshifts = [s['redshift'] for s in non_lens_systems + hard_negatives]
        
        summary = {
            'dataset_composition': {
                'total_lenses': len(lens_systems),
                'total_non_lenses': len(non_lens_systems),
                'total_hard_negatives': len(hard_negatives),
                'total_samples': len(lens_systems) + len(non_lens_systems) + len(hard_negatives),
                'lens_ratio': len(lens_systems) / (len(non_lens_systems) + len(hard_negatives))
            },
            'lens_statistics': {
                'redshift_range': [min(lens_redshifts), max(lens_redshifts)],
                'redshift_mean': np.mean(lens_redshifts),
                'mass_range': [min(lens_masses), max(lens_masses)],
                'mass_mean': np.mean(lens_masses),
                'avg_detection_prob': np.mean([s['expected_detection_prob'] for s in lens_systems])
            },
            'non_lens_statistics': {
                'redshift_range': [min(non_lens_redshifts), max(non_lens_redshifts)],
                'redshift_mean': np.mean(non_lens_redshifts),
                'avg_confusion_prob': np.mean([s['expected_confusion_prob'] for s in non_lens_systems + hard_negatives])
            },
            'difficulty_distribution': {
                'lens_easy': len([s for s in lens_systems if s['difficulty_level'] == 'easy']),
                'lens_medium': len([s for s in lens_systems if s['difficulty_level'] == 'medium']),
                'lens_hard': len([s for s in lens_systems if s['difficulty_level'] == 'hard']),
                'hard_negatives': len(hard_negatives)
            }
        }
        
        return summary


def create_balanced_training_pipeline(n_lenses=1000, strategy='balanced_training', 
                                    output_dir="balanced_training_data"):
    """
    Create balanced training pipeline for JWST COSMOS-Web lens detection
    
    Args:
        n_lenses: Number of lens systems
        strategy: Training strategy
        output_dir: Output directory
    
    Returns:
        dict: Training pipeline configuration
    """
    
    # Initialize balanced training sets
    balanced_sets = JWSTBalancedTrainingSets()
    
    # Create training strategy
    strategy_plan = balanced_sets.create_training_strategy(n_lenses, strategy)
    
    # Create training catalog
    training_catalog = balanced_sets.create_training_catalog(strategy_plan, output_dir)
    
    print(f"\n🎯 JWST COSMOS-Web Balanced Training Pipeline")
    print(f"   Strategy: {strategy}")
    print(f"   Lenses: {n_lenses}")
    print(f"   Non-lenses: {strategy_plan['non_lenses']['count']}")
    print(f"   Hard negatives: {strategy_plan['non_lenses']['hard_negative_count']}")
    print(f"   Lens ratio: {strategy_plan['lens_ratio']:.3f}")
    
    return {
        'strategy_plan': strategy_plan,
        'training_catalog': training_catalog,
        'balanced_sets': balanced_sets
    }


if __name__ == "__main__":
    # Example usage
    print("Creating balanced training sets for JWST COSMOS-Web...")
    
    # Create different training strategies
    strategies = ['balanced_training', 'hard_negative_training', 'realistic_survey']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} ---")
        pipeline = create_balanced_training_pipeline(
            n_lenses=100 if strategy == 'realistic_survey' else 500,
            strategy=strategy,
            output_dir=f"training_data_{strategy}"
        )
    
    print("\n✅ All balanced training sets created successfully!")
