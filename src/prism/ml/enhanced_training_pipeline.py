#!/usr/bin/env python3
"""
Enhanced Training Pipeline for JWST COSMOS-Web Lens Detection

This module integrates:
1. Hard negative mining
2. Data augmentation
3. Balanced training sets
4. Survey-specific parameters

Designed for JWST COSMOS-Web survey lens detection ML training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our enhancement modules
try:
    from prism.ml.ml_training_enhancements import (
        JWSTHardNegativeMiner, JWSTDataAugmentation, 
        JWSTBalancedTrainingSets, JWSTSurveyMetrics
    )
    from prism.ml.balanced_training_sets import JWSTBalancedTrainingSets as BalancedSets
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    print("Warning: ML enhancements not available")

class JWSTEnhancedTrainingPipeline:
    """
    Comprehensive training pipeline for JWST COSMOS-Web lens detection
    """
    
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        
        if ENHANCEMENTS_AVAILABLE:
            self.hard_negative_miner = JWSTHardNegativeMiner(rng)
            self.data_augmentation = JWSTDataAugmentation(rng)
            self.balanced_sets = BalancedSets(rng)
            self.survey_metrics = JWSTSurveyMetrics()
        else:
            print("Warning: Using fallback implementations")
    
    def create_comprehensive_training_set(self, n_lenses=1000, 
                                        training_strategy='balanced_training',
                                        hard_negative_ratio=0.2,
                                        augmentation_factor=2.0,
                                        output_dir="enhanced_training_data"):
        """
        Create comprehensive training set with all enhancements
        
        Args:
            n_lenses: Number of lens systems
            training_strategy: Training strategy
            hard_negative_ratio: Fraction of hard negatives
            augmentation_factor: Data augmentation factor
            output_dir: Output directory
        
        Returns:
            dict: Comprehensive training configuration
        """
        
        print(f"🚀 Creating Enhanced Training Pipeline for JWST COSMOS-Web")
        print(f"   Lenses: {n_lenses}")
        print(f"   Strategy: {training_strategy}")
        print(f"   Hard negative ratio: {hard_negative_ratio}")
        print(f"   Augmentation factor: {augmentation_factor}")
        
        # Create balanced training strategy
        if ENHANCEMENTS_AVAILABLE:
            strategy_plan = self.balanced_sets.create_training_strategy(n_lenses, training_strategy)
        else:
            # Fallback strategy
            strategy_plan = self._create_fallback_strategy(n_lenses, training_strategy)
        
        # Create training catalog
        training_catalog = self._create_enhanced_training_catalog(
            strategy_plan, hard_negative_ratio, augmentation_factor, output_dir
        )
        
        # Create data augmentation plan
        augmentation_plan = self._create_augmentation_plan(
            training_catalog, augmentation_factor
        )
        
        # Create quality metrics
        quality_metrics = self._create_quality_metrics(training_catalog)
        
        # Save comprehensive configuration
        config = {
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'pipeline_version': 'v1.0',
                'survey': 'JWST_COSMOS_Web',
                'strategy': training_strategy,
                'n_lenses': n_lenses,
                'hard_negative_ratio': hard_negative_ratio,
                'augmentation_factor': augmentation_factor
            },
            'strategy_plan': strategy_plan,
            'training_catalog': training_catalog,
            'augmentation_plan': augmentation_plan,
            'quality_metrics': quality_metrics
        }
        
        # Save configuration
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        config_path = output_path / "enhanced_training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Enhanced training pipeline created: {config_path}")
        return config
    
    def _create_enhanced_training_catalog(self, strategy_plan, hard_negative_ratio, 
                                        augmentation_factor, output_dir):
        """Create enhanced training catalog with hard negatives and augmentation"""
        
        training_catalog = {
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
                ),
                'augmentation_candidates': self._select_augmentation_candidates(
                    strategy_plan['lenses']['redshift_distribution'][i],
                    strategy_plan['lenses']['mass_distribution'][i]
                )
            }
            training_catalog['lens_systems'].append(lens_system)
        
        # Generate non-lens systems
        n_non_lenses = strategy_plan['non_lenses']['count']
        n_hard_negatives = int(n_non_lenses * hard_negative_ratio)
        n_regular_negatives = n_non_lenses - n_hard_negatives
        
        # Regular non-lens systems
        for i in range(n_regular_negatives):
            non_lens_system = {
                'system_id': f"nonlens_{i:06d}",
                'system_type': 'non_lens',
                'is_lens': 0,
                'non_lens_type': 'regular',
                'is_hard_negative': False,
                'redshift': float(strategy_plan['non_lenses']['redshift_distribution'][i]),
                'difficulty_level': self._assign_difficulty_level('non_lens', False),
                'expected_confusion_prob': self._calculate_confusion_probability(
                    'regular', strategy_plan['non_lenses']['redshift_distribution'][i]
                )
            }
            training_catalog['non_lens_systems'].append(non_lens_system)
        
        # Hard negative systems
        if ENHANCEMENTS_AVAILABLE:
            hard_negative_types = list(self.hard_negative_miner.hard_negative_types.keys())
        else:
            hard_negative_types = ['edge_on_spiral', 'merging_galaxies', 'star_forming_ring']
        
        for i in range(n_hard_negatives):
            hard_negative_type = self.rng.choice(hard_negative_types)
            hard_negative_system = {
                'system_id': f"hardneg_{i:06d}",
                'system_type': 'non_lens',
                'is_lens': 0,
                'non_lens_type': 'hard_negative',
                'hard_negative_type': hard_negative_type,
                'is_hard_negative': True,
                'redshift': float(strategy_plan['non_lenses']['redshift_distribution'][i + n_regular_negatives]),
                'difficulty_level': self._assign_difficulty_level('non_lens', True),
                'expected_confusion_prob': self._calculate_confusion_probability(
                    'hard_negative', strategy_plan['non_lenses']['redshift_distribution'][i + n_regular_negatives]
                )
            }
            training_catalog['hard_negatives'].append(hard_negative_system)
        
        return training_catalog
    
    def _create_augmentation_plan(self, training_catalog, augmentation_factor):
        """Create data augmentation plan"""
        
        if not ENHANCEMENTS_AVAILABLE:
            return {'augmentation_available': False}
        
        augmentation_plan = {
            'augmentation_available': True,
            'augmentation_factor': augmentation_factor,
            'augmentation_types': ['rotation', 'noise', 'psf', 'magnification', 'color'],
            'lens_augmentation': [],
            'negative_augmentation': []
        }
        
        # Plan lens augmentation
        for lens in training_catalog['lens_systems']:
            if lens['augmentation_candidates']:
                for aug_type in self.rng.choice(
                    augmentation_plan['augmentation_types'], 
                    size=int(augmentation_factor),
                    replace=True
                ):
                    augmentation_plan['lens_augmentation'].append({
                        'base_system_id': lens['system_id'],
                        'augmentation_type': aug_type,
                        'augmentation_params': self._generate_augmentation_params(aug_type)
                    })
        
        # Plan negative augmentation
        all_negatives = training_catalog['non_lens_systems'] + training_catalog['hard_negatives']
        for negative in all_negatives:
            for aug_type in self.rng.choice(
                augmentation_plan['augmentation_types'], 
                size=int(augmentation_factor * 0.5),  # Less augmentation for negatives
                replace=True
            ):
                augmentation_plan['negative_augmentation'].append({
                    'base_system_id': negative['system_id'],
                    'augmentation_type': aug_type,
                    'augmentation_params': self._generate_augmentation_params(aug_type)
                })
        
        return augmentation_plan
    
    def _create_quality_metrics(self, training_catalog):
        """Create quality metrics for training set"""
        
        quality_metrics = {
            'dataset_balance': self._calculate_dataset_balance(training_catalog),
            'difficulty_distribution': self._calculate_difficulty_distribution(training_catalog),
            'redshift_coverage': self._calculate_redshift_coverage(training_catalog),
            'expected_performance': self._estimate_expected_performance(training_catalog)
        }
        
        return quality_metrics
    
    def _create_fallback_strategy(self, n_lenses, strategy):
        """Create fallback strategy when enhancements not available"""
        return {
            'strategy': strategy,
            'lens_ratio': 0.5 if strategy == 'balanced_training' else 0.1,
            'total_samples': n_lenses * 2,
            'lenses': {
                'count': n_lenses,
                'redshift_distribution': np.random.uniform(0.2, 2.0, n_lenses),
                'mass_distribution': np.random.normal(11.2, 0.3, n_lenses)
            },
            'non_lenses': {
                'count': n_lenses,
                'types': ['central_galaxy'] * n_lenses,
                'redshift_distribution': np.random.uniform(0.1, 3.0, n_lenses)
            }
        }
    
    def _assign_difficulty_level(self, system_type, is_hard_negative=False):
        """Assign difficulty level"""
        if system_type == 'lens':
            return self.rng.choice(['easy', 'medium', 'hard'], p=[0.4, 0.4, 0.2])
        else:
            if is_hard_negative:
                return self.rng.choice(['medium', 'hard'], p=[0.3, 0.7])
            else:
                return self.rng.choice(['easy', 'medium'], p=[0.7, 0.3])
    
    def _calculate_detection_probability(self, redshift, mass_log10):
        """Calculate detection probability"""
        mass_factor = (mass_log10 - 10.5) / 1.5
        redshift_factor = 1.0 - (redshift - 0.5) / 1.5
        base_prob = 0.5
        detection_prob = base_prob + 0.3 * mass_factor + 0.2 * redshift_factor
        return np.clip(detection_prob, 0.1, 0.9)
    
    def _calculate_confusion_probability(self, non_lens_type, redshift):
        """Calculate confusion probability"""
        if non_lens_type == 'hard_negative':
            return self.rng.uniform(0.3, 0.7)
        else:
            return self.rng.uniform(0.05, 0.3)
    
    def _select_augmentation_candidates(self, redshift, mass_log10):
        """Select augmentation candidates based on system properties"""
        # Higher mass, lower redshift = more augmentation candidates
        if mass_log10 > 11.0 and redshift < 1.0:
            return True
        elif mass_log10 > 10.8 and redshift < 1.5:
            return self.rng.random() < 0.7
        else:
            return self.rng.random() < 0.4
    
    def _generate_augmentation_params(self, aug_type):
        """Generate augmentation parameters"""
        if aug_type == 'rotation':
            return {'max_angle': 360}
        elif aug_type == 'noise':
            return {'noise_factor_range': (0.5, 2.0)}
        elif aug_type == 'psf':
            return {'psf_variation_range': (0.8, 1.2)}
        elif aug_type == 'magnification':
            return {'mag_range': (0.8, 1.2)}
        elif aug_type == 'color':
            return {'color_shift_range': (-0.2, 0.2)}
        else:
            return {}
    
    def _calculate_dataset_balance(self, training_catalog):
        """Calculate dataset balance metrics"""
        n_lenses = len(training_catalog['lens_systems'])
        n_non_lenses = len(training_catalog['non_lens_systems'])
        n_hard_negatives = len(training_catalog['hard_negatives'])
        
        return {
            'lens_ratio': n_lenses / (n_non_lenses + n_hard_negatives),
            'hard_negative_ratio': n_hard_negatives / (n_non_lenses + n_hard_negatives),
            'total_samples': n_lenses + n_non_lenses + n_hard_negatives
        }
    
    def _calculate_difficulty_distribution(self, training_catalog):
        """Calculate difficulty distribution"""
        all_systems = (training_catalog['lens_systems'] + 
                      training_catalog['non_lens_systems'] + 
                      training_catalog['hard_negatives'])
        
        difficulty_counts = {}
        for system in all_systems:
            level = system['difficulty_level']
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
        
        return difficulty_counts
    
    def _calculate_redshift_coverage(self, training_catalog):
        """Calculate redshift coverage"""
        all_redshifts = []
        for system in (training_catalog['lens_systems'] + 
                      training_catalog['non_lens_systems'] + 
                      training_catalog['hard_negatives']):
            all_redshifts.append(system['redshift'])
        
        return {
            'redshift_range': [min(all_redshifts), max(all_redshifts)],
            'redshift_mean': np.mean(all_redshifts),
            'redshift_std': np.std(all_redshifts)
        }
    
    def _estimate_expected_performance(self, training_catalog):
        """Estimate expected ML performance"""
        n_lenses = len(training_catalog['lens_systems'])
        n_hard_negatives = len(training_catalog['hard_negatives'])
        
        # Estimate based on dataset characteristics
        base_accuracy = 0.85
        hard_negative_penalty = n_hard_negatives / (n_lenses + n_hard_negatives) * 0.1
        
        estimated_accuracy = base_accuracy - hard_negative_penalty
        
        return {
            'estimated_accuracy': estimated_accuracy,
            'estimated_precision': estimated_accuracy * 0.9,
            'estimated_recall': estimated_accuracy * 0.85,
            'hard_negative_impact': hard_negative_penalty
        }


def create_enhanced_training_pipeline(n_lenses=1000, strategy='balanced_training',
                                    hard_negative_ratio=0.2, augmentation_factor=2.0,
                                    output_dir="enhanced_training_data"):
    """
    Create enhanced training pipeline for JWST COSMOS-Web lens detection
    
    Args:
        n_lenses: Number of lens systems
        strategy: Training strategy
        hard_negative_ratio: Fraction of hard negatives
        augmentation_factor: Data augmentation factor
        output_dir: Output directory
    
    Returns:
        dict: Enhanced training pipeline configuration
    """
    
    pipeline = JWSTEnhancedTrainingPipeline()
    
    config = pipeline.create_comprehensive_training_set(
        n_lenses=n_lenses,
        training_strategy=strategy,
        hard_negative_ratio=hard_negative_ratio,
        augmentation_factor=augmentation_factor,
        output_dir=output_dir
    )
    
    return config


if __name__ == "__main__":
    # Example usage
    print("Creating Enhanced Training Pipeline for JWST COSMOS-Web...")
    
    # Create different training configurations
    configurations = [
        {'n_lenses': 500, 'strategy': 'balanced_training', 'hard_negative_ratio': 0.2},
        {'n_lenses': 1000, 'strategy': 'hard_negative_training', 'hard_negative_ratio': 0.4},
        {'n_lenses': 200, 'strategy': 'realistic_survey', 'hard_negative_ratio': 0.1}
    ]
    
    for i, config in enumerate(configurations):
        print(f"\n--- Configuration {i+1} ---")
        pipeline_config = create_enhanced_training_pipeline(
            n_lenses=config['n_lenses'],
            strategy=config['strategy'],
            hard_negative_ratio=config['hard_negative_ratio'],
            augmentation_factor=2.0,
            output_dir=f"enhanced_training_{i+1}"
        )
    
    print("\n✅ All enhanced training pipelines created successfully!")
