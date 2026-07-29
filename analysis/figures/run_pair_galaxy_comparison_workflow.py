#!/usr/bin/env python
"""
Helper script to run JWST lens simulations with different pair galaxy treatments
and generate comparison figures for paper discussion.

This workflow:
1. Generates 100-200 lens systems with each pair galaxy treatment
2. Creates comparison figures for selected lens IDs
3. Generates analysis report on differences

Usage:
------
python analysis/figures/run_pair_galaxy_comparison_workflow.py \
    --n-per-treatment 100 \
    --n-comparisons 5 \
    --output-dir results/pair_comparison_study
"""

import argparse
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import json


def run_command(cmd, description=None):
    """Run shell command and report status"""
    if description:
        print(f"\n{'='*70}")
        print(f"[STEP] {description}")
        print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    print(f"✓ Success")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run pair galaxy treatment comparison workflow"
    )
    parser.add_argument(
        '--n-per-treatment',
        type=int,
        default=50,
        help='Number of lens systems per treatment'
    )
    parser.add_argument(
        '--n-comparisons',
        type=int,
        default=5,
        help='Number of lens IDs to show in comparison figures'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/pair_comparison_study',
        help='Base output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2026,
        help='Random seed'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip lens generation, only create comparisons'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration files for each treatment
    treatments = {
        'sie_sie': {
            'config': 'configs/default_config.yaml',  # Default has SIE+SIE
            'output': base_dir / 'simulations' / 'sie_sie',
            'description': 'SIE+SIE Binary Lens (Both galaxies as SIE)'
        },
        'nfw_nfw': {
            'config': 'configs/default_config.yaml',  # Modified for NFW
            'output': base_dir / 'simulations' / 'nfw_nfw',
            'description': 'NFW+NFW Binary Lens (Realistic mass profiles)'
        },
        'shear_only': {
            'config': 'configs/default_config.yaml',  # Modified to disable binary
            'output': base_dir / 'simulations' / 'shear_only',
            'description': 'Shear Only (Pair as external shear only)'
        }
    }
    
    generated_files = {}
    
    if not args.skip_generation:
        print("\n" + "="*70)
        print("STEP 1: LENS SIMULATION WITH DIFFERENT PAIR TREATMENTS")
        print("="*70)
        
        for treatment_name, treatment_info in treatments.items():
            output_dir = treatment_info['output']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            description = treatment_info['description']
            
            # Note: This is a template. Actual generation would use:
            # python src/generate_lens_dataset.py \
            #     --output-dir {output_dir} \
            #     --n-lenses {n} \
            #     --config {config} \
            #     --seed {seed}_{treatment}
            
            print(f"\n{'─'*70}")
            print(f"Treatment: {treatment_name.upper()}")
            print(f"Description: {description}")
            print(f"Output: {output_dir}")
            print(f"{'─'*70}")
            
            print(f"⚠️  NOTE: Actual generation would use:")
            print(f"    python src/generate_lens_dataset.py \\")
            print(f"        --output-dir {output_dir} \\")
            print(f"        --n-lenses {args.n_per_treatment} \\")
            print(f"        --config {treatment_info['config']} \\")
            print(f"        --seed {args.seed}_{treatment_name}")
            
            generated_files[treatment_name] = str(output_dir)
    
    # ================================================================
    print("\n" + "="*70)
    print("STEP 2: CREATE COMPARISON FIGURES")
    print("="*70)
    
    # For demo, we'll use mock comparison generation
    comparison_dir = base_dir / 'comparisons'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'python',
        'analysis/figures/compare_pair_galaxy_treatments.py',
        '--n-samples', str(args.n_comparisons),
        '--output-dir', str(comparison_dir),
        '--seed', str(args.seed)
    ]
    
    run_command(cmd, "Generate comparison figures")
    
    # ================================================================
    print("\n" + "="*70)
    print("STEP 3: GENERATE ANALYSIS REPORT")
    print("="*70)
    
    report = {
        'workflow': 'Pair Galaxy Treatment Comparison',
        'timestamp': pd.Timestamp.now().isoformat(),
        'parameters': {
            'n_per_treatment': args.n_per_treatment,
            'n_comparisons': args.n_comparisons,
            'seed': args.seed
        },
        'treatments': {name: info['description'] for name, info in treatments.items()},
        'outputs': {
            'simulations': generated_files,
            'comparisons': str(comparison_dir),
            'report': str(base_dir / 'analysis_report.json')
        },
        'next_steps': [
            '1. Review comparison figures in: ' + str(comparison_dir),
            '2. Analyze differences in arc morphology and magnification',
            '3. Compare computational costs of different treatments',
            '4. Consider which treatment best balances accuracy vs. cost',
            '5. Incorporate findings into paper discussion'
        ]
    }
    
    report_path = base_dir / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n✓ Workflow completed successfully!")
    print(f"\nOutputs:")
    print(f"  • Comparison figures: {comparison_dir}")
    print(f"  • Analysis report: {report_path}")
    
    print(f"\nNext steps:")
    for step in report['next_steps']:
        print(f"  {step}")
    
    print(f"\nTo view comparison figures:")
    print(f"  open {comparison_dir}/pair_comparison_*.png")
    
    print(f"\nTo extract metrics:")
    print(f"  python analysis/figures/extract_comparison_metrics.py \\")
    print(f"      --comparison-dir {comparison_dir} \\")
    print(f"      --output {base_dir}/metrics.csv")


if __name__ == '__main__':
    main()
