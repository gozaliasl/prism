#!/usr/bin/env python3
"""
Size Evolution Analysis using existing redshift-binned mass-size relations

This script uses the previously fitted mass-size relations in redshift bins to:
1. Extract sizes at fixed stellar masses for each redshift bin
2. Fit the size evolution: log₁₀(Reff/kpc) = A - α log(1 + z)

We use the median stellar mass of each sample and the existing fit results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')

class SizeEvolutionFromExistingFits:
    def __init__(self):
        """Initialize the analyzer"""
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
    def load_existing_fits(self):
        """Load existing redshift-binned fit results"""
        print("Loading existing redshift-binned fit results...")
        
        # Load COWLS redshift-binned relations
        cowls_file = 'cowls_redshift_binned_relations.csv'
        cowls_fits = pd.read_csv(cowls_file)
        print(f"COWLS redshift bins: {len(cowls_fits)}")
        
        # Load massive galaxy redshift-binned relations  
        massive_file = 'massive_galaxy_redshift_binned_relations.csv'
        massive_fits = pd.read_csv(massive_file)
        print(f"Massive galaxy redshift bins: {len(massive_fits)}")
        
        return cowls_fits, massive_fits
    
    def load_median_masses(self):
        """Load median stellar masses from existing analysis"""
        print("Loading median stellar masses...")
        
        # Load COWLS fit results
        cowls_file = 'cowls_fit_results.csv'
        cowls_results = pd.read_csv(cowls_file)
        
        # Load massive galaxy fit results
        massive_file = 'massive_galaxy_fit_results.csv'
        massive_results = pd.read_csv(massive_file)
        
        # Extract median masses
        cowls_median = cowls_results[cowls_results['Sample'] == 'COWLS']['Median_Mass_log'].iloc[0]
        cowls_early_median = cowls_results[cowls_results['Sample'] == 'COWLS Early-type']['Median_Mass_log'].iloc[0]
        cowls_late_median = cowls_results[cowls_results['Sample'] == 'COWLS Late-type']['Median_Mass_log'].iloc[0]
        
        massive_median = massive_results[massive_results['Sample'] == 'CWMGs']['Median_Mass_log'].iloc[0]
        massive_early_median = massive_results[massive_results['Sample'] == 'CWMGs Early-type']['Median_Mass_log'].iloc[0]
        massive_late_median = massive_results[massive_results['Sample'] == 'CWMGs Late-type']['Median_Mass_log'].iloc[0]
        
        print(f"COWLS median mass: {cowls_median:.3f}")
        print(f"COWLS Early-type median mass: {cowls_early_median:.3f}")
        print(f"COWLS Late-type median mass: {cowls_late_median:.3f}")
        print(f"CWMGs median mass: {massive_median:.3f}")
        print(f"CWMGs Early-type median mass: {massive_early_median:.3f}")
        print(f"CWMGs Late-type median mass: {massive_late_median:.3f}")
        
        return {
            'COWLS': cowls_median,
            'COWLS Early-type': cowls_early_median,
            'COWLS Late-type': cowls_late_median,
            'CWMGs': massive_median,
            'CWMGs Early-type': massive_early_median,
            'CWMGs Late-type': massive_late_median
        }
    
    def extract_sizes_at_fixed_mass(self, fits_df, sample_name, median_mass):
        """Extract sizes at fixed mass for each redshift bin"""
        print(f"Extracting sizes at fixed mass for {sample_name}...")
        
        # Filter fits for this sample
        sample_fits = fits_df[fits_df['Sample'] == sample_name].copy()
        
        if len(sample_fits) == 0:
            print(f"  No fits found for {sample_name}")
            return pd.DataFrame()
        
        # Calculate sizes at fixed mass using the fitted relations
        # log₁₀(Reff/kpc) = alpha * log₁₀(M*/M☉) + beta
        # where beta = logA - alpha * log(5e10) for normalized relation
        
        sizes = []
        redshifts = []
        z_centers = []
        
        for _, fit in sample_fits.iterrows():
            alpha = fit['alpha']
            beta = fit['beta']  # This is already the intercept for log₁₀(M*/M☉)
            
            # Calculate size at median mass
            log_size = alpha * median_mass + beta
            
            # Convert to physical size
            size_kpc = 10**log_size
            
            sizes.append(size_kpc)
            redshifts.append(fit['z_center'])
            z_centers.append(fit['z_center'])
        
        result_df = pd.DataFrame({
            'z_center': z_centers,
            'log_size': np.log10(sizes),
            'size_kpc': sizes,
            'median_mass': median_mass
        })
        
        print(f"  Extracted {len(result_df)} redshift bins")
        return result_df
    
    def fit_size_evolution(self, size_data, sample_name):
        """Fit size evolution: log₁₀(Reff/kpc) = A - α log(1 + z)"""
        print(f"Fitting size evolution for {sample_name}...")
        
        if len(size_data) < 3:
            print(f"  Not enough data points ({len(size_data)}) for {sample_name}")
            return None
        
        # Prepare data
        log_size = size_data['log_size'].values
        log_redshift = np.log10(1 + size_data['z_center'].values)
        
        print(f"  Using {len(size_data)} redshift bins")
        print(f"  Redshift range: {size_data['z_center'].min():.2f} - {size_data['z_center'].max():.2f}")
        print(f"  Size range: {size_data['size_kpc'].min():.2f} - {size_data['size_kpc'].max():.2f} kpc")
        
        # Fit with PyMC
        with pm.Model() as model:
            A = pm.Normal("A", mu=0.5, sigma=1.0)
            alpha = pm.Normal("alpha", mu=0.2, sigma=0.5)
            sigma = pm.HalfNormal("sigma", sigma=0.3)
            
            # Model: log_size = A - alpha * log(1 + z)
            mu = A - alpha * log_redshift
            
            pm.Normal("log_size_obs", mu=mu, sigma=sigma, observed=log_size)
            
            trace = pm.sample(
                draws=2000,
                tune=500,
                target_accept=0.95,
                chains=4,
                cores=2,
                progressbar=False
            )
        
        # Extract results
        A_samples = trace.posterior['A'].values.flatten()
        alpha_samples = trace.posterior['alpha'].values.flatten()
        
        A_mean = np.mean(A_samples)
        A_err = np.std(A_samples)
        alpha_mean = np.mean(alpha_samples)
        alpha_err = np.std(alpha_samples)
        
        # Calculate R²
        y_pred = A_mean - alpha_mean * log_redshift
        y_true = log_size
        r_squared = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        # Calculate confidence intervals
        A_ci = np.percentile(A_samples, [2.5, 97.5])
        alpha_ci = np.percentile(alpha_samples, [2.5, 97.5])
        
        print(f"  {sample_name} size evolution: log₁₀(Reff/kpc) = {A_mean:.3f} ± {A_err:.3f} - {alpha_mean:.3f} ± {alpha_err:.3f} × log(1+z)")
        print(f"  R² = {r_squared:.3f}, N = {len(size_data)}")
        print(f"  95% CI: A = [{A_ci[0]:.3f}, {A_ci[1]:.3f}], α = [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]")
        
        return {
            'A': A_mean, 'A_err': A_err,
            'alpha': alpha_mean, 'alpha_err': alpha_err,
            'A_ci': A_ci, 'alpha_ci': alpha_ci,
            'r_squared': r_squared, 'n_bins': len(size_data),
            'log_size': log_size, 'log_redshift': log_redshift,
            'z_centers': size_data['z_center'].values,
            'median_mass': size_data['median_mass'].iloc[0],
            'trace': trace
        }
    
    def create_size_evolution_plot(self, results):
        """Create size evolution plot"""
        print("Creating size evolution plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Colors for different samples
        colors = {
            'COWLS': 'red',
            'COWLS Early-type': 'darkred', 
            'COWLS Late-type': 'lightcoral',
            'CWMGs': 'blue',
            'CWMGs Early-type': 'darkblue',
            'CWMGs Late-type': 'lightblue'
        }
        
        # Plot data points and fits
        for name, result in results.items():
            if result is None:
                continue
                
            color = colors[name]
            
            # Plot data points
            ax.scatter(result['log_redshift'], result['log_size'], 
                      c=color, alpha=0.8, s=60, label=f'{name} data', zorder=3)
            
            # Plot fit line
            z_range = np.linspace(result['log_redshift'].min(), 
                                 result['log_redshift'].max(), 100)
            size_pred = result['A'] - result['alpha'] * z_range
            
            ax.plot(z_range, size_pred, color=color, linewidth=2.5,
                   label=f'{name}: α={result["alpha"]:.3f}±{result["alpha_err"]:.3f}', zorder=4)
            
            # Add confidence band
            A_samples = result['trace'].posterior['A'].values.flatten()
            alpha_samples = result['trace'].posterior['alpha'].values.flatten()
            
            size_samples = []
            for z in z_range:
                size_pred_samples = A_samples - alpha_samples * z
                size_samples.append(size_pred_samples)
            
            size_samples = np.array(size_samples).T
            size_ci_lower = np.percentile(size_samples, 2.5, axis=0)
            size_ci_upper = np.percentile(size_samples, 97.5, axis=0)
            
            ax.fill_between(z_range, size_ci_lower, size_ci_upper, 
                           alpha=0.2, color=color, zorder=1)
        
        # Formatting
        ax.set_xlabel('log₁₀(1 + z)', fontsize=14)
        ax.set_ylabel('log₁₀(Reff/kpc)', fontsize=14)
        ax.grid(False)
        ax.legend(fontsize=10, loc='upper right')
        
        # Add text box with median masses
        text_lines = ['Median Stellar Masses:']
        for name, result in results.items():
            if result is not None:
                text_lines.append(f'{name}: log₁₀(M*/M☉) = {result["median_mass"]:.3f}')
        
        text_str = '\n'.join(text_lines)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('size_evolution_from_existing_fits.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("Size evolution plot saved as 'size_evolution_from_existing_fits.png'")
    
    def save_results(self, results):
        """Save fit results to CSV"""
        print("Saving size evolution results...")
        
        # Prepare results for CSV
        csv_data = []
        for name, result in results.items():
            if result is not None:
                csv_data.append({
                    'Sample': name,
                    'Median_Mass_log': result['median_mass'],
                    'A': result['A'],
                    'A_err': result['A_err'],
                    'A_ci_lower': result['A_ci'][0],
                    'A_ci_upper': result['A_ci'][1],
                    'alpha': result['alpha'],
                    'alpha_err': result['alpha_err'],
                    'alpha_ci_lower': result['alpha_ci'][0],
                    'alpha_ci_upper': result['alpha_ci'][1],
                    'r_squared': result['r_squared'],
                    'n_redshift_bins': result['n_bins']
                })
        
        df_results = pd.DataFrame(csv_data)
        df_results.to_csv('size_evolution_from_existing_fits_results.csv', index=False)
        print("Size evolution results saved as 'size_evolution_from_existing_fits_results.csv'")
    
    def run_analysis(self):
        """Run the complete size evolution analysis using existing fits"""
        print("=== Size Evolution Analysis from Existing Fits ===")
        print("Model: log₁₀(Reff/kpc) = A - α log(1 + z) at fixed stellar mass")
        print("Using existing redshift-binned mass-size relations")
        print()
        
        # Load existing fits and median masses
        cowls_fits, massive_fits = self.load_existing_fits()
        median_masses = self.load_median_masses()
        
        print()
        print("=== Extracting Sizes at Fixed Masses ===")
        
        # Extract sizes for each sample
        results = {}
        
        # COWLS samples
        for sample_name in ['COWLS', 'COWLS Early-type', 'COWLS Late-type']:
            median_mass = median_masses[sample_name]
            size_data = self.extract_sizes_at_fixed_mass(cowls_fits, sample_name, median_mass)
            if len(size_data) > 0:
                results[sample_name] = self.fit_size_evolution(size_data, sample_name)
            else:
                results[sample_name] = None
        
        # CWMGs samples
        for sample_name in ['CWMGs', 'CWMGs Early-type', 'CWMGs Late-type']:
            median_mass = median_masses[sample_name]
            size_data = self.extract_sizes_at_fixed_mass(massive_fits, sample_name, median_mass)
            if len(size_data) > 0:
                results[sample_name] = self.fit_size_evolution(size_data, sample_name)
            else:
                results[sample_name] = None
        
        print()
        print("=== Creating Visualizations ===")
        
        # Create plot
        self.create_size_evolution_plot(results)
        
        # Save results
        self.save_results(results)
        
        print()
        print("=== Size Evolution Analysis Complete ===")
        
        # Print summary
        for name, result in results.items():
            if result is not None:
                print(f"{name}: α = {result['alpha']:.3f} ± {result['alpha_err']:.3f}, R² = {result['r_squared']:.3f}")

if __name__ == "__main__":
    analyzer = SizeEvolutionFromExistingFits()
    analyzer.run_analysis()
