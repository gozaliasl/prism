#!/usr/bin/env python3
"""
Size Evolution Analysis: log₁₀(Reff/kpc) vs log(1+z) at fixed stellar mass

This script models the size evolution of galaxies with redshift using:
log₁₀(R_e/kpc) = A - α log(1 + z)

For each sample (COWLS full, ET, LT; CWMGs full, ET, LT), we use the median 
stellar mass to analyze size evolution at fixed mass.
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

class SizeEvolutionAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
    def load_cowls_data(self):
        """Load and process COWLS lens data"""
        print("Loading COWLS lens structural properties...")
        
        # Load COWLS data
        cowls_file = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/data/cosmos_web_lens_structural_properties.csv'
        df = pd.read_csv(cowls_file)
        
        print(f"COWLS lenses: {len(df)} lenses")
        
        # Apply rest-frame structural parameter selection
        print("Applying rest-frame structural parameter selection...")
        rest_frame_data = df.apply(self.select_restframe_structures, axis=1)
        df_processed = pd.concat([df, rest_frame_data], axis=1)
        
        # Debug: check what columns we have
        print(f"Columns after rest-frame selection: {df_processed.columns.tolist()}")
        print(f"Size_kpc values: {df_processed['size_kpc'].describe()}")
        print(f"Nsersic values: {df_processed['nsersic'].describe()}")
        
        # Remove galaxies without valid rest-frame measurements
        df_processed = df_processed.dropna(subset=['size_kpc', 'nsersic'])
        print(f"After processing: {len(df_processed)} COWLS lenses")
        
        # Additional quality cuts
        df_processed = df_processed[df_processed['size_kpc'] > 0.01]  # Minimum physical size
        df_processed = df_processed[df_processed['size_kpc'] < 100]   # Maximum physical size
        
        # Use the converted physical sizes directly
        df_processed['Reff_kpc'] = df_processed['size_kpc']
        df_processed['nsersic_rest'] = df_processed['nsersic']
        
        return df_processed
    
    def load_massive_galaxies(self):
        """Load and process massive galaxy data"""
        print("Loading massive galaxies with log M* > 9.5...")
        
        # Load massive galaxy catalog
        catalog_file = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/data/galaxy_catalog.csv'
        df = pd.read_csv(catalog_file)
        
        print(f"Total galaxies in catalog: {len(df):,}")
        
        # Apply mass cut
        mass_cut = df['LP_mass_med_PDF'] > 9.5
        df_massive = df[mass_cut].copy()
        print(f"Massive galaxies (log M* > 9.5): {len(df_massive):,}")
        
        # Remove warning flags
        df_massive = df_massive[df_massive['LP_warn_fl'] == 0]
        print(f"After removing warning flags: {len(df_massive):,}")
        
        # Basic cleaning
        df_massive = df_massive[
            (df_massive['LP_zfinal'] > 0.05) & 
            (df_massive['LP_zfinal'] < 12.0) &
            (df_massive['LP_mass_med_PDF'] > 8.0) &
            (df_massive['LP_mass_med_PDF'] < 13.0)
        ]
        print(f"After basic cleaning: {len(df_massive):,}")
        
        # Apply rest-frame structural parameter selection
        print("Applying rest-frame structural parameter selection...")
        rest_frame_data = df_massive.apply(self.select_restframe_structures, axis=1)
        df_processed = pd.concat([df_massive, rest_frame_data], axis=1)
        
        # Remove galaxies without valid rest-frame measurements
        df_processed = df_processed.dropna(subset=['size_kpc', 'nsersic'])
        print(f"After rest-frame selection and quality cuts: {len(df_processed):,}")
        
        # Additional quality cuts
        df_processed = df_processed[df_processed['size_kpc'] > 0.01]  # Minimum physical size
        df_processed = df_processed[df_processed['size_kpc'] < 100]   # Maximum physical size
        
        print(f"After physical size cuts: {len(df_processed):,} massive galaxies")
        
        # Use the converted physical sizes directly
        df_processed['Reff_kpc'] = df_processed['size_kpc']
        df_processed['nsersic_rest'] = df_processed['nsersic']
        
        return df_processed
    
    def select_restframe_structures(self, row):
        """Select rest-frame structural parameters based on redshift"""
        z = row.get('LP_zfinal', np.nan)
        
        # Select filter based on redshift
        if 0.05 < z <= 0.4:
            band = 'f115w'
        elif 0.4 < z <= 1.0:
            band = 'f150w'
        elif 1.0 < z <= 3.0:
            band = 'f277w'
        elif 3.0 < z <= 12.0:  # Extended range using F444W for high-z galaxies
            band = 'f444w'
        else:
            return pd.Series({'best_filter': None, 'rearc_arcsec': np.nan, 'rearc_arcsec_err': np.nan,
                              'size_kpc': np.nan, 'size_kpc_err': np.nan, 'nsersic': np.nan, 'nsersic_err': np.nan,
                              'qratio': np.nan, 'qratio_err': np.nan, 'mag': np.nan, 'mag_err': np.nan})
        
        output = {'best_filter': band}
        
        # Structural parameters and their errors
        rearc_col = f'rearc_{band}'
        rearc_err_col = f'rearc_{band}_err'
        nsersic_col = f'nsersic_{band}'
        nsersic_err_col = f'nsersic_{band}_err'
        qratio_col = f'qratio_{band}'
        qratio_err_col = f'qratio_{band}_err'
        mag_col = f'mag_{band}'
        mag_err_col = f'mag_{band}_err'
        
        # Extract values with error handling
        rearc_arcsec = row.get(rearc_col, np.nan)
        rearc_arcsec_err = row.get(rearc_err_col, np.nan)
        nsersic = row.get(nsersic_col, np.nan)
        nsersic_err = row.get(nsersic_err_col, np.nan)
        qratio = row.get(qratio_col, np.nan)
        qratio_err = row.get(qratio_err_col, np.nan)
        mag = row.get(mag_col, np.nan)
        mag_err = row.get(mag_err_col, np.nan)
        
        # Quality cuts
        if (pd.notna(rearc_arcsec) and pd.notna(nsersic) and pd.notna(mag) and
            rearc_arcsec > 0.01 and rearc_arcsec < 10.0 and
            nsersic > 0.1 and nsersic < 8.0 and
            mag > 15 and mag < 30):
            
            # Convert angular size to physical size
            da = self.cosmo.angular_diameter_distance(z)
            size_kpc = rearc_arcsec * (da.to(u.kpc) / u.arcsec).value
            size_kpc_err = rearc_arcsec_err * (da.to(u.kpc) / u.arcsec).value
            
            output.update({
                'rearc_arcsec': rearc_arcsec,
                'rearc_arcsec_err': rearc_arcsec_err,
                'size_kpc': size_kpc,
                'size_kpc_err': size_kpc_err,
                'nsersic': nsersic,
                'nsersic_err': nsersic_err,
                'qratio': qratio,
                'qratio_err': qratio_err,
                'mag': mag,
                'mag_err': mag_err
            })
        else:
            output.update({
                'rearc_arcsec': np.nan, 'rearc_arcsec_err': np.nan,
                'size_kpc': np.nan, 'size_kpc_err': np.nan,
                'nsersic': np.nan, 'nsersic_err': np.nan,
                'qratio': np.nan, 'qratio_err': np.nan,
                'mag': np.nan, 'mag_err': np.nan
            })
        
        return pd.Series(output)
    
    
    def separate_by_sersic(self, df, name):
        """Separate galaxies by Sersic index"""
        early_mask = df['nsersic_rest'] <= 2.5
        late_mask = df['nsersic_rest'] > 2.5
        
        early_df = df[early_mask].copy()
        late_df = df[late_mask].copy()
        
        print(f"Separating {name} by Sersic index...")
        print(f"  Early-type (n ≤ 2.5): {len(early_df)} galaxies")
        print(f"  Late-type (n > 2.5): {len(late_df)} galaxies")
        
        return early_df, late_df
    
    def calculate_median_mass(self, df, name):
        """Calculate median stellar mass for a sample"""
        median_mass = df['LP_mass_med_PDF'].median()
        print(f"{name} median stellar mass: log₁₀(M*/M☉) = {median_mass:.3f}")
        return median_mass
    
    def fit_size_evolution(self, df, name, median_mass):
        """Fit size evolution: log₁₀(Reff/kpc) = A - α log(1 + z)"""
        print(f"Fitting size evolution for {name}...")
        
        # Filter galaxies near the median mass (±0.2 dex)
        mass_mask = np.abs(df['LP_mass_med_PDF'] - median_mass) <= 0.2
        df_fit = df[mass_mask].copy()
        
        if len(df_fit) < 10:
            print(f"  Warning: Only {len(df_fit)} galaxies near median mass, using all galaxies")
            df_fit = df.copy()
        
        print(f"  Using {len(df_fit)} galaxies for size evolution fit")
        
        # Prepare data
        log_size = np.log10(df_fit['Reff_kpc'].values)
        log_redshift = np.log10(1 + df_fit['LP_zfinal'].values)
        
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
        
        print(f"  {name} size evolution: log₁₀(Reff/kpc) = {A_mean:.3f} ± {A_err:.3f} - {alpha_mean:.3f} ± {alpha_err:.3f} × log(1+z)")
        print(f"  R² = {r_squared:.3f}, N = {len(df_fit)}")
        print(f"  95% CI: A = [{A_ci[0]:.3f}, {A_ci[1]:.3f}], α = [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]")
        
        return {
            'A': A_mean, 'A_err': A_err,
            'alpha': alpha_mean, 'alpha_err': alpha_err,
            'A_ci': A_ci, 'alpha_ci': alpha_ci,
            'r_squared': r_squared, 'n_galaxies': len(df_fit),
            'log_size': log_size, 'log_redshift': log_redshift,
            'median_mass': median_mass,
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
            color = colors[name]
            
            # Skip if no data
            if result['n_galaxies'] == 0:
                print(f"Skipping {name} - no galaxies")
                continue
            
            # Plot data points
            ax.scatter(result['log_redshift'], result['log_size'], 
                      c=color, alpha=0.6, s=20, label=f'{name} data', zorder=1)
            
            # Plot fit line
            z_range = np.linspace(result['log_redshift'].min(), 
                                 result['log_redshift'].max(), 100)
            size_pred = result['A'] - result['alpha'] * z_range
            
            ax.plot(z_range, size_pred, color=color, linewidth=2.5,
                   label=f'{name}: α={result["alpha"]:.3f}±{result["alpha_err"]:.3f}', zorder=3)
            
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
            if result['n_galaxies'] > 0:
                text_lines.append(f'{name}: log₁₀(M*/M☉) = {result["median_mass"]:.3f}')
        
        text_str = '\n'.join(text_lines)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('size_evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("Size evolution plot saved as 'size_evolution_analysis.png'")
    
    def save_results(self, results):
        """Save fit results to CSV"""
        print("Saving size evolution results...")
        
        # Prepare results for CSV
        csv_data = []
        for name, result in results.items():
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
                'n_galaxies': result['n_galaxies']
            })
        
        df_results = pd.DataFrame(csv_data)
        df_results.to_csv('size_evolution_results.csv', index=False)
        print("Size evolution results saved as 'size_evolution_results.csv'")
    
    def run_analysis(self):
        """Run the complete size evolution analysis"""
        print("=== Size Evolution Analysis ===")
        print("Model: log₁₀(Reff/kpc) = A - α log(1 + z) at fixed stellar mass")
        print()
        
        # Load data
        cowls_data = self.load_cowls_data()
        massive_data = self.load_massive_galaxies()
        
        # Separate by morphology
        cowls_early, cowls_late = self.separate_by_sersic(cowls_data, "COWLS")
        massive_early, massive_late = self.separate_by_sersic(massive_data, "CWMGs")
        
        print()
        print("=== Calculating Median Stellar Masses ===")
        
        # Calculate median masses
        cowls_median = self.calculate_median_mass(cowls_data, "COWLS")
        cowls_early_median = self.calculate_median_mass(cowls_early, "COWLS Early-type")
        cowls_late_median = self.calculate_median_mass(cowls_late, "COWLS Late-type")
        massive_median = self.calculate_median_mass(massive_data, "CWMGs")
        massive_early_median = self.calculate_median_mass(massive_early, "CWMGs Early-type")
        massive_late_median = self.calculate_median_mass(massive_late, "CWMGs Late-type")
        
        print()
        print("=== Fitting Size Evolution ===")
        
        # Fit size evolution for each sample
        results = {}
        results['COWLS'] = self.fit_size_evolution(cowls_data, "COWLS", cowls_median)
        results['COWLS Early-type'] = self.fit_size_evolution(cowls_early, "COWLS Early-type", cowls_early_median)
        results['COWLS Late-type'] = self.fit_size_evolution(cowls_late, "COWLS Late-type", cowls_late_median)
        results['CWMGs'] = self.fit_size_evolution(massive_data, "CWMGs", massive_median)
        results['CWMGs Early-type'] = self.fit_size_evolution(massive_early, "CWMGs Early-type", massive_early_median)
        results['CWMGs Late-type'] = self.fit_size_evolution(massive_late, "CWMGs Late-type", massive_late_median)
        
        print()
        print("=== Creating Visualizations ===")
        
        # Create plot
        self.create_size_evolution_plot(results)
        
        # Save results
        self.save_results(results)
        
        print()
        print("=== Size Evolution Analysis Complete ===")
        print(f"COWLS: {len(cowls_data)} galaxies")
        print(f"CWMGs: {len(massive_data)} galaxies")

if __name__ == "__main__":
    analyzer = SizeEvolutionAnalyzer()
    analyzer.run_analysis()
