#!/usr/bin/env python3
"""
Focused COWLS Mass-Size Relations Analysis
Creates streamlined plots: size-mass best fit with confidence intervals and literature comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import warnings
warnings.filterwarnings('ignore')

class COWLSFocusedAnalyzer:
    def __init__(self):
        # Cosmology from the paper
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
        # Literature relations for comparison (removed Yang et al. SF/QG relations - different sample selection)
        # COWLS lenses are massive galaxies selected for lensing, not by star formation activity
        self.literature_relations = {
            'Shen et al. (2003)': {'alpha': 0.56, 'beta': -5.06, 'color': 'blue', 'linestyle': '-.'},
            'van der Wel et al. (2014)': {'alpha': 0.75, 'beta': -6.25, 'color': 'green', 'linestyle': ':'}
        }
        
    def load_and_process_data(self):
        """Load and process COWLS lens data"""
        print("Loading COWLS lens structural properties...")
        
        # Load COWLS lens structural properties
        cowls_lenses = pd.read_csv('/Users/gozalig1/Projects/jwst-mock-lens-simulator/data/cosmos_web_lens_structural_properties.csv')
        print(f"COWLS lenses: {len(cowls_lenses)} lenses")
        
        # Clean data
        df_clean = cowls_lenses[cowls_lenses['LP_warn_fl'] == 0].copy()
        
        # Remove unrealistic values
        unrealistic_values = [-99, -999, 99, 999]
        for col in ['LP_zfinal', 'LP_mass_med_PDF', 'rearc_f277w', 'mag_f277w']:
            if col in df_clean.columns:
                df_clean = df_clean[~df_clean[col].isin(unrealistic_values)]
        
        # Remove negative or zero values for physical parameters
        df_clean = df_clean[df_clean['LP_zfinal'] > 0]
        df_clean = df_clean[df_clean['LP_mass_med_PDF'] > 0]
        df_clean = df_clean[df_clean['rearc_f277w'] > 0]
        
        # Focus on lens-appropriate mass range (log M* > 9.5)
        df_clean = df_clean[df_clean['LP_mass_med_PDF'] > 9.5]
        
        def convert_arcsec_to_kpc(re_arcsec, z):
            """Convert effective radius from arcseconds to kiloparsecs at redshift z"""
            d_a = self.cosmo.angular_diameter_distance(z).to('kpc').value
            arcsec_to_rad = np.pi / (180.0 * 3600.0)
            size_kpc = re_arcsec * arcsec_to_rad * d_a
            return size_kpc
        
        def select_restframe_structures(row):
            """Select rest-frame structural parameters based on redshift"""
            z = row.get('LP_zfinal', np.nan)
            
            # Select filter based on redshift (same as your example)
            if z < 0.4:
                band = 'f115w'
            elif z < 1.0:
                band = 'f150w'
            elif z < 3.0:
                band = 'f277w'
            else:
                band = 'f444w'
            
            output = {'best_filter': band}
            
            # Structural parameters
            struct_params = ['rearc', 'nsersic']
            for param in struct_params:
                key = f'{param}_{band}'
                value = row.get(key, np.nan)
                
                if param == 'rearc':
                    output['rearc_arcsec'] = value
                    output['size_kpc'] = convert_arcsec_to_kpc(value, z)
                else:
                    output[f'{param}'] = value
            
            return pd.Series(output)
        
        # Apply rest-frame structural parameter selection
        print("Applying rest-frame structural parameter selection...")
        rest_frame_data = df_clean.apply(select_restframe_structures, axis=1)
        
        # Add rest-frame measurements to dataframe
        df_clean = pd.concat([df_clean, rest_frame_data], axis=1)
        
        # Remove galaxies without valid rest-frame measurements
        df_clean = df_clean.dropna(subset=['size_kpc', 'nsersic'])
        
        # Use the converted physical sizes directly
        df_clean['Reff_kpc'] = df_clean['size_kpc']
        
        # Rename nsersic to nsersic_rest for consistency
        df_clean['nsersic_rest'] = df_clean['nsersic']
        
        # Classify structural types based on Sersic index
        df_clean['structural_type'] = 'Unknown'
        
        # Structural type definitions based on Sersic index
        structural_types = {
            'Elliptical': {'n_min': 2.5, 'n_max': 10.0},
            'Lenticular': {'n_min': 1.5, 'n_max': 2.5},
            'Spiral': {'n_min': 0.5, 'n_max': 1.5},
            'Irregular': {'n_min': 0.0, 'n_max': 0.5}
        }
        
        for stype, params in structural_types.items():
            mask = (df_clean['nsersic_rest'] >= params['n_min']) & (df_clean['nsersic_rest'] < params['n_max'])
            df_clean.loc[mask, 'structural_type'] = stype
        
        print(f"After processing: {len(df_clean)} COWLS lenses")
        return df_clean
    
    def mass_size_relation(self, mass, alpha, beta):
        """Mass-size relation: log10(Reff) = alpha * log10(M*) + beta"""
        # Note: mass is already in log10 scale (LP_mass_med_PDF)
        return alpha * mass + beta
    
    def fit_mass_size_relation(self, df):
        """Fit mass-size relation with confidence intervals"""
        print("Fitting COWLS mass-size relation...")
        
        # Mass is already in log10 scale (LP_mass_med_PDF)
        log_mass = df['LP_mass_med_PDF'].values
        # Convert size to log10 scale
        log_size = np.log10(df['Reff_kpc'].values)
        
        # Fit the relation: log_size = alpha * log_mass + beta
        popt, pcov = curve_fit(
            self.mass_size_relation, 
            log_mass, 
            log_size,
            p0=[0.2, -1.0],
            maxfev=10000
        )
        
        alpha, beta = popt
        alpha_err, beta_err = np.sqrt(np.diag(pcov))
        
        # Calculate R²
        y_pred = self.mass_size_relation(log_mass, alpha, beta)
        y_true = log_size
        r_squared = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        # Calculate confidence intervals
        mass_range = np.linspace(log_mass.min(), log_mass.max(), 100)
        size_pred = self.mass_size_relation(mass_range, alpha, beta)
        
        # Bootstrap for confidence intervals
        n_bootstrap = 1000
        bootstrap_alphas = []
        bootstrap_betas = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(log_mass), size=len(log_mass), replace=True)
            mass_boot = log_mass[indices]
            size_boot = log_size[indices]
            
            try:
                popt_boot, _ = curve_fit(
                    self.mass_size_relation, 
                    mass_boot, 
                    size_boot,
                    p0=[alpha, beta],
                    maxfev=1000
                )
                bootstrap_alphas.append(popt_boot[0])
                bootstrap_betas.append(popt_boot[1])
            except:
                continue
        
        bootstrap_alphas = np.array(bootstrap_alphas)
        bootstrap_betas = np.array(bootstrap_betas)
        
        # Calculate confidence intervals
        alpha_ci = np.percentile(bootstrap_alphas, [2.5, 97.5])
        beta_ci = np.percentile(bootstrap_betas, [2.5, 97.5])
        
        # Calculate confidence band for the fit
        size_ci_lower = []
        size_ci_upper = []
        
        for m in mass_range:
            size_samples = []
            for a, b in zip(bootstrap_alphas, bootstrap_betas):
                size_samples.append(self.mass_size_relation(m, a, b))
            size_samples = np.array(size_samples)
            size_ci_lower.append(np.percentile(size_samples, 2.5))
            size_ci_upper.append(np.percentile(size_samples, 97.5))
        
        results = {
            'alpha': alpha, 'alpha_err': alpha_err, 'alpha_ci': alpha_ci,
            'beta': beta, 'beta_err': beta_err, 'beta_ci': beta_ci,
            'r_squared': r_squared, 'n_lenses': len(df),
            'mass_range': mass_range, 'size_pred': size_pred,
            'size_ci_lower': np.array(size_ci_lower),
            'size_ci_upper': np.array(size_ci_upper)
        }
        
        print(f"COWLS fit: log₁₀(Reff/kpc) = {alpha:.3f} ± {alpha_err:.3f} × log₁₀(M*/M☉) + {beta:.3f} ± {beta_err:.3f}")
        print(f"R² = {r_squared:.3f}, N = {len(df)}")
        print(f"95% CI: α = [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}], β = [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]")
        
        return results
    
    def print_structural_type_statistics(self, df):
        """Print structural type statistics for the paper"""
        print("\n=== COWLS Structural Type Statistics ===")
        type_counts = df['structural_type'].value_counts()
        total_lenses = len(df)
        
        print(f"Total COWLS lenses: {total_lenses}")
        print("\nStructural type distribution:")
        for stype in ['Elliptical', 'Lenticular', 'Spiral', 'Irregular']:
            if stype in type_counts:
                count = type_counts[stype]
                percentage = count / total_lenses * 100
                print(f"  {stype}: {count} ({percentage:.1f}%)")
        
        print(f"\nMass range: log M* = {df['LP_mass_med_PDF'].min():.2f} - {df['LP_mass_med_PDF'].max():.2f}")
        print(f"Redshift range: z = {df['LP_zfinal'].min():.2f} - {df['LP_zfinal'].max():.2f}")
        print(f"Size range: Reff = {df['Reff_kpc'].min():.2f} - {df['Reff_kpc'].max():.2f} kpc")
    
    def load_elliptical_results(self):
        """Load elliptical-specific results from CSV files"""
        try:
            # Load structural type relations
            structural_df = pd.read_csv('cowls_structural_type_relations.csv')
            elliptical_row = structural_df[structural_df['structural_type'] == 'Elliptical'].iloc[0]
            
            # Load size evolution results
            evolution_df = pd.read_csv('cowls_size_evolution.csv')
            
            return {
                'alpha': elliptical_row['alpha'],
                'alpha_err': elliptical_row['alpha_err'],
                'beta': elliptical_row['beta'],
                'beta_err': elliptical_row['beta_err'],
                'r_squared': elliptical_row['r_squared'],
                'n_lenses': elliptical_row['n_lenses'],
                'evolution_results': evolution_df
            }
        except Exception as e:
            print(f"Warning: Could not load elliptical results: {e}")
            return None
    
    def elliptical_size_evolution(self, z, gamma, delta):
        """Elliptical size evolution: log10(Reff) = gamma * log10(1+z) + delta"""
        return gamma * np.log10(1 + z) + delta
    
    def create_focused_plots(self, df, fit_results):
        """Create focused plots: combined COWLS mass-size relation with confidence intervals and literature comparison"""
        print("Creating focused visualizations...")
        
        # Calculate structural type statistics
        type_counts = df['structural_type'].value_counts()
        total_lenses = len(df)
        
        # Load elliptical-specific results
        elliptical_results = self.load_elliptical_results()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        #fig.suptitle('COWLS Lens Mass-Size Relations Analysis', fontsize=16, fontweight='bold')
        
        # Plot all COWLS data points
        ax.scatter(df['LP_mass_med_PDF'], df['Reff_kpc'], 
                   c='gray', alpha=0.6, s=20, label=f'COWLS Lenses (N={total_lenses})', zorder=1)
        
        # Plot COWLS best fit
        mass_range = fit_results['mass_range']
        size_pred = 10**fit_results['size_pred']  # Convert back from log scale
        ax.plot(mass_range, size_pred, 'k-', linewidth=2, 
                label=f'COWLS Best Fit\nα = {fit_results["alpha"]:.3f} ± {fit_results["alpha_err"]:.3f}', zorder=3)
        
        # Plot confidence interval
        size_ci_lower = 10**fit_results['size_ci_lower']
        size_ci_upper = 10**fit_results['size_ci_upper']
        ax.fill_between(mass_range, size_ci_lower, size_ci_upper, 
                        alpha=0.3, color='gray', label='95% Confidence Interval', zorder=2)
        
        # Plot literature relations
        for name, params in self.literature_relations.items():
            lit_size = 10**(params['alpha'] * mass_range + params['beta'])
            ax.plot(mass_range, lit_size, 
                    color=params['color'], linestyle=params['linestyle'], 
                    linewidth=2, label=f'{name}\nα = {params["alpha"]:.2f}', zorder=2)
        
        ax.set_xlabel('log₁₀($M_*$/$M_\odot$)', fontsize=12)
        ax.set_ylabel('Effective Radius (kpc)', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=9, loc='upper right')
        
        # Add text box with sample info including structural type statistics
        type_stats = '\n'.join([f'{stype}: {count} ({count/total_lenses*100:.1f}%)' 
                               for stype, count in type_counts.items()])
        textstr = f'COWLS Sample (N={total_lenses})\nz = {df["LP_zfinal"].min():.2f} - {df["LP_zfinal"].max():.2f}\n\nStructural Types:\n{type_stats}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('cowls_mass_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Focused visualizations saved as 'cowls_mass_size_analysis.png'")
    
    def save_results(self, df, fit_results):
        """Save results to CSV files"""
        print("Saving results...")
        
        # Save processed COWLS catalog
        output_cols = ['LP_zfinal', 'LP_mass_med_PDF', 'Reff_kpc', 'structural_type', 'nsersic_f277w', 'rearc_f277w', 'mag_f277w']
        df[output_cols].to_csv('cowls_processed_catalog.csv', index=False)
        
        # Save fit results
        fit_df = pd.DataFrame([{
            'alpha': fit_results['alpha'],
            'alpha_err': fit_results['alpha_err'],
            'alpha_ci_lower': fit_results['alpha_ci'][0],
            'alpha_ci_upper': fit_results['alpha_ci'][1],
            'beta': fit_results['beta'],
            'beta_err': fit_results['beta_err'],
            'beta_ci_lower': fit_results['beta_ci'][0],
            'beta_ci_upper': fit_results['beta_ci'][1],
            'r_squared': fit_results['r_squared'],
            'n_lenses': fit_results['n_lenses']
        }])
        fit_df.to_csv('cowls_fit_results.csv', index=False)
        
        print("Results saved to CSV files")
    
    def run_focused_analysis(self):
        """Run the focused COWLS analysis"""
        print("=== COWLS Focused Mass-Size Relations Analysis ===")
        
        # Load and process data
        df = self.load_and_process_data()
        
        # Print structural type statistics
        self.print_structural_type_statistics(df)
        
        # Fit mass-size relation
        fit_results = self.fit_mass_size_relation(df)
        
        # Create focused visualizations
        self.create_focused_plots(df, fit_results)
        
        # Save results
        self.save_results(df, fit_results)
        
        print("\n=== Focused Analysis Complete ===")
        print(f"Processed {len(df)} COWLS lenses")
        print(f"Mass range: log M* = {df['LP_mass_med_PDF'].min():.2f} - {df['LP_mass_med_PDF'].max():.2f}")
        print(f"Redshift range: z = {df['LP_zfinal'].min():.2f} - {df['LP_zfinal'].max():.2f}")
        print(f"Size range: Reff = {df['Reff_kpc'].min():.2f} - {df['Reff_kpc'].max():.2f} kpc")

if __name__ == "__main__":
    analyzer = COWLSFocusedAnalyzer()
    analyzer.run_focused_analysis()
