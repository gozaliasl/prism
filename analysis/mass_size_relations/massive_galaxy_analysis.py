#!/usr/bin/env python3
"""
Massive Galaxy Size-Mass Relations Analysis
Using galaxy catalog with M* > 9.5, analyzing in redshift bins without SF/passive classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import warnings
warnings.filterwarnings('ignore')

class MassiveGalaxyAnalyzer:
    def __init__(self):
        # Cosmology from the paper
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
    def convert_arcsec_to_kpc(self, re_arcsec, z):
        """
        Convert effective radius (Re) from arcseconds to kiloparsecs at redshift z.
        
        Parameters:
        - re_arcsec: angular size in arcseconds (float or array)
        - z: redshift (float or array of same shape)
        
        Returns:
        - size_kpc: size in kiloparsecs
        """
        # Angular diameter distance in kpc
        d_a = self.cosmo.angular_diameter_distance(z).to('kpc').value  # scalar or array
        # Convert arcsec to radians: 1 arcsec = (1/3600)*(pi/180) radians
        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        
        # Linear size in kpc = angle [radians] × D_A [kpc]
        size_kpc = re_arcsec * arcsec_to_rad * d_a
        return size_kpc
    
    def select_restframe_structures(self, row):
        """Select rest-frame structural parameters based on redshift, following the paper's methodology"""
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
        
        # Structural parameters and their errors
        struct_params = ['rearc', 'nsersic', 'qratio', 'mag']
        for param in struct_params:
            key = f'{param}_{band}'
            err_key = f'{param}_{band}_err'
            
            value = row.get(key, np.nan)
            error = row.get(err_key, np.nan)
            
            if param == 'rearc':
                output['rearc_arcsec'] = value
                output['rearc_arcsec_err'] = error
                output['size_kpc'] = self.convert_arcsec_to_kpc(value, z)
                output['size_kpc_err'] = self.convert_arcsec_to_kpc(error, z) if not np.isnan(error) else np.nan
            else:
                output[f'{param}'] = value
                output[f'{param}_err'] = error
        
        return pd.Series(output)
    
    def load_massive_galaxies(self, mass_cut=9.5):
        """Load massive galaxies from galaxy catalog with proper rest-frame filter selection"""
        print(f"Loading massive galaxies with log M* > {mass_cut}...")
        
        # Load galaxy catalog
        df = pd.read_csv('/Users/gozalig1/Projects/jwst-mock-lens-simulator/data/galaxy_catalog.csv')
        print(f"Total galaxies in catalog: {len(df)}")
        
        # Apply mass cut
        df_massive = df[df['LP_mass_med_PDF'] > mass_cut].copy()
        print(f"Massive galaxies (log M* > {mass_cut}): {len(df_massive)}")
        
        # Clean data - remove warning flags
        df_clean = df_massive[df_massive['LP_warn_fl'] == 0].copy()
        print(f"After removing warning flags: {len(df_clean)}")
        
        # Remove unrealistic values
        unrealistic_values = [-99, -999, 99, 999]
        for col in ['LP_zfinal', 'LP_mass_med_PDF']:
            if col in df_clean.columns:
                df_clean = df_clean[~df_clean[col].isin(unrealistic_values)]
        
        # Remove negative or zero values for physical parameters
        df_clean = df_clean[df_clean['LP_zfinal'] > 0]
        df_clean = df_clean[df_clean['LP_mass_med_PDF'] > 0]
        
        # Apply redshift range from paper (0.05 < z ≤ 4.0)
        df_clean = df_clean[df_clean['LP_zfinal'] > 0.05]
        df_clean = df_clean[df_clean['LP_zfinal'] <= 4.0]
        
        print(f"After basic cleaning: {len(df_clean)}")
        
        # Apply rest-frame structural parameter selection (following your example)
        print("Applying rest-frame structural parameter selection...")
        rest_frame_data = df_clean.apply(self.select_restframe_structures, axis=1)
        
        # Add rest-frame measurements to dataframe
        df_clean = pd.concat([df_clean, rest_frame_data], axis=1)
        
        # Remove galaxies without valid rest-frame measurements
        df_clean = df_clean.dropna(subset=['size_kpc', 'nsersic'])
        print(f"After rest-frame selection and quality cuts: {len(df_clean)}")
        
        # Use the converted physical sizes directly
        df_clean['Reff_kpc'] = df_clean['size_kpc']
        
        # Additional quality cuts for reasonable physical sizes
        df_clean = df_clean[df_clean['Reff_kpc'] > 0.1]  # Minimum reasonable size
        df_clean = df_clean[df_clean['Reff_kpc'] < 20]   # Maximum reasonable size
        
        print(f"After physical size cuts: {len(df_clean)} massive galaxies")
        
        # Print filter usage statistics
        filter_counts = df_clean['best_filter'].value_counts()
        print("Rest-frame filter usage:")
        for filter_name, count in filter_counts.items():
            print(f"  {filter_name}: {count} galaxies")
        
        # Print size statistics
        print(f"Size range: {df_clean['Reff_kpc'].min():.2f} - {df_clean['Reff_kpc'].max():.2f} kpc")
        print(f"Redshift range: z = {df_clean['LP_zfinal'].min():.2f} - {df_clean['LP_zfinal'].max():.2f}")
        
        return df_clean
    
    def mass_size_relation(self, mass, alpha, beta):
        """Mass-size relation: log10(Reff) = alpha * log10(M*) + beta"""
        # Note: mass is already in log10 scale (LP_mass_med_PDF)
        return alpha * mass + beta
    
    def define_redshift_bins(self):
        """Define redshift bins for analysis"""
        return [
            (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), 
            (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)
        ]
    
    def fit_mass_size_relation(self, df):
        """Fit mass-size relation for the full sample"""
        print("Fitting massive galaxy mass-size relation...")
        
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
            'r_squared': r_squared, 'n_galaxies': len(df),
            'mass_range': mass_range, 'size_pred': size_pred,
            'size_ci_lower': np.array(size_ci_lower),
            'size_ci_upper': np.array(size_ci_upper)
        }
        
        print(f"Massive galaxies fit: log₁₀(Reff/kpc) = {alpha:.3f} ± {alpha_err:.3f} × log₁₀(M*/M☉) + {beta:.3f} ± {beta_err:.3f}")
        print(f"R² = {r_squared:.3f}, N = {len(df)}")
        print(f"95% CI: α = [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}], β = [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]")
        
        return results
    
    def fit_redshift_binned_relations(self, df):
        """Fit mass-size relations in redshift bins"""
        print("\n=== Fitting Redshift-Binned Mass-Size Relations ===")
        
        redshift_bins = self.define_redshift_bins()
        results = []
        
        for z_min, z_max in redshift_bins:
            subset = df[(df['LP_zfinal'] >= z_min) & (df['LP_zfinal'] < z_max)]
            
            if len(subset) < 20:  # Need sufficient data points
                print(f"z{z_min}-{z_max}: Insufficient data ({len(subset)} galaxies)")
                continue
            
            try:
                # Mass is already in log10 scale
                log_mass = subset['LP_mass_med_PDF'].values
                # Convert size to log10 scale
                log_size = np.log10(subset['Reff_kpc'].values)
                z = subset['LP_zfinal'].values
                
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
                
                results.append({
                    'z_min': z_min, 'z_max': z_max, 'z_center': (z_min + z_max) / 2,
                    'alpha': alpha, 'alpha_err': alpha_err,
                    'beta': beta, 'beta_err': beta_err,
                    'r_squared': r_squared, 'n_galaxies': len(subset)
                })
                
                z_center = (z_min + z_max) / 2
                print(f"z{z_min}-{z_max} (z={z_center:.2f}):")
                print(f"  log₁₀(Reff/kpc) = {alpha:.3f} × log₁₀(M*/M☉) + {beta:.3f}")
                print(f"  R² = {r_squared:.3f}, N = {len(subset)}")
                
            except Exception as e:
                print(f"z{z_min}-{z_max}: Fitting failed - {e}")
        
        return pd.DataFrame(results)
    
    def create_visualizations(self, df, fit_results, redshift_results):
        """Create visualizations"""
        print("Creating visualizations...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle('Massive Galaxy Size-Mass Relations by Redshift', fontsize=16, fontweight='bold')
        
        # Define colors for different redshift bins
        colors = plt.cm.viridis(np.linspace(0, 1, len(redshift_results)))
        
        # Plot data points colored by redshift
        scatter = ax.scatter(df['LP_mass_med_PDF'], df['Reff_kpc'], 
                           c=df['LP_zfinal'], cmap='viridis', alpha=0.6, s=20, 
                           label=f'All Massive Galaxies (N={len(df)})', zorder=1)
        
        # Plot redshift-binned relations
        if not redshift_results.empty:
            for i, (_, row) in enumerate(redshift_results.iterrows()):
                z_min, z_max = row['z_min'], row['z_max']
                z_center = row['z_center']
                alpha, beta = row['alpha'], row['beta']
                
                # Create mass range for this redshift bin
                subset = df[(df['LP_zfinal'] >= z_min) & (df['LP_zfinal'] < z_max)]
                if len(subset) > 0:
                    mass_range = np.linspace(subset['LP_mass_med_PDF'].min(), 
                                           subset['LP_mass_med_PDF'].max(), 100)
                    size_pred = 10**(alpha * mass_range + beta)
                    
                    ax.plot(mass_range, size_pred, color=colors[i], linewidth=2,
                           label=f'z={z_min}-{z_max} (α={alpha:.3f})', zorder=3)
        
        # Plot overall best fit
        mass_range = fit_results['mass_range']
        size_pred = 10**fit_results['size_pred']
        ax.plot(mass_range, size_pred, 'k--', linewidth=3, 
               label=f'Overall Fit (α={fit_results["alpha"]:.3f})', zorder=4)
        
        ax.set_xlabel('log₁₀($M_*$/$M_\odot$)', fontsize=12)
        ax.set_ylabel('Effective Radius (kpc)', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=9, loc='upper left', ncol=2)
        
        # Add colorbar for redshift
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Redshift (z)', rotation=270, labelpad=15)
        
        # Add text box with sample info
        textstr = f'Massive Galaxies\nN = {len(df)}\nz = {df["LP_zfinal"].min():.2f} - {df["LP_zfinal"].max():.2f}\nR² = {fit_results["r_squared"]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('massive_galaxy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'massive_galaxy_analysis.png'")
    
    def save_results(self, df, fit_results, redshift_results):
        """Save results to CSV files"""
        print("Saving results...")
        
        # Save processed catalog
        output_cols = ['LP_zfinal', 'LP_mass_med_PDF', 'Reff_kpc', 'nsersic', 'rearc_arcsec', 'mag', 'best_filter']
        df[output_cols].to_csv('massive_galaxy_catalog.csv', index=False)
        
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
            'n_galaxies': fit_results['n_galaxies']
        }])
        fit_df.to_csv('massive_galaxy_fit_results.csv', index=False)
        
        # Save redshift binned results
        if not redshift_results.empty:
            redshift_results.to_csv('massive_galaxy_redshift_binned_relations.csv', index=False)
        
        print("Results saved to CSV files")
    
    def run_analysis(self, mass_cut=9.5):
        """Run the complete massive galaxy analysis"""
        print("=== Massive Galaxy Size-Mass Relations Analysis ===")
        print(f"Using galaxies with log M* > {mass_cut}")
        
        # Load and process data
        df = self.load_massive_galaxies(mass_cut)
        
        # Print sample statistics
        print(f"\nSample Statistics:")
        print(f"Total galaxies: {len(df)}")
        print(f"Mass range: log M* = {df['LP_mass_med_PDF'].min():.2f} - {df['LP_mass_med_PDF'].max():.2f}")
        print(f"Redshift range: z = {df['LP_zfinal'].min():.2f} - {df['LP_zfinal'].max():.2f}")
        print(f"Size range: Reff = {df['Reff_kpc'].min():.2f} - {df['Reff_kpc'].max():.2f} kpc")
        
        # Fit relations
        fit_results = self.fit_mass_size_relation(df)
        redshift_results = self.fit_redshift_binned_relations(df)
        
        # Create visualizations
        self.create_visualizations(df, fit_results, redshift_results)
        
        # Save results
        self.save_results(df, fit_results, redshift_results)
        
        print("\n=== Analysis Complete ===")
        return df, fit_results, redshift_results

if __name__ == "__main__":
    analyzer = MassiveGalaxyAnalyzer()
    
    # Run analysis with different mass cuts
    print("Running analysis with log M* > 9.5...")
    df_95, fit_95, redshift_95 = analyzer.run_analysis(mass_cut=9.5)
    
    print("\n" + "="*50)
    print("Running analysis with log M* > 10.0...")
    df_10, fit_10, redshift_10 = analyzer.run_analysis(mass_cut=10.0)
