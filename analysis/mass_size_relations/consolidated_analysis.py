#!/usr/bin/env python3
"""
Consolidated Mass-Size Relations Analysis
Two focused plots: 1) Mass-size relations with all samples, 2) Redshift evolution of slopes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
import os
import pymc as pm
import arviz as az

class ConsolidatedAnalyzer:
    def __init__(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
    def convert_arcsec_to_kpc(self, arcsec, redshift):
        """Convert angular size to physical size in kpc"""
        if np.isnan(arcsec) or np.isnan(redshift) or redshift <= 0:
            return np.nan
        
        # Angular diameter distance
        da = self.cosmo.angular_diameter_distance(redshift).value  # Mpc
        
        # Convert arcsec to kpc
        kpc_per_arcsec = da * 1000 * np.pi / (180 * 3600)  # kpc/arcsec
        return arcsec * kpc_per_arcsec
    
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
    
    def load_cowls_data(self):
        """Load and process COWLS data"""
        print("Loading COWLS lens structural properties...")
        
        cowls_file = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/data/cosmos_web_lens_structural_properties.csv'
        df = pd.read_csv(cowls_file)
        print(f"COWLS lenses: {len(df)} lenses")
        
        # Basic cleaning
        df_clean = df.copy()
        
        # Remove warning flags
        if 'LP_warn_fl' in df_clean.columns:
            df_clean = df_clean[df_clean['LP_warn_fl'] == 0]
        
        # Remove unrealistic values
        df_clean = df_clean[df_clean['LP_zfinal'] > 0]
        df_clean = df_clean[df_clean['LP_zfinal'] < 5]
        df_clean = df_clean[df_clean['LP_mass_med_PDF'] > 8]
        df_clean = df_clean[df_clean['LP_mass_med_PDF'] < 13]
        
        # Apply rest-frame structural parameter selection
        print("Applying rest-frame structural parameter selection...")
        rest_frame_data = df_clean.apply(self.select_restframe_structures, axis=1)
        df_clean = pd.concat([df_clean, rest_frame_data], axis=1)
        
        # Remove galaxies without valid rest-frame measurements
        df_clean = df_clean.dropna(subset=['size_kpc', 'nsersic'])
        
        # Use the converted physical sizes directly
        df_clean['Reff_kpc'] = df_clean['size_kpc']
        
        # Rename nsersic to nsersic_rest for consistency
        df_clean['nsersic_rest'] = df_clean['nsersic']
        
        print(f"After processing: {len(df_clean)} COWLS lenses")
        return df_clean
    
    def load_massive_galaxies(self, mass_cut=9.0):
        """Load and process massive galaxy data"""
        print(f"Loading massive galaxies with log M* > {mass_cut}...")
        
        catalog_file = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/data/galaxy_catalog.csv'
        df = pd.read_csv(catalog_file)
        print(f"Total galaxies in catalog: {len(df)}")
        
        # Apply mass cut
        df = df[df['LP_mass_med_PDF'] > mass_cut]
        print(f"Massive galaxies (log M* > {mass_cut}): {len(df)}")
        
        # Remove warning flags
        if 'LP_warn_fl' in df.columns:
            df = df[df['LP_warn_fl'] == 0]
        print(f"After removing warning flags: {len(df)}")
        
        # Basic cleaning (extended redshift range)
        df = df[df['LP_zfinal'] > 0]
        df = df[df['LP_zfinal'] < 12]  # Extended to z=12
        df = df[df['LP_mass_med_PDF'] > 8]
        df = df[df['LP_mass_med_PDF'] < 13]
        
        # Remove unrealistic structural measurements
        for band in ['f115w', 'f150w', 'f277w', 'f444w']:
            rearc_col = f'rearc_{band}'
            if rearc_col in df.columns:
                df = df[df[rearc_col] > 0]
                df = df[df[rearc_col] < 3]  # Remove very large angular sizes
        
        print(f"After basic cleaning: {len(df)}")
        
        # Apply rest-frame structural parameter selection
        print("Applying rest-frame structural parameter selection...")
        rest_frame_data = df.apply(self.select_restframe_structures, axis=1)
        df = pd.concat([df, rest_frame_data], axis=1)
        
        # Remove galaxies without valid rest-frame measurements
        df = df.dropna(subset=['size_kpc', 'nsersic'])
        print(f"After rest-frame selection and quality cuts: {len(df)}")
        
        # Additional quality cuts
        df = df[df['size_kpc'] > 0.1]  # Minimum physical size
        df = df[df['size_kpc'] < 20]   # Maximum physical size
        
        print(f"After physical size cuts: {len(df)} massive galaxies")
        
        # Use the converted physical sizes directly
        df['Reff_kpc'] = df['size_kpc']
        
        # Rename nsersic to nsersic_rest for consistency
        df['nsersic_rest'] = df['nsersic']
        
        return df
    
    def separate_by_sersic(self, df, name):
        """Separate galaxies into early-type (n ≤ 2.5) and late-type (n > 2.5)"""
        print(f"\nSeparating {name} by Sersic index...")
        
        # Early-type: Sersic ≤ 2.5
        early_type = df[df['nsersic_rest'] <= 2.5].copy()
        early_type['galaxy_type'] = 'Early-type'
        
        # Late-type: Sersic > 2.5
        late_type = df[df['nsersic_rest'] > 2.5].copy()
        late_type['galaxy_type'] = 'Late-type'
        
        print(f"  Early-type (n ≤ 2.5): {len(early_type)} galaxies")
        print(f"  Late-type (n > 2.5): {len(late_type)} galaxies")
        
        return early_type, late_type
    
    def fit_mass_size_relation(self, df, name):
        """Fit mass-size relation using PyMC"""
        print(f"Fitting {name} mass-size relation with PyMC...")
        
        # Mass is already in log10 scale (LP_mass_med_PDF)
        log_mass = df['LP_mass_med_PDF'].values
        # Convert size to log10 scale
        log_size = np.log10(df['Reff_kpc'].values)
        
        # Reference mass (similar to your notebook)
        mass_ref = 5e10  # solar masses
        log_mass_ref = np.log10(mass_ref)
        
        # PyMC model
        with pm.Model() as model:
            # Priors
            logA = pm.Normal("logA", mu=0.5, sigma=1.0)
            alpha = pm.Normal("alpha", mu=0.3, sigma=0.5)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            
            # Model: log_size = logA + alpha * (log_mass - log_mass_ref)
            mu = logA + alpha * (log_mass - log_mass_ref)
            
            # Likelihood
            pm.Normal("log_size_obs", mu=mu, sigma=sigma, observed=log_size)
            
            # Sample
            trace = pm.sample(
                draws=2000,
                tune=500,
                target_accept=0.95,
                chains=4,
                cores=2,
                progressbar=False
            )
        
        # Extract results
        alpha_samples = trace.posterior['alpha'].values.flatten()
        logA_samples = trace.posterior['logA'].values.flatten()
        sigma_samples = trace.posterior['sigma'].values.flatten()
        
        # Convert logA to beta (intercept)
        beta_samples = logA_samples - alpha_samples * log_mass_ref
        
        # Calculate statistics
        alpha_mean = np.mean(alpha_samples)
        alpha_err = np.std(alpha_samples)
        beta_mean = np.mean(beta_samples)
        beta_err = np.std(beta_samples)
        
        # Calculate R²
        y_pred = alpha_mean * log_mass + beta_mean
        y_true = log_size
        r_squared = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        # Calculate confidence intervals
        alpha_ci = np.percentile(alpha_samples, [2.5, 97.5])
        beta_ci = np.percentile(beta_samples, [2.5, 97.5])
        
        # Calculate confidence band for the fit
        mass_range = np.linspace(log_mass.min(), log_mass.max(), 100)
        size_samples = []
        
        for m in mass_range:
            size_pred_samples = logA_samples + alpha_samples * (m - log_mass_ref)
            size_samples.append(size_pred_samples)
        
        size_samples = np.array(size_samples).T
        size_ci_lower = np.percentile(size_samples, 2.5, axis=0)
        size_ci_upper = np.percentile(size_samples, 97.5, axis=0)
        
        print(f"{name} fit: log₁₀(Reff/kpc) = {alpha_mean:.3f} ± {alpha_err:.3f} × log₁₀(M*/M☉) + {beta_mean:.3f} ± {beta_err:.3f}")
        print(f"R² = {r_squared:.3f}, N = {len(df)}")
        print(f"95% CI: α = [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}], β = [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]")
        
        return {
            'alpha': alpha_mean, 'alpha_err': alpha_err,
            'beta': beta_mean, 'beta_err': beta_err,
            'alpha_ci': alpha_ci, 'beta_ci': beta_ci,
            'r_squared': r_squared, 'n_galaxies': len(df),
            'log_mass': log_mass, 'log_size': log_size,
            'mass_range': (log_mass.min(), log_mass.max()),
            'mass_range_plot': mass_range,
            'size_ci_lower': size_ci_lower,
            'size_ci_upper': size_ci_upper,
            'trace': trace
        }
    
    def fit_redshift_binned_relations(self, df, name, min_samples=10):
        """Fit mass-size relations in redshift bins using PyMC"""
        print(f"Fitting redshift-binned relations for {name}...")
        
        # Define redshift bins (fewer bins for COWLS due to smaller sample)
        if 'COWLS' in name:
            z_bins = [(0.2, 0.8), (0.8, 1.5), (1.5, 4.0)]
        else:
            z_bins = [(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), 
                      (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0)]
        
        results = []
        
        for z_min, z_max in z_bins:
            subset = df[(df['LP_zfinal'] >= z_min) & (df['LP_zfinal'] < z_max)]
            
            if len(subset) < min_samples:  # Need sufficient data points
                print(f"  z{z_min}-{z_max}: Insufficient data ({len(subset)} galaxies)")
                continue
            
            try:
                # Mass is already in log10 scale
                log_mass = subset['LP_mass_med_PDF'].values
                # Convert size to log10 scale
                log_size = np.log10(subset['Reff_kpc'].values)
                
                # Reference mass
                mass_ref = 5e10
                log_mass_ref = np.log10(mass_ref)
                
                # PyMC model
                with pm.Model() as model:
                    logA = pm.Normal("logA", mu=0.5, sigma=1.0)
                    alpha = pm.Normal("alpha", mu=0.3, sigma=0.5)
                    sigma = pm.HalfNormal("sigma", sigma=0.5)
                    
                    mu = logA + alpha * (log_mass - log_mass_ref)
                    pm.Normal("log_size_obs", mu=mu, sigma=sigma, observed=log_size)
                    
                    trace = pm.sample(
                        draws=1000, tune=250, target_accept=0.95,
                        chains=2, cores=1, progressbar=False
                    )
                
                # Extract results
                alpha_samples = trace.posterior['alpha'].values.flatten()
                logA_samples = trace.posterior['logA'].values.flatten()
                beta_samples = logA_samples - alpha_samples * log_mass_ref
                
                alpha_mean = np.mean(alpha_samples)
                alpha_err = np.std(alpha_samples)
                beta_mean = np.mean(beta_samples)
                beta_err = np.std(beta_samples)
                
                # Calculate R²
                y_pred = alpha_mean * log_mass + beta_mean
                y_true = log_size
                r_squared = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
                
                results.append({
                    'z_min': z_min, 'z_max': z_max, 'z_center': (z_min + z_max) / 2,
                    'alpha': alpha_mean, 'alpha_err': alpha_err,
                    'beta': beta_mean, 'beta_err': beta_err,
                    'r_squared': r_squared, 'n_galaxies': len(subset)
                })
                
                print(f"  z{z_min}-{z_max}: α={alpha_mean:.3f}±{alpha_err:.3f}, β={beta_mean:.3f}±{beta_err:.3f}, R²={r_squared:.3f}, N={len(subset)}")
                
            except Exception as e:
                print(f"  z{z_min}-{z_max}: Fit failed - {e}")
                continue
        
        return results
    
    def mass_size_relation(self, mass, alpha, beta):
        """Mass-size relation: log10(Reff) = alpha * log10(M*) + beta"""
        return alpha * mass + beta
    
    def create_consolidated_mass_size_plot(self, cowls_data, cowls_early, cowls_late, massive_data,
                                          cowls_fit, cowls_early_fit, cowls_late_fit, 
                                          massive_fit, massive_early_fit, massive_late_fit):
        """Create consolidated mass-size relations plot"""
        print("Creating consolidated mass-size relations plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Determine common mass range
        all_mass_ranges = [
            cowls_fit['mass_range'], cowls_early_fit['mass_range'], cowls_late_fit['mass_range'],
            massive_fit['mass_range'], massive_early_fit['mass_range'], massive_late_fit['mass_range']
        ]
        mass_min = max([r[0] for r in all_mass_ranges])
        mass_max = min([r[1] for r in all_mass_ranges])
        mass_range = np.linspace(mass_min, mass_max, 100)
        
        print(f"Common mass range: log M* = {mass_min:.2f} - {mass_max:.2f}")
        
        # Plot COWLS data points by type with redshift color-coding
        cowls_early_scatter = ax.scatter(cowls_early_fit['log_mass'], cowls_early_fit['log_size'], 
                                       c=cowls_early['LP_zfinal'], cmap='Reds', alpha=0.7, s=50, 
                                       marker='o', label='COWLS Early-type', zorder=3, edgecolors='darkred', linewidth=0.5)
        
        cowls_late_scatter = ax.scatter(cowls_late_fit['log_mass'], cowls_late_fit['log_size'], 
                                      c=cowls_late['LP_zfinal'], cmap='Blues', alpha=0.7, s=50, 
                                      marker='s', label='COWLS Late-type', zorder=3, edgecolors='darkblue', linewidth=0.5)
        
        # Plot CWMGs data points (smaller, more transparent) with redshift color-coding
        massive_early_mask = massive_data['nsersic_rest'] <= 2.5
        massive_late_mask = massive_data['nsersic_rest'] > 2.5
        
        ax.scatter(massive_data.loc[massive_early_mask, 'LP_mass_med_PDF'], 
                  np.log10(massive_data.loc[massive_early_mask, 'Reff_kpc']), 
                  c=massive_data.loc[massive_early_mask, 'LP_zfinal'], cmap='Reds', alpha=0.1, s=3, 
                  marker='o', zorder=1)
        
        ax.scatter(massive_data.loc[massive_late_mask, 'LP_mass_med_PDF'], 
                  np.log10(massive_data.loc[massive_late_mask, 'Reff_kpc']), 
                  c=massive_data.loc[massive_late_mask, 'LP_zfinal'], cmap='Blues', alpha=0.1, s=3, 
                  marker='s', zorder=1)
        
        # Plot fits
        # COWLS overall fit
        cowls_pred = self.mass_size_relation(mass_range, cowls_fit['alpha'], cowls_fit['beta'])
        ax.plot(mass_range, cowls_pred, 'k-', linewidth=3, 
                label=f'COWLS: α={cowls_fit["alpha"]:.3f}±{cowls_fit["alpha_err"]:.3f}', zorder=4)
        
        # COWLS confidence interval
        cowls_ci_lower = np.interp(mass_range, cowls_fit['mass_range_plot'], cowls_fit['size_ci_lower'])
        cowls_ci_upper = np.interp(mass_range, cowls_fit['mass_range_plot'], cowls_fit['size_ci_upper'])
        ax.fill_between(mass_range, cowls_ci_lower, cowls_ci_upper, 
                       alpha=0.2, color='black', zorder=1)
        
        # COWLS Early-type fit
        cowls_early_pred = self.mass_size_relation(mass_range, cowls_early_fit['alpha'], cowls_early_fit['beta'])
        ax.plot(mass_range, cowls_early_pred, 'r--', linewidth=2, 
                label=f'COWLS Early: α={cowls_early_fit["alpha"]:.3f}±{cowls_early_fit["alpha_err"]:.3f}', zorder=4)
        
        # COWLS Late-type fit
        cowls_late_pred = self.mass_size_relation(mass_range, cowls_late_fit['alpha'], cowls_late_fit['beta'])
        ax.plot(mass_range, cowls_late_pred, 'b--', linewidth=2, 
                label=f'COWLS Late: α={cowls_late_fit["alpha"]:.3f}±{cowls_late_fit["alpha_err"]:.3f}', zorder=4)
        
        # CWMGs overall fit
        massive_pred = self.mass_size_relation(mass_range, massive_fit['alpha'], massive_fit['beta'])
        ax.plot(mass_range, massive_pred, 'g-', linewidth=3,
                label=f'CWMGs: α={massive_fit["alpha"]:.3f}±{massive_fit["alpha_err"]:.3f}', zorder=4)
        
        # CWMGs confidence interval
        massive_ci_lower = np.interp(mass_range, massive_fit['mass_range_plot'], massive_fit['size_ci_lower'])
        massive_ci_upper = np.interp(mass_range, massive_fit['mass_range_plot'], massive_fit['size_ci_upper'])
        ax.fill_between(mass_range, massive_ci_lower, massive_ci_upper, 
                       alpha=0.2, color='green', zorder=1)
        
        # CWMGs Early-type fit
        massive_early_pred = self.mass_size_relation(mass_range, massive_early_fit['alpha'], massive_early_fit['beta'])
        ax.plot(mass_range, massive_early_pred, 'orange', linestyle='--', linewidth=2,
                label=f'CWMGs Early: α={massive_early_fit["alpha"]:.3f}±{massive_early_fit["alpha_err"]:.3f}', zorder=4)
        
        # CWMGs Late-type fit
        massive_late_pred = self.mass_size_relation(mass_range, massive_late_fit['alpha'], massive_late_fit['beta'])
        ax.plot(mass_range, massive_late_pred, 'purple', linestyle='--', linewidth=2,
                label=f'CWMGs Late: α={massive_late_fit["alpha"]:.3f}±{massive_late_fit["alpha_err"]:.3f}', zorder=4)
        
        # Formatting
        ax.set_xlabel('log₁₀(M*/M☉)', fontsize=14)
        ax.set_ylabel('log₁₀(Reff/kpc)', fontsize=14)
        #ax.set_title('Consolidated Mass-Size Relations: COWLS vs CWMGs', fontsize=16)
        
        # Set axis limits
        ax.set_xlim(mass_min - 0.1, mass_max + 0.1)
        
        # Add symbol legend first
        from matplotlib.lines import Line2D
        symbol_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Early-type (n ≤ 2.5)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Late-type (n > 2.5)')
        ]
        symbol_legend = ax.legend(handles=symbol_legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add main legend
        main_legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
        ax.add_artist(symbol_legend)  # Keep symbol legend visible
        
        # Remove grid
        ax.grid(False)
        
        # Add horizontal colorbar for redshift
        cbar = plt.colorbar(cowls_early_scatter, ax=ax, orientation='horizontal', shrink=0.8, pad=0.1)
        cbar.set_label('Redshift (z)', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        output_file = 'consolidated_mass_size_relations.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Consolidated mass-size relations saved as '{output_file}'")
        
        plt.show()
    
    def create_redshift_evolution_plot(self, cowls_fits, cowls_early_fits, cowls_late_fits, 
                                      massive_fits, massive_early_fits, massive_late_fits):
        """Create redshift evolution plot"""
        print("Creating redshift evolution plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot all trends on the same axes
        if cowls_fits:
            z_centers = [fit['z_center'] for fit in cowls_fits]
            alphas = [fit['alpha'] for fit in cowls_fits]
            alpha_errs = [fit['alpha_err'] for fit in cowls_fits]
            ax.errorbar(z_centers, alphas, yerr=alpha_errs, fmt='ko-', linewidth=2, 
                       markersize=8, capsize=5, label='COWLS')
        
        if cowls_early_fits:
            z_centers = [fit['z_center'] for fit in cowls_early_fits]
            alphas = [fit['alpha'] for fit in cowls_early_fits]
            alpha_errs = [fit['alpha_err'] for fit in cowls_early_fits]
            ax.errorbar(z_centers, alphas, yerr=alpha_errs, fmt='ro-', linewidth=2, 
                       markersize=8, capsize=5, label='COWLS Early-type')
        
        if cowls_late_fits:
            z_centers = [fit['z_center'] for fit in cowls_late_fits]
            alphas = [fit['alpha'] for fit in cowls_late_fits]
            alpha_errs = [fit['alpha_err'] for fit in cowls_late_fits]
            ax.errorbar(z_centers, alphas, yerr=alpha_errs, fmt='bo-', linewidth=2, 
                       markersize=8, capsize=5, label='COWLS Late-type')
        
        if massive_fits:
            z_centers = [fit['z_center'] for fit in massive_fits]
            alphas = [fit['alpha'] for fit in massive_fits]
            alpha_errs = [fit['alpha_err'] for fit in massive_fits]
            ax.errorbar(z_centers, alphas, yerr=alpha_errs, fmt='gs--', linewidth=2, 
                       markersize=8, capsize=5, label='CWMGs')
        
        if massive_early_fits:
            z_centers = [fit['z_center'] for fit in massive_early_fits]
            alphas = [fit['alpha'] for fit in massive_early_fits]
            alpha_errs = [fit['alpha_err'] for fit in massive_early_fits]
            ax.errorbar(z_centers, alphas, yerr=alpha_errs, fmt='orange', linestyle='--', linewidth=2, 
                       markersize=8, capsize=5, label='CWMGs Early-type')
        
        if massive_late_fits:
            z_centers = [fit['z_center'] for fit in massive_late_fits]
            alphas = [fit['alpha'] for fit in massive_late_fits]
            alpha_errs = [fit['alpha_err'] for fit in massive_late_fits]
            ax.errorbar(z_centers, alphas, yerr=alpha_errs, fmt='purple', linestyle='--', linewidth=2, 
                       markersize=8, capsize=5, label='CWMGs Late-type')
        
        #ax.set_title('Redshift Evolution of Mass-Size Relations', fontsize=16)
        ax.set_xlabel('Redshift (z)', fontsize=14)
        ax.set_ylabel('Mass-Size Slope (α)', fontsize=14)
        ax.grid(False)
        ax.legend(fontsize=12)
        
        # Add horizontal line at α=0 for reference
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        output_file = 'redshift_evolution_consolidated.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Redshift evolution plot saved as '{output_file}'")
        
        plt.show()
    
    def calculate_median_masses(self, cowls_data, cowls_early, cowls_late, massive_data, massive_early, massive_late):
        """Calculate median stellar masses for each sample"""
        samples = {
            'COWLS': cowls_data,
            'COWLS Early-type': cowls_early,
            'COWLS Late-type': cowls_late,
            'CWMGs': massive_data,
            'CWMGs Early-type': massive_early,
            'CWMGs Late-type': massive_late
        }
        
        median_masses = {}
        for name, data in samples.items():
            if len(data) > 0:
                median_masses[name] = data['LP_mass_med_PDF'].median()
                print(f"{name} median mass: {median_masses[name]:.2f} log₁₀(M☉)")
            else:
                median_masses[name] = np.nan
                print(f"{name}: No galaxies")
        
        return median_masses
    
    def fit_size_evolution_from_binned_fits(self, redshift_fits, sample_name, median_mass):
        """Fit size evolution using existing redshift-binned mass-size relations"""
        if not redshift_fits or len(redshift_fits) < 2:
            print(f"  Not enough redshift bins for {sample_name}")
            return None
        
        # Extract data from redshift-binned fits
        z_centers = []
        log_sizes_at_fixed_mass = []
        log_size_errors = []
        
        for fit in redshift_fits:
            z_center = fit['z_center']
            alpha = fit['alpha']  # slope of mass-size relation
            beta = fit['beta']    # intercept of mass-size relation
            alpha_err = fit['alpha_err']
            beta_err = fit['beta_err']
            
            # Calculate log₁₀(R_e/kpc) at fixed stellar mass using the mass-size relation
            # log₁₀(R_e/kpc) = beta + alpha * log₁₀(M*/M☉)
            log_size = beta + alpha * median_mass
            
            # Error propagation for log_size
            log_size_err = np.sqrt(beta_err**2 + (alpha_err * median_mass)**2)
            
            z_centers.append(z_center)
            log_sizes_at_fixed_mass.append(log_size)
            log_size_errors.append(log_size_err)
        
        z_centers = np.array(z_centers)
        log_sizes_at_fixed_mass = np.array(log_sizes_at_fixed_mass)
        log_size_errors = np.array(log_size_errors)
        
        # Fit size evolution: log₁₀(R_e/kpc) = A - α log(1 + z)
        log_one_plus_z = np.log10(1 + z_centers)
        
        # Use PyMC for fitting
        with pm.Model() as model:
            A = pm.Normal("A", mu=0.5, sigma=1.0)
            alpha = pm.Normal("alpha", mu=0.0, sigma=0.5)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            
            mu = A - alpha * log_one_plus_z
            pm.Normal("log_size_obs", mu=mu, sigma=sigma, observed=log_sizes_at_fixed_mass)
            
            trace = pm.sample(
                draws=2000,
                tune=500,
                target_accept=0.95,
                chains=4,
                cores=2,
                progressbar=False
            )
        
        A_samples = trace.posterior['A'].values.flatten()
        alpha_samples = trace.posterior['alpha'].values.flatten()
        
        A_mean = np.mean(A_samples)
        A_err = np.std(A_samples)
        alpha_mean = np.mean(alpha_samples)
        alpha_err = np.std(alpha_samples)
        
        # Calculate R²
        y_pred = A_mean - alpha_mean * log_one_plus_z
        y_true = log_sizes_at_fixed_mass
        r_squared = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        A_ci = np.percentile(A_samples, [2.5, 97.5])
        alpha_ci = np.percentile(alpha_samples, [2.5, 97.5])
        
        print(f"  {sample_name} size evolution: log₁₀(Reff/kpc) = {A_mean:.3f} ± {A_err:.3f} - {alpha_mean:.3f} ± {alpha_err:.3f} × log(1+z)")
        print(f"  R² = {r_squared:.3f}, N_bins = {len(z_centers)}")
        print(f"  95% CI: A = [{A_ci[0]:.3f}, {A_ci[1]:.3f}], α = [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]")
        
        return {
            'sample': sample_name,
            'A': A_mean, 'A_err': A_err,
            'alpha': alpha_mean, 'alpha_err': alpha_err,
            'A_ci': A_ci, 'alpha_ci': alpha_ci,
            'r_squared': r_squared, 'n_bins': len(z_centers),
            'median_mass': median_mass,
            'z_centers': z_centers,
            'log_sizes': log_sizes_at_fixed_mass,
            'log_size_errors': log_size_errors,
            'trace': trace,
            'A_samples': A_samples,
            'alpha_samples': alpha_samples
        }
    
    def create_size_evolution_plot(self, size_evolution_results):
        """Create plot showing size evolution for all samples"""
        if not size_evolution_results:
            print("No size evolution results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Size evolution parameters
        samples = [result['sample'] for result in size_evolution_results]
        A_values = [result['A'] for result in size_evolution_results]
        A_errors = [result['A_err'] for result in size_evolution_results]
        alpha_values = [result['alpha'] for result in size_evolution_results]
        alpha_errors = [result['alpha_err'] for result in size_evolution_results]
        
        # Plot A parameter
        ax1.errorbar(range(len(samples)), A_values, yerr=A_errors, 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('A (log₁₀ R_e/kpc at z=0)')
        ax1.set_title('Size Evolution Normalization')
        ax1.set_xticks(range(len(samples)))
        ax1.set_xticklabels(samples, rotation=45, ha='right')
        ax1.grid(False)
        
        # Plot α parameter
        ax2.errorbar(range(len(samples)), alpha_values, yerr=alpha_errors, 
                    fmt='s', capsize=5, capthick=2, markersize=8, color='red')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('α (size evolution parameter)')
        ax2.set_title('Size Evolution Parameter')
        ax2.set_xticks(range(len(samples)))
        ax2.set_xticklabels(samples, rotation=45, ha='right')
        ax2.grid(False)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        output_file = 'size_evolution_from_binned_fits.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Size evolution plot saved as '{output_file}'")
        
        plt.show()
        
        # Create detailed size evolution curves plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'teal', 'goldenrod', 'magenta', 'slategray']
        
        for i, result in enumerate(size_evolution_results):
            sample = result['sample']
            A = result['A']
            alpha = result['alpha']
            median_mass = result['median_mass']
            
            # Generate redshift range for plotting (extend for massive galaxies)
            z_min, z_max = 0.2, 4.0
            if 'CWMG' in sample:
                z_max = 10.0
            z_plot = np.linspace(z_min, z_max, 200)
            log_one_plus_z = np.log10(1 + z_plot)
            log_size_plot = A - alpha * log_one_plus_z
            
            ax.plot(z_plot, log_size_plot, color=colors[i % len(colors)], 
                   linewidth=2, label=f'{sample} (M*={median_mass:.1f})')
            
            # Add 68% credible interval from posterior samples if available
            A_samples = result.get('A_samples')
            alpha_samples = result.get('alpha_samples')
            if A_samples is not None and alpha_samples is not None:
                pred_samples = A_samples[:, None] - alpha_samples[:, None] * log_one_plus_z[None, :]
                lower = np.percentile(pred_samples, 16, axis=0)
                upper = np.percentile(pred_samples, 84, axis=0)
                ax.fill_between(z_plot, lower, upper, color=colors[i % len(colors)], alpha=0.2)
        
        ax.set_xlabel('Redshift (z)', fontsize=14)
        ax.set_ylabel('log₁₀(R_e/kpc)', fontsize=14)
        ax.set_title('Size Evolution at Fixed Stellar Mass', fontsize=16)
        ax.grid(False)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.1, 11.0)
        
        plt.tight_layout()
        
        # Save detailed plot
        output_file = 'size_evolution_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Size evolution curves plot saved as '{output_file}'")
        
        plt.show()
    
    def save_size_evolution_results(self, size_evolution_results):
        """Save size evolution results to CSV"""
        if not size_evolution_results:
            return
        
        results_data = []
        for result in size_evolution_results:
            results_data.append({
                'sample': result['sample'],
                'A': result['A'],
                'A_err': result['A_err'],
                'alpha': result['alpha'],
                'alpha_err': result['alpha_err'],
                'r_squared': result['r_squared'],
                'n_bins': result['n_bins'],
                'median_mass': result['median_mass']
            })
        
        df = pd.DataFrame(results_data)
        output_file = 'size_evolution_from_binned_fits.csv'
        df.to_csv(output_file, index=False)
        print(f"Size evolution results saved to '{output_file}'")
    
    def run_consolidated_analysis(self):
        """Run the complete consolidated analysis"""
        print("=== Consolidated Mass-Size Relations Analysis ===")
        
        # Load data
        cowls_data = self.load_cowls_data()
        massive_data = self.load_massive_galaxies(mass_cut=9.0)
        
        # Separate by Sersic index
        cowls_early, cowls_late = self.separate_by_sersic(cowls_data, "COWLS")
        massive_early, massive_late = self.separate_by_sersic(massive_data, "CWMGs")
        
        # Fit relations for each sample
        print("\n=== Fitting Mass-Size Relations ===")
        cowls_fit = self.fit_mass_size_relation(cowls_data, "COWLS")
        cowls_early_fit = self.fit_mass_size_relation(cowls_early, "COWLS Early-type")
        cowls_late_fit = self.fit_mass_size_relation(cowls_late, "COWLS Late-type")
        massive_fit = self.fit_mass_size_relation(massive_data, "CWMGs")
        massive_early_fit = self.fit_mass_size_relation(massive_early, "CWMGs Early-type")
        massive_late_fit = self.fit_mass_size_relation(massive_late, "CWMGs Late-type")
        
        # Fit redshift-binned relations
        print("\n=== Fitting Redshift-Binned Relations ===")
        cowls_fits = self.fit_redshift_binned_relations(cowls_data, "COWLS", min_samples=5)
        cowls_early_fits = self.fit_redshift_binned_relations(cowls_early, "COWLS Early-type", min_samples=5)
        cowls_late_fits = self.fit_redshift_binned_relations(cowls_late, "COWLS Late-type", min_samples=5)
        massive_fits = self.fit_redshift_binned_relations(massive_data, "CWMGs", min_samples=20)
        massive_early_fits = self.fit_redshift_binned_relations(massive_early, "CWMGs Early-type", min_samples=20)
        massive_late_fits = self.fit_redshift_binned_relations(massive_late, "CWMGs Late-type", min_samples=20)
        
        # Create visualizations
        self.create_consolidated_mass_size_plot(cowls_data, cowls_early, cowls_late, massive_data,
                                               cowls_fit, cowls_early_fit, cowls_late_fit, 
                                               massive_fit, massive_early_fit, massive_late_fit)
        self.create_redshift_evolution_plot(cowls_fits, cowls_early_fits, cowls_late_fits, 
                                           massive_fits, massive_early_fits, massive_late_fits)
        
        # Size evolution analysis using existing redshift-binned fits
        print("\n=== Size Evolution Analysis ===")
        median_masses = self.calculate_median_masses(cowls_data, cowls_early, cowls_late, 
                                                    massive_data, massive_early, massive_late)
        
        size_evolution_results = []
        
        # Fit size evolution for each sample using existing redshift-binned fits
        if cowls_fits and median_masses['COWLS'] is not None:
            result = self.fit_size_evolution_from_binned_fits(cowls_fits, "COWLS", median_masses['COWLS'])
            if result:
                size_evolution_results.append(result)
        
        if cowls_early_fits and median_masses['COWLS Early-type'] is not None:
            result = self.fit_size_evolution_from_binned_fits(cowls_early_fits, "COWLS Early-type", median_masses['COWLS Early-type'])
            if result:
                size_evolution_results.append(result)
        
        if cowls_late_fits and median_masses['COWLS Late-type'] is not None:
            result = self.fit_size_evolution_from_binned_fits(cowls_late_fits, "COWLS Late-type", median_masses['COWLS Late-type'])
            if result:
                size_evolution_results.append(result)
        
        if massive_fits and median_masses['CWMGs'] is not None:
            result = self.fit_size_evolution_from_binned_fits(massive_fits, "CWMGs", median_masses['CWMGs'])
            if result:
                size_evolution_results.append(result)
        
        if massive_early_fits and median_masses['CWMGs Early-type'] is not None:
            result = self.fit_size_evolution_from_binned_fits(massive_early_fits, "CWMGs Early-type", median_masses['CWMGs Early-type'])
            if result:
                size_evolution_results.append(result)
        
        if massive_late_fits and median_masses['CWMGs Late-type'] is not None:
            result = self.fit_size_evolution_from_binned_fits(massive_late_fits, "CWMGs Late-type", median_masses['CWMGs Late-type'])
            if result:
                size_evolution_results.append(result)
        
        # Create size evolution plots and save results
        if size_evolution_results:
            self.create_size_evolution_plot(size_evolution_results)
            self.save_size_evolution_results(size_evolution_results)
        
        print("\n=== Consolidated Analysis Complete ===")
        print(f"COWLS: {cowls_fit['n_galaxies']} galaxies")
        print(f"CWMGs: {massive_fit['n_galaxies']} galaxies")

if __name__ == "__main__":
    analyzer = ConsolidatedAnalyzer()
    analyzer.run_consolidated_analysis()
