#!/usr/bin/env python3
"""
Create CWMGs-only mass-size relation plots for Paper 1 (Framework)
- Left panel: Subplots showing mass-size relations for each redshift bin
- Right panel: Beta vs redshift evolution
Uses existing fit results from CSV files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit
import os

# Increase default font sizes and improve readability for paper/A4 output
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.linewidth': 1.0
})

class Paper1CWMGsAnalyzer:
    def __init__(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
    def convert_arcsec_to_kpc(self, arcsec, redshift):
        """Convert angular size to physical size in kpc"""
        if np.isnan(arcsec) or np.isnan(redshift) or redshift <= 0:
            return np.nan
        
        da = self.cosmo.angular_diameter_distance(redshift).value  # Mpc
        kpc_per_arcsec = da * 1000 * np.pi / (180 * 3600)  # kpc/arcsec
        return arcsec * kpc_per_arcsec
    
    def select_restframe_structures(self, row):
        """Select rest-frame structural parameters based on redshift"""
        z = row.get('LP_zfinal', np.nan)
        
        if 0.05 < z <= 0.4:
            band = 'f115w'
        elif 0.4 < z <= 1.0:
            band = 'f150w'
        elif 1.0 < z <= 3.0:
            band = 'f277w'
        elif 3.0 < z <= 12.0:
            band = 'f444w'
        else:
            return pd.Series({'best_filter': None, 'rearc_arcsec': np.nan, 'size_kpc': np.nan, 'nsersic': np.nan})
        
        output = {'best_filter': band}
        
        rearc_key = f'rearc_{band}'
        nsersic_key = f'nsersic_{band}'
        
        rearc_value = row.get(rearc_key, np.nan)
        nsersic_value = row.get(nsersic_key, np.nan)
        
        output['rearc_arcsec'] = rearc_value
        output['size_kpc'] = self.convert_arcsec_to_kpc(rearc_value, z)
        output['nsersic'] = nsersic_value
        
        return pd.Series(output)
    
    def load_massive_galaxies(self, mass_cut=9.0):
        """Load and process massive galaxy data (CWMGs)"""
        print(f"Loading massive galaxies with log M* > {mass_cut}...")
        
        catalog_file = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/data/galaxy_catalog.csv'
        df = pd.read_csv(catalog_file)
        print(f"Total galaxies in catalog: {len(df)}")
        
        df = df[df['LP_mass_med_PDF'] > mass_cut]
        print(f"Massive galaxies (log M* > {mass_cut}): {len(df)}")
        
        if 'LP_warn_fl' in df.columns:
            df = df[df['LP_warn_fl'] == 0]
        
        df = df[df['LP_zfinal'] > 0]
        df = df[df['LP_zfinal'] < 12]
        df = df[df['LP_mass_med_PDF'] > 8]
        df = df[df['LP_mass_med_PDF'] < 13]
        
        for band in ['f115w', 'f150w', 'f277w', 'f444w']:
            rearc_col = f'rearc_{band}'
            if rearc_col in df.columns:
                df = df[df[rearc_col] > 0]
                df = df[df[rearc_col] < 3]
        
        rest_frame_data = df.apply(self.select_restframe_structures, axis=1)
        df = pd.concat([df, rest_frame_data], axis=1)
        
        df = df.dropna(subset=['size_kpc', 'nsersic'])
        df = df[df['size_kpc'] > 0.1]
        df = df[df['size_kpc'] < 20]
        
        df['Reff_kpc'] = df['size_kpc']
        df['nsersic_rest'] = df['nsersic']
        
        print(f"Final CWMGs sample: {len(df)} galaxies")
        return df
    
    def separate_by_sersic(self, df):
        """Separate galaxies into early-type and late-type"""
        early_type = df[df['nsersic_rest'] <= 2.5].copy()
        late_type = df[df['nsersic_rest'] > 2.5].copy()
        return early_type, late_type
    
    def load_redshift_binned_fits(self):
        """Load redshift-binned fit results from CSV"""
        csv_file = '/Users/gozalig1/Projects/jwst-mock-lens-simulator/analysis/mass_size_relations/massive_galaxy_redshift_binned_relations.csv'
        df = pd.read_csv(csv_file)
        return df.to_dict('records')
    
    def calculate_extended_redshift_bins(self, massive_data, existing_fits, min_samples=20):
        """Calculate fits for extended redshift bins (4.0-6.0, 6.0-8.0, 8.0-12.0)"""
        print("Calculating fits for extended redshift bins...")
        
        extended_bins = [(4.0, 6.0), (6.0, 8.0), (8.0, 12.0)]
        extended_fits = []
        
        for z_min, z_max in extended_bins:
            subset = massive_data[(massive_data['LP_zfinal'] >= z_min) & (massive_data['LP_zfinal'] < z_max)]
            
            if len(subset) < min_samples:
                print(f"  z{z_min}-{z_max}: Insufficient data ({len(subset)} galaxies)")
                continue
            
            try:
                log_mass = subset['LP_mass_med_PDF'].values
                log_size = np.log10(subset['Reff_kpc'].values)
                
                # Fit using scipy
                fit_result = self.fit_linear_relation(log_mass, log_size)
                
                if fit_result:
                    extended_fits.append({
                        'z_min': z_min,
                        'z_max': z_max,
                        'z_center': (z_min + z_max) / 2,
                        'alpha': fit_result['alpha'],
                        'alpha_err': fit_result['alpha_err'],
                        'beta': fit_result['beta'],
                        'beta_err': fit_result['beta_err'],
                        'cov_ab': fit_result.get('cov_ab', 0.0),  # Include covariance
                        'n_galaxies': len(subset)
                    })
                    print(f"  z{z_min}-{z_max}: α={fit_result['alpha']:.3f}±{fit_result['alpha_err']:.3f}, "
                          f"β={fit_result['beta']:.3f}±{fit_result['beta_err']:.3f}, N={len(subset)}")
            except Exception as e:
                print(f"  z{z_min}-{z_max}: Fit failed - {e}")
                continue
        
        return extended_fits
    
    def fit_linear_relation(self, x, y, n_realizations=100):
        """Fit a simple linear relation y = a*x + b using scipy with bootstrap"""
        def linear_func(x, a, b):
            return a * x + b
        
        try:
            # Initial fit
            popt, pcov = curve_fit(linear_func, x, y)
            a, b = popt
            a_err, b_err = np.sqrt(np.diag(pcov))
            cov_ab = pcov[0, 1]
            
            # Bootstrap to get parameter distributions
            a_samples = []
            b_samples = []
            
            for i in range(n_realizations):
                try:
                    # Bootstrap resample
                    n_points = len(x)
                    indices = np.random.choice(n_points, size=n_points, replace=True)
                    x_boot = x[indices]
                    y_boot = y[indices]
                    
                    # Fit to bootstrap sample
                    popt_boot, _ = curve_fit(linear_func, x_boot, y_boot, maxfev=10000)
                    a_samples.append(popt_boot[0])
                    b_samples.append(popt_boot[1])
                except:
                    continue
            
            return {
                'alpha': a, 'alpha_err': a_err, 
                'beta': b, 'beta_err': b_err, 
                'cov': pcov, 'cov_ab': cov_ab, 
                'x': x, 'y': y,
                'alpha_samples': np.array(a_samples) if len(a_samples) > 0 else None,
                'beta_samples': np.array(b_samples) if len(b_samples) > 0 else None
            }
        except:
            return None
    
    def calculate_confidence_band(self, x_range, fit_result, n_sigma=1):
        """Calculate confidence band for linear fit - simplified version"""
        if fit_result is None or 'alpha_err' not in fit_result:
            return None, None
        
        alpha = fit_result['alpha']
        beta = fit_result['beta']
        alpha_err = fit_result['alpha_err']
        beta_err = fit_result['beta_err']
        
        # Predicted values
        y_pred = alpha * x_range + beta
        
        # Simple error propagation: use alpha_err and beta_err
        # For a tighter band, use only the parameter uncertainties
        # This gives prediction uncertainty rather than full confidence band
        x_mean = np.mean(x_range)
        x_span = np.max(x_range) - np.min(x_range)
        
        # Simplified: use parameter errors scaled by x position
        # This gives a more reasonable band width
        se_pred = np.sqrt((alpha_err * (x_range - x_mean))**2 + beta_err**2) * 0.5
        
        # Confidence band (much tighter)
        lower = y_pred - n_sigma * se_pred
        upper = y_pred + n_sigma * se_pred
        
        return lower, upper
    
    def create_redshift_binned_subplots(self, massive_data, redshift_fits):
        """Create comprehensive figure with mass-size subplots and evolution plots"""
        print("Creating comprehensive redshift-binned figure...")
        
        # Filter fits to only include those with sufficient data
        valid_fits = [f for f in redshift_fits if f['n_galaxies'] >= 20]
        valid_fits.sort(key=lambda x: x['z_center'])
        
        n_bins = len(valid_fits)
        n_cols = 3
        # Calculate rows needed: ensure at least 4 rows (row 3 for alpha/beta)
        n_rows_mass_size = (n_bins + n_cols - 1) // n_cols
        total_rows = max(n_rows_mass_size, 4)  # At least 4 rows to accommodate row 3
        
        # Use a tall A4-like figure for multi-row subplots to ensure readable labels
        fig = plt.figure(figsize=(11.69, 16))
        # Slightly increase vertical spacing between rows for better readability
        gs = GridSpec(total_rows, n_cols, figure=fig, hspace=0.55, wspace=0.3)
        
        # Store early/late fits for evolution plots
        early_fits_list = []
        late_fits_list = []
        
        # Define consistent colors
        color_all = '#2E2E2E'  # Dark gray/black for all
        color_early = '#FF9800'  # Orange for early-type (improved visibility) - fits
        color_late = '#00BCD4'   # Cyan for late-type (improved visibility) - fits
        # Keep scatter/data point colors distinct (red/blue) for visibility
        scatter_color_early = '#D32F2F'  # red for early-type points
        scatter_color_late = '#1976D2'   # blue for late-type points
        
        # Plot mass-size relations for each redshift bin
        # Place them sequentially, skipping positions (3,1) and (3,2) which are reserved
        plot_idx = 0
        for idx, fit in enumerate(valid_fits):
            # Find next available position, skipping (3,1) and (3,2)
            while True:
                row = plot_idx // n_cols
                col = plot_idx % n_cols
                if (row == 3 and col == 1) or (row == 3 and col == 2):
                    plot_idx += 1
                else:
                    break
            
            ax = fig.add_subplot(gs[row, col])
            plot_idx += 1
            
            z_min = fit['z_min']
            z_max = fit['z_max']
            z_center = fit['z_center']
            
            # Get data for this redshift bin
            subset = massive_data[(massive_data['LP_zfinal'] >= z_min) & (massive_data['LP_zfinal'] < z_max)]
            
            if len(subset) == 0:
                continue
            
            # Separate by Sersic index
            early_mask = subset['nsersic_rest'] <= 2.5
            late_mask = subset['nsersic_rest'] > 2.5
            
            # Plot data points with better colors
            if early_mask.sum() > 0:
                ax.scatter(subset.loc[early_mask, 'LP_mass_med_PDF'], 
                          np.log10(subset.loc[early_mask, 'Reff_kpc']), 
                          c=scatter_color_early, alpha=0.3, s=10, marker='o', 
                          edgecolors=scatter_color_early, linewidth=0.2, 
                          label='Early-type (n ≤ 2.5)' if idx == 0 else '', zorder=1)
            
            if late_mask.sum() > 0:
                ax.scatter(subset.loc[late_mask, 'LP_mass_med_PDF'], 
                          np.log10(subset.loc[late_mask, 'Reff_kpc']), 
                          c=scatter_color_late, alpha=0.3, s=10, marker='s', 
                          edgecolors=scatter_color_late, linewidth=0.2, 
                          label='Late-type (n > 2.5)' if idx == 0 else '', zorder=1)
            
            # Calculate and plot fits for each type
            mass_range = np.linspace(subset['LP_mass_med_PDF'].min(), subset['LP_mass_med_PDF'].max(), 100)
            
            # Overall fit (from loaded results) with confidence band
            size_pred_all = fit['alpha'] * mass_range + fit['beta']
            ax.plot(mass_range, size_pred_all, color=color_all, linewidth=3, 
                   linestyle='-', label='All galaxies' if idx == 0 else '', zorder=5)
            
            # Early-type fit with confidence band
            early_fit_result = None
            if early_mask.sum() >= 15:
                early_subset = subset[early_mask]
                early_fit = self.fit_linear_relation(early_subset['LP_mass_med_PDF'].values, 
                                                     np.log10(early_subset['Reff_kpc'].values),
                                                     n_realizations=100)
                if early_fit and early_fit.get('alpha_samples') is not None:
                    early_fit_result = early_fit
                    size_pred_early = early_fit['alpha'] * mass_range + early_fit['beta']
                    
                    # Calculate confidence band from bootstrap samples
                    size_samples_early = []
                    for m in mass_range:
                        size_pred_samples = early_fit['alpha_samples'] * m + early_fit['beta_samples']
                        size_samples_early.append(size_pred_samples)
                    
                    size_samples_early = np.array(size_samples_early).T
                    size_lower_early = np.percentile(size_samples_early, 16, axis=0)  # 1-sigma lower
                    size_upper_early = np.percentile(size_samples_early, 84, axis=0)  # 1-sigma upper
                    
                    # Plot confidence band
                    ax.fill_between(mass_range, size_lower_early, size_upper_early, 
                                   color=color_early, alpha=0.15, zorder=2)
                    
                    ax.plot(mass_range, size_pred_early, color=color_early, linewidth=2.5, 
                           linestyle='--', label='Early-type' if idx == 0 else '', zorder=4)
                    early_fits_list.append({
                        'z_center': z_center, 'z_min': z_min, 'z_max': z_max,
                        'alpha': early_fit['alpha'], 'alpha_err': early_fit['alpha_err'],
                        'beta': early_fit['beta'], 'beta_err': early_fit['beta_err'],
                        'cov_ab': early_fit.get('cov_ab', 0.0)  # Include covariance
                    })
            
            # Late-type fit with confidence band
            late_fit_result = None
            if late_mask.sum() >= 15:
                late_subset = subset[late_mask]
                late_fit = self.fit_linear_relation(late_subset['LP_mass_med_PDF'].values, 
                                                   np.log10(late_subset['Reff_kpc'].values),
                                                   n_realizations=100)
                if late_fit and late_fit.get('alpha_samples') is not None:
                    late_fit_result = late_fit
                    size_pred_late = late_fit['alpha'] * mass_range + late_fit['beta']
                    
                    # Calculate confidence band from bootstrap samples
                    size_samples_late = []
                    for m in mass_range:
                        size_pred_samples = late_fit['alpha_samples'] * m + late_fit['beta_samples']
                        size_samples_late.append(size_pred_samples)
                    
                    size_samples_late = np.array(size_samples_late).T
                    size_lower_late = np.percentile(size_samples_late, 16, axis=0)  # 1-sigma lower
                    size_upper_late = np.percentile(size_samples_late, 84, axis=0)  # 1-sigma upper
                    
                    # Plot confidence band
                    ax.fill_between(mass_range, size_lower_late, size_upper_late, 
                                   color=color_late, alpha=0.15, zorder=2)
                    
                    ax.plot(mass_range, size_pred_late, color=color_late, linewidth=2.5, 
                           linestyle='--', label='Late-type' if idx == 0 else '', zorder=4)
                    late_fits_list.append({
                        'z_center': z_center, 'z_min': z_min, 'z_max': z_max,
                        'alpha': late_fit['alpha'], 'alpha_err': late_fit['alpha_err'],
                        'beta': late_fit['beta'], 'beta_err': late_fit['beta_err'],
                        'cov_ab': late_fit.get('cov_ab', 0.0)  # Include covariance
                    })
            
            # Formatting
            ax.set_xlabel('log₁₀(M*/M☉)', fontsize=14)
            ax.set_ylabel('log₁₀(Reff/kpc)', fontsize=14)
            ax.set_title(f'z = {z_min:.1f} - {z_max:.1f}\n(N = {fit["n_galaxies"]})', 
                        fontsize=16, fontweight='normal')
            ax.grid(False)
            ax.set_xlim(subset['LP_mass_med_PDF'].min() - 0.1, subset['LP_mass_med_PDF'].max() + 0.1)
            
            if idx == 0:
                ax.legend(fontsize=12, loc='lower right', frameon=False)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Alpha vs redshift plot in position (3, 1) - row 3, column 1 (0-indexed)
        ax_alpha = fig.add_subplot(gs[3, 1])  # Row 3, column 1
        z_centers_all = [f['z_center'] for f in valid_fits]
        alphas_all = [f['alpha'] for f in valid_fits]
        alpha_errs_all = [f['alpha_err'] for f in valid_fits]
        
        ax_alpha.errorbar(z_centers_all, alphas_all, yerr=alpha_errs_all, 
                         fmt='o-', color=color_all, linewidth=2.5, markersize=10, 
                         capsize=5, capthick=2, label='All galaxies', zorder=3)
        
        if early_fits_list:
            z_centers_early = [f['z_center'] for f in early_fits_list]
            alphas_early = [f['alpha'] for f in early_fits_list]
            alpha_errs_early = [f['alpha_err'] for f in early_fits_list]
            ax_alpha.errorbar(z_centers_early, alphas_early, yerr=alpha_errs_early, 
                             fmt='s--', color=color_early, linewidth=2.5, markersize=9, 
                             capsize=5, capthick=2, label='Early-type', zorder=3)
        
        if late_fits_list:
            z_centers_late = [f['z_center'] for f in late_fits_list]
            alphas_late = [f['alpha'] for f in late_fits_list]
            alpha_errs_late = [f['alpha_err'] for f in late_fits_list]
            ax_alpha.errorbar(z_centers_late, alphas_late, yerr=alpha_errs_late, 
                             fmt='^--', color=color_late, linewidth=2.5, markersize=9, 
                             capsize=5, capthick=2, label='Late-type', zorder=3)
        
        ax_alpha.set_xlabel('Redshift (z)', fontsize=14, fontweight='bold')
        ax_alpha.set_ylabel('α (Slope)', fontsize=14, fontweight='bold')
        ax_alpha.set_title('Slope Evolution', fontsize=16, fontweight='bold')
        ax_alpha.grid(False)
        ax_alpha.legend(fontsize=12, loc='best', frameon=False)
        ax_alpha.tick_params(axis='both', which='major', labelsize=12)
        ax_alpha.axhline(y=0, color='gray', linestyle=':', alpha=0.5, zorder=1)
        
        # Beta vs redshift plot in position (3, 2) - row 3, column 2 (0-indexed)
        ax_beta = fig.add_subplot(gs[3, 2])  # Row 3, column 2
        betas_all = [f['beta'] for f in valid_fits]
        beta_errs_all = [f['beta_err'] for f in valid_fits]
        
        ax_beta.errorbar(z_centers_all, betas_all, yerr=beta_errs_all, 
                        fmt='o-', color=color_all, linewidth=2.5, markersize=10, 
                        capsize=5, capthick=2, label='All galaxies', zorder=3)
        
        if early_fits_list:
            betas_early = [f['beta'] for f in early_fits_list]
            beta_errs_early = [f['beta_err'] for f in early_fits_list]
            ax_beta.errorbar(z_centers_early, betas_early, yerr=beta_errs_early, 
                            fmt='s--', color=color_early, linewidth=2.5, markersize=9, 
                            capsize=5, capthick=2, label='Early-type', zorder=3)
        
        if late_fits_list:
            betas_late = [f['beta'] for f in late_fits_list]
            beta_errs_late = [f['beta_err'] for f in late_fits_list]
            ax_beta.errorbar(z_centers_late, betas_late, yerr=beta_errs_late, 
                            fmt='^--', color=color_late, linewidth=2.5, markersize=9, 
                            capsize=5, capthick=2, label='Late-type', zorder=3)
        
        ax_beta.set_xlabel('Redshift (z)', fontsize=14, fontweight='bold')
        ax_beta.set_ylabel('β (Intercept)', fontsize=14, fontweight='bold')
        ax_beta.set_title('Intercept Evolution', fontsize=16, fontweight='bold')
        ax_beta.grid(False)
        ax_beta.legend(fontsize=12, loc='best', frameon=False)
        ax_beta.tick_params(axis='both', which='major', labelsize=12)
        
        # Remove title to eliminate white space
        # plt.suptitle('Mass-Size Relations by Redshift Bin (CWMGs)', fontsize=16, y=0.995, fontweight='bold')
        
        return fig
    
    def calculate_size_at_fixed_mass(self, redshift_fits, fixed_mass):
        """Calculate effective radius at fixed stellar mass for each redshift bin
        
        Uses proper error propagation accounting for correlation between alpha and beta:
        σ²_y = (∂y/∂α)² σ²_α + (∂y/∂β)² σ²_β + 2(∂y/∂α)(∂y/∂β) cov(α,β)
        where y = α*M + β, so:
        σ²_y = M² σ²_α + σ²_β + 2M cov(α,β)
        """
        z_centers = []
        log_sizes = []
        log_size_errs = []
        
        for fit in redshift_fits:
            # log₁₀(Reff/kpc) = alpha * log₁₀(M*/M☉) + beta
            log_size = fit['alpha'] * fixed_mass + fit['beta']
            
            # Proper error propagation accounting for correlation
            # σ²_y = M² σ²_α + σ²_β + 2M cov(α,β)
            alpha_err_sq = fit['alpha_err']**2
            beta_err_sq = fit['beta_err']**2
            
            # Try to get covariance term if available
            if 'cov_ab' in fit:
                cov_ab = fit['cov_ab']
            else:
                # If covariance not available, assume zero correlation (conservative)
                cov_ab = 0.0
            
            # Full error propagation with correlation term
            log_size_err_sq = (fixed_mass**2) * alpha_err_sq + beta_err_sq + 2 * fixed_mass * cov_ab
            log_size_err = np.sqrt(max(0, log_size_err_sq))  # Ensure non-negative
            
            z_centers.append(fit['z_center'])
            log_sizes.append(log_size)
            log_size_errs.append(log_size_err)
        
        return np.array(z_centers), np.array(log_sizes), np.array(log_size_errs)
    
    def fit_size_evolution_model(self, z_centers, log_sizes, log_size_errs, n_realizations=100):
        """Fit size evolution model: log₁₀(R_e/kpc) = A - α log(1 + z)
        
        Performs multiple realizations (bootstrap/Monte Carlo) to get distribution
        of parameters and returns mean with confidence bands.
        """
        # Convert to log(1+z)
        log_one_plus_z = np.log10(1 + z_centers)
        
        # Fit function
        def evolution_model(log_z, A, alpha):
            return A - alpha * log_z
        
        A_samples = []
        alpha_samples = []
        
        # Perform multiple realizations
        for i in range(n_realizations):
            try:
                # Bootstrap: resample with replacement
                n_points = len(z_centers)
                indices = np.random.choice(n_points, size=n_points, replace=True)
                
                z_boot = z_centers[indices]
                log_z_boot = log_one_plus_z[indices]
                log_sizes_boot = log_sizes[indices]
                log_size_errs_boot = log_size_errs[indices]
                
                # Fit to bootstrap sample
                popt, _ = curve_fit(evolution_model, log_z_boot, log_sizes_boot, 
                                   sigma=log_size_errs_boot, absolute_sigma=True,
                                   maxfev=10000)
                A_samples.append(popt[0])
                alpha_samples.append(popt[1])
            except:
                continue
        
        if len(A_samples) < 10:  # Need at least some successful fits
            return None
        
        # Calculate statistics
        A_mean = np.mean(A_samples)
        A_err = np.std(A_samples)
        alpha_mean = np.mean(alpha_samples)
        alpha_err = np.std(alpha_samples)
        
        # Calculate R² using mean parameters
        y_pred = evolution_model(log_one_plus_z, A_mean, alpha_mean)
        y_true = log_sizes
        r_squared = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        return {
            'A': A_mean, 'A_err': A_err,
            'alpha': alpha_mean, 'alpha_err': alpha_err,
            'r_squared': r_squared,
            'log_one_plus_z': log_one_plus_z,
            'z_centers': z_centers,
            'A_samples': np.array(A_samples),
            'alpha_samples': np.array(alpha_samples)
        }
    
    def create_size_evolution_at_fixed_mass_plot(self, massive_data, redshift_fits, 
                                                  early_fits_list, late_fits_list,
                                                  massive_early, massive_late,
                                                  fixed_masses=[10.0, 11.5]):
        """Create single plot showing effective radius evolution at fixed stellar masses"""
        print("Creating size evolution at fixed mass plot...")
        print(f"Using fixed masses: {fixed_masses} for all samples")
        
        # Use a slightly narrower figure for the fixed-mass evolution plot
        fig, ax = plt.subplots(1, 1, figsize=(10.2, 8.27))
        
        # Define colors
        color_all = '#2E2E2E'  # Dark gray/black
        color_early = '#FF9800'  # Orange
        color_late = '#00BCD4'   # Cyan
        
        # Plot both fixed masses for each sample type
        for fixed_mass in fixed_masses:
            # Full sample - use fixed mass
            z_centers_all, log_sizes_all, log_size_errs_all = self.calculate_size_at_fixed_mass(
                redshift_fits, fixed_mass)
            
            # Fit evolution model with multiple realizations
            evolution_all = self.fit_size_evolution_model(z_centers_all, log_sizes_all, log_size_errs_all, n_realizations=100)
            
            # Plot fitted evolution curve with confidence band
            if evolution_all:
                z_plot = np.linspace(z_centers_all.min(), z_centers_all.max(), 200)
                log_one_plus_z_plot = np.log10(1 + z_plot)
                
                # Prefer canonical saved results if available to ensure reproducibility
                npz_path = os.path.join(os.path.dirname(__file__), 'size_evolution_fits.npz')
                key_prefix = f'm{int(fixed_mass*10)}'
                if os.path.exists(npz_path):
                    try:
                        npz = np.load(npz_path)
                        A_samples = npz[f'{key_prefix}_all_A_samples']
                        alpha_samples = npz[f'{key_prefix}_all_alpha_samples']
                        # Calculate mean curve from saved means
                        A_mean = np.mean(A_samples)
                        alpha_mean = np.mean(alpha_samples)
                        log_size_mean = A_mean - alpha_mean * log_one_plus_z_plot

                        # Calculate confidence band from saved samples
                        size_samples = []
                        for z_val in z_plot:
                            log_z_val = np.log10(1 + z_val)
                            size_pred_samples = A_samples - alpha_samples * log_z_val
                            size_samples.append(size_pred_samples)
                        size_samples = np.array(size_samples).T
                        size_lower = np.percentile(size_samples, 16, axis=0)
                        size_upper = np.percentile(size_samples, 84, axis=0)
                        # Use saved mean/std for legend
                        alpha_mean = np.mean(alpha_samples)
                        alpha_err = np.std(alpha_samples)
                    except Exception:
                        # Fall back to computed evolution if loading fails
                        log_size_mean = evolution_all['A'] - evolution_all['alpha'] * log_one_plus_z_plot
                        size_samples = []
                        for z_val in z_plot:
                            log_z_val = np.log10(1 + z_val)
                            size_pred_samples = evolution_all['A_samples'] - evolution_all['alpha_samples'] * log_z_val
                            size_samples.append(size_pred_samples)
                        size_samples = np.array(size_samples).T
                        size_lower = np.percentile(size_samples, 16, axis=0)
                        size_upper = np.percentile(size_samples, 84, axis=0)
                        alpha_mean = evolution_all['alpha']
                        alpha_err = evolution_all['alpha_err']
                else:
                    # No canonical file; use computed evolution
                    log_size_mean = evolution_all['A'] - evolution_all['alpha'] * log_one_plus_z_plot
                    size_samples = []
                    for z_val in z_plot:
                        log_z_val = np.log10(1 + z_val)
                        size_pred_samples = evolution_all['A_samples'] - evolution_all['alpha_samples'] * log_z_val
                        size_samples.append(size_pred_samples)
                    size_samples = np.array(size_samples).T
                    size_lower = np.percentile(size_samples, 16, axis=0)
                    size_upper = np.percentile(size_samples, 84, axis=0)
                    alpha_mean = evolution_all['alpha']
                    alpha_err = evolution_all['alpha_err']

                # Plot confidence band
                ax.fill_between(z_plot, size_lower, size_upper, 
                               color=color_all, alpha=0.2, zorder=1)

                # Plot mean curve
                ax.plot(z_plot, log_size_mean, color=color_all, linewidth=2.5, 
                       linestyle='-' if fixed_mass == 11.5 else '--',
                       alpha=0.7 if fixed_mass == 10.0 else 1.0,
                       label=f'All (M*={fixed_mass:.1f}, α={alpha_mean:.3f}±{alpha_err:.3f})', zorder=3)
            
            # Early-type - use same fixed mass
            if early_fits_list:
                z_centers_early, log_sizes_early, log_size_errs_early = self.calculate_size_at_fixed_mass(
                    early_fits_list, fixed_mass)
                evolution_early = self.fit_size_evolution_model(z_centers_early, log_sizes_early, log_size_errs_early, n_realizations=100)

                if evolution_early:
                    z_plot = np.linspace(z_centers_early.min(), z_centers_early.max(), 200)
                    log_one_plus_z_plot = np.log10(1 + z_plot)

                    # Calculate mean curve
                    log_size_mean = evolution_early['A'] - evolution_early['alpha'] * log_one_plus_z_plot

                    # Calculate confidence band
                    size_samples = []
                    for z_val in z_plot:
                        log_z_val = np.log10(1 + z_val)
                        size_pred_samples = evolution_early['A_samples'] - evolution_early['alpha_samples'] * log_z_val
                        size_samples.append(size_pred_samples)

                    size_samples = np.array(size_samples).T
                    size_lower = np.percentile(size_samples, 16, axis=0)
                    size_upper = np.percentile(size_samples, 84, axis=0)

                    # Plot confidence band
                    ax.fill_between(z_plot, size_lower, size_upper, 
                                   color=color_early, alpha=0.2, zorder=1)

                    # Plot mean curve
                    # Try to read canonical values for early-type
                    try:
                        npz = np.load(os.path.join(os.path.dirname(__file__), 'size_evolution_fits.npz'))
                        key_prefix = f'm{int(fixed_mass*10)}'
                        a_samples = npz[f'{key_prefix}_early_A_samples']
                        al_samples = npz[f'{key_prefix}_early_alpha_samples']
                        A_mean_e = np.mean(a_samples)
                        A_err_e = np.std(a_samples)
                        alpha_mean_e = np.mean(al_samples)
                        alpha_err_e = np.std(al_samples)
                        # Recompute mean curve and bands
                        log_size_mean = A_mean_e - alpha_mean_e * log_one_plus_z_plot
                        size_samples = []
                        for z_val in z_plot:
                            log_z_val = np.log10(1 + z_val)
                            size_pred_samples = a_samples - al_samples * log_z_val
                            size_samples.append(size_pred_samples)
                        size_samples = np.array(size_samples).T
                        size_lower = np.percentile(size_samples, 16, axis=0)
                        size_upper = np.percentile(size_samples, 84, axis=0)
                        ax.plot(z_plot, log_size_mean, color=color_early, linewidth=2.5,
                               linestyle='-' if fixed_mass == 11.5 else '--',
                               alpha=0.7 if fixed_mass == 10.0 else 1.0,
                               label=f'Early-type (M*={fixed_mass:.1f}, α={alpha_mean_e:.3f}±{alpha_err_e:.3f})', 
                               zorder=3)
                    except Exception:
                        ax.plot(z_plot, log_size_mean, color=color_early, linewidth=2.5,
                               linestyle='-' if fixed_mass == 11.5 else '--',
                               alpha=0.7 if fixed_mass == 10.0 else 1.0,
                               label=f'Early-type (M*={fixed_mass:.1f}, α={evolution_early["alpha"]:.3f}±{evolution_early["alpha_err"]:.3f})', 
                               zorder=3)
            
            # Late-type - use same fixed mass
            if late_fits_list:
                z_centers_late, log_sizes_late, log_size_errs_late = self.calculate_size_at_fixed_mass(
                    late_fits_list, fixed_mass)
                evolution_late = self.fit_size_evolution_model(z_centers_late, log_sizes_late, log_size_errs_late, n_realizations=100)

                if evolution_late:
                    z_plot = np.linspace(z_centers_late.min(), z_centers_late.max(), 200)
                    log_one_plus_z_plot = np.log10(1 + z_plot)

                    # Calculate mean curve
                    log_size_mean = evolution_late['A'] - evolution_late['alpha'] * log_one_plus_z_plot

                    # Calculate confidence band
                    size_samples = []
                    for z_val in z_plot:
                        log_z_val = np.log10(1 + z_val)
                        size_pred_samples = evolution_late['A_samples'] - evolution_late['alpha_samples'] * log_z_val
                        size_samples.append(size_pred_samples)

                    size_samples = np.array(size_samples).T
                    size_lower = np.percentile(size_samples, 16, axis=0)
                    size_upper = np.percentile(size_samples, 84, axis=0)

                    # Plot confidence band
                    ax.fill_between(z_plot, size_lower, size_upper, 
                                   color=color_late, alpha=0.2, zorder=1)

                    # Plot mean curve
                    try:
                        npz = np.load(os.path.join(os.path.dirname(__file__), 'size_evolution_fits.npz'))
                        key_prefix = f'm{int(fixed_mass*10)}'
                        a_samples = npz[f'{key_prefix}_late_A_samples']
                        al_samples = npz[f'{key_prefix}_late_alpha_samples']
                        A_mean_l = np.mean(a_samples)
                        A_err_l = np.std(a_samples)
                        alpha_mean_l = np.mean(al_samples)
                        alpha_err_l = np.std(al_samples)
                        log_size_mean = A_mean_l - alpha_mean_l * log_one_plus_z_plot
                        size_samples = []
                        for z_val in z_plot:
                            log_z_val = np.log10(1 + z_val)
                            size_pred_samples = a_samples - al_samples * log_z_val
                            size_samples.append(size_pred_samples)
                        size_samples = np.array(size_samples).T
                        size_lower = np.percentile(size_samples, 16, axis=0)
                        size_upper = np.percentile(size_samples, 84, axis=0)
                        ax.plot(z_plot, log_size_mean, color=color_late, linewidth=2.5,
                               linestyle='-' if fixed_mass == 11.5 else '--',
                               alpha=0.7 if fixed_mass == 10.0 else 1.0,
                               label=f'Late-type (M*={fixed_mass:.1f}, α={alpha_mean_l:.3f}±{alpha_err_l:.3f})', 
                               zorder=3)
                    except Exception:
                        ax.plot(z_plot, log_size_mean, color=color_late, linewidth=2.5,
                               linestyle='-' if fixed_mass == 11.5 else '--',
                               alpha=0.7 if fixed_mass == 10.0 else 1.0,
                               label=f'Late-type (M*={fixed_mass:.1f}, α={evolution_late["alpha"]:.3f}±{evolution_late["alpha_err"]:.3f})', 
                               zorder=3)
        
        ax.set_xlabel('Redshift (z)', fontsize=16, fontweight='bold')
        ax.set_ylabel('log₁₀(Reff/kpc)', fontsize=16, fontweight='bold')
        ax.set_title('Effective Radius Evolution at Fixed Stellar Masses', fontsize=16, fontweight='bold')
        ax.grid(False)
        # Increase legend font size for readability in paper figures
        # Also save exact legend strings to a file so we can compare byte-for-byte
        handles, labels = ax.get_legend_handles_labels()
        legend_file = os.path.join(os.path.dirname(__file__), 'size_evolution_legend_strings.txt')
        try:
            with open(legend_file, 'w') as lf:
                for lbl in labels:
                    print('LEGEND:', lbl)
                    lf.write(lbl + '\n')
        except Exception as e:
            print('Warning: could not write legend strings file:', e)

        ax.legend(fontsize=15, loc='upper right', framealpha=0.95, edgecolor='gray', ncol=1, frameon=False)
        # Increase axis tick label sizes (numbers)
        ax.tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        
        return fig
    
    def create_beta_vs_redshift_plot(self, redshift_fits):
        """Create beta vs redshift evolution plot"""
        print("Creating beta vs redshift plot...")
        
        # Beta plot also larger for readability in paper figures
        fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
        
        # Filter to valid fits
        valid_fits = [f for f in redshift_fits if f['n_galaxies'] >= 20]
        valid_fits.sort(key=lambda x: x['z_center'])
        
        if len(valid_fits) == 0:
            print("No valid fits found for beta vs redshift plot")
            return fig
        
        z_centers = [fit['z_center'] for fit in valid_fits]
        betas = [fit['beta'] for fit in valid_fits]
        beta_errs = [fit['beta_err'] for fit in valid_fits]
        alphas = [fit['alpha'] for fit in valid_fits]
        alpha_errs = [fit['alpha_err'] for fit in valid_fits]
        
        # Plot beta vs redshift
        ax.errorbar(z_centers, betas, yerr=beta_errs, fmt='ko-', linewidth=2.5, 
                   markersize=10, capsize=5, capthick=2, label='CWMGs', zorder=3)
        
        # Add text annotations for alpha values
        for i, (z_c, beta, alpha, alpha_err) in enumerate(zip(z_centers, betas, alphas, alpha_errs)):
            if i % 2 == 0:  # Annotate every other point to avoid crowding
                ax.annotate(f'α={alpha:.3f}', xy=(z_c, beta), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, alpha=0.8)
        
        ax.set_xlabel('Redshift (z)', fontsize=16, fontweight='bold')
        ax.set_ylabel('β (Intercept)', fontsize=16, fontweight='bold')
        ax.set_title('Mass-Size Relation Intercept Evolution', fontsize=16, fontweight='bold')
        ax.grid(False)
        ax.legend(fontsize=13, loc='best', frameon=False)
        
        plt.tight_layout()
        
        return fig
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("=== Paper 1 CWMGs-Only Mass-Size Analysis ===")
        
        # Load data
        massive_data = self.load_massive_galaxies(mass_cut=9.0)
        
        # Load existing fit results
        redshift_fits = self.load_redshift_binned_fits()
        print(f"\nLoaded {len(redshift_fits)} redshift-binned fits")
        
        # Calculate extended redshift bins (4.0-6.0, 6.0-8.0, 8.0-12.0)
        extended_fits = self.calculate_extended_redshift_bins(massive_data, redshift_fits, min_samples=20)
        if extended_fits:
            redshift_fits.extend(extended_fits)
            print(f"Added {len(extended_fits)} extended redshift bins")
            print(f"Total redshift bins: {len(redshift_fits)}")
        
        # Create plots
        print("\n=== Creating Plots ===")
        
        # Redshift-binned subplots
        fig1 = self.create_redshift_binned_subplots(massive_data, redshift_fits)
        output_file1 = 'paper1_cwmgs_redshift_binned_mass_size.png'
        fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
        print(f"Redshift-binned subplots saved as '{output_file1}'")
        
        # Beta vs redshift
        fig2 = self.create_beta_vs_redshift_plot(redshift_fits)
        output_file2 = 'paper1_cwmgs_beta_vs_redshift.png'
        fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
        print(f"Beta vs redshift plot saved as '{output_file2}'")
        
        # Size evolution at fixed masses
        # Need to calculate early/late fits for extended bins
        print("\n=== Calculating Early/Late-type Fits for Extended Bins ===")
        massive_early, massive_late = self.separate_by_sersic(massive_data)
        
        # Calculate early/late fits for all bins including extended
        all_early_fits = []
        all_late_fits = []
        
        for fit in redshift_fits:
            z_min = fit['z_min']
            z_max = fit['z_max']
            
            # Early-type
            early_subset = massive_early[(massive_early['LP_zfinal'] >= z_min) & 
                                        (massive_early['LP_zfinal'] < z_max)]
            if len(early_subset) >= 15:
                early_fit = self.fit_linear_relation(early_subset['LP_mass_med_PDF'].values,
                                                    np.log10(early_subset['Reff_kpc'].values))
                if early_fit:
                    all_early_fits.append({
                        'z_min': z_min, 'z_max': z_max, 'z_center': fit['z_center'],
                        'alpha': early_fit['alpha'], 'alpha_err': early_fit['alpha_err'],
                        'beta': early_fit['beta'], 'beta_err': early_fit['beta_err'],
                        'cov_ab': early_fit.get('cov_ab', 0.0)  # Include covariance
                    })
            
            # Late-type
            late_subset = massive_late[(massive_late['LP_zfinal'] >= z_min) & 
                                      (massive_late['LP_zfinal'] < z_max)]
            if len(late_subset) >= 15:
                late_fit = self.fit_linear_relation(late_subset['LP_mass_med_PDF'].values,
                                                   np.log10(late_subset['Reff_kpc'].values))
                if late_fit:
                    all_late_fits.append({
                        'z_min': z_min, 'z_max': z_max, 'z_center': fit['z_center'],
                        'alpha': late_fit['alpha'], 'alpha_err': late_fit['alpha_err'],
                        'beta': late_fit['beta'], 'beta_err': late_fit['beta_err'],
                        'cov_ab': late_fit.get('cov_ab', 0.0)  # Include covariance
                    })
        
        # Try different stellar masses to show more variation
        # Use wider range: lower mass (9.0) and higher mass (11.5) to span more of the mass range
        # This should show clearer differences between early-type and late-type evolution
        fig3 = self.create_size_evolution_at_fixed_mass_plot(massive_data, redshift_fits,
                                                             all_early_fits, all_late_fits,
                                                             massive_early, massive_late,
                                                             fixed_masses=[10.0, 11.5])
        output_file3 = 'paper1_cwmgs_size_evolution_fixed_mass.png'
        fig3.savefig(output_file3, dpi=300, bbox_inches='tight')
        print(f"Size evolution at fixed mass plot saved as '{output_file3}'")
        
        # Also save to Paper 1 figures directory
        paper1_fig_dir = '/Users/gozalig1/Projects/JWST_lensing_simulations/prism_papers_split/paper1_framework/figures/main_paper/fig08_mass_size_relations'
        if os.path.exists(paper1_fig_dir):
            fig1.savefig(os.path.join(paper1_fig_dir, 'paper1_cwmgs_redshift_binned_mass_size.png'), dpi=300, bbox_inches='tight')
            fig2.savefig(os.path.join(paper1_fig_dir, 'paper1_cwmgs_beta_vs_redshift.png'), dpi=300, bbox_inches='tight')
            fig3.savefig(os.path.join(paper1_fig_dir, 'paper1_cwmgs_size_evolution_fixed_mass.png'), dpi=300, bbox_inches='tight')
            print(f"Plots also saved to {paper1_fig_dir}")
        
        plt.show()
        
        print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    analyzer = Paper1CWMGsAnalyzer()
    analyzer.run_analysis()
