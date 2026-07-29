#!/usr/bin/env python3
import runpy
import numpy as np
m = runpy.run_path('analysis/mass_size_relations/create_paper1_cwmgs_only_plots.py')
Analyzer = m['Paper1CWMGsAnalyzer']
A = Analyzer()
massive = A.load_massive_galaxies()
early_df, late_df = A.separate_by_sersic(massive)
# load fits (includes extended bins)
fits = A.load_redshift_binned_fits()
# build early/late fits lists similar to the script
all_early_fits = []
all_late_fits = []
for fit in fits:
    zmin = fit['z_min']; zmax = fit['z_max']
    early_subset = early_df[(early_df['LP_zfinal'] >= zmin) & (early_df['LP_zfinal'] < zmax)]
    if len(early_subset) >= 15:
        ef = A.fit_linear_relation(early_subset['LP_mass_med_PDF'].values, np.log10(early_subset['Reff_kpc'].values))
        if ef:
            all_early_fits.append({'z_min':zmin,'z_max':zmax,'z_center':fit['z_center'],'alpha':ef['alpha'],'alpha_err':ef['alpha_err'],'beta':ef['beta'],'beta_err':ef['beta_err']})
    late_subset = late_df[(late_df['LP_zfinal'] >= zmin) & (late_df['LP_zfinal'] < zmax)]
    if len(late_subset) >= 15:
        lf = A.fit_linear_relation(late_subset['LP_mass_med_PDF'].values, np.log10(late_subset['Reff_kpc'].values))
        if lf:
            all_late_fits.append({'z_min':zmin,'z_max':zmax,'z_center':fit['z_center'],'alpha':lf['alpha'],'alpha_err':lf['alpha_err'],'beta':lf['beta'],'beta_err':lf['beta_err']})
# Full-sample fits
full_fit = A.fit_linear_relation(massive['LP_mass_med_PDF'].values, np.log10(massive['Reff_kpc'].values))
early_full = A.fit_linear_relation(early_df['LP_mass_med_PDF'].values, np.log10(early_df['Reff_kpc'].values))
late_full = A.fit_linear_relation(late_df['LP_mass_med_PDF'].values, np.log10(late_df['Reff_kpc'].values))
print('%% Full sample fits')
print('CWMGs (full) & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & 0.1--10.0 \\\n'.format(full_fit['alpha'], full_fit['alpha_err'], full_fit['beta'], full_fit['beta_err']))
print('CWMGs Early-type (full) & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & 0.1--10.0 \\\n'.format(early_full['alpha'], early_full['alpha_err'], early_full['beta'], early_full['beta_err']))
print('CWMGs Late-type (full) & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & 0.1--10.0 \\\n'.format(late_full['alpha'], late_full['alpha_err'], late_full['beta'], late_full['beta_err']))
# Redshift bins: print full, early, late when available
print('%% Redshift bins (full / early / late where available)')
for fit in fits:
    zmin,zmax = fit['z_min'], fit['z_max']
    print('CWMGs & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.1f}--{:.1f} \\\n'.format(fit['alpha'], fit['alpha_err'], fit['beta'], fit['beta_err'], zmin, zmax))
    ef = next((x for x in all_early_fits if x['z_center']==fit['z_center']), None)
    if ef:
        print('CWMGs Early-type & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.1f}--{:.1f} \\\n'.format(ef['alpha'],ef['alpha_err'],ef['beta'],ef['beta_err'], zmin, zmax))
    lf = next((x for x in all_late_fits if x['z_center']==fit['z_center']), None)
    if lf:
        print('CWMGs Late-type & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.1f}--{:.1f} \\\n'.format(lf['alpha'],lf['alpha_err'],lf['beta'],lf['beta_err'], zmin, zmax))
# Size evolution at fixed masses 10.0 and 11.5
fixed_masses = [10.0, 11.5]
# helper to get evolution results for a list of fits
def evol_for_fitlist(fitlist, fixed_mass):
    if len(fitlist)==0:
        return None
    zc, ls, lse = A.calculate_size_at_fixed_mass(fitlist, fixed_mass)
    if len(zc)<5:
        return None
    res = A.fit_size_evolution_model(zc, ls, lse, n_realizations=200)
    return res
for fm in fixed_masses:
    res_all = evol_for_fitlist(fits, fm)
    res_early = evol_for_fitlist(all_early_fits, fm)
    res_late = evol_for_fitlist(all_late_fits, fm)
    print('\n%% Fixed mass = {:.1f}'.format(fm))
    if res_all:
        print('All: A={:.3f} ± {:.3f}, alpha={:.3f} ± {:.3f}'.format(res_all['A'], res_all['A_err'], res_all['alpha'], res_all['alpha_err']))
    else:
        print('All: insufficient data')
    if res_early:
        print('Early: A={:.3f} ± {:.3f}, alpha={:.3f} ± {:.3f}'.format(res_early['A'], res_early['A_err'], res_early['alpha'], res_early['alpha_err']))
    else:
        print('Early: insufficient data')
    if res_late:
        print('Late: A={:.3f} ± {:.3f}, alpha={:.3f} ± {:.3f}'.format(res_late['A'], res_late['A_err'], res_late['alpha'], res_late['alpha_err']))
    else:
        print('Late: insufficient data')
# Print median masses
print('\n%% Median log masses:')
print('All median logM = {:.2f}'.format(np.median(massive['LP_mass_med_PDF'].values)))
print('Early median logM = {:.2f}'.format(np.median(early_df['LP_mass_med_PDF'].values)))
print('Late median logM = {:.2f}'.format(np.median(late_df['LP_mass_med_PDF'].values)))
