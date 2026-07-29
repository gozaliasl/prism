#!/usr/bin/env python3
import runpy, numpy as np
m = runpy.run_path('analysis/mass_size_relations/create_paper1_cwmgs_only_plots.py')
Analyzer = m['Paper1CWMGsAnalyzer']
A = Analyzer()
# Prepare data
massive = A.load_massive_galaxies()
early_df, late_df = A.separate_by_sersic(massive)
fits = A.load_redshift_binned_fits()
# Build early/late fits lists
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
# Helper: calculate size evolution for given fitlist and fixed_mass using 100 realizations
from math import isnan

def evol_for_fitlist(fitlist, fixed_mass):
    if len(fitlist) == 0:
        return None
    zc, ls, lse = A.calculate_size_at_fixed_mass(fitlist, fixed_mass)
    if len(zc) < 5:
        return None
    res = A.fit_size_evolution_model(zc, ls, lse, n_realizations=100)
    return res

fixed_masses = [10.0, 11.5]
results = {}
for fm in fixed_masses:
    res_all = evol_for_fitlist(fits, fm)
    res_early = evol_for_fitlist(all_early_fits, fm)
    res_late = evol_for_fitlist(all_late_fits, fm)
    results[fm] = {'all': res_all, 'early': res_early, 'late': res_late}

# Save canonical results to a compressed numpy file for reproducibility and plotting
out_file_npz = 'analysis/mass_size_relations/size_evolution_fits.npz'
save_dict = {}
for fm, d in results.items():
    key_prefix = f'm{int(fm*10)}'  # 10.0 -> m100, 11.5 -> m115
    for k, v in d.items():
        if v is None:
            continue
        save_dict[f'{key_prefix}_{k}_A'] = np.array([v['A']])
        save_dict[f'{key_prefix}_{k}_A_err'] = np.array([v['A_err']])
        save_dict[f'{key_prefix}_{k}_alpha'] = np.array([v['alpha']])
        save_dict[f'{key_prefix}_{k}_alpha_err'] = np.array([v['alpha_err']])
        # Save sample arrays if present
        if 'A_samples' in v:
            save_dict[f'{key_prefix}_{k}_A_samples'] = v['A_samples']
        if 'alpha_samples' in v:
            save_dict[f'{key_prefix}_{k}_alpha_samples'] = v['alpha_samples']

try:
    np.savez_compressed(out_file_npz, **save_dict)
    print('Saved canonical size-evolution fits to', out_file_npz)
except Exception as e:
    print('Warning: could not save canonical fits:', e)

# Print in a concise, parseable format (also write LaTeX rows to a file)
latex_file = 'analysis/mass_size_relations/size_evolution_table_rows.tex'
with open(latex_file, 'w') as lf:
    for fm, d in results.items():
        print(f"FIXED_MASS {fm}")
        for k, v in d.items():
            if v is None:
                print(f"{k.upper()} INSUFFICIENT")
            else:
                print(f"{k.upper()} A {v['A']:.6f} Aerr {v['A_err']:.6f} alpha {v['alpha']:.6f} alpha_err {v['alpha_err']:.6f}")
                # Write a LaTeX table row using fixed mass label formatting
                mass_label = f"{fm:.2f}" if (fm != int(fm)) else f"{fm:.1f}"
                sample_label = 'CWMGs' if k == 'all' else ('CWMGs (Early-type)' if k == 'early' else 'CWMGs (Late-type)')
                lf.write(f"{sample_label} & ${v['A']:.3f} \\pm {v['A_err']:.3f}$ & ${v['alpha']:.3f} \\pm {v['alpha_err']:.3f}$ & {mass_label} \\\\n+")

# Also print median masses for reference
print('MEDIANS')
print('ALL_MEDIAN', np.median(massive['LP_mass_med_PDF'].values))
print('EARLY_MEDIAN', np.median(early_df['LP_mass_med_PDF'].values))
print('LATE_MEDIAN', np.median(late_df['LP_mass_med_PDF'].values))

print('Wrote LaTeX table rows to', latex_file)
