# Mass-Size Relations Analysis

This directory contains the final analysis of galaxy mass-size relations for COWLS lenses and COSMOS WEB Massive Galaxies (CWMGs).

## Final Results

### Main Analysis Script
- **`consolidated_analysis.py`** - Main analysis script that combines COWLS and CWMGs data
  - Creates two final plots with redshift color-coding and morphological symbols
  - Uses PyMC for robust Bayesian fitting
  - Combines highest redshift bins for better statistics

### Final Plots
- **`consolidated_mass_size_relations.png`** - Combined mass-size relations plot
  - COWLS data points (circles/squares) color-coded by redshift
  - CWMGs background data (small points) 
  - All fit lines with confidence intervals
  - Two legends: symbols (upper right) and fits (upper left)
  - Horizontal colorbar for redshift scale

- **`redshift_evolution_consolidated.png`** - Redshift evolution of mass-size slopes
  - All 6 trends: COWLS, COWLS ET/LT, CWMGs, CWMGs ET/LT
  - Shows how correlations weaken with redshift
  - Clear morphological differences

## Individual Analysis Scripts (Reference)

### COWLS Analysis
- **`cowls_mass_size_analysis.py`** - Individual COWLS lens analysis
- **`cowls_fit_results.csv`** - COWLS fit parameters and statistics
- **`cowls_processed_catalog.csv`** - Processed COWLS lens catalog
- **`cowls_redshift_binned_relations.csv`** - COWLS redshift-binned fit results
- **`cowls_size_evolution.csv`** - COWLS size evolution data
- **`cowls_structural_type_relations.csv`** - COWLS structural type statistics

### Massive Galaxy Analysis  
- **`massive_galaxy_analysis.py`** - Individual CWMGs analysis
- **`massive_galaxy_catalog.csv`** - Processed massive galaxy catalog
- **`massive_galaxy_fit_results.csv`** - CWMGs fit parameters and statistics
- **`massive_galaxy_redshift_binned_relations.csv`** - CWMGs redshift-binned fit results

## Key Scientific Results

### Mass-Size Relations
- **Late-type galaxies** show stronger correlations than early-type
- **COWLS lenses** follow similar trends to CWMGs
- **CWMGs Late-type** shows strongest correlation (α = 0.190±0.007)

### Redshift Evolution
- **Strongest correlations at low-z** (z < 1) for all types
- **Late-type galaxies** maintain stronger correlations across redshifts
- **Early-type galaxies** show weaker correlations that break down at high-z
- **COWLS and CWMGs** show consistent evolutionary patterns

### Morphological Differences
- **Early-type** (ellipticals/lenticulars): Weaker mass-size coupling
- **Late-type** (spirals/irregulars): Stronger disk scaling relations
- **Redshift evolution** affects different morphologies differently

## Usage

To reproduce the analysis:
```bash
python consolidated_analysis.py
```

This will generate both final plots and all intermediate data files.
