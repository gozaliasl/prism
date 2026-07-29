# Simulation vs. Observation Comparison Framework

## Objective
Systematically compare simulated strong lensing systems with real JWST-observed lenses to identify gaps in simulation fidelity and recommend improvements.

**Key Finding**: ML models trained on simulated + observed lenses significantly outperform models trained on simulated lenses alone. This indicates that simulations are missing or under-representing specific features crucial for robust lens detection.

## Structure

```
analysis/sim_obs_comparison/
├── FRAMEWORK.md (this file)
├── catalogs/
│   ├── simulated_lens_properties.csv          # Extracted from training output
│   ├── real_lens_properties.csv               # Extracted from FITS headers / manual
│   └── matched_lens_pairs.csv                 # Simulated vs observed matches
├── visualizations/
│   ├── side_by_side_COSJ*.png                 # Sim vs Real for each lens
│   └── comparison_grid.png                    # Gallery of best matches
├── metrics/
│   ├── morphology_statistics.csv              # Sersic profile, axis ratio, etc.
│   ├── flux_statistics.csv                    # Brightness, SNR per band
│   ├── noise_characteristics.csv              # Background RMS, PSF contamination
│   └── feature_analysis.csv                   # Artefacts, symmetry, sharpness
├── scripts/
│   ├── extract_real_lens_properties.py        # Parse FITS, catalog metadata
│   ├── match_sim_to_obs.py                    # Pair lenses by redshift, theta_E
│   ├── compute_morphology_metrics.py          # Sersic fit, axis ratios, sizes
│   ├── compute_flux_metrics.py                # SNR, background, per-band stats
│   ├── visualize_comparisons.py               # Side-by-side figures
│   └── generate_report.py                     # Summary & recommendations
└── reports/
    ├── summary_findings.md                    # Key differences & gaps
    └── improvement_recommendations.md         # Actionable next steps
```

## Comparison Strategy

### 1. Real Lens Properties Extraction
**Source**: `/data/real_lenses/*.fits` + manual catalog
- Extract FITS headers: photometry, astrometry, redshifts
- Parse filename convention: `COSJ[RA]+[DEC]_[BAND].fits`
- Collect: θ_E, z_lens, z_source, magnitudes, morphology

### 2. Simulated Lens Properties
**Source**: `outputs/custom_20260213_155632/cosmos_lens_training_catalog.csv`
- Already available: θ_E, z_lens, z_source, magnitudes, fluxes
- Available in .npz files: raw pixel data for morphology fitting

### 3. Matching Strategy
Match simulated lenses to observed lenses by:
- **Primary**: Einstein radius θ_E (±20% tolerance)
- **Secondary**: z_lens, z_source (±0.1)
- **Tertiary**: magnitudes in F150W (±1 mag)

This creates "equivalent systems" for direct comparison.

### 4. Quantitative Metrics

#### A. Morphology (from image pixels)
- **Sersic index**: How smooth/concentrated the lens light?
- **Axis ratio**: How elongated?
- **Size (effective radius)**: PSF convolution effects?
- **Symmetry**: Real lenses have asymmetries from environment?

#### B. Flux & Brightness
- **Total flux per band**: Magnitude match?
- **Peak brightness**: Dynamic range?
- **Signal-to-noise ratio**: Implicit in flux levels?
- **Magnitude dispersion**: Colors realistic?

#### C. Noise & Background Characteristics
- **Background RMS**: Sky subtraction accurate?
- **PSF effects**: Are simulated PSFs too perfect?
- **Artefacts**: Spikes, halos, diffraction patterns?
- **Detector artifacts**: Hot pixels, cosmic rays in simulation?

#### D. Lens System Features
- **Einstein radius measured from image**: How well-matched?
- **Arc morphology**: Smooth or clumpy?
- **Multiple images**: Are all clearly separated?
- **Central image (for quasar lensing)**: Detectable?

### 5. Qualitative Assessment

Inspect pairs visually for:
- **Over-smoothness**: Are simulated arcs too perfect/mathematical?
- **Over-brightness**: Are simulated lenses brighter than observed?
- **Missing substructure**: Does simulation capture galaxy morphology nuances?
- **Background galaxies**: Are field galaxies realistic enough?
- **Noise texture**: Is Poisson noise accurate? Any missing realistic noise?
- **PSF artifacts**: Simulated PSFs too perfect?

## Expected Findings (Hypotheses)

Based on ML performance gap:

1. **Simulation may under-represent morphological diversity**
   - Real lenses have dustier, more complex structures
   - Simulated Sérsic profiles too idealized

2. **Noise texture may be unrealistic**
   - Real JWST images have specific noise patterns (readout, cosmic rays)
   - Simulation uses pure Poisson + Gaussian

3. **PSF effects may be incomplete**
   - Real JWST PSF has subtle features (diffraction spikes, core/wings mismatch)
   - Simulated PSF may not capture all effects

4. **Background field galaxies may need refinement**
   - Real fields have more complexity (overlapping galaxies, low-surface-brightness)
   - Simulated fields may be too sparse or too structured

5. **Photometry calibration**
   - Real vs simulated magnitudes may have systematic offsets
   - Color distributions might differ

## Improvement Recommendations (To Be Developed)

Once differences identified, prioritize improvements:

- [ ] More realistic Sérsic profiles (fit from real galaxy surveys)
- [ ] Better noise model (study real JWST backgrounds)
- [ ] Improved PSF (use actual JWST PSF kernels)
- [ ] Richer background field population
- [ ] Add subtle artefacts (cosmic rays, diffraction)
- [ ] Refine color calibration

## Next Steps

1. Extract real lens properties from FITS + manual catalog
2. Match simulated lenses to observed equivalents
3. Compute morphology metrics (Sersic fitting)
4. Compute flux metrics and noise characteristics
5. Generate visual comparisons (5×5 grid per system)
6. Write summary findings document
7. Prioritize improvements based on impact

---

**Owner**: User  
**Start Date**: 2026-02-13  
**Status**: Framework defined, implementation pending
