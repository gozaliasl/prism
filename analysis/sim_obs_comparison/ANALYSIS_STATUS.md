# Simulation vs. Observation Comparison - Analysis Initialized

**Status**: Framework complete and initial pipeline executed  
**Date**: 2026-02-13  
**Location**: `/Users/gozalig1/Projects/jwst-mock-lens-simulator/analysis/sim_obs_comparison/`

---

## What's Been Created

### 1. **Comprehensive Framework** (`FRAMEWORK.md`)
- Detailed strategy for comparing simulated and observed lensing systems
- Motivating context: ML models trained on simulated + observed lenses significantly outperform simulated-only models
- This indicates simulation gaps that need to be identified and fixed

### 2. **Directory Structure**
```
analysis/sim_obs_comparison/
├── FRAMEWORK.md                          # Strategic planning document
├── catalogs/
│   ├── real_lens_properties.csv          # ✅ GENERATED (435 real lenses)
│   ├── matched_lens_pairs.csv            # ✅ GENERATED (1305 match records)
│   └── simulated_lens_properties.csv     # (Future: extracted from .npz)
├── scripts/
│   ├── extract_real_lens_properties.py   # ✅ EXECUTED (435 lenses processed)
│   ├── match_sim_to_obs.py               # ✅ EXECUTED (matched all pairs)
│   ├── visualize_comparisons.py          # Ready to run
│   ├── compute_morphology_metrics.py     # Stub template
│   ├── compute_flux_metrics.py           # Stub template
│   ├── generate_report.py                # Stub template
│   └── run_analysis.sh                   # Master orchestration script
├── visualizations/                       # (Generated comparison images)
├── metrics/                              # (Future: computed statistics)
└── reports/                              # (Future: findings documents)
```

---

## Data Generated

### `real_lens_properties.csv` (435 real lenses)
Extracted from FITS files in `/data/real_lenses/`:
- **Columns**: name, n_bands_available, per-band flux/max/mean/std for F115W/F150W/F277W/F444W
- **Example**: COSJ095846+020304 has 4 bands with measurable statistics

### `matched_lens_pairs.csv` (1305 match records)
Paired real lenses with best-matching simulated lenses:
- **Columns**: match_rank, real_name, sim_lens_id, sim_filename_base, match_score, theta_E/z_lens/z_source/mag_sim
- **Strategy**: Ranked by combined parameter distance (θ_E, z_lens, z_source, magnitude)
- **Interpretation**: Top 3 matches per real lens for potential visual inspection

---

## What This Enables

### Immediate Capabilities

1. **Visual Comparison**: Generate side-by-side figures of matched real vs simulated systems
   ```bash
   python analysis/sim_obs_comparison/scripts/visualize_comparisons.py --max-pairs 20 --stretch log
   ```

2. **Quantitative Assessment**: Compute morphological/flux metrics once image processing libraries available
   - Sersic profile fitting → compare lens structure
   - SNR & background statistics → identify noise model gaps
   - Flux distribution → validate photometry calibration

3. **Feature Analysis**: Inspect for:
   - Over-smoothness in simulated arcs
   - PSF artifact realism
   - Morphological diversity
   - Background field complexity

### Hypothesized Gaps (To Test)

Based on ML performance difference:
1. Simulated profiles too idealized (Sérsic math not real enough)
2. Noise texture unrealistic (Poisson-only vs real JWST noise)
3. PSF effects incomplete (diffraction spikes, core shape)
4. Background galaxies too sparse/regular
5. Photometry systematic offsets

---

## Next Steps (Prioritized)

### Phase 1: Visual Inspection (High Priority)
- [ ] Generate visualization grid for 20-30 top matched pairs
- [ ] Manually inspect for obvious differences
- [ ] Document qualitative observations (smoothness, artefacts, complexity)

### Phase 2: Quantitative Metrics (Medium Priority)
- [ ] Implement `compute_morphology_metrics.py` (Sersic fitting)
- [ ] Implement `compute_flux_metrics.py` (SNR, colors)
- [ ] Generate statistical comparison plots

### Phase 3: Root Cause Analysis (High Priority)
- [ ] For each gap identified, trace back to simulator code
- [ ] Propose targeted improvements (Sérsic realism, noise model, PSF)
- [ ] Estimate impact of each improvement on ML performance

### Phase 4: Improvement Implementation (Ongoing)
- [ ] Prioritize highest-impact changes
- [ ] Implement in `src/jwst_lens_simulator.py`
- [ ] Re-run comparison to validate improvement
- [ ] Retrain ML models to confirm performance gain

---

## Files Ready to Inspect

**Quick Start Analysis**:
1. View extracted real lens statistics:
   ```bash
   head -20 analysis/sim_obs_comparison/catalogs/real_lens_properties.csv
   ```

2. View matched pairs:
   ```bash
   head -30 analysis/sim_obs_comparison/catalogs/matched_lens_pairs.csv
   ```

3. Generate first 10 visual comparisons:
   ```bash
   cd /Users/gozalig1/Projects/jwst-mock-lens-simulator
   python analysis/sim_obs_comparison/scripts/visualize_comparisons.py --max-pairs 10 --stretch log
   ```

---

## Key Insight

The framework systematically maps **simulated system properties** (Einstein radius, redshift, magnitudes, pixel data) to **observed system properties** (FITS photometry, extracted statistics), enabling:
- Direct visual comparison of "equivalent" systems
- Quantitative metric comparison (morphology, noise, flux)
- Identification of simulation gaps
- Prioritized improvement roadmap

This is a repeatable cycle: **Compare → Identify Gaps → Improve → Retrain ML → Validate**.

---

## Contact & Documentation

- **Framework Document**: [FRAMEWORK.md](FRAMEWORK.md)
- **Analysis Scripts**: `scripts/`
- **Catalogs**: `catalogs/` (auto-generated CSVs)
- **Outputs**: `visualizations/`, `metrics/`, `reports/`

All code is self-documenting with inline comments explaining comparison strategy and output interpretation.
