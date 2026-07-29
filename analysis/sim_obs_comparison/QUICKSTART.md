# Quick Start: Sim vs Obs Comparison

Complete pipeline for comparing simulated vs observed strong lenses to identify simulation improvements.

## What's Ready Now (Data Extracted)

✅ **435 real lenses** cataloged from FITS files  
✅ **1,305 matched pairs** (sim→obs) based on physical parameters  
✅ **Framework document** explaining comparison strategy  

## Run Visualizations (5 min)

```bash
cd /Users/gozalig1/Projects/jwst-mock-lens-simulator

# Generate comparison figures for first 10 matched pairs
python analysis/sim_obs_comparison/scripts/visualize_comparisons.py \
  --max-pairs 10 \
  --stretch log

# View results
open analysis/sim_obs_comparison/visualizations/
```

**What to Look For:**
- Are simulated lenses too smooth/perfect?
- Do backgrounds have realistic structure?
- Are arcs crisp vs fuzzy in reality?
- What morphological details are missing?

## Inspect Matched Catalogs

```bash
# See real lens flux statistics
head -20 analysis/sim_obs_comparison/catalogs/real_lens_properties.csv

# See sim-to-real matches (top match per real lens)
grep ',1,' analysis/sim_obs_comparison/catalogs/matched_lens_pairs.csv | head -20
```

## Full Pipeline (Using Orchestration Script)

```bash
# Run all steps 1-3 with defaults
bash analysis/sim_obs_comparison/scripts/run_analysis.sh --step all --max-pairs 20

# Run specific steps
bash analysis/sim_obs_comparison/scripts/run_analysis.sh --step 1  # Extract real lenses
bash analysis/sim_obs_comparison/scripts/run_analysis.sh --step 2  # Match pairs
bash analysis/sim_obs_comparison/scripts/run_analysis.sh --step 3  # Generate visuals
```

## Analysis Workflow

**Phase 1: Visual Inspection** (Manual, ~30 min)
1. Generate 20-30 comparisons
2. Inspect for obvious differences
3. Document observations (smoothness, complexity, artefacts)

**Phase 2: Quantitative Metrics** (Automated, ~20 min once code ready)
1. Sersic profile fitting on image pixels
2. SNR & background statistics
3. Flux distribution analysis
4. Statistical comparison plots

**Phase 3: Root Cause Analysis** (Manual + Code Exploration)
1. For each gap, trace to simulator code
2. Propose improvement options
3. Estimate impact on ML performance

**Phase 4: Implementation & Validation** (Iterative)
1. Implement improvements in `src/jwst_lens_simulator.py`
2. Re-run comparison pipeline
3. Measure ML performance gain

## Key Files

```
analysis/sim_obs_comparison/
├── FRAMEWORK.md                         ← Read first: strategy & metrics
├── ANALYSIS_STATUS.md                   ← Overview & status
├── catalogs/
│   ├── real_lens_properties.csv         ← 435 real lenses from FITS
│   └── matched_lens_pairs.csv           ← 1,305 sim-to-real matches
├── scripts/
│   ├── visualize_comparisons.py         ← Main comparison tool
│   ├── compute_morphology_metrics.py    ← Sersic fitting (stub)
│   ├── compute_flux_metrics.py          ← SNR analysis (stub)
│   └── run_analysis.sh                  ← Master orchestrator
└── visualizations/                      ← Output comparison PNGs
```

## Expected Findings (Hypotheses to Test)

1. **Morphology Too Idealized**
   - Simulated Sérsic profiles perfectly smooth
   - Real lenses have dust, substructure, asymmetries
   
2. **Unrealistic Noise**
   - Simulation: Poisson + Gaussian only
   - Real JWST: detector-specific patterns, cosmic rays, readout noise
   
3. **PSF Artifacts Incomplete**
   - Simulated PSF too mathematically perfect
   - Real PSF has diffraction features, core/wings mismatch
   
4. **Background Galaxies Too Sparse**
   - Simulated fields feel too ordered
   - Real COSMOS has overlaps, low-surface-brightness galaxies
   
5. **Photometry Calibration**
   - Systematic magnitude offsets between sim and real
   - Color distributions don't match

## Why This Matters

ML models trained on sim+real lenses **significantly outperform** sim-only models. This comparison pipeline identifies **what's missing** in pure simulation, enabling targeted improvements. Each fix → retrain ML → measure performance gain.

---

**Next Action**: Run visualizations and visually inspect first 20 pairs.
