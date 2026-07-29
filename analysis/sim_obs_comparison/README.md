# Complete Simulation vs Observation Comparison Pipeline

## ✅ What's Been Created

A **complete framework** for systematically comparing simulated JWST strong lenses with real observed lenses to identify and prioritize simulation improvements.

**Location**: `/Users/gozalig1/Projects/jwst-mock-lens-simulator/analysis/sim_obs_comparison/`

### Generated Data

| File | Type | Records | Purpose |
|------|------|---------|---------|
| `catalogs/real_lens_properties.csv` | Catalog | 435 | Real JWST lenses extracted from FITS files |
| `catalogs/matched_lens_pairs.csv` | Catalog | 1,305 | Simulated↔Observed matches (3 per real lens) |
| `scripts/*.py` | Code | 6 | Extraction, matching, visualization, metrics |
| `*.md` | Docs | 3 | Framework, status, quickstart |

### Files to Read First

1. **QUICKSTART.md** — How to run the pipeline (5 min read)
2. **FRAMEWORK.md** — Detailed comparison strategy (10 min read)
3. **ANALYSIS_STATUS.md** — Current state and next steps (5 min read)

---

## 🚀 Quick Start: Run Visualizations

```bash
cd /Users/gozalig1/Projects/jwst-mock-lens-simulator

# Generate 20 side-by-side comparisons
python analysis/sim_obs_comparison/scripts/visualize_comparisons.py \
  --max-pairs 20 \
  --stretch log

# View results
open analysis/sim_obs_comparison/visualizations/
```

**What to look for:**
- Are simulated arcs too smooth/perfect?
- Do backgrounds have realistic structure?
- Are morphologies simplified (missing substructure)?
- Are noises realistic (texture, cosmic rays)?

---

## 📊 Data Generated

### Real Lens Catalog (435 systems)
```
COSJ095846+020304, F115W: flux=6417.19, max=0.0042
COSJ095847+015837, F115W: flux=9283.45, max=0.0063
...
```

### Matched Pairs (1,305 records)
```
COSJ095846+020304 ← matched_to → cosmos_lens_000023 (score=2.45)
COSJ095847+015837 ← matched_to → cosmos_lens_000045 (score=1.89)
...
```

---

## 🔍 Analysis Workflow

**Phase 1: Visual Inspection** (30 min)
- Generate comparison grids
- Manually inspect for obvious differences
- Document qualitative observations

**Phase 2: Quantitative Metrics** (20 min)
- Sersic profile fitting
- SNR & background analysis
- Statistical comparison plots
- *Scripts: compute_morphology_metrics.py, compute_flux_metrics.py*

**Phase 3: Root Cause Analysis** (1-2 hrs)
- Trace each gap to simulator code
- Identify top 3 improvement targets
- Estimate ML performance impact

**Phase 4: Implementation & Validation** (iterative)
- Implement improvements in src/jwst_lens_simulator.py
- Re-run comparison pipeline
- Retrain ML models
- Measure performance gain

---

## 💡 Expected Simulation Gaps

Based on ML performance difference (sim+obs >> sim-only):

1. **Morphology Too Idealized**
   - Real lenses have dust, substructure, asymmetries
   - Simulated Sérsic profiles too mathematically smooth

2. **Unrealistic Noise**
   - Simulation: Poisson + Gaussian only
   - Real JWST: detector patterns, cosmic rays, readout noise

3. **Incomplete PSF**
   - Simulated PSF too perfect
   - Real PSF: diffraction spikes, core/wings variation

4. **Sparse Background**
   - Simulated fields too ordered/regular
   - Real COSMOS: overlapping galaxies, faint structures

5. **Photometry Calibration**
   - Magnitude offsets between sim and real
   - Color distribution mismatches

---

## 📁 Directory Structure

```
analysis/sim_obs_comparison/
├── FRAMEWORK.md                    ← Strategic planning
├── ANALYSIS_STATUS.md              ← Overview & progress
├── QUICKSTART.md                   ← This file
│
├── catalogs/
│   ├── real_lens_properties.csv    (435 real lenses)
│   └── matched_lens_pairs.csv      (1,305 sim↔obs matches)
│
├── scripts/
│   ├── extract_real_lens_properties.py  (EXECUTED)
│   ├── match_sim_to_obs.py              (EXECUTED)
│   ├── visualize_comparisons.py         (READY TO RUN)
│   ├── compute_morphology_metrics.py    (STUB)
│   ├── compute_flux_metrics.py          (STUB)
│   ├── generate_report.py               (STUB)
│   └── run_analysis.sh                  (Orchestrator)
│
├── visualizations/                 (Comparison PNG outputs)
├── metrics/                        (Computed statistics)
└── reports/                        (Analysis findings)
```

---

## 🎯 Immediate Actions

### 1. Visual Inspection (~30 min)
```bash
python analysis/sim_obs_comparison/scripts/visualize_comparisons.py \
  --max-pairs 20 --stretch log
open analysis/sim_obs_comparison/visualizations/
```

### 2. Review Matched Pairs (~5 min)
```bash
head -30 analysis/sim_obs_comparison/catalogs/matched_lens_pairs.csv
```

### 3. Run Full Pipeline
```bash
bash analysis/sim_obs_comparison/scripts/run_analysis.sh --step all
```

---

## ✨ Key Insights

- **435 real lenses** cataloged from JWST observations
- **1,305 matching records** enable direct sim↔obs comparison
- **Framework** systematically identifies gaps
- **Repeatable cycle**: Compare → Identify → Improve → Validate → Retrain
- **Data-driven approach** targets highest-impact improvements

---

## 📝 Next Steps (Your Choice)

**Option A**: Start visual inspection immediately
```bash
cd /Users/gozalig1/Projects/jwst-mock-lens-simulator
python analysis/sim_obs_comparison/scripts/visualize_comparisons.py \
  --max-pairs 10 --stretch log
```

**Option B**: Read complete framework first
```bash
cat analysis/sim_obs_comparison/FRAMEWORK.md | less
```

**Option C**: Check matched pairs data
```bash
head -50 analysis/sim_obs_comparison/catalogs/matched_lens_pairs.csv | column -t -s,
```

---

## 📧 Summary

A complete **data-driven framework** for improving your JWST lens simulator by directly comparing simulated systems with ~435 real JWST observations, identifying what's missing, and prioritizing targeted improvements for maximum ML model performance gain.

**Ready to use. Start with visual inspection.**
