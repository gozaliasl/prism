# Phase 0: Validation-First Gap Analysis
## Toward the "Most Realistic JWST Strong Lens Simulation"

**Date**: 2026-06-12
**Scope**: Architecture review of `src/jwst_lens_simulator.py` + `src/detector_chain.py`,
quantitative statistics from the 435 real COSMOS-Web lenses in `data/real_lenses/`,
and context from COWLS I/II (Nightingale/Mulroy et al. 2025, arXiv:2503.08777,
arXiv:2503.08782) and COSMOS2025 (Shuntov et al. 2025, arXiv:2506.03243).

This is **Phase 0** of a multi-phase effort. It is a status/gap report, not an
implementation. No simulator code was modified as part of this report.

---

## 1. What Is Already Implemented

### Lens mass modeling
- **SIE + external shear** macromodel (standard configuration)
- Binary lens systems: `sie_sie`, `nfw_nfw`, `shear_only` (configurable mixture fractions)
- **NFW dark-matter subhalo population** generator (`generate_subhalo_population`,
  `CONFIG['subhalos']`) — implemented but **disabled by default**
- **Environment-dependent external shear**, calibrated to COWLS shear ranges
  (`shear_min`/`shear_max` per environment class)
- Fundamental-Plane / Faber-Jackson based mass-redshift scaling (Bernardi+2003,
  Singh+2021, Sonnenfeld+2023)
- Time-delay multi-epoch lenses (10% fraction by default)

**Not found**: elliptical power-law (PEMD/EPL) profiles (SIE/SPEMD referenced
but PEMD/EPL lens models not wired in), multipole perturbations (m=3,4 isophote
twists/boxiness), explicit line-of-sight perturbers (separate from subhalos),
satellite galaxies as distinct lensing components (companions exist as
*light*, not as additional deflectors).

### Source modeling
- Sersic-based sources with bulge+disk components, "enhanced disk" structure
- Keywords present for clumpy/irregular morphology, mergers, dust — present in
  the morphological-enhancement code path (`[v11] Applied ... enhancements`)
- Empirical SED templates (BC03 + Calzetti+2000 + Chary & Elbaz 2001) drive
  band-dependent colors
- Source redshift distribution z~0.8-15, lens redshift z~0.2-6 (Euclid config
  additionally supports a detection-completeness mixture)

**Not found**: direct use of *real JWST galaxy cutouts* as source templates
(everything is parametric/analytic).

### Multi-band JWST realism
- Real JWST NIRCam filter transmission curves (29 filters loaded)
- Real WebbPSF-derived PSF kernels for F115W/F150W/F277W/F444W
  (`data/psf_v5_30mas`); other bands (F070W/F090W/F200W/F356W, etc.) currently
  use **analytic Gaussian PSF approximations** (no diffraction rings/spikes) —
  added this session for the JADES extension
- Empirical noise model (`configs/default_config.yaml` `noise:` dict) derived
  from the same 435 real lenses analyzed here — **validated below**: median
  F150W background RMS in our new real-lens analysis (0.0242) matches the
  config's empirical median (0.0230) to ~5%

### Detector & observation effects (`detector_chain.py`)
Full chain implemented and enabled: IPC, charge diffusion, brighter-fatter,
non-linearity, dark current, Poisson shot noise, read noise, 1/f noise,
saturation, PRNU, gain/ADC, optional persistence. Verified this session to be
correctly invoked for all bands and to add realistic background pedestal +
structured (row/column-correlated) noise.

**Not found**: drizzling/mosaic artifacts (resampling correlated-noise
covariance — real JADES images show power-law noise scaling index β≈1.29 vs
β=1 for white noise; our images are currently closer to β≈1), cosmic-ray hits
as a *configurable artifact rate matched to real exposure times* (a generic
"moderate" artifact mode exists via `add_cosmos_web_artifacts`).

---

## 2. Quantitative Statistics from the 435 Real COSMOS-Web Lenses

New script: `analysis/sim_obs_comparison/scripts/compute_real_lens_morphology.py`
Output: `analysis/sim_obs_comparison/catalogs/real_lens_morphology.csv` (435 rows)
Summary: `analysis/sim_obs_comparison/reports/real_lens_statistics_summary.json`

Method: photutils segmentation on an SNR-combined F150W+F277W detection image
(3σ-per-band threshold), restricted to the central 6"×6" of each 9"×9" cutout.

| Quantity | Median | 16th-84th pctile | Notes |
|---|---|---|---|
| Components in central 6" | 7 | 4–10 | includes lens, arcs, companions, field galaxies |
| Max pairwise separation (proxy 2θ_E) | 4.69" | 3.26–5.84" | i.e. θ_E ~ 1.6-2.9" typical *image* separation — much larger than our θ_E config range (0.3-2.4") suggests many "components" are field galaxies, not lensed images; needs refinement of the detection-image threshold in a follow-up pass |
| Arc length (FWHM-like) | 0.28" | 0.14-0.62" | |
| Arc width | 0.17" | 0.09-0.35" | |
| Arc length/width | 1.61 | 1.23-2.31 | most "arcs" are mildly elongated, not long thin arcs — consistent with COWLS noting many systems are compact/group-scale rather than giant-arc systems |
| Lens galaxy effective size | 0.77" | 0.56-1.22" | |
| Lens axis ratio (b/a) | 0.71 | 0.50-0.86 | |
| F150W background RMS | 0.0242 | 0.0193-0.0292 | **matches empirical noise config (0.0230) to ~5%** |
| Color F115W-F444W | 0.50 | -0.15 to 1.26 | broad — consistent with mixed high-z source / low-z lens populations |
| Multiplicity histogram | 86% have ≥4 components | | dominated by crowded fields, not strict "quad lenses" — segmentation threshold likely too permissive for true image-multiplicity classification |

**Key takeaway**: the empirical noise calibration already in
`configs/default_config.yaml` is well-matched to the real sample (background
RMS within 5%). The "max separation" and "multiplicity" numbers above are
contaminated by field-galaxy detections and need a tighter, lens-model-aware
definition (Phase 2) before they can be used as simulator targets.

---

## 3. Discrepancies Likely Dominating Sim-vs-Real Gaps (Ranked)

Based on the architecture review + the existing `sim_obs_comparison/FRAMEWORK.md`
hypotheses (ML trained on sim+real >> sim-only):

1. **Noise correlation structure** (β≈1.29 in real drizzled mosaics vs ~white
   noise in current detector chain) — likely the single largest discriminator
   for a real-vs-sim classifier, since it's a global per-pixel statistical
   signature independent of source morphology.
2. **Field/companion realism** — real cutouts have ~7 detected components in
   6"; simulated fields use a fixed `n_field_max` with simplified placement.
3. **Source morphology richness** — parametric Sersic+bulge/disk vs. real
   clumpy high-z galaxies (JADES/COSMOS-Web sources are frequently irregular,
   multi-clump, with color gradients from dust).
4. **PSF fidelity for non-core JWST bands** — newly-added analytic Gaussian
   PSFs (F070W/F090W/F200W/F356W) lack diffraction rings/spikes; only
   F115W/F150W/F277W/F444W have real WebbPSF kernels.
5. **Lens mass model richness** — no multipoles/PEMD/explicit LOS perturbers;
   subhalos implemented but off by default. Affects arc curvature/asymmetry
   statistics specifically.
6. **No real-galaxy source templates** — fully parametric sources may miss
   non-Gaussian/non-Sersic structure present in real high-z JWST sources.

---

## 4. Proposed Phased Roadmap

This cannot be completed in one session; suggested phasing:

- **Phase 1** (next): Tighten the real-lens statistics (lens-model-aware
  multiplicity/θ_E using the COWLS catalog redshifts/θ_E where available
  rather than raw segmentation), and build the **simulated-lens equivalent**
  of `compute_real_lens_morphology.py` so the same metrics can be computed on
  JADES/COSMOS-Web sim outputs for direct comparison.
- **Phase 2**: Noise-correlation matching — measure and replicate the β≈1.29
  power-law noise scaling (likely via a drizzle/resampling step or correlated
  noise injection in `detector_chain.py`).
- **Phase 3**: Lens-model richness — add PEMD/EPL + multipole option, enable
  subhalos by default at COWLS-calibrated rates, add explicit LOS perturbers.
- **Phase 4**: Source realism — clumpy multi-component sources; evaluate using
  real JWST galaxy cutouts (e.g. from COSMOS2025) as source light templates.
- **Phase 5**: Real-lens forward modeling loop (fit → reconstruct source →
  re-lens → re-observe → residual) for a subset of the 435 real lenses.
- **Phase 6**: Automated classifier-based validator (real vs sim
  indistinguishability test), SSIM/feature-space metrics, summary dashboard.

All deliverables live under `analysis/sim_obs_comparison/` per your request.
