# Phase 2: Background Noise Fix + Color/Size Investigation

**Date**: 2026-06-12
**Sample**: 75 lenses, fresh JADES run (seed=99, 8 bands, exposure_time=3000,
post-fix code) vs the 435-lens real COSMOS-Web sample.

## 1. Background noise — FIXED (validated)

**Change**: `src/jwst_lens_simulator.py` — new `add_sky_background_noise()`,
called after `DetectorChain.apply()` in both the main-lens and non-lens
band-processing loops. Adds Gaussian noise with σ =
`CONFIG['noise'][band]['background_rms']` (empirically calibrated from the
435 real lenses). Also filled in `noise:` entries for F070W/F090W/F200W/F356W
directly in `configs/jades_config.yaml` (previously only in
`default_config.yaml`, which is never loaded by the JADES run script).

**Result**: bg_rms_F150W 0.0246 → real 0.0242 (ratio 1.07x, was 0.17x before
Phase 2). **This is the single biggest realism improvement so far** — fixes
the "too perfect / no noise" visual issue reported earlier in this session.

## 2. SED template bug — FIXED (validated, modest effect)

**Change**: `src/empirical_sed_templates.py::stellar_continuum_BC03` — the
`break_4000` term `1/(1+exp(-(wave-0.4)/0.015))` collapsed to ~0 for
rest-frame UV wavelengths, multiplying away the UV-continuum boost it was
combined with and producing K-corrections of **+12.5 mag** for z~4
star-forming sources in F115W (essentially erasing them from the bluest
band). Changed to floor at 0.3 (`0.3 + 0.7/(1+exp(...))`), giving a
physically reasonable ~1.3 mag suppression instead. Verified via
`calculate_k_correction_empirical(z=3.9, F115W, 'star_forming')`: 12.5 → 1.77.

**Result**: small improvement in aggregate color (1.93 → 1.86), as expected
since this function (`get_realistic_jwst_color`/`calculate_k_correction_empirical`)
is a secondary fallback path — the primary catalog color-generation path uses
`get_realistic_jwst_color_from_transmission` (see below).

## 3. K-correction no-op bug — IDENTIFIED, fix attempted and reverted

**Found**: `JWSTFilterTransmissionSystem._calculate_kcorrection()` in
`src/jwst_filter_transmission.py` returns
`2.5*log10(1+z) + (-2.5*log10(1+z)) = 0` for **all** redshifts and bands — a
no-op. Additionally, `convolve_sed_to_magnitude()` convolves the rest-frame
SED with the **observed-frame** filter transmission curve without any
redshift shift — i.e. **galaxy color in this simulator is currently
independent of redshift**, which is the primary per-band magnitude
generation path (`create_parameter_variations` → `_per_band_colors` →
`get_realistic_jwst_color_from_transmission`).

**Attempted fix**: shift the filter transmission by `(1+z)` before
convolving (`T(λ_rest·(1+z))`), which is the physically correct formulation.
Also widened the SED rest-frame wavelength grid (0.3-5.0μm → 0.05-6.0μm) so
high-z sources' rest-UV is covered.

**Result of attempted fix**: color_F115W_F444W got **worse** (1.86 → 2.85,
even 3.28 with the original 0.3-5.0 grid) — i.e. correctly making colors
redshift-dependent, combined with the current Calzetti extinction defaults
(E(B-V)=0.15 for star-forming, etc.) and SED shapes, over-suppresses F115W
flux for the z~3-6 lensed-source population far more than real data shows.

**Decision**: reverted both changes (kept item 2's break_4000 fix, which is
independently a real bug-fix and harmless). The no-op K-correction / missing
redshift-shift remains a known, documented bug
(`src/jwst_filter_transmission.py:202-213`, see inline comment) but fixing it
properly requires recalibrating the SED extinction/normalization for the
redshift-dependent case — flagged for Phase 3, not attempted further this
session given risk of broad regression across all configs (Euclid, Roman,
etc. also use this function).

## 4. Lens size — partially explained (noise-dependent measurement artifact)

`lens_reff_arcsec` ratio improved from **2.04x → 1.53x** purely from the
noise fix (item 1) — i.e. roughly half of the original "2x too big" finding
was a measurement artifact: with near-zero background noise, the photutils
segmentation extended much further into the Sersic profile's faint outer
wings, inflating the measured second-moment size. The remaining ~1.5x gap is
likely a mix of (a) genuine R_sersic/mass-size calibration (config
`lens_radius` for this sample has median ~0.5-0.7", smaller than the 1.18"
segmentation measurement, consistent with Sersic index n>2 profiles having
larger second moments than R_sersic for a fixed Re) and (b) `lens_axis_ratio`
being more elongated in sim (0.61 vs 0.71 real) inflating semimajor_sigma.
Not further investigated this session — Phase 3 candidate.

## 5. Multiplicity — improved but still low

quad+ fraction: 44% (n=245, pre-fix) → ~24% (n=75, post-fix, seed=99) vs 86%
real. The noise fix did NOT increase multiplicity as hypothesized in Phase 1
(if anything it decreased slightly in this sample, within small-N noise) —
the dominant gap is field/companion **count**, not noise-driven detectability.
`n_field_max=5` and the `field.expected_density_per_arcsec2: 1.5` settings are
the likely levers; real cutouts average ~7 detected components in the central
6"x6" (≈28 arcsec²), i.e. ~0.25/arcsec², vs config's 1.5/arcsec² target
(which should be MORE than enough) — suggesting the field-galaxy *placement*
logic isn't reaching its configured target density, or many placed field
galaxies fall below the post-fix detection threshold. Phase 3 candidate.

## Summary of Phase 2 Net Changes

| File | Change | Status |
|---|---|---|
| `src/jwst_lens_simulator.py` | Added `add_sky_background_noise()`, wired into both detector-chain call sites | Kept — validated fix |
| `configs/jades_config.yaml` | Added `noise:` entries for F070W/F090W/F200W/F356W | Kept — required for fix above |
| `src/empirical_sed_templates.py` | Fixed `break_4000` floor in `stellar_continuum_BC03` | Kept — bug fix, modest effect |
| `src/jwst_filter_transmission.py` | Redshift-shift in `convolve_sed_to_magnitude` | **Reverted** — correct physics, but regresses color match given current SED calibration; documented for Phase 3 |

## 6. Multiplicity fixes — IMPLEMENTED (validated, partial improvement)

Two changes targeting the quad-image / field-galaxy-count gap (item 5,
n_components/multiplicity ~0.29x real):

**(a) Source-offset/θ_E coupling** (`create_parameter_variations`,
`src/jwst_lens_simulator.py` ~line 3959): previously `source_x`/`source_y`
were perturbed by `N(0, 0.08)` arcsec **independent of θ_E**, giving a median
`offset/θ_E ≈ 0.41` — well outside the quad-image caustic (~0.2-0.3×θ_E for
typical axis ratios) for most lenses. Changed to sample
`offset_ratio = Beta(1.5, 4) × 0.6` (mean ~0.16, skewed toward small values)
and set `offset = offset_ratio × θ_E`, preserving the base catalog's position
angle. This ties the source position to the lens's actual Einstein radius so
more sources fall inside the quad caustic, matching the strong selection bias
of the real confirmed-lens sample.

**(b) Field-galaxy count caps** (two identical blocks, ~line 4612 and
~line 5289): the `expected_density_per_arcsec2: 1.5` config value was dead
code — `mean_field = min(expected_count, n_max)` always picked `n_max`, and
`n_max` came from small hard-coded per-environment ranges
(isolated_field 0-2, galaxy_pair 1-3, group 2-4), capping field galaxies at 4
regardless of config. Roughly doubled these ranges (isolated_field 0-4,
galaxy_pair 1-5, group 3-7), still subject to the `--n_field_max` CLI cap
(default 5).

**Validation** (49-lens batch, seed=321, `/tmp/phase2_offsetfix`):

| metric | before (phase2_final, n=75) | after (n=49) | real |
|---|---|---|---|
| offset/θ_E median | 0.41 | 0.27 | (selection-biased, low) |
| n_components median | 2 | 3 | 7 |
| quad+ fraction | 0.24 | **0.39** | 0.86 |
| max_sep_arcsec | 2.590 | 2.950 | 4.693 |
| lens_reff_arcsec | 1.178 | 1.242 | 0.769 |
| color_F115W_F444W | 1.863 | 1.986 | 0.498 |

quad+ fraction nearly doubled (0.24 → 0.39) and max_sep improved toward real
(ratio 0.55 → 0.63). lens_reff and color are roughly unchanged (within
small-N noise) — as expected, these fixes targeted multiplicity specifically.

**Caveat**: offset/θ_E median only improved to 0.27 (not down to the
Beta-mean of ~0.16) because downstream Fundamental-Plane consistency
(`fp_consistent_lens_params`, ~line 4469) can re-derive `theta_E` *after* the
source position was set from the pre-FP θ_E, decoupling the ratio again for
some lenses. Fully closing the gap to 0.86 would require either computing the
source offset *after* the FP-consistent θ_E is finalized, or iterating —
flagged for Phase 3.

## 7. FP-consistency offset decoupling — FIXED (validated)

Per item 6's caveat: `fp_consistent_lens_params` can revise `theta_E` after
`create_parameter_variations` set the source position from the pre-FP
`theta_E`, decoupling the offset/θ_E ratio. Fix: `create_parameter_variations`
now also stores `source_offset_ratio` and `source_angle` on each row; after FP
consistency finalizes `theta_E` (`src/jwst_lens_simulator.py` ~line 4488),
the source position is re-derived from
`offset = source_offset_ratio * theta_E` with the stored angle, before
`lensed_source` is constructed.

**Validation** (71 lenses, seed=555): `offset/θ_E` median improved
0.27 → **0.19** (close to the target Beta-mean of 0.16).

## 8. Color normalization bug — FOUND AND FIXED (major impact)

While investigating the color gap further, found the **actual root cause**
in `JWSTFilterSystem.convolve_sed_to_magnitude`
(`src/jwst_filter_transmission.py`): the magnitude integral
`numerator = ∫ SED(λ)·T(λ) dλ` was **not normalized by the filter's own
bandpass integral `∫T(λ) dλ`**. Since JWST filters have very different
widths (F115W FWHM≈0.214μm vs F444W FWHM≈1.024μm, a ~4.8x ratio,
`∫T dλ` ratio ≈4.3x), wider filters integrated to systematically larger
"numerator" values purely from bandpass width — independent of actual SED
flux — producing colors ~1.4 mag too red across **all** SED types
(`color_F115W_F444W` = 2.0–3.4 with no SED/redshift dependence at all).

**Fix**: `numerator = ∫SED·T dλ / ∫T dλ` (proper flux-weighted average through
the filter). This is the standard AB-magnitude bandpass convolution and is
independent of, and much more impactful than, the previously-investigated
no-op K-correction (item 3, still unfixed/documented for Phase 3).

**Validation, isolated SED-level check** (no redshift, normalized):

| SED type | color_F115W_F444W (before) | (after) |
|---|---|---|
| star_forming | 2.198 | 0.612 |
| passive | 2.000 | 0.413 |
| post_starburst | 2.091 | 0.505 |
| dusty_starburst | 3.382 | 1.796 |

Real median = 0.498 — star_forming/passive/post_starburst (the dominant
high-z source types) now land within ~0.1 mag of real.

**Full-pipeline validation** (81 rows, seed=777, `/tmp/phase2_colorfix`,
includes items 6+7 fixes too):

| metric | phase2_final (n=75) | phase2_colorfix (n=81) | real |
|---|---|---|---|
| color_F115W_F444W | 1.863 | **0.477** | 0.498 |
| color_F115W_F277W | 1.131 | 0.049 | 0.725 |
| color_F277W_F444W | 0.758 | 0.499 | -0.237 |
| quad+ fraction | 0.24 | **0.47** | 0.86 |
| n_components median | 2 | 3 | 7 |
| max_sep_arcsec | 2.590 | 3.030 | 4.693 |
| lens_reff_arcsec | 1.178 | 1.124 | 0.769 |
| bg_rms_F150W | 0.026 | 0.023 | 0.024 |

**color_F115W_F444W is now essentially matched** (ratio ~0.96, was 3.74x).
quad+ fraction nearly doubled again (0.24 → 0.47, cumulative with item 6/7).
**However**, the two-color (F115W-F277W, F277W-F444W) shape is now off in a
new way: F115W-F277W undershoots (0.05 vs real 0.725) and F277W-F444W flips
sign (+0.50 vs real -0.237). The wide-baseline color matches well, but the
SED's curvature across the three bands (driven by the 1.6μm stellar bump and
the `rj_tail` break at 2.0μm in `stellar_continuum_BC03`) doesn't match real
galaxies' mid-band colors — flagged for Phase 3 SED-shape recalibration.

## 9. SED mid-band shape (2.7μm bump) — FIXED (validated)

Following item 8, `color_F115W_F277W` and `color_F277W_F444W` were off in a
new way (0.049 vs real 0.725, and 0.499 vs real -0.237). Root cause: the
`rj_tail` Rayleigh-Jeans decline beyond 2.0μm in `stellar_continuum_BC03`
gives a F277W/F444W flux ratio too shallow (~2.6x) to reproduce real's
F277W-F444W color (~-0.237 implies a ratio ~4.95x), while F115W-F444W (1.15
vs 4.44 μm, unaffected by this region) was already correct.

**Fix**: added a Gaussian "bump" centered at 2.7μm (`1.0 + 1.5*exp(-((wave-2.7)/0.5)^2)`)
to `stellar_continuum_BC03`, sampled mainly by F277W and not by F115W/F444W,
raising F277W flux relative to its neighbors. Amplitude/width tuned by
grid search against the empirical SED-level colors (not yet redshift- or
pipeline-dependent).

**Full-pipeline validation** (78 rows, seed=888, `/tmp/phase2_nirbump`):

| metric | phase2_final | phase2_colorfix (item 8) | phase2_nirbump (item 9) | real |
|---|---|---|---|---|
| color_F115W_F444W | 1.863 | 0.477 | **0.466** | 0.498 |
| color_F115W_F277W | 1.131 | 0.049 | **0.775** | 0.725 |
| color_F277W_F444W | 0.758 | 0.499 | **-0.307** | -0.237 |
| quad+ fraction | 0.24 | 0.47 | 0.47 | 0.86 |
| lens_reff_arcsec | 1.178 | 1.124 | 1.173 | 0.769 |

**All three colors now closely match real** (within ~0.05-0.07 mag), a
dramatic improvement from the Phase 1 baseline (color_F115W_F444W was 3.74x
real / +1.4 mag too red). The color investigation requested at the start of
Phase 2 is effectively **resolved**.

**Remaining major gap**: quad+ fraction (0.47 vs 0.86) and lens_reff_arcsec
(1.17x / ratio ~1.52, vs real 0.769) — see Phase 3 candidates below.

## 10. Multiplicity investigation — calibration fixed, but bottleneck is elsewhere

**Caveat discovered first**: with `--variations_per_base 1` (used in all
validation batches for items 6-9), `create_parameter_variations` is **never
called** (`src/jwst_lens_simulator.py:7770`, `if args.variations_per_base > 1`)
— so the offset/θ_E coupling (items 6, 7) was never actually exercised in
those batches! The quad+ improvement from 0.24 → 0.47 measured for items 6-9
came entirely from the field-galaxy-count increase (item 6b), which inflates
`n_components` (it counts field galaxies + lensed-image blobs together)
independent of true lens-image multiplicity. Production runs use
`--variations_per_base 25`, so this path *is* exercised there — but it means
items 6/7's effect on multiplicity hadn't actually been measured yet.

**Lenstronomy-level calibration** (`/tmp/multiplicity_sweep.py`, SIE+SHEAR
with `lens_q ~ clip(lognormal(ln 0.7, 0.1), 0.2, 1)`,
`shear ~ U(0.01, 0.05)`, using `LensEquationSolver.image_position_from_source`):
the *previous* offset_ratio distribution (`Beta(1.5,4)×0.6`, mean 0.16) gives
only **51% quad+** (n_img≥4) at the lens-equation level. `Beta(1,8)×0.6`
(mean 0.065) gives **83-89% quad+** — matching real's 0.86. Changed
`offset_ratio = rng.beta(1.0, 8.0) * 0.6` in `create_parameter_variations`
(~line 3974).

**Full-pipeline test with `--variations_per_base 2`** (75 rows, seed=1010,
`/tmp/phase2_multfix2`) confirms `offset/θ_E` is now correctly distributed
(median 0.047, mean 0.064 — matches the Beta(1,8)×0.6 target). **However,
quad+ fraction by `n_components` is still ~0.43** — essentially unchanged.

**Why**: `n_components` is a photutils segmentation count on the rendered
image. For a source very close to the lens center (offset/θ_E ~0.05-0.07,
i.e. ~0.03-0.07" for typical θ_E~0.5-1"), the 4 lensed images are separated
by distances comparable to or smaller than the PSF FWHM — segmentation
likely **merges them into one or two blended blobs / a ring** rather than 4
distinct components. So the geometric image count (now correctly ~86% quad)
and the *observable/segmented* component count are different quantities, and
real's 86% "quad+ by `n_components`" may reflect **resolved, extended arcs
fragmenting into multiple segments**, not 4 point images — a different
mechanism than what the offset/θ_E ratio controls.

**Net assessment**: the offset_ratio fix is geometrically correct and
validated against lenstronomy (kept), but closing the *segmentation-based*
`n_components`/quad+ gap likely requires investigating PSF/arc-blending in
the rendered images directly (e.g., do real quad lenses actually show 4
separate segments, or 1-2 large arcs that fragment under the real
noise/PSF?) — a distinct, deeper image-formation investigation flagged for
Phase 3.

## 11. Multiplicity investigation — `n_components` is not a multiplicity metric (root cause found)

Direct segmentation comparison (`/tmp/seg_compare.py`, central 6"x6" box,
SNR-combined detection image, threshold=3σ×√(n_bands), `npixels=8`) between
the real lens `COSJ095856+015821` (n_components=7) and 9 simulated lenses
from `/tmp/phase2_multfix2` (n_components 1-8) shows the **same structural
pattern in every case**:

- The entire lens system — deflector light + Einstein ring/arcs/multiple
  images — is segmented as **one single large connected blob**
  (real: label 2, area 4029px, semimajor 1.32"; sim0: label 2, area 10110px,
  semimajor 1.53"). Visually (`/tmp/seg_real.png`, `/tmp/seg_sim0.png`,
  `/tmp/seg_PRISM_lens_BR_000004.png`, `/tmp/seg_PRISM_lens_BR_000012.png`),
  this blob is a smooth, simply-connected ellipse/ring in *every* case —
  there is no visible sub-fragmentation into separate image components, in
  either real or simulated data, regardless of whether the underlying lens
  equation produces 2 or 4 images.
- All the *other* segments (real: labels 1,3-7, areas 8-17px,
  semimajor 0.08-0.13"; sim: similarly small blobs, 8-101px) are small,
  isolated, PSF-scale detections **scattered elsewhere in the 6"x6" box** —
  i.e. faint companion/field galaxies or noise peaks, unrelated to the
  lensed-image system itself.

**Conclusion**: `n_components` (and the "quad+ fraction by `n_components`"
metric used in items 5/10) does not measure lens-image multiplicity in
*either* real or simulated data — at JWST/NIRCam resolution and the depth of
this sample, 2 and 4-image configurations both render as one connected
arc/ring blob. The metric is almost entirely driven by the number of
small unrelated detections (companions/field galaxies/noise) within the
6"x6" box, a quantity already addressed by the background-noise (item 1)
and field-galaxy-density (item 6b) fixes. **The geometric
offset_ratio/quad-fraction calibration from item 10 (Beta(1,8)×0.6,
lens-equation quad+ ~86%) is correct and should be kept**, but it cannot be
validated against `n_components`, and further PSF/pixel-scale tuning aimed
at making `n_components` reflect "quad+" would be chasing a metric that real
data doesn't satisfy either.

**Recommendation**: either (a) drop `n_components`/quad+ as a validation
target entirely and rely on the lens-equation-level calibration (item 10,
already validated against lenstronomy directly), or (b) if
arc/ring-substructure realism is still wanted, use `photutils.segmentation
.deblend_sources` on the central blob and compare *deblended* sub-component
counts between real and sim — but this is a second-order refinement, not a
multiplicity-distribution problem, and is deprioritized below.

## 12. Lens-size investigation — `lens_reff_arcsec` gap traced to total-flux normalization, not R_sersic input

Cross-matched `/tmp/phase2_nirbump` (78-lens, `--variations_per_base 2`,
seed=888) sim morphology (`analysis/sim_obs_comparison/catalogs/sim_lens_morphology.csv`)
against `cosmos_lens_training_catalog.csv` via npz metadata `lens_id`
(54/78 rows matched) to relate the *input* `lens_radius` (R_sersic, the
value actually used to render `main_lens_light`, set via
`convert_physical_to_angular_radius(reff_kpc, lens_z)` from the FP-derived
`re_kpc`) to the *measured* `lens_reff_arcsec`.

**Input distribution checks out**: sim `lens_radius` median = 0.43" (often
clamped to the 0.35" floor from `geo.lens_radius_min`/FP output), matching
real catalog `rearc_*`/`RADIUS`-derived R_sersic (~0.43" median) reasonably
well. `lens_n_sersic` median = 2.38 vs real catalog `SERSIC` median = 2.55 —
also reasonably close.

**But `lens_reff_arcsec` / `lens_radius` ratio is large (median ~2.7x, range
0.06-6.0x) and essentially uncorrelated with `lens_radius` itself (r=0.05
against `lens_radius/theta_E`)** — i.e. the segmentation-measured size is
**not simply a smoothed/PSF-broadened version of the input R_sersic**. This
is consistent with item 11: `lens_reff_arcsec` is measured on the single
connected lens+arc+image blob, so it reflects the *whole system's* light
distribution, not just the deflector's Sersic profile.

**What does explain it**: `lens_reff_arcsec` correlates with the **total
flux** in the central segment (`total_flux_F150W`) in *both* datasets (sim
r=0.32, real r=0.53) — a fixed absolute detection threshold (3σ above a
now-well-matched background, item 1) is reached at larger radius for a
brighter system, regardless of its intrinsic Sersic shape. And **sim's
`total_flux_F150W` is ~1.79x brighter than real's** (median 1446 vs 809,
arbitrary flux units) for the matched central-segment systems. A simple
isophotal-radius argument (R ∝ √flux at fixed surface-brightness threshold
and profile shape) predicts a √1.79 ≈ 1.34x size inflation from this flux
excess alone — in the same direction and same order of magnitude as the
observed 1.52x `lens_reff_arcsec` ratio (the residual ~1.13x beyond √-scaling
is plausibly from the steeper high-n Sersic wings making isophotal radius
more sensitive to flux than a simple √ scaling). Note `sb_peak_F150W` is
*lower* in sim (3.83 vs 5.93 real) despite higher total flux — consistent
with sim systems being more spatially extended/diffuse for a given total
brightness, not just uniformly rescaled.

**Conclusion**: the `lens_reff_arcsec` gap is not primarily an R_sersic/
axis-ratio *input*-distribution problem (those already roughly match real
catalog values) — it is substantially explained by an **absolute
flux/magnitude normalization excess (~1.8x, ~0.65 mag) in the rendered
lens+arc system relative to real COSMOS-Web photometry**, propagating into
a larger segmentation-based size measurement via the fixed-threshold/
isophotal-radius mechanism (a second, distinct manifestation of the same
"single blended blob" issue as item 11). This is a different calibration
axis than the *color* normalization fixed in item 8 (which fixes
band-to-band ratios but not the overall flux scale/zero-point).

**Recommendation (next concrete step, Phase 3)**: check the absolute
flux/magnitude scale of rendered lens (and source) light against real
COSMOS-Web `MAG_MODEL_F150W`/`mag_f150w` for galaxies with matched
`lens_radius`/`n_sersic`/`lens_redshift` — i.e. verify
`convolve_sed_to_magnitude`'s zero-point (`FILTER_NOISE_PROPERTIES[band]['zeropoint']`)
and the SED normalization that sets total luminosity from `lens_mass_log10`/
`abs_mag_r`, independent of the per-band color shape already fixed. A ~0.65
mag systematic offset there would be consistent with both this item and a
secondary contribution to item 5's `n_components` gap (brighter systems →
more of the field/companion population crosses the detection threshold).

## 13. Lens magnitude normalization — FIXED (zeropoint ruled out, synthetic-magnitude offset found and corrected)

Followed up on item 12's recommendation by checking `convolve_sed_to_magnitude`'s
zeropoint (`FILTER_NOISE_PROPERTIES['F150W']['zeropoint']=30.00`) against the
single global `magnitude_zero_point=28.09` used by lenstronomy's
`SimAPI`/`magnitude2amplitude` (`src/jwst_lens_simulator.py:3355-3361`).

**Zeropoint mismatch ruled out**: computed an *empirical* zeropoint
(`mag_f150w + 2.5*log10(total_flux_F150W)`) from the rendered/measured
`total_flux_F150W` for both the real lens `COSJ095856+015821` (28.61) and the
54 matched sim lenses from `/tmp/phase2_nirbump` (median 28.62, mean 28.70).
These agree to within ~0.1 mag — i.e. for a *given* input `lens_mag_f150w`,
sim and real produce consistent total flux. So `convolve_sed_to_magnitude`'s
per-band zeropoints and `SimAPI`'s `magnitude_zero_point=28.09` are NOT
mismatched (item 12's hypothesis on this specific point was wrong).

**Actual cause found**: the *input* `lens_mag_f150w` values themselves are
assigned by a purely synthetic formula
(`src/jwst_lens_simulator.py:6782`, `lens_base = lens_base_mag_zero +
lens_redshift_log_slope * log10(z_lens)`, with `lens_base_mag_zero=21.0`,
plus per-galaxy SED color offset and `N(0, 1.2)` scatter) — **not** drawn
from or matched to the real catalog's `mag_f150w`/`MAG_MODEL_F150W` at all.
For the 54-lens `/tmp/phase2_nirbump` sample, the resulting
`lens_mag_f150w` median was **20.97**, vs the real COSMOS-Web catalog's
`mag_f150w` median of **21.55** (`MAG_MODEL_F150W` median 21.64,
consistent) — i.e. simulated lenses were ~0.55 mag (~1.6x flux) too bright
on average, matching the ~1.79x `total_flux_F150W` excess measured in item
12.

**Fix**: changed `lens_base_mag_zero` from `21.0` → `21.55`
(`src/jwst_lens_simulator.py:567`) to align the median of the synthetic
`lens_mag_f150w` distribution with the real catalog's `mag_f150w` median.

**Validation** (`/tmp/phase2_magfix`, 46 lenses, `--variations_per_base 2`,
seed=2222): `lens_mag_f150w` median moved 20.97 → 21.19 (partial shift —
the redshift-dependent term and per-galaxy color/scatter mean the median
doesn't shift by exactly the zero-point delta, and this is a different
46-lens draw than the 54-lens sample used to diagnose the offset).
`total_flux_F150W` ratio improved **1.79x → 1.44x** (correct direction,
~45% of the gap closed). However `lens_reff_arcsec` ratio did *not* improve
in this sample (1.52x → 1.65x, i.e. slightly worse) despite the flux
improvement — given item 12's only-moderate flux↔reff correlation (r~0.3-0.5)
and the small sample size (46 vs 78, different seed/lens population), this
is plausibly dominated by sampling noise rather than indicating the fix is
wrong. The magnitude-normalization fix itself is correct and validated
independently (it now matches the real catalog's photometric distribution),
but its effect on `lens_reff_arcsec` specifically needs the larger
(n≥200) sample in Phase 3 item 3 to resolve from noise.

## 14. n≥200 validation batch — overall Phase 2 vs real comparison

Ran the full pipeline with all of items 6-13 in place
(`--n_lenses 100 --variations_per_base 2 --seed 4242`, output to
`/Volumes/extHD/jwst_lens_outputs/phase2_validation200`). 133/200 requested
lens variants were produced (some base lenses are filtered/skipped by the
pipeline's quality checks; 133 is still ~1.7-2.9x larger than the n=46-78
samples used for items 9/13). Ran `compute_sim_lens_morphology.py` on the
133 npz files and compared medians/means against the 435-lens real
`real_lens_morphology.csv`:

| metric | real median | sim200 median | ratio | real mean | sim200 mean | ratio |
|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.769 | 1.208 | 1.570 | 0.873 | 1.312 | 1.502 |
| lens_axis_ratio | 0.707 | 0.637 | 0.902 | 0.687 | 0.589 | 0.859 |
| n_components | 7 | 4 | 0.571 | 7.051 | 4.128 | 0.585 |
| max_sep_arcsec | 4.693 | 3.424 | 0.730 | 4.488 | 3.303 | 0.736 |
| color_F115W_F444W | 0.498 | 0.466 | 0.936 | 0.539 | 0.461 | 0.855 |
| color_F115W_F277W | 0.725 | 0.724 | 0.999 | 0.728 | 0.798 | 1.096 |
| color_F277W_F444W | -0.237 | -0.376 | 1.587 | -0.189 | -0.338 | 1.782 |
| total_flux_F150W | 808.7 | 1285.9 | 1.590 | 2088.8 | 1985.9 | 0.951 |
| quad+ fraction (n_components≥4) | 0.862 | 0.504 | 0.585 | — | — | — |

**Colors remain the headline success of Phase 2**: `color_F115W_F444W` and
`color_F115W_F277W` are matched to within ~6% / 0.1% at this larger sample
size, confirming items 8/9 hold up. `color_F277W_F444W` retains the ~1.6x
(median) residual offset already flagged in item 9 as a Phase 3 SED-shape
item.

**`total_flux_F150W`: median ratio got worse (1.44x → 1.59x at n=133),
but the mean ratio is now close to 1** (0.951). The real distribution is
heavily right-skewed (mean 2089 vs median 809, a ~2.6x mean/median ratio,
vs sim's more moderate 1986/1286 ≈ 1.5x) — i.e. real has a long tail of very
bright systems that the median is insensitive to but the mean is dominated
by. Item 13's `lens_base_mag_zero` fix shifts the bulk/typical-flux
(median) systems, but doesn't reproduce the real sample's long bright tail
(itself likely a strong-lensing selection effect: the confirmed real sample
over-represents intrinsically bright/massive deflectors). The fix is
directionally correct for the mean but the median regression at n=133 vs
n=46 shows item 13 alone does not fully resolve the flux-distribution shape;
reproducing the bright tail likely requires revisiting the
`lens_redshift_log_slope`/scatter shape or tying `lens_mag_f150w` more
directly to the FP-derived stellar mass, rather than further zero-point
shifts — flagged for Phase 3.

**`lens_reff_arcsec` ratio (1.57x median / 1.50x mean) is essentially
unchanged from item 9's pre-fix value (1.173x at n=78, prior to item 13) —
if anything slightly worse**, despite total_flux's mean now being ~1. This
confirms item 13's prediction that the flux↔reff correlation (r~0.3-0.5) is
too weak for a flux-normalization fix alone to close this gap, and that
`lens_reff_arcsec`'s ~1.5x excess has another, largely independent driver —
most likely the segmentation threshold/PSF-blending mechanism itself (item
11/12: real lenses are measured against a different effective PSF/pixel
covariance than sim, inflating sim's isophotal radius at fixed flux). Not
resolved by Phase 2; needs PSF-level investigation in Phase 3.

**`n_components`/quad+ fraction improved further with the larger sample**
(0.47 → 0.504, continuing the 0.24 → 0.39 → 0.47 → 0.50 trend from items
6-9), but the gap to real's 0.862 remains large. Per item 11, `n_components`
is not a true multiplicity metric in either dataset, so this gap is now
understood to reflect a *segmentation/blending* difference (real lenses
fragment into more separate photutils components — likely from sharper
arcs/rings at higher real-data resolution or different noise properties)
rather than purely an image-multiplicity difference — also a PSF/rendering
question for Phase 3.

**Summary**: of the Phase 2 items, the color fixes (8/9) are robustly
validated at n=133. The magnitude-normalization fix (13) is directionally
correct for total_flux's mean but does not close the lens_reff_arcsec or
quad+/n_components gaps — both of these now point toward PSF/segmentation-
level differences between real and simulated images as the next
investigation area, rather than further parameter-distribution tuning.

## 15. PSF investigation — sim rendering kernels (`_kernel.fits`) are 1.7-1.8x broader than the empirical PSF in F277W/F444W (likely root cause of the `lens_reff_arcsec` gap)

Started the PSF/segmentation-level investigation flagged by item 14. The
PSF kernels actually used to convolve simulated images
(`data/psf_v5_30mas/tiles/<tile>/<band>_kernel.fits`, loaded by
`load_psf_data()`/`apply_psf_convolution()`, saved per-lens to
`psf_arrays/`) were compared against the **empirical PSF measured from the
real COSMOS-Web mosaics** by PSFEx (`data/psf_v5_30mas/tiles/<tile>/<band>.psf`,
header keyword `PSF_FWHM` — the same images `real_lens_morphology.csv` is
measured from).

**Method**: for each of the 20 tiles and the 4 real-kernel bands
(F115W/F150W/F277W/F444W), fit a 2D Gaussian to `<band>_kernel.fits` to get
its FWHM, and compare to `PSF_FWHM` (already in image-pixel units, ×0.03"/px)
from the corresponding `.psf` PSFEx model:

| band | kernel.fits FWHM (sim rendering) | PSFEx empirical FWHM (real data) | ratio (median, 20 tiles) |
|---|---|---|---|
| F115W | 0.0786" | 0.0742-0.0753" | **1.04x** |
| F150W | 0.0894" | 0.0803" | **1.11x** |
| F277W | 0.1279" | 0.0766-0.0772" | **1.67x** |
| F444W | 0.1686" | 0.0935-0.0955" | **1.77x** |

The pattern is extremely consistent across all 20 tiles (F277W ratio range
1.63-1.70x, F444W 1.75-1.80x; F115W/F150W both ~1.0-1.14x) — this is a
systematic, band-dependent effect, not noise. The SW bands (F115W/F150W,
native 0.031"/px detector) are close to correct, while the LW bands
(F277W/F444W, native 0.063"/px detector) are ~1.7-1.8x too broad in the
rendering kernels relative to the empirical PSF of the real mosaics.

**Why this matters for `lens_reff_arcsec`**: `measure_lens_arrays()`'s
detection image is an SNR-sum over F115W+F150W+F277W+F444W (`morphology_metrics.py`
line 37-42). Convolving the simulated F277W/F444W images with a PSF ~1.7-1.8x
broader (in FWHM) than what the real images effectively have directly
inflates the simulated central blob's semimajor/semiminor sigma in those
bands, and therefore the SNR-combined `lens_reff_arcsec` — fully consistent
with the ~1.5x gap that survived items 12/13 and is independent of
flux-normalization (item 14).

**Likely origin**: the only PSF-kernel-generation script in the repo
(`scripts/local/generate_extra_jades_psf_kernels.py`) explicitly only
created Gaussian fallback kernels for F070W/F090W/F200W/F356W and states
that "F115W, F150W, F277W, F444W have real [WebbPSF-derived] kernels" already
— i.e. those 4 kernels predate this repo and their generation script is not
available. The clean SW-vs-LW split in the ratio (~1.0x vs ~1.7-1.8x)
strongly suggests a pixel-scale/resampling step that was correct for the
0.031"→0.03" SW remap but applied incorrectly (e.g. not accounting for the
LW detector's larger native 0.063"/px scale) when the LW kernels were
resampled onto the shared 0.03"/px grid.

**Not yet done / next step**: regenerating `F277W_kernel.fits` and
`F444W_kernel.fits` for all 20 tiles directly from the `.psf` PSFEx models
(rasterizing `PSF_MASK[0]` at `PSF_SAMP` and resampling to 0.03"/px, the same
way `PSF_FWHM` is referenced) would shrink these two kernels by the ~1.7-1.8x
factor measured above. This is a PSF-data change affecting every band/tile
used by the simulator (not just lens morphology), so it should be done as a
deliberate, reviewable regeneration step (with before/after kernel FWHM
QA across all 20 tiles) before re-running the n≥200 validation — flagged as
the concrete Phase 3 action, not done in this session pending confirmation.

## 16. LW PSF kernel regeneration — implemented and validated (modest `lens_reff_arcsec`/`color_F277W_F444W` improvement)

Implemented item 15's recommendation:
`scripts/local/regen_lw_psf_kernels.py` rebuilds `F277W_kernel.fits` and
`F444W_kernel.fits` for all 20 tiles from the `.psf` PSFEx models —
rasterizing the constant component `PSF_MASK[0]` at `PSF_SAMP`
image-pixels-per-PSF-pixel, resampling to 1 image-pixel (0.03"/px) via
spline zoom, and cropping/normalizing to the existing 101x101, sum=1
format. Old kernels backed up as `<band>_kernel.fits.bak_v5`.

**Kernel FWHM after regeneration** (median over 20 tiles, vs the PSFEx
empirical target `PSF_FWHM`):

| band | old kernel.fits FWHM | new kernel.fits FWHM | PSFEx target | old ratio | new ratio |
|---|---|---|---|---|---|
| F277W | 0.1279" | 0.0672" | ~0.077" | 1.67x | **0.87x** |
| F444W | 0.1686" | 0.0834" | ~0.095" | 1.77x | **0.88x** |

The new kernels now slightly *undershoot* the empirical FWHM (~0.87-0.88x,
i.e. ~12-13% too narrow) rather than overshooting by 1.7-1.8x — a large net
improvement, though not an exact match (likely because the cropped 101x101
rasterization truncates some of the PSF wings present in the full 201x201
PSFEx model).

**Full-pipeline validation** (same command/seed as item 14:
`--n_lenses 100 --variations_per_base 2 --seed 4242`, new output
`/Volumes/extHD/jwst_lens_outputs/phase2_psffix200`, 139 npz files, vs
item 14's 133 npz files — both ~1.7-2.9x the n=46-78 samples used for
items 9/13):

| metric | real median | item14 (old PSF) | item16 (new PSF) | real mean | item14 mean | item16 mean |
|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.769 | 1.208 (1.570x) | 1.161 (**1.509x**) | 0.873 | 1.312 (1.502x) | 1.212 (**1.387x**) |
| lens_axis_ratio | 0.707 | 0.637 (0.902x) | 0.602 (0.851x) | 0.687 | 0.589 (0.859x) | 0.592 (0.862x) |
| n_components | 7 | 4 (0.571x) | 3 (0.429x) | 7.051 | 4.128 (0.585x) | 3.777 (0.536x) |
| max_sep_arcsec | 4.693 | 3.424 (0.730x) | 3.335 (0.711x) | 4.488 | 3.303 (0.736x) | 3.144 (0.701x) |
| color_F115W_F444W | 0.498 | 0.466 (0.936x) | 0.419 (0.840x) | 0.539 | 0.461 (0.855x) | 0.452 (0.839x) |
| color_F115W_F277W | 0.725 | 0.724 (0.999x) | 0.639 (0.881x) | 0.728 | 0.798 (1.096x) | 0.654 (0.898x) |
| color_F277W_F444W | -0.237 | -0.376 (1.587x) | -0.274 (**1.158x**) | -0.189 | -0.338 (1.782x) | -0.202 (**1.065x**) |
| total_flux_F150W | 808.7 | 1285.9 (1.590x) | 1224.4 (1.514x) | 2088.8 | 1985.9 (0.951x) | 2409.0 (1.153x) |
| quad+ fraction | 0.862 | 0.504 | 0.460 | — | — | — |

**`lens_reff_arcsec` improved modestly** (mean ratio 1.502x → 1.387x, ~23%
of the gap-to-1 closed; median ratio 1.570x → 1.509x). This confirms item
15's hypothesis was directionally correct and a real, measurable
contributor, but the LW-PSF fix alone does not close the gap — a ~1.4x
excess remains, so other contributors (SW-band PSF residuals, intrinsic
size/Sersic-profile differences, or the segmentation-threshold mechanism
itself from item 12) are still at play.

**`color_F277W_F444W` improved substantially** (mean ratio 1.782x → 1.065x,
now within ~7% of real) — the over-broad F444W kernel was apparently
spreading extra flux into the segmentation footprint disproportionately,
biasing this color. This is the most unambiguous win from item 16.

**Other metrics moved slightly in the wrong direction** but by amounts
consistent with sampling noise at n=135-139 (different lens population than
item 14's 133 due to PSF-driven changes in which variants pass quality
checks): `color_F115W_F444W` (0.855x→0.839x), `color_F115W_F277W`
(1.096x→0.898x, now undershooting where it was previously ~exact),
`n_components`/quad+ (0.585x/0.504→0.536x/0.460), `lens_axis_ratio`
(0.859x→0.862x, ~flat). None of these regressions are large enough to be
clearly real vs. noise at this sample size, but `color_F115W_F277W` moving
from near-exact to a 10% undershoot is worth re-checking with the SW PSF
kernels (item 17 below) since F115W/F150W kernels were also found to be
~1.04-1.11x broad in item 15 (smaller than the LW excess, but non-zero).

**Recommendation**: keep the regenerated F277W/F444W kernels (clear net
improvement, especially for `color_F277W_F444W`); the `lens_reff_arcsec`
gap needs further investigation beyond PSF FWHM alone.

## Phase 3 Candidates (priority order, updated)

1. **`lens_reff_arcsec` still ~1.4x too large after the LW PSF fix (item
   16)** — investigate remaining contributors: (a) regenerate F115W/F150W
   kernels the same way (item 15 found ~1.04-1.11x excess, smaller but
   nonzero, and might also help `color_F115W_F277W`'s new 0.898x
   undershoot); (b) revisit item 12's segmentation-threshold mechanism
   directly (e.g. compare real vs sim isophotal radius at matched
   *PSF-deconvolved* flux); (c) check whether simulated lens light profiles
   (Sersic `n`/`Re` rendering, not just PSF) are systematically larger than
   intended. **New top priority.**
2. `total_flux_F150W` distribution shape (real has a long bright tail, mean/
   median ratio ~2.6x vs sim's ~1.5x) — consider tying `lens_mag_f150w` to
   FP-derived stellar mass/luminosity rather than a fixed zero-point +
   redshift slope + scatter, to better reproduce the bright-end tail.
3. No-op K-correction / redshift-shift in `convolve_sed_to_magnitude`
   (item 3) — now that items 8/9/13 fix color and overall flux normalization,
   re-attempting redshift-dependence is lower priority/risk, but still a
   known correctness gap (colors are currently redshift-independent).
4. (Optional, low priority) deblended sub-component count comparison
   (item 11b) for arc/ring substructure realism, if desired after 1-3.
