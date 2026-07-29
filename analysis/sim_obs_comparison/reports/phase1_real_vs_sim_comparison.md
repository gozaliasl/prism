# Phase 1: Real vs. Simulated Lens Morphology Comparison

**Date**: 2026-06-12
**Real sample**: 435 COSMOS-Web lenses (`catalogs/real_lens_morphology.csv`)
**Sim sample**: 245 lenses from the in-progress JADES 10k run
(`/Volumes/extHD/jwst-lens-similator-output/jades_10000_20260612_133258`,
exposure_time=3000, 8-band, `catalogs/sim_lens_morphology.csv`)

Both measured with the same pipeline
(`scripts/morphology_metrics.py::measure_lens_arrays`): photutils segmentation
on an SNR-combined F115W+F150W+F277W+F444W detection image, 3σ/band combined
threshold, restricted to central 6"x6".

## Headline numbers (median [16th-84th pctile])

| Metric | Real | Sim | Ratio (sim/real) |
|---|---|---|---|
| n_components (central 6") | 7 [4-10] | 3 [1-7] | 0.43x |
| max_sep_arcsec (proxy 2θ_E) | 4.69" [3.26-5.84] | 3.28" [0.0-5.17] | 0.70x |
| arc_length_arcsec | 0.282" [0.145-0.616] | 0.152" [0.116-0.263] | 0.54x |
| arc_width_arcsec | 0.172" [0.091-0.351] | 0.078" [0.053-0.199] | 0.45x |
| arc_length_to_width | 1.61 [1.23-2.31] | 1.82 [1.22-2.87] | 1.13x |
| lens_reff_arcsec | 0.769" [0.562-1.218] | 1.573" [1.091-2.194] | **2.04x** |
| lens_axis_ratio | 0.706 [0.496-0.860] | 0.617 [0.462-0.748] | 0.87x |
| **bg_rms_F150W** | **0.0242** [0.0193-0.0292] | **0.0041** [0.0035-0.0055] | **0.17x** |
| color_F115W-F444W | 0.498 [-0.146-1.258] | 1.848 [1.318-2.311] | **+1.35 mag redder** |

Multiplicity histogram (n_components classification):
| | single (1) | double (2) | triple (3) | quad+ (≥4) |
|---|---|---|---|---|
| Real (n=435) | 10 (2.3%) | 19 (4.4%) | 31 (7.1%) | 375 (86.2%) |
| Sim (n=245) | 40 (16.3%) | 44 (18.0%) | 33 (13.5%) | 108 (44.1%) |

## Interpretation — Top Findings

1. **Background noise level is ~6x too low** (bg_rms_F150W sim=0.0041 vs
   real=0.0242, ratio 0.17x). This is the single most important quantitative
   result of Phase 1, and directly explains the "too perfect" visual
   impression from earlier in this session — even after reducing
   `exposure_time` from 200000 → 3000, the *measured* per-pixel background
   RMS in the actual output images is still far below both the real data and
   the simulator's own `noise:` config dict (F150W background_rms=0.023,
   which itself matches the real value to 5%). This implies the
   `detector_chain.py` noise floor at exposure_time=3000 is not reproducing
   the configured `noise:` dict values — **the noise dict values are
   apparently not being applied as the effective background RMS when
   `detector_chain.enabled: true`** (consistent with the Phase 0 finding that
   detector_chain doesn't consume `noise.background_rms` directly). This is
   the highest-priority fix for Phase 2.

2. **Lens galaxies are ~2x too large** (lens_reff median 1.57" sim vs 0.77"
   real). Combined with smaller arcs (0.54x length, 0.45x width) and fewer
   detected components, this suggests simulated lens galaxies are
   over-extended relative to real COSMOS-Web lens galaxies, possibly
   swamping/blending nearby arc/companion light in the segmentation (fewer
   components detected as separate sources).

3. **Sim lenses are far redder** (F115W-F444W color +1.35 mag redder than
   real median). Could indicate: (a) the lens-galaxy SED/dust assumptions are
   too red, (b) the lower background noise floor changes flux measured in
   F115W (bluest, faintest band) disproportionately via the asinh/segmentation
   flux sums, or (c) a real population mix effect (real sample includes more
   blue star-forming lens galaxies / blended high-z blue arcs).

4. **Fewer detected components / lower multiplicity** in sim (86% of real are
   "quad+" vs only 44% of sim). Likely a *combination* of: lower background
   noise (fainter field sources don't clear the detection threshold as often
   in real images because real noise itself doesn't suppress them — need to
   re-examine), larger/brighter lens galaxies blending nearby sources, and/or
   genuinely fewer field/companion galaxies being placed
   (`n_field_max=5` vs real crowded fields).

5. **Arc length/width ratio is comparable** (1.82 sim vs 1.61 real) — this is
   a *relative* shape metric, less sensitive to the absolute noise/size
   issues above, and is reassuring: the basic arc elongation statistics are
   roughly in the right regime once items 1-2 are corrected.

## Root Cause Confirmed for Finding #1

Read `DetectorChain.apply()` (`src/detector_chain.py:297-357`): it converts
the input flux image to electrons (`im_e = im * t_exp`), then applies IPC,
charge diffusion, brighter-fatter, non-linearity, **dark current**, Poisson
shot noise (on signal + dark current only), read noise, 1/f noise,
saturation, PRNU, persistence, and gain/ADC — then divides back by `t_exp`.

**There is no step that adds a sky/zodiacal background flux level.** The
`noise:` dict's `background_level`/`background_rms` (e.g. F150W
background_rms=0.023, matching the real data to 5%) is computed but never
injected into `im_e` before the detector chain runs. The measured sim
bg_rms_F150W≈0.004 is therefore just dark current (0.0022 e-/s ×
t_exp=3000 → ~6.6 e- mean) + CDS read noise (13 e-) + shot noise, converted
back to e-/s — i.e. a "dark frame" noise floor with no sky background at all.

**This is the concrete Phase 2 fix**: add a sky-background flux term (from
`CONFIG['noise'][band]['background_level']`, in e-/s, the same units as
`flux_image`) to `im` *before* `DetectorChain.apply()` multiplies by
`t_exp` — so Poisson noise on the sky background is correctly propagated
through the whole chain (shot noise scales as sqrt(signal+sky+dark), not just
sqrt(signal+dark)). This single change should resolve both the bg_rms gap
(item 1) and likely much of the multiplicity/component-count gap (item 4,
since real images include a noise floor below which faint field sources are
*also* not detected — but our current 6x-too-low noise floor makes the
detection threshold artificially low relative to real, which should mean MORE
not fewer detections; the lower multiplicity in sim is therefore probably
dominated by the larger lens galaxy / fewer placed field sources, not the
noise level — worth re-testing after the noise fix).

## Caveats

- Sim sample (n=245) is a small subset of the still-running 10k generation;
  re-run `compute_sim_lens_morphology.py` on the full set once complete for
  tighter statistics.
- Pixel-value *units* between real FITS (calibrated electron-rate from JWST
  pipeline) and sim `image_final` (simulator-native units) may not be
  identical 1:1, which could contribute to the bg_rms and color offsets in
  addition to the noise-pipeline issue identified in (1). Before implementing
  a Phase 2 noise fix, verify what units `image_final` is in and whether a
  simple flux-calibration/zeropoint check (using `magnitude_zero_point: 28.09`
  vs the real data's zeropoint) is also needed.

## Next Steps (Phase 2 candidates, in priority order)

1. Trace why `detector_chain.py` at `exposure_time=3000` produces
   bg_rms_F150W ≈ 0.004 instead of the configured 0.023 — likely the
   dominant fix needed to close the largest sim-vs-real gap.
2. Re-check lens-galaxy size/flux normalization (lens_reff 2x too large).
3. Re-check lens-galaxy / source color calibration (SED templates, dust)
   against the +1.35 mag color offset.
4. Once (1)-(3) are addressed, re-run this comparison on a fresh sample and
   re-assess multiplicity/field-density gap (item 4).
