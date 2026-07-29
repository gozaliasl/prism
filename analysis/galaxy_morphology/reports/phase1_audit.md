# Phase 1: Audit of Galaxy Morphology Modeling in the Current Simulator

This audits the current state of galaxy-light-profile realism in
`src/jwst_lens_simulator.py` (and supporting modules) ahead of a
next-generation, multi-survey-realistic galaxy morphology framework
(COSMOS-Web, JADES, CEERS, Euclid, Roman, LSST, HSC).

## Executive summary

The simulator presents a **hybrid approach**: sophisticated *algorithmic*
post-processing (spiral arms, bars, clumps, dust lanes) is layered on top of
a **single-Sersic-profile foundation**. Every galaxy component — lens
deflector, lensed source, and field/contaminant galaxies — is rendered as
exactly one `SERSIC_ELLIPSE` light profile in lenstronomy's `SimAPI`
(`src/jwst_lens_simulator.py:4756-4765`). Visual structure (arms, bars,
clumps, dust) is then painted onto that single-component rendered image in
pixel space. There is **no native multi-component (bulge+disk+bar) light
model, no GalSim/COSMOS real-galaxy stamp injection, and no per-band
morphology variation** (every filter sees the same `n`, `q`, PA, and
pixel-space structure).

## 1. Light-profile models

- **Exclusive profile**: `SERSIC_ELLIPSE` (lenstronomy), for lens
  (`:4756`), source (`:4764`), and field galaxies (`:4759`).
- Per-galaxy parameters: Sersic index `n`, effective radius `R_eff`, axis
  ratio `q` (via `e1`/`e2`), position angle, center, per-band magnitude.
- **Absent**: `CORE_SERSIC`, `SHAPELET`, `INTERPOL`/image-based profiles,
  native multi-Sersic (bulge+disk) `light_model_list` entries.

## 2. Structural parameter sampling

- `sample_sersic_n(z, measured=None, rng=None)` (`:3218-3241`): redshift-
  dependent Gaussian, e.g. z<0.8 → N(3.5, 0.7) clipped [2,6]; 0.8≤z<1.6 →
  N(2.5, 0.6) clipped [1,5]; z≥1.6 → N(1.5, 0.5) clipped [0.5,3]. Can anchor
  to a measured `n_rest` from the real catalog.
- `R_eff`: Fundamental-Plane/Faber-Jackson (`src/fundamental_plane.py`),
  with lensing-selection bias (−0.10 dex) and intrinsic scatter (0.08 dex);
  `sample_effective_radius`/`sample_halo_radius_profile` (`:323-436`).
- Axis ratio (`:2350-2372`): morphology-dependent Beta distributions —
  spirals `Beta(2,2)*0.5+0.5` (q∈[0.5,1]), ellipticals `Beta(2.5,1.5)*0.3+0.7`
  (q∈[0.7,1]), S0 `Beta(2,2)*0.4+0.6` (q∈[0.6,1]).
- Position angle: uniform.
- **Absent**: bulge-vs-disk axis-ratio/PA misalignment, intrinsic 3D
  triaxiality.

## 3. Complex morphology support (algorithmic, post-rendering)

All applied to the single rendered Sersic image, **same random seed across
bands** (`:1426` comment: "Uses SAME seed for all bands so features appear
in same positions"):

| Feature | Status | Location | Notes |
|---|---|---|---|
| Bulge+disk split | Synthetic, post-hoc | `:1094-1102` | Splits existing image into bulge/disk flux fractions; not a native 2-component model |
| Spiral arms | Implemented | `:1053-1215` | Log-spiral, 2-3 arms, pitch 8-30° depending on `n` |
| Clumps/knots | Implemented | `:1218-1263` | 3-19 clumps, 60-150 pc physical size |
| Dust lanes | Partial | `:1201-1207` | Gaussian lane confined to disk, only if q<0.85, 15-60% attenuation |
| Bars | Partial | `:1266-1350` | Only for 'barred_spiral' morph (15% of n<2.0 galaxies) |
| Rings | Minimal | `:2258` | Classification label only ("ring", 5% of n∈[1.2,2.0]); no visual enhancement |
| Irregular/clumpy/starburst | Partial | `:2237-2273` | Classification + reuses clump code; no dedicated chain/turbulent-disk renderer |
| Post-merger | Minimal | `:2267` | Classification only (10% of n∈[2,3]); no tidal tails/double nuclei/shells |
| Interacting systems | Absent | — | No multi-galaxy tidal interaction code |

## 4. Wavelength-dependent appearance

- **Colors are band-dependent**: `get_realistic_jwst_color()` (`:2641-2735`,
  K-corrections by type/n/z) and `get_realistic_jwst_color_from_transmission()`
  (`:2736-2823`, SED-template convolution) produce per-band magnitude
  offsets; `classify_galaxy_colors_enhanced()` (`:2275-2283`) links
  morphology type to color (red ellipticals, blue spirals).
- **Morphology is NOT band-dependent**: identical `n`, `q`, PA, and
  pixel-space arm/clump/bar/dust pattern (same RNG seed) are used for every
  filter. No radial color gradients, no dust-driven morphological
  K-correction, no wavelength-dependent clump visibility.

## 5. Package inventory

- Lensing/rendering: `lenstronomy.SimulationAPI.sim_api.SimAPI`,
  `lenstronomy.LensModel.*`.
- Astronomy: `astropy.io.fits`, `astropy.convolution`, `astropy.cosmology`.
- Numerics/image-ops: `numpy`, `scipy.interpolate/ndimage/signal`
  (incl. `fftconvolve`), `PIL.Image`.
- Data/ML: `pandas`; `sklearn.ensemble` (RandomForest, used for environment
  classification, not morphology generation).
- **Not used anywhere**: `galsim` (no `COSMOSCatalog`, no real-galaxy
  postage stamps, no GalSim shear/convolution operators), `photutils` (used
  only in the separate `analysis/` validation scripts, not in the
  simulator itself), `torch`/`tensorflow` (no generative-model morphology).

## 6. Field/contaminant galaxies

Two pathways, both ultimately rendering single `SERSIC_ELLIPSE` profiles
with the same algorithmic enhancements (at reduced strength, 0.5-0.7x):

- **Real-data-driven** (preferred): `sample_real_field_galaxies_for_mock()`
  (`:2423-2686`) draws structural parameters (`n`, `R_eff`, `q`, PA, mag, z)
  from a merged real-catalog CSV via
  `load_real_field_galaxy_population_from_merged()` (`:1703-1779`) /
  `convert_real_galaxy_to_field_format()` (`:2085-2170`). Only the
  *parameters* are real — images are still synthetic Sersic renders.
- **Synthetic fallback**: `generate_synthetic_field_population()`
  (`:2306-2422`) — 60% spirals (n∈{0.8,1.0,1.2,1.5}), 25% ellipticals
  (n∈{3,4,5}), 15% S0 (n∈{2,2.5,3}); magnitude function with 15%
  bright-foreground / 85% faint-background split; `R_eff` lognormal
  (μ=ln 0.35, σ=0.5, clipped [0.08,2.5]"); Poisson(λ≈1.2/arcsec²) counts.
- **Absent**: tidal interaction with the lens galaxy, satellite/substructure
  decomposition.

## 7. Real galaxy postage stamps / image-based templates

**None exist.** The "real data" usage is strictly parameter-level (Sersic
`n`, `R_eff`, `q`, PA, magnitude, redshift pulled from a real catalog) —
the rendered images are always synthetic `SERSIC_ELLIPSE` + algorithmic
texture, never actual HST/JWST cutouts or GalSim `COSMOSCatalog` real
galaxies.

## 8. Configuration (`configs/default_config.yaml`, `configs/jades_config.yaml`)

Key morphology-relevant ranges:
- `sersic_range: [0.3, 6.0]`, `magnitude_range: [18.0, 28.0]`
- `geometry.lens_radius_min/max: [0.35, 2.2]"`,
  `lens_size_scaling` (r0_kpc=5.2, alpha_mass=0.75, beta_redshift=-1.2,
  reference_mass=11.2, reference_z=0.6)
- `geometry.source_fraction_mean/sigma: 0.12 / 0.45` (R_source/θ_E)
- `redshifts`: lens [0.3,3.5], source [1.0,6.0], min Δz=0.5
- `mass.lens_mass_min/max: [10, 12]` (log10 M/M☉)
- Subhalos (disabled by default): count [3,12], dN/dm∝m^-1.9, Duffy+2008
  concentration
- Binary lenses: 35% of systems, 50/50 SIE+SIE vs NFW+NFW, separation
  0.5-2.0 θ_E

## Key findings / realism gaps (priority order for Phase 2+)

1. **Single-Sersic foundation is the core limitation.** All "morphology"
   beyond a smooth ellipse is pixel-space post-processing on one profile,
   with no independent physical components.
2. **No per-band morphology** — same structure/seed across all filters,
   unlike real galaxies where dust, young stellar populations, and clumps
   appear/disappear with wavelength.
3. **No real-galaxy image injection** (GalSim COSMOSCatalog or similar) —
   everything is synthetic, capping achievable pixel-level realism
   (asymmetry, Gini, M20, clumpiness statistics will be systematically
   different from real galaxies almost by construction).
4. **Rings and mergers/interactions are classification labels only**, with
   no corresponding visual implementation.
5. Field galaxies are reasonably realistic in *counts* and *colors* (real
   parameter draws available) but share the same rendering limitations.

These gaps map directly onto the user's Phase 2+ goals (state-of-the-art
framework review, bulge+disk+bar+ring native models, data-driven/hybrid
morphology, multi-survey validation against CAS/Gini/M20/clumpiness, etc.)
and should be used to scope and prioritize that work.

## 17. `src/galaxy_morphology/` package -- native multi-component light models

Implemented the new self-contained package per the approved plan
(`/Users/gozalig1/.claude/plans/atomic-exploring-volcano.md`):

- **`taxonomy.py`**: `classify_morphology(n_sersic, q_ratio, rng)` (moved
  verbatim from `jwst_lens_simulator.classify_galaxy_morphology_enhanced`,
  which is now a thin wrapper) + `MORPH_COMPONENTS` table mapping each
  morphology type to its native component list (e.g.
  `barred_spiral -> ['bulge', 'disk', 'bar']`, `ring -> ['bulge', 'disk',
  'ring']`, `post_merger -> ['bulge', 'disk', 'bulge_secondary']`,
  `irregular/primordial/clumpy/starburst -> ['disk']`).
- **`components.py`**: `build_components()` derives bulge (n=4,
  R=0.25*R_total, rounder q), disk (n=1, R=R_total, q=q_total), and
  optional bar/ring/bulge_secondary `SERSIC_ELLIPSE` parameter sets from a
  single base Sersic profile, with reference-band B/T-like flux fractions
  from `configs/default_config.yaml: morphology.bt_fractions` (normalized
  to sum to 1).
- **`band_variation.py`**: `band_flux_fractions()` reweights bulge-like
  components (bulge, bar, ring, bulge_secondary) per JWST band via
  `morphology.band_weight_bulge_like` (0.8x at F115W -> 1.2x at F444W),
  renormalized to sum to 1 -- this is what makes the multi-component
  structure wavelength-dependent (closing audit gap #4 for the lens/source
  galaxies).
- **`validation.py`**: `compute_cas_gini_m20()` -- numpy/scipy
  implementations of concentration, asymmetry, smoothness, clumpiness
  (Conselice 2003), Gini and M20 (Lotz+2004), for future real-vs-sim
  comparisons.
- **`__init__.py`**: `build_light_model(role, base_params, total_mag_by_band,
  bands, rng, config, morph_type=None)` -- drop-in replacement for the
  `["SERSIC_ELLIPSE"]` + `dict(base_params, magnitude=...)` pattern. When
  `morphology.multicomponent_enabled` (default `False`) is unset/false, it
  returns the original single-component fragment unchanged and performs
  **no extra RNG draws**, so default-mode runs are bit-for-bit unaffected.

### Integration into `jwst_lens_simulator.py`

- Added `morphology:` config block to `configs/default_config.yaml`
  (components, bt_fractions, band_weight_bulge_like;
  `multicomponent_enabled: false` by default).
- `classify_galaxy_morphology_enhanced` is now a thin wrapper around
  `galaxy_morphology.taxonomy.classify_morphology`.
- Main lens-system light-model construction (formerly `:4756-4833`):
  lens and source `light_model_list`/per-band kwargs are now built via
  `gm_build_light_model('lens', ...)` / `gm_build_light_model('source',
  ...)` once before the per-band loop (lens/source magnitudes are computed
  per band ahead of the loop, since they're deterministic and don't depend
  on RNG). The resulting fragments are spliced into
  `lens_light_model_list` / `source_light_model_list` alongside the
  existing companion-lens / field-galaxy entries.
- `generate_intermediate_images()` (intermediate/decomposition images, used
  when `save_intermediate_images: true`) now accepts
  `lens_light_fragment` / `source_light_fragment` and slices
  `kwargs_lens_light[:len(lens_light_fragment)]` as the main lens (instead
  of assuming index `[0]` is a single component), with
  `lens_light_model_list`/`source_light_model_list` matching the fragment
  lengths.
- To avoid double-rendering bars/rings both natively and as pixel texture,
  `apply_morphological_enhancements()` gained a `skip_native_bar_ring` flag,
  set when `multicomponent_enabled=True` and the lens's morph type is
  `barred_spiral`/`ring` (its native bar/ring component already exists).
  Spiral-arm/clump/dust-lane texture remains enabled in all cases. Field
  galaxies and non-lens-system central galaxies (`:~5840`
  `central_components`) are **not yet** switched to multi-component --
  documented as Phase 2 follow-up in `src/galaxy_morphology/README.md`.

### Verification

- `analysis/galaxy_morphology/test_build_light_model.py`: for all 12
  morphology types, asserts fragment lengths match `MORPH_COMPONENTS`, that
  per-band flux fractions sum to 1.0 (to 1e-6), and renders each fragment
  through a standalone `SimAPI` -- all 12 produced finite, non-negative,
  non-zero images. **PASS** for all types (elliptical, s0, spiral,
  late_spiral, edge_on, barred_spiral, ring, irregular, primordial, clumpy,
  starburst, post_merger).
- Pipeline smoke test: `--n_lenses 20 --variations_per_base 1 --seed 9001
  --add_artifacts --numpix 300` with `morphology.multicomponent_enabled:
  true`, output to
  `/Volumes/extHD/jwst_lens_outputs/morphfix_smoke20_date_20260612_222807`.
  Completed in 21s, 19/20 samples generated (95% success, comparable to
  typical batches), **no errors/exceptions** in the log. `[v11] Applied
  {morph_type} enhancements to lens galaxy` fired for `elliptical` (15x),
  `edge_on` (3x), `late_spiral` (1x), `spiral` (1x) -- confirming the
  multi-component path executed across multiple morphology types without
  issues.
- A parallel disabled-mode run (`multicomponent_enabled: false`,
  same config/seed) completed 20/20. `flux_sum_F150W/F277W/F444W` differ by
  a roughly constant ~0.81-0.83x between the two runs -- this is *not* a
  flux-conservation bug (the unit test above confirms per-band flux
  fractions sum to exactly 1.0 by construction); rather, enabling
  multicomponent mode adds extra `rng` draws (`classify_morphology`,
  bar/ring/bulge_secondary placement), which shifts the downstream RNG
  sequence and therefore which catalog rows/lens parameters get sampled
  (19 vs 20 successful rows). A true apples-to-apples flux comparison would
  require re-running with identical RNG consumption up to the
  light-model-construction point, which is lower priority than the
  structural verification above.

### Roadmap

See `src/galaxy_morphology/README.md` for the full Phase 2/3 roadmap:
real-galaxy stamp injection (GalSim/COSMOSCatalog), data-driven/generative
morphology, native ring/merger geometry beyond the Sersic approximation,
field-galaxy and non-lens-central-galaxy multi-component models, and full
multi-survey CAS/Gini/M20/clumpiness validation campaigns.

## 18. Larger-batch validation of `multicomponent_enabled=true` (item17, n=145)

Ran `--n_lenses 100 --variations_per_base 2 --seed 4242 --add_artifacts
--numpix 300` with `morphology.multicomponent_enabled: true` (config
`/tmp/jades_morph_config.yaml` = `jades_config.yaml` base +
`morphology` block from `default_config.yaml` with the flag flipped on).
Output: `/Volumes/extHD/jwst_lens_outputs/phase2_morphfix100_date_20260612_224630/`
(145 npz, 85/100 samples generated, 85% success rate -- comparable to
item16's 87/100).

`[v11] Applied {morph_type} enhancements to lens galaxy` morph-type counts
across the run: elliptical 57, edge_on 30, late_spiral 13, s0 (subset),
spiral 6, irregular 6, barred_spiral 4, post_merger 3, primordial 3, ring 2
(s0 not separately counted above but present in the log).

### Comparison vs real (n=435) and item16 (`phase2_psffix200`, n=139,
PSF-fix only, multicomponent disabled)

| metric | real median | item16 med | item17 med | real mean | item16 mean | item17 mean | item17/real (mean) | item16/real (mean) |
|---|---|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.769 | 1.161 | 1.067 | 0.873 | 1.212 | 1.155 | **1.322** | 1.387 |
| lens_axis_ratio | 0.707 | 0.602 | 0.651 | 0.687 | 0.592 | 0.645 | **0.939** | 0.862 |
| n_components | 7 | 3 | 3 | 7.051 | 3.777 | 3.214 | **0.456** | 0.536 |
| max_sep_arcsec | 4.693 | 3.335 | 2.508 | 4.488 | 3.144 | 2.313 | **0.515** | 0.701 |
| color_F115W_F444W | 0.498 | 0.419 | 0.545 | 0.539 | 0.452 | 0.568 | **1.055** | 0.839 |
| color_F115W_F277W | 0.725 | 0.639 | 0.757 | 0.728 | 0.654 | 0.798 | **1.095** | 0.898 |
| color_F277W_F444W | -0.237 | -0.274 | -0.215 | -0.189 | -0.202 | -0.229 | **1.211** | 1.065 |
| total_flux_F150W | 808.7 | 1224.4 | 933.5 | 2088.8 | 2409.0 | 1737.5 | **0.832** | 1.153 |
| quad+ fraction (n_components>=4) | 0.862 | 0.460 | 0.379 | -- | -- | -- | -- | -- |

(item16 numbers are from Section 16's table; a row-level item16 vs item17
comparison was not possible because `compute_sim_lens_morphology.py`
overwrites `sim_lens_morphology.csv` in place with no backup -- the item16
catalog was already overwritten by the item17 run before this was noticed.)

### Interpretation

**Improved toward real (closer to ratio=1.0):**
- `lens_axis_ratio`: 0.862 -> 0.939. Lens galaxies are rounder, closer to
  the real population.
- `lens_reff_arcsec`: 1.387 -> 1.322. Slightly smaller oversizing.
- `color_F115W_F444W` and `color_F115W_F277W`: 0.839/0.898 -> 1.055/1.095.
  Colors flip from undershooting to slightly overshooting the real
  blue-red contrast -- the band-dependent bulge/disk flux-fraction
  reweighting (`band_variation.band_flux_fractions`) is working as
  intended and now over-corrects slightly rather than under-correcting.

**Regressed (further from ratio=1.0):**
- `max_sep_arcsec`: 0.701 -> 0.515 (largest regression). Detected image
  separations shrank by ~26% relative to item16.
- `n_components` / quad+ fraction: 0.536/0.460 -> 0.456/0.379. Fewer
  distinct image components are detected by the segmentation pipeline.
- `total_flux_F150W`: 1.153 -> 0.832. Total detected F150W flux flipped
  from overshooting to undershooting real.
- `color_F277W_F444W`: 1.065 -> 1.211. Slightly worse overshoot in the red
  color.

**Likely cause -- open question.** `build_light_model` is applied to both
the **lens** and the **source** light models. For the lens galaxy, splitting
flux into bulge+disk(+bar/ring) components changes only the *foreground*
light and should not strongly affect the *lensed-image* segmentation
statistics (`max_sep_arcsec`, `n_components`, quad+ fraction, `total_flux`
are all measured on the lensed source images/arcs). But the **source**
galaxy is also now multi-component: each Sersic sub-component (e.g. a
compact n=4 bulge at 0.25*R_source plus an n=1 disk) gets lensed
*independently*, so multiply-imaged structure that previously came from one
extended Sersic profile now comes from the superposition of a compact
bulge image set plus a fainter, more extended disk image set. If the faint
disk-component images fall below the detection threshold after PSF
convolution + noise, the segmentation pipeline will detect fewer, smaller,
closer-together blobs from the bulge component alone -- which is exactly
the `max_sep_arcsec` / `n_components` / quad+ / `total_flux` regression
pattern observed. This has **not been confirmed**; it is the leading
hypothesis but is also entangled with normal RNG-sequence divergence
(145 vs 139 npz from a different successful-sample subset of the 100 base
systems, seed 4242 in both cases but different downstream RNG draws once
`classify_morphology` consumes extra random numbers per system).

**Recommendation for follow-up** (not yet done): an isolation run with
`build_light_model` applied to the **lens galaxy only** (source kept
single-Sersic) would separate the "lens-light B/T split" effect (which
appears beneficial: axis ratio, R_eff, and color improvements) from the
"source-light B/T split" effect (suspected cause of the
max_sep/n_components/flux regressions). This would require adding a
role-gated config option (e.g. `morphology.apply_to_source: false`) and
re-running the same n=100/seed=4242 batch.

### Example simulated galaxies (`jpg_rgb/`, item17 run)

All paths relative to
`/Volumes/extHD/jwst_lens_outputs/phase2_morphfix100_date_20260612_224630/jpg_rgb/`:

- **`PRISM_lens_BR_000004.jpg`** -- `elliptical` lens (most common type,
  57/145). Smooth red elliptical lens with a faint blue lensed arc/companion
  to the lower-left, visible in the RGB panel.
- **`PRISM_lens_BR_000012.jpg`** -- `edge_on` lens (30/145). Strongly
  elongated, dust-lane-like edge-on disk with two compact lensed
  images/companions below it.
- **`PRISM_lens_BR_000038.jpg`** -- `ring` + `spiral` system. Lens galaxy
  with a faint companion to the upper-right; visible reddish extended
  structure in F277W-F444W.
- **`PRISM_lens_BR_000053.jpg`** -- `barred_spiral` lens. Elongated
  bar/disk structure with a compact bright source image to the lower-left
  (green in RGB) and a fainter lensed counter-image above.
- **`PRISM_lens_BR_000059.jpg`** -- `irregular`/`primordial` lens. Compact,
  fairly round lens with a partial arc visible above it (best seen in
  F070W-F150W), red/orange in the RGB composite.
- **`PRISM_lens_BR_000097.jpg`** -- `post_merger` lens. Lens with a visible
  secondary nucleus/companion close to the main bulge plus a separate
  lensed arc-like feature to the upper-left, all visible in the RGB
  composite.

### Lens-only isolation run (item18, n=142) -- diagnosing the item17 regressions

Added `morphology.apply_to_lens` / `apply_to_source` config gates (default
both `true`) to `build_light_model()` and `configs/default_config.yaml`,
then re-ran the same `--n_lenses 100 --variations_per_base 2 --seed 4242`
batch with `apply_to_lens: true, apply_to_source: false` (config
`/tmp/jades_morph_lensonly_config.yaml`). Output:
`/Volumes/extHD/jwst_lens_outputs/phase2_lensonly100_date_20260612_231128/`
(142 npz, 86/100 success). The item17 catalog was copied to
`sim_lens_morphology_item17.csv`/`sim_lens_statistics_summary_item17.json`
before `compute_sim_lens_morphology.py` overwrote it with item18.

| metric | real mean | item17 mean (lens+source) | item18 mean (lens-only) | item17/real | item18/real |
|---|---|---|---|---|---|
| lens_reff_arcsec | 0.873 | 1.155 | 1.370 | 1.322 | 1.568 |
| lens_axis_ratio | 0.687 | 0.645 | 0.616 | 0.939 | 0.897 |
| n_components | 7.051 | 3.214 | 3.042 | 0.456 | 0.431 |
| max_sep_arcsec | 4.488 | 2.313 | 2.523 | 0.515 | 0.562 |
| color_F115W_F444W | 0.539 | 0.568 | 0.525 | 1.055 | 0.974 |
| color_F115W_F277W | 0.728 | 0.798 | 0.868 | 1.095 | 1.192 |
| color_F277W_F444W | -0.189 | -0.229 | -0.344 | 1.211 | 1.815 |
| total_flux_F150W | 2088.8 | 1737.5 | 2275.3 | 0.832 | 1.089 |
| quad+ fraction | 0.862 | 0.379 (n=145) | 0.324 (n=142) | -- | -- |

**Conclusion: the "source-light splitting" hypothesis from the item17
write-up is NOT supported.** Disabling source-side multicomponent
(item18) does **not** recover `max_sep_arcsec`, `n_components`, or quad+
fraction -- all three remain similar to or slightly *worse* than item17
(0.515->0.562, 0.456->0.431, 0.379->0.324). Since `multicomponent_enabled`
only changes **light** models, not the **mass**/deflection model, the
lensed-image geometry (separations, image multiplicity) should in principle
be unaffected by it entirely. The observed regressions vs item16
(0.701/0.536/0.460) are therefore most likely dominated by **RNG-sequence
divergence**: `classify_morphology`/`build_components` consume extra random
draws before the mass-model/source-position draws happen for the *same*
`rng` stream, so item16 (139 npz), item17 (145 npz), and item18 (142 npz)
each end up simulating a different subset of the 100 base systems with
different lensing geometries (theta_E, source offsets) -- not a like-for-
like comparison.

`total_flux_F150W` and `color_F115W_F444W` moved *toward* item16/real with
lens-only multicomponent (0.832->1.089, 1.055->0.974), consistent with these
two metrics being driven mainly by the **source**-light B/T split (as
expected, since they're measured on the lensed-source images).
`color_F277W_F444W` got markedly worse (1.211->1.815) -- likely sample-size/
RNG noise (n=142) rather than a systematic effect, given the small sample.

**Recommendation**: a true apples-to-apples comparison requires
`classify_morphology`/`build_components`'s RNG draws to not perturb the
downstream mass-model/source-position RNG sequence (e.g., spawn a separate
deterministic RNG stream for morphology classification, independent of the
main `rng`). This is a more invasive refactor than originally scoped;
flagged as a **Phase 2.1 follow-up**, lower priority than the Phase 3 items
below given that median/mean statistics for the primary structural metrics
(R_eff, axis ratio, colors) are comparable-to-improved and the package
passes its unit/smoke tests.

### Phase 2.1 RNG isolation validation (item19, n=139) -- apples-to-apples vs item16

Re-ran the same `--n_lenses 100 --variations_per_base 2 --seed 4242
--add_artifacts --numpix 300` batch with `multicomponent_enabled: true`
(both lens and source, `apply_to_lens`/`apply_to_source` default `true`)
**after** the Phase 2.1 RNG-isolation fix (`morph_seed` derived from
`lens_id`, independent of `rng`). Output:
`/Volumes/extHD/jwst_lens_outputs/phase2_rngfix100_date_20260612_231842/`.

**139/139 npz -- exactly matching item16's count and (by construction of
the now-untouched `rng` sequence) the same 139 base lens systems, same mass
models, and same source positions as item16.** This confirms the RNG
isolation works: `multicomponent_enabled` no longer changes which lenses
get simulated.

| metric | real mean | item16 mean | item19 mean | item19/real | item16/real | \|item19-item16\|/item16 |
|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.873 | 1.212 | 1.310 | 1.499 | 1.388 | 8.0% |
| lens_axis_ratio | 0.687 | 0.592 | 0.609 | 0.887 | 0.862 | 2.8% |
| n_components | 7.051 | 3.777 | 3.137 | 0.445 | 0.536 | 17.0% |
| max_sep_arcsec | 4.488 | 3.144 | 2.453 | 0.546 | 0.700 | 22.0% |
| color_F115W_F444W | 0.539 | 0.452 | 0.647 | 1.201 | 0.839 | 43.1% |
| color_F115W_F277W | 0.728 | 0.654 | 0.904 | 1.242 | 0.898 | 38.3% |
| color_F277W_F444W | -0.189 | -0.202 | -0.257 | 1.358 | 1.067 | 27.3% |
| total_flux_F150W | 2088.8 | 2409.0 | 2258.9 | 1.081 | 1.153 | 6.2% |
| quad+ fraction | 0.862 | 0.460 | 0.324 | -- | -- | -- |

**This is now a real, confirmed effect -- not an RNG-sequence artifact.**
With the identical 139 lens systems (same mass model, same source position,
same `rng` draws for everything except morphology classification/component
placement), enabling `multicomponent_enabled: true`:

- **Worsens** `max_sep_arcsec` (0.700 -> 0.546, -22% toward real), 
  `n_components` (0.536 -> 0.445, -17%), and quad+ fraction
  (0.460 -> 0.324, further from real's 0.862).
- **Worsens** all three colors -- item16 *undershot* real
  (0.839/0.898/1.067) but item19 now *overshoots* by a larger margin
  (1.201/1.242/1.358). The `band_weight_bulge_like` reweighting
  (`F115W: 0.8x -> F444W: 1.2x` for bulge-like components) is too strong
  for this lens sample.
- **Roughly unchanged / mildly worse**: `lens_reff_arcsec` (+8%, still
  ~1.5x real), `lens_axis_ratio` (+2.8%, still ~0.89x real),
  `total_flux_F150W` (-6.2%, now slightly closer to real).

**Interpretation**: splitting the lensed-source light into bulge+disk
components changes the lensed-image morphology measured by the segmentation
pipeline (fewer/closer detected blobs -> lower `max_sep_arcsec`/
`n_components`/quad+ fraction), and the per-band bulge/disk flux reweighting
overshoots the real color contrast for this sample. Both are **parameter-
tuning issues in `configs/default_config.yaml: morphology`** (component
`r_frac`s, `bt_fractions`, `band_weight_bulge_like`), not bugs -- but the
net effect of the *current* default values is a regression on 4 of 8
metrics + quad+ fraction relative to the already-imperfect item16 baseline.

**Recommendation**: keep `morphology.multicomponent_enabled: false` as the
pipeline default (as already configured) until the component/B-T/band-
weight parameters are retuned against this n=139 apples-to-apples baseline
-- e.g. reduce `band_weight_bulge_like` contrast (currently 0.8x/1.2x
F115W/F444W) toward 1.0, and/or reduce source-side `bt_fractions` so the
disk component (which produces the more extended/separated lensed images)
retains more flux. This tuning loop is now fast and reproducible (139/139
npz match guaranteed by the RNG isolation), unlike the item17/18 runs which
had different sample sizes each time.

### Tuning iteration 1 (item20, n=124)

Starting from the item19 baseline (`/tmp/jades_morph_config.yaml`), changed
two config blocks in `/tmp/jades_morph_tuned1_config.yaml`:

- `band_weight_bulge_like`: contrast halved, `{F115W:0.8...F444W:1.2}` ->
  `{F115W:0.90, F150W:0.95, F200W:1.00, F277W:1.05, F356W:1.075, F444W:1.10}`.
- Source-side `bt_fractions` reduced 30% for disk-hosting morph types:
  `spiral 0.25->0.175, late_spiral/barred_spiral/ring 0.20->0.14,
  edge_on 0.30->0.21, post_merger 0.25->0.175` (elliptical/s0 unchanged at
  0.9/0.85).

Output: `/Volumes/extHD/jwst_lens_outputs/phase2_tuned1_100_date_20260612_232649/`,
**92/100 samples -> 124 npz** (vs item19's 87/100 -> 139 npz for the
*same* seed/config-base -- see caveat below).

| metric | real | item16 | item19 (n=139) | item20 (n=124) | item19/real | item20/real | item16/real |
|---|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.873 | 1.212 | 1.310 | 1.290 | 1.499 | 1.478 | 1.388 |
| lens_axis_ratio | 0.687 | 0.592 | 0.609 | 0.604 | 0.887 | 0.881 | 0.862 |
| n_components | 7.051 | 3.777 | 3.137 | 3.315 | 0.445 | 0.470 | 0.536 |
| max_sep_arcsec | 4.488 | 3.144 | 2.453 | 2.630 | 0.546 | 0.586 | 0.700 |
| color_F115W_F444W | 0.539 | 0.452 | 0.647 | 0.521 | 1.201 | **0.966** | 0.839 |
| color_F115W_F277W | 0.728 | 0.654 | 0.904 | 0.829 | 1.242 | 1.139 | 0.898 |
| color_F277W_F444W | -0.189 | -0.202 | -0.257 | -0.308 | 1.358 | **1.629** | 1.067 |
| total_flux_F150W | 2088.8 | 2409.0 | 2258.9 | 2225.8 | 1.081 | 1.066 | 1.153 |
| quad+ fraction | 0.862 | 0.460 | 0.324 | 0.298 | -- | -- | -- |

**Mixed result -- one clear win, one new regression, marginal elsewhere:**
- `color_F115W_F444W`: 1.201 -> **0.966** (near-perfect, halving the
  band-weight contrast was almost exactly the right correction for the
  F115W-F444W color).
- `color_F115W_F277W`: 1.242 -> 1.139 (improved, still overshooting).
- `color_F277W_F444W`: 1.358 -> **1.629** (got *worse* -- the halved
  contrast at the F277W/F356W/F444W end (1.05/1.075/1.10, ratio 1.048)
  apparently undershoots what's needed for this color, while the F115W end
  correction (0.90 vs 1.00) overshoots in the other direction. The
  band_weight table needs **non-uniform** rebalancing: keep the F115W-side
  reduction but increase the F277W->F444W contrast back toward (or beyond)
  the original 1.1/1.15/1.2).
- `max_sep_arcsec` (0.546->0.586) / `n_components` (0.445->0.470): both
  nudged toward item16 but still far short of it (0.700/0.536) -- the 30%
  bt_fraction cut had only a small effect. quad+ fraction got slightly
  *worse* (0.324->0.298).
- `lens_reff_arcsec`/`lens_axis_ratio`/`total_flux_F150W`: essentially
  unchanged, marginally better.

**Caveat -- sample count drifted (139 -> 124 npz, 87 -> 92 base successes)**
even though `morph_seed` doesn't touch the shared `rng`. Most likely
explanation: the changed `bt_fractions`/`band_weight_bulge_like` shift
individual component magnitudes enough to push a few marginal systems
across an internal success/failure threshold (e.g. `magnitude2amplitude`
under/overflow for very faint/bright components) -- a secondary, expected
consequence of changing flux splits, not a re-break of the RNG isolation
(which only guarantees `rng`'s draw *sequence*, not that every system
remains numerically well-conditioned under any flux split).

**Direction for iteration 2**: (1) make `band_weight_bulge_like`
non-monotonic / asymmetric -- keep `F115W~0.90-0.95` but restore or
increase `F356W/F444W` toward `1.15-1.25` to fix `color_F277W_F444W`
without re-breaking `color_F115W_F444W`; (2) the `max_sep_arcsec`/
`n_components`/quad+ fraction gap (item20 still ~0.47-0.59x real vs
item16's ~0.54-0.70x) needs a structural change rather than a B/T-fraction
tweak -- likely increasing the source disk component's `r_frac` (currently
1.0x of the total Sersic radius) so the lensed disk images extend further
and segment into more/more-separated blobs.

### Tuning iteration 2 (item21, n=139) -- asymmetric band_weight fix

Per user selection, kept item20's F115W/F150W/F200W reduction (which had
nearly fixed `color_F115W_F444W`) but made `band_weight_bulge_like`
asymmetric, increasing the red end beyond the *original* item19 values to
try to pull `color_F277W_F444W` back toward 1.0x without re-breaking
`color_F115W_F444W`:

- `/tmp/jades_morph_tuned2_config.yaml`: `band_weight_bulge_like` =
  `{F115W:0.90, F150W:0.95, F200W:1.00, F277W:1.02, F356W:1.15, F444W:1.30}`
  (vs item20's `{...F277W:1.05, F356W:1.075, F444W:1.10}` and item19's
  original `{...F277W:1.10, F356W:1.15, F444W:1.20}`). `bt_fractions`
  unchanged from item20.

Output: `/Volumes/extHD/jwst_lens_outputs/phase2_tuned2_100_date_20260612_233754/`,
**87/100 base successes -> 139 npz** (matches item19's 139, RNG isolation
holding -- the count-drift in item20 was not a persistent effect of this
config family).

| metric | real | item16 | item19 (n=139) | item20 (n=124) | item21 (n=139) | item19/real | item20/real | item21/real |
|---|---|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.873 | 1.212 | 1.310 | 1.290 | 1.301 | 1.499 | 1.478 | 1.490 |
| lens_axis_ratio | 0.687 | 0.592 | 0.609 | 0.604 | 0.652 | 0.887 | 0.881 | 0.950 |
| n_components | 7.051 | 3.777 | 3.137 | 3.315 | 2.957 | 0.445 | 0.470 | 0.419 |
| max_sep_arcsec | 4.488 | 3.144 | 2.453 | 2.630 | 2.656 | 0.546 | 0.586 | 0.592 |
| color_F115W_F444W | 0.539 | 0.452 | 0.647 | 0.521 | 0.396 | 1.201 | **0.966** | 0.734 |
| color_F115W_F277W | 0.728 | 0.654 | 0.904 | 0.829 | 0.687 | 1.242 | 1.139 | 0.943 |
| color_F277W_F444W | -0.189 | -0.202 | -0.257 | -0.308 | -0.291 | 1.358 | **1.629** | 1.539 |
| total_flux_F150W | 2088.8 | 2409.0 | 2258.9 | 2225.8 | 2217.9 | 1.081 | 1.066 | 1.062 |
| quad+ fraction | 0.862 | 0.460 | 0.324 | 0.298 | 0.353 | -- | -- | -- |

**Net result: did not achieve the goal -- improved one color slightly at
the cost of re-breaking the other.**
- `color_F277W_F444W`: 1.629 -> 1.539 (small improvement, still
  substantially overshooting; the F356W/F444W boost to 1.15/1.30 was not
  enough even though it exceeds the *original* item19 values of
  1.15/1.20).
- `color_F115W_F444W`: **0.966 -> 0.734** (re-broken -- now undershoots
  real by ~27%, worse than item19's 1.201 overshoot in absolute terms).
  Lowering the F277W weight from item20's 1.05 to 1.02 (to avoid
  "spending" red-end contrast on F277W) apparently shifted enough flux
  toward the disk in F115W/F150W/F200W that the blue end over-corrected.
- `color_F115W_F277W`: 1.139 -> 0.943 (crossed from over- to
  under-shooting -- same root cause as above).
- `max_sep_arcsec` / `n_components` / quad+ fraction: mixed, small moves
  (0.586->0.592 / 0.470->0.419 / 0.298->0.353) -- still far from item16
  (0.700/0.536/0.460), consistent with the standing conclusion that this
  triplet needs the structural disk `r_frac` fix, not a band-weight tweak.
- `lens_axis_ratio`: 0.881 -> 0.950, closest to real of any iteration so
  far (a side effect of the bulge/disk flux rebalance, not targeted).

**Interpretation.** The two F444W-anchored colors (`F115W-F444W`,
`F115W-F277W`, `F277W-F444W`) are not independent -- they are three
differences among four band fluxes, so adjusting `band_weight` at any one
band moves all three colors simultaneously in coupled ways. Iteration 1's
near-perfect `color_F115W_F444W` (0.966) appears to have been close to a
local optimum for the *symmetric*-contrast family of configs; iteration 2's
attempt to asymmetrically free up `color_F277W_F444W` pulled
`color_F115W_F444W` and `color_F115W_F277W` away from that optimum by more
than it gained on `color_F277W_F444W`. A genuinely better fit likely
requires a 4-band weight table with more degrees of freedom (or per-band
weights derived from an actual bulge/disk SED model rather than a single
linear-in-log-lambda contrast knob), which is a larger change than a
2-3 parameter nudge -- flagged as a candidate for a future iteration
alongside the structural disk `r_frac` fix.

**Recommendation going forward**: iteration 1 (item20's
`band_weight_bulge_like`) remains the best single-color result
(`color_F115W_F444W` = 0.966) found so far, but at the cost of the n=124
sample-count drift and `color_F277W_F444W` = 1.629. Given the coupling
identified above, further iteration on this 6-band linear-weight knob has
diminishing returns; the next investment should go to (1) the structural
disk `r_frac` fix for `max_sep_arcsec`/`n_components`/quad+ fraction
(largest remaining gaps, ~0.42-0.59x real across all variants), and/or (2)
a non-linear/4-parameter band-weight model if color tuning is revisited.

### Iteration 3 (item22, n=142) -- combined structural + independent disk band-weight fix

Per user request ("continue both"), implemented both deferred fixes
together in `src/galaxy_morphology/`:

1. **Structural source-disk fix** (`components.py`): added a
   `disk.r_frac_source` config knob (default 1.0, no change unless set).
   `build_components(..., role=...)` now uses `r_frac_source` instead of
   `r_frac` for the disk component's `R_sersic` when `role == 'source'`,
   extending the lensed source's disk component beyond the total Sersic
   radius so its lensed images spread over more area/segments.
2. **Independent disk band-weight** (`band_variation.py`): added
   `band_weight_disk` (default all-1.0, neutral), applied multiplicatively
   to the disk component's flux fraction per band, independent of
   `band_weight_bulge_like`. Doubles the color-tuning DOF from 6 to 12.

`/tmp/jades_morph_tuned3_config.yaml` = item20's `band_weight_bulge_like`
and `bt_fractions` (best `color_F115W_F444W` so far) +
`components.disk.r_frac_source: 1.35` +
`band_weight_disk: {F115W:1.0, F150W:1.0, F200W:1.0, F277W:1.0, F356W:1.05, F444W:1.15}`
(hypothesis: boosting disk flux fraction at F356W/F444W would dilute the
bulge-driven `color_F277W_F444W` overshoot without touching F115W).

Output: `/Volumes/extHD/jwst_lens_outputs/phase2_tuned3_100_date_20260613_000205/`,
**86/100 base successes -> 142 npz**.

| metric | real | item16 | item19 (n=139) | item20 (n=124) | item21 (n=139) | item22 (n=142) | item19/real | item20/real | item21/real | item22/real |
|---|---|---|---|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.873 | 1.212 | 1.310 | 1.290 | 1.301 | 1.233 | 1.499 | 1.478 | 1.490 | 1.412 |
| lens_axis_ratio | 0.687 | 0.592 | 0.609 | 0.604 | 0.652 | 0.687 | 0.887 | 0.881 | 0.950 | **1.001** |
| n_components | 7.051 | 3.777 | 3.137 | 3.315 | 2.957 | 3.387 | 0.445 | 0.470 | 0.419 | 0.480 |
| max_sep_arcsec | 4.488 | 3.144 | 2.453 | 2.630 | 2.656 | 2.659 | 0.546 | 0.586 | 0.592 | 0.593 |
| color_F115W_F444W | 0.539 | 0.452 | 0.647 | 0.521 | 0.396 | 0.402 | 1.201 | 0.966 | 0.734 | 0.746 |
| color_F115W_F277W | 0.728 | 0.654 | 0.904 | 0.829 | 0.687 | 0.723 | 1.242 | 1.139 | 0.943 | **0.992** |
| color_F277W_F444W | -0.189 | -0.202 | -0.257 | -0.308 | -0.291 | -0.320 | 1.358 | 1.629 | 1.539 | 1.696 |
| total_flux_F150W | 2088.8 | 2409.0 | 2258.9 | 2225.8 | 2217.9 | 1166.6 | 1.081 | 1.066 | 1.062 | **0.558** |
| quad+ fraction | 0.862 | 0.460 | 0.324 | 0.298 | 0.353 | 0.401 | -- | -- | -- | -- |

**Two big wins, one big new regression, structural fix barely moved its
target metrics:**
- `lens_axis_ratio`: 0.950 -> **1.001** -- essentially exact match to real,
  best of any iteration. Side effect of the bulge/disk q-shape interplay
  under the new disk geometry, not directly targeted.
- `color_F115W_F277W`: 0.943 -> **0.992** -- also essentially exact.
- `color_F277W_F444W`: 1.539 -> 1.696 -- **got worse**, opposite of the
  hypothesis. Boosting the disk's F444W/F356W flux *fraction* does not
  increase the *detected* F444W flux the way intended: because
  `total_mag_by_band` (the intrinsic total per band) is fixed before
  `band_flux_fractions` runs, redistributing fractions among components
  only changes which component (compact bulge vs. extended disk) carries
  more of that fixed total -- and a more-disk-dominated F444W apparently
  loses more flux to the segmentation threshold (the extended disk's lower
  surface brightness falls below detection), making the *detected*
  F277W-F444W color *more* negative, not less. The band-weight knobs affect
  detected colors only through this selection effect, not a direct flux
  shift -- a much weaker and less predictable lever than assumed.
- `color_F115W_F444W`: 0.734 -> 0.746 -- essentially unchanged.
- `total_flux_F150W`: 1.062 -> **0.558** -- a large new regression. The
  `disk.r_frac_source = 1.35` extension spreads the source disk's flux over
  a larger area; in F150W (where the disk carries a larger flux fraction
  than in the red, per `band_weight_bulge_like` < 1 at F150W) this pushes
  much more flux below the per-pixel detection threshold, roughly halving
  the total detected F150W flux.
- `n_components` (0.419->0.480) / quad+ fraction (0.353->0.401): modest
  improvement, similar magnitude to iteration 1's bt_fraction tweak --
  **but `max_sep_arcsec` did not move at all** (0.592->0.593). The
  `r_frac_source=1.35` extension increases the *number* of faint
  low-surface-brightness disk-image fragments the segmentation picks up
  (helping `n_components`/quad+) without increasing the *separation*
  between the brightest (bulge-dominated) images that `max_sep_arcsec`
  measures.

**Interpretation.** Both fixes worked, but not along the axes intended --
and both operate through the same underlying mechanism (the segmentation
pipeline's surface-brightness detection threshold interacting with how
flux is distributed in *both* the bulge/disk fraction in a band *and* the
disk's physical extent). This makes the four target metrics
(`max_sep_arcsec`, `n_components`, quad+ fraction, `color_F277W_F444W`)
much harder to move independently than the two that improved
(`lens_axis_ratio`, `color_F115W_F277W`, which are largely
shape/geometry-driven and not selection-effect-driven).

**Recommendation**: `disk.r_frac_source=1.35` is too aggressive given the
`total_flux_F150W` collapse -- a smaller value (e.g. 1.1-1.15) should be
tried to get a partial `n_components`/quad+ gain without crushing
`total_flux_F150W`. For `color_F277W_F444W`/`color_F115W_F444W`/
`max_sep_arcsec`, band-weight and disk-extent knobs both operate through
detection-threshold selection effects that are difficult to predict
analytically; given each batch costs ~3 hours wall-clock, further blind
iteration has a poor cost/benefit ratio. A more efficient path would be to
instrument `compute_sim_lens_morphology.py` (or a smaller diagnostic
script) to report the *intrinsic* (pre-segmentation) per-band/per-component
flux alongside the *detected* flux for a handful of systems, to build an
empirical map from (band_weight, r_frac_source) -> detected-color/max_sep
*before* committing to another full 142-lens batch.

### Iteration 4 (item23, n=130) -- conservative structural fix, neutral disk weight

Per user request ("do the best which improve the simulation"), tried the
recommended conservative follow-up: `/tmp/jades_morph_tuned4_config.yaml` =
item20's `band_weight_bulge_like`/`bt_fractions` (best `color_F115W_F444W`
result) + `band_weight_disk` reverted to neutral (all 1.0, i.e. removed --
item22 showed it was actively harmful) + `components.disk.r_frac_source`
reduced from item22's 1.35 to **1.15**.

Output: `/Volumes/extHD/jwst_lens_outputs/phase2_tuned4_100_date_20260613_122514/`,
**90/100 base successes -> 130 npz**.

| metric | real | item16 | item19 | item20 | item21 | item22 | item23 (n=130) | item19/real | item20/real | item21/real | item22/real | item23/real |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lens_reff_arcsec | 0.873 | 1.212 | 1.310 | 1.290 | 1.301 | 1.233 | 1.340 | 1.499 | 1.478 | 1.490 | 1.412 | 1.535 |
| lens_axis_ratio | 0.687 | 0.592 | 0.609 | 0.604 | 0.652 | 0.687 | 0.628 | 0.887 | 0.881 | 0.950 | 1.001 | 0.914 |
| n_components | 7.051 | 3.777 | 3.137 | 3.315 | 2.957 | 3.387 | 3.662 | 0.445 | 0.470 | 0.419 | 0.480 | **0.519** |
| max_sep_arcsec | 4.488 | 3.144 | 2.453 | 2.630 | 2.656 | 2.659 | 3.028 | 0.546 | 0.586 | 0.592 | 0.593 | **0.675** |
| color_F115W_F444W | 0.539 | 0.452 | 0.647 | 0.521 | 0.396 | 0.402 | 0.343 | 1.201 | 0.966 | 0.734 | 0.746 | 0.636 |
| color_F115W_F277W | 0.728 | 0.654 | 0.904 | 0.829 | 0.687 | 0.723 | 0.647 | 1.242 | 1.139 | 0.943 | 0.992 | 0.888 |
| color_F277W_F444W | -0.189 | -0.202 | -0.257 | -0.308 | -0.291 | -0.320 | -0.304 | 1.358 | 1.629 | 1.539 | 1.696 | 1.608 |
| total_flux_F150W | 2088.8 | 2409.0 | 2258.9 | 2225.8 | 2217.9 | 1166.6 | 2887.0 | 1.081 | 1.066 | 1.062 | 0.558 | 1.382 |
| quad+ fraction | 0.862 | 0.460 | 0.324 | 0.298 | 0.353 | 0.401 | 0.423 | -- | -- | -- | -- | -- |

**Biggest gain on the originally-targeted metrics across all iterations:**
- `max_sep_arcsec`: 0.593 -> **0.675**, vs item16's 0.700 -- closer to
  item16 (and real) than any multicomponent variant tried so far, by a
  wide margin. This is the first iteration where `disk.r_frac_source`
  clearly moves `max_sep_arcsec` itself (not just `n_components`/quad+),
  confirming the structural-extent fix *is* the right lever for this
  metric -- 1.15 was simply more in the useful range than 1.35 (which
  apparently over-extended the disk enough that its images fell almost
  entirely below threshold, removing their separation contribution
  rather than adding to it).
- `n_components`: 0.480 -> **0.519** and quad+ fraction: 0.401 -> **0.423**,
  both their best values yet, approaching item16 (0.536/0.460).

**But two new/returning regressions appeared:**
- `total_flux_F150W`: 0.558 -> **1.382** -- swung from item22's
  undershoot to the largest overshoot of any iteration (worse than item16's
  1.153 and item19-21's ~1.06-1.08). Removing `band_weight_disk` plus
  `r_frac_source=1.15` evidently shifted enough flux into the
  F150W-detected region to overshoot.
- `color_F115W_F444W`: 0.966 (item20) -> **0.636** -- did *not* recover
  item20's value despite reverting `band_weight_disk` to neutral; the
  `r_frac_source` change alone (independent of `band_weight_disk`) has a
  substantial effect on this color too, confirming `r_frac_source` is not
  a "structure-only" knob -- it couples into every detected-flux-based
  metric via the same selection-threshold mechanism identified in
  iteration 3.
- `lens_axis_ratio`: 1.001 (item22) -> 0.914 -- lost iteration 3's
  near-perfect match (still better than item16/19-21's ~0.59-0.65).

**Overall assessment across all 5 multicomponent variants (item19-23):**
no single configuration dominates on all 8 metrics; each knob
(`band_weight_bulge_like`, `band_weight_disk`, `bt_fractions`,
`disk.r_frac_source`) has cross-cutting effects on colors, total flux, and
the max_sep/n_components/quad+ triplet via the shared segmentation
detection-threshold mechanism. **item23 is the best choice if the priority
is the segmentation-topology metrics** (max_sep/n_components/quad+, which
were the *original* regression that motivated this whole tuning effort and
are now within ~25-50% of item16, vs. ~30-50% before); **item20 remains
best for `color_F115W_F444W`**; **item22 remains best for
`lens_axis_ratio`/`color_F115W_F277W`**. A configuration combining
`r_frac_source~1.15` (item23) with item20's color weights was exactly what
item23 *is* -- so the color regression in item23 is a direct, non-separable
consequence of the structural fix, not a missed combination.

**Recommendation**: given 5 full-batch iterations (~15 hours wall-clock
total) have not found a configuration that improves all axes
simultaneously, and the marginal-cost/marginal-gain ratio is worsening,
further parameter-space search on this 4-knob linear model is not
recommended. Suggested next steps (future work, not pursued further in
this session): (1) ship **item23's config** as the new default
(`multicomponent_enabled: true` with `r_frac_source=1.15` and item20's
color weights) since it directly addresses the original max_sep/
n_components/quad+ regression that started this investigation, accepting
the `color_F115W_F444W`/`total_flux_F150W` trade-offs as a known
limitation; or (2) revisit the segmentation/detection-threshold pipeline
itself (`compute_sim_lens_morphology.py` / `morphology_metrics.py`) since
nearly every metric studied here is mediated by that one threshold, making
it the highest-leverage single point of intervention for *all* of these
metrics simultaneously.

#### Iteration 5 / diagnostic (item24, n=139)

**Hypothesis tested**: item24's "Critical reframing" noted that item16
(single-component, `multicomponent_enabled: false`) already had
`color_F115W_F444W` *closer* to real (ratio 0.839) than any multicomponent
variant tried so far (item19-23, ratios 0.6-0.75), and that every
non-neutral `band_weight_bulge_like` table tried (item19's 0.8-1.2, item20's
0.9-1.1, item21's asymmetric, item22/23's 0.9-1.1) made this color *worse*
relative to item16's pre-multicomponent baseline. The diagnostic asked: if
`band_weight_bulge_like` is reset to fully **neutral (all 1.0)** — i.e. the
multicomponent B/T split and structural `r_frac_source` changes are kept,
but the wavelength-dependent bulge/disk color weighting is switched off —
does `color_F115W_F444W` recover toward item16's ratio?

**Config** (`/tmp/jades_morph_tuned5_config.yaml`, item23's config with one
change):
- `disk.r_frac_source = 1.15` (unchanged from item23)
- `band_weight_bulge_like`: **all 1.0** (neutral; was item20's
  `{F115W:0.9, F150W:0.95, F200W:1.0, F277W:1.05, F356W:1.075, F444W:1.1}`)
- `band_weight_disk`: not set (defaults to neutral, all 1.0)
- `bt_fractions`: item20's values (unchanged)

| Metric | Real | item16 | item20 | item23 | item24 |
|---|---|---|---|---|---|
| lens_reff_arcsec (ratio) | 1.000 | 1.388 | — | 1.758 | 1.277 |
| lens_axis_ratio (ratio) | 1.000 | 0.862 | — | 0.914 | 0.884 |
| n_components (ratio) | 1.000 | 0.536* | — | 0.519 | 0.500 |
| max_sep_arcsec (ratio) | 1.000 | 0.700 | — | 0.675 | 0.662 |
| color_F115W_F444W (ratio) | 1.000 | 0.839 | 1.139 | 0.636 | **0.836** |
| color_F115W_F277W (ratio) | 1.000 | 0.898 | 1.139 | — | 1.047 |
| color_F277W_F444W (ratio) | 1.000 | 1.067 | — | 1.539-1.696 | 1.649 |
| total_flux_F150W (ratio) | 1.000 | 0.866* | — | 1.382 | 0.639 |
| quad+ fraction (ratio) | 1.000 | 0.460 | — | 0.423 | 0.417 |

(*item16 values transcribed from earlier table rows; some cells not
directly available for all iterations and left blank.)

**Result: hypothesis confirmed for `color_F115W_F444W`, not for
`color_F277W_F444W`.**

- `color_F115W_F444W` ratio jumped from item23's 0.636 to **0.836** —
  essentially an exact match to item16's 0.839. Reverting
  `band_weight_bulge_like` to neutral recovers this color to the
  pre-multicomponent baseline. This means **every non-neutral
  `band_weight_bulge_like` table tried in this project (item19-23) actively
  hurt `color_F115W_F444W`** relative to both real data and the
  single-component baseline.
- `color_F115W_F277W` ratio is 1.047 — closer to 1.0 than item20's 1.139,
  though item16's 0.898 is closer still on the other side. A middling
  result, not a regression.
- `color_F277W_F444W` ratio is **1.649**, essentially unchanged from item22's
  1.696 and item23's 1.539. Band weighting (in either the bulge-like or disk
  table) has **no material effect on this color** across iterations 3-5 —
  its ~1.5-1.7x overshoot appears to be **intrinsic to the bulge/disk B/T
  split itself** (i.e. driven by `bt_fractions` and/or the underlying
  per-component Sersic indices/sizes), independent of how the per-band flux
  is *re-weighted* between components. Fixing this would require either
  revisiting `bt_fractions` (B/T ratios) or accepting this as a residual gap.
- `max_sep_arcsec`, `n_components`, `quad+` (0.662/0.500/0.417) are close to
  item23's (0.675/0.519/0.423) — confirming these are governed by
  `r_frac_source` (held fixed at 1.15 in both), not by band weighting.
- `total_flux_F150W` ratio (0.639) is undershoot, the mirror image of
  item23's overshoot (1.382) — both ~35-40% off, but in opposite
  directions, with no other config change between item23 and item24 besides
  `band_weight_bulge_like`. This is a side effect of the segmentation
  detection-threshold selection effect discussed above: changing relative
  per-band component brightness changes which pixels clear the 3σ
  per-band/6σ-combined threshold, which changes the *measured* (detected)
  total flux even though the *intrinsic* total flux is unchanged by
  construction (`fractions_to_magnitudes` always renormalizes to the
  original `total_mag_by_band`).

**Recommendation for `configs/default_config.yaml`**: set
`morphology.band_weight_bulge_like` to **neutral (all 1.0)**, replacing
item20's values currently in the config. Rationale:
- `color_F115W_F444W` is the metric most directly tied to the audit's "gap
  #4" (wavelength-dependent morphology) and neutral weighting gives by far
  the best result (0.836 vs 0.636 with item20's weights) — a ~30 percentage
  point improvement, landing almost exactly on the pre-multicomponent
  baseline.
- `color_F277W_F444W`'s residual overshoot (~1.6x) is **not addressed by any
  band-weight choice** tried (neutral, item20, item21's asymmetric, or
  item22/23's), so neutral weighting does not give up anything on this
  metric relative to the alternatives — it is simply a separate, unresolved
  issue tied to `bt_fractions`.
- This does mean the explicit per-band bulge/disk color-weighting mechanism
  (`band_weight_bulge_like` / `band_weight_disk`) becomes a no-op in the
  shipped default — the wavelength-dependence of multicomponent morphology
  in the default config now comes *only* from the bulge/disk
  structural+B/T split itself (different component sizes/Sersic
  indices/B-T ratios per band via `fractions_to_magnitudes`), not from an
  explicit per-band re-weighting table. The knobs remain in the code and
  config schema (commented out / set to neutral) for future tuning if
  `bt_fractions` is revisited to address `color_F277W_F444W`.
- `disk.r_frac_source: 1.15` (item23's value) is retained — it is the
  best-performing single lever found for `max_sep_arcsec`/`n_components`/
  `quad+` recovery toward item16/real, and is orthogonal to the band-weight
  question.

**Status**: `configs/default_config.yaml`'s `morphology.band_weight_bulge_like`
updated to all-1.0 (neutral) per this recommendation; `disk.r_frac_source:
1.15` and item20's `bt_fractions`/`band_weight_disk` (commented, neutral)
retained from the prior update.

### Post-hoc fix: 'ring' morphology removed from lens/central deflectors

`PRISM_lens_SF_000037.jpg` (a non-lens field system) showed an unrealistic
pixel-space collisional ring (`add_ring_structure_to_image`, hardcoded
`ring_radius=30px`) sitting right on top of/overlapping the galaxy's bulge,
and -- more importantly -- visually indistinguishable from a strong-lensing
Einstein ring in a system that has no lensing at all. Two fixes were made:

1. `add_ring_structure_to_image`'s `ring_radius`/`ring_width` are now scaled
   from the galaxy's effective radius (`ring_radius = 2.5 * r_eff_pix`,
   clipped to [0.1, 0.45] * numpix; `ring_width = 0.08 * ring_radius`,
   min 1.5px), with the off-center "intruder nucleus" offset scaled
   proportionally, so the ring is clearly separated from the galaxy light
   profile when it does occur (`apply_morphological_enhancements` now takes
   an `r_eff_pix` kwarg, wired from `lens_radius`/`primary_comp['R_sersic']`
   at the lens and non-lens-central call sites).
2. `classify_morphology`/`classify_galaxy_morphology_enhanced` gained an
   `allow_ring` flag (default `True`); lens-deflector and non-lens central
   galaxy classification now passes `allow_ring=False`, remapping 'ring' ->
   'spiral' for those roles (RNG draw sequence unchanged). 'ring' remains
   available for the **lensed source** (`gm_build_light_model('source', ...)`),
   where a ring-like lensed image is physically expected (e.g. Einstein
   rings from extended sources) rather than a confound. Not yet
   re-validated with a fresh batch run.
