# galaxy_morphology

Native multi-component (bulge/disk/bar/ring/post-merger) light models for
the strong-lens pipeline, plus a CAS/Gini/M20/clumpiness validation module.

## Why this exists

The Phase 1 audit
(`analysis/galaxy_morphology/reports/phase1_audit.md`) found that every
galaxy in `jwst_lens_simulator.py` was rendered as a single
`SERSIC_ELLIPSE` profile, with all visual structure (arms, bars, clumps,
dust) applied as pixel-space post-processing using the **same RNG seed for
every band** (no wavelength-dependent morphology), and with rings/mergers
existing only as classification labels.

## What this package does

`build_light_model(role, base_params, total_mag_by_band, bands, rng,
config, morph_type=None)` is a drop-in replacement for the
`["SERSIC_ELLIPSE"]` + `dict(base_params, magnitude=...)` pattern used at
the lens/source/field light-model construction sites in
`jwst_lens_simulator.py`.

- `taxonomy.py`: morphology classification (`classify_morphology`, moved
  from `jwst_lens_simulator.classify_galaxy_morphology_enhanced`) and the
  `MORPH_COMPONENTS` table mapping morphology type -> native component list.
- `components.py`: `build_components()` generates bulge/disk/bar/ring/
  bulge_secondary `SERSIC_ELLIPSE` parameter sets from a single base Sersic
  profile, with reference-band (B/T-like) flux fractions.
- `band_variation.py`: `band_flux_fractions()` makes bulge-like components
  relatively redder than the disk via a per-band weight table, producing
  wavelength-dependent structure without re-rendering pixel textures.
- `validation.py`: `compute_cas_gini_m20()` -- Concentration, Asymmetry,
  Smoothness, Clumpiness, Gini, M20, implemented directly with numpy/scipy
  (Conselice 2003 / Lotz et al. 2004 definitions), for real-vs-sim
  comparison.

Behavior is gated by `config['morphology']['multicomponent_enabled']`
(default `False`): when off, `build_light_model` returns the original
single-`SERSIC_ELLIPSE` fragment unchanged, so it is safe to wire into all
call sites ahead of validation.

The existing pixel-space `add_spiral_arms_to_image` /
`add_clumpy_structure_to_image` / `add_dust_lane_to_image` /
`add_bar_structure_to_image` / `add_ring_structure_to_image` /
`apply_morphological_enhancements` functions are unchanged and continue to
run on top of the (now possibly multi-component) rendered image, adding
texture on top of the new structural/SED realism. When
`multicomponent_enabled=True`, the pixel-space bar/ring additions in
`apply_morphological_enhancements` are skipped for morph types that already
received a native bar/ring component, to avoid double-rendering.

Note: `add_ring_structure_to_image` implements a *collision (Cartwheel-type)
ring* -- a different physical feature from the *inner/outer ring* structural
component modeled here as a thin annular Sersic component. Both can in
principle coexist (a barred/ring galaxy with a faint collisional ring is
rare but not impossible); currently the native `ring` component only gates
the pixel-space ring path off to avoid two unrelated ring features stacking.

## TNG particle-driven physical morphology (`tng_particle_light.py`)

`galaxygenius_stamps.py` (SKIRT-rendered, pre-baked image cutouts pasted in
and rescaled) is **disabled by default** as a production path
(`galaxygenius_stamps.enabled/source_enabled/lens_enabled: false` in
`configs/default_config.yaml`) -- pasting a fixed-size cutout and rescaling
it to very different angular sizes repeatedly produced edge artifacts
(square crops, then circular crops from the edge taper), and a single fixed
subhalo cannot represent the diversity of the simulated population.

`tng_particle_light.build_tng_particle_interpol_kwargs` replaces that
approach: for a TNG subhalo with a locally-downloaded particle cutout
(`tng_galaxy_selector.local_particle_path`), it procedurally bins the
subhalo's star/gas particles into a fresh per-band surface-brightness map
each time it's called -- projected at a random viewing angle, colored by
per-particle age/metallicity via an approximate SSP table, and
dust-attenuated using the projected gas distribution -- then returns the
same `INTERPOL` kwargs shape as the stamp builders (`image`, `center_x`,
`center_y`, `phi_G`, `scale`, `magnitude`). This is a forward model from
particle data, not image compositing, so it can be rendered at any angular
size/orientation without introducing cutout-edge artifacts.

Gated by `tng_mode.particle_morphology` (off by default):

```yaml
tng_mode:
  particle_morphology:
    enabled: false
    lens_enabled: true
    lens_fraction: 1.0
    source_enabled: true
    source_fraction: 1.0
```

At the lens-light and lensed-source `build_light_model` call sites in
`jwst_lens_simulator.py`, if enabled and the matched `tng_lens`/`tng_source`
subhalo has a local particle file, the particle-driven `INTERPOL` profile is
used; otherwise it falls back to `build_light_model` (Sersic
bulge/disk/bar/ring), which remains the default/fast baseline. Verified with
a 10-lens run (`configs/tng_mode_test10.yaml`, seed 4242): the one matched
subhalo with local particle data (lens BR/SF_000005's lensed source,
snap21/subhalo8108) rendered as a clumpy, asymmetric ring with no edge
artifacts; the other 9 lenses (no local particle file for their matched
subhalos) fell back to the Sersic models unchanged.

Currently only **820/31500** (2.6%) local-catalog subhalos have downloaded
particle cutouts (`/Volumes/extHD/galaxygenius_build/workspace/data/`), via
`scripts/batch_fetch_galaxygenius_stamps.py` -- so most lenses/sources fall
back to Sersic until more particle cutouts are downloaded.

### Phase 1 simplifications (documented, not yet upgraded)

- **Approximate SSP**: `_stellar_band_luminosity` uses a small hand-tuned
  age/metallicity -> luminosity table per JWST band, not a real stellar
  population synthesis code. A drop-in upgrade would replace this function
  with FSPS or BPASS lookups while keeping the same per-particle interface.
- **Simplified dust**: a single Calzetti-like power law applied to a
  gas-surface-density-derived V-band optical depth map, with one tunable
  constant (`_DUST_K`); not a radiative-transfer calculation.
- **Random orientation**: each subhalo gets one random inclination/PA per
  process (cached), independent of the lens/source's own assigned
  ellipticity (`phi_G` only sets the *placement* angle of the resulting
  image, not the intrinsic 3D orientation of the particle projection).

## Phase 2: field/companion particle morphology

Extends the Phase 1 mechanism to the two remaining light-model roles:

- **Field galaxies**: `tag_field_galaxies_with_tng_particles` runs after
  `apply_tng_field_overrides` and, for galaxies with a TNG match
  (`tng_info`) that aren't already a GalaxyGenius stamp, tags a fraction
  (`particle_morphology.field_fraction`) with `_tng_particle_file` if the
  matched subhalo has a local particle cutout. `field_galaxy_light_model_types`
  then returns `"INTERPOL"` for these, and
  `apply_real_jwst_colors_to_field_galaxies` builds the INTERPOL kwargs via
  `build_tng_particle_interpol_kwargs` using the K-corrected magnitude as
  `magnitude_ref` and `R_sersic * galaxygenius_stamps.field_target_size_factor`
  (default `4.0`) as the rendered angular size.
- **Companion lens light** (binary/group lens pairs): immediately after
  `companion_lens_light` is built, a TNG subhalo is matched at the
  companion's mass-ratio-scaled stellar mass
  (`query_tng_properties(lens_z, lens_mass_log10 + log10(mass_ratio), ...)`).
  If `particle_morphology.companion_enabled` and a local particle cutout
  exists for the match, the companion's `lens_light_model_list` entry and
  per-band kwargs become `INTERPOL` (via `build_tng_particle_interpol_kwargs`,
  same `field_target_size_factor` sizing) instead of `SERSIC_ELLIPSE`.

Both gated by new `tng_mode.particle_morphology.{field,companion}_enabled`/
`{field,companion}_fraction` (off by default, mirroring the Phase 1
`lens_*`/`source_*` keys). Verified via a standalone integration script
exercising `tag_field_galaxies_with_tng_particles`,
`field_galaxy_light_model_types`, `apply_real_jwst_colors_to_field_galaxies`,
and the companion `build_tng_particle_interpol_kwargs` call with known local
cutouts (logM~9.2 for the field case, logM~11.1 for the companion case),
including an end-to-end lenstronomy render. A 10-lens pipeline run
(`configs/tng_mode_test10.yaml`, seed 4242, all four `particle_morphology`
flags on) completed without errors but did not naturally trigger either new
branch -- consistent with the ~1.6% local-cutout coverage for field-relevant
masses (logM 9-10) measured for Phase 1.

## Phase 3: multi-component morphology for field galaxies + non-lens centrals

Extends `build_light_model` to the two remaining single-Sersic call sites:

- **Field galaxies** (Sersic fallback in `apply_real_jwst_colors_to_field_galaxies`):
  `field_galaxy_light_model_types` now returns the flattened per-galaxy
  fragment (e.g. `['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']` for a bulge+disk
  field galaxy) instead of a single `"SERSIC_ELLIPSE"`/`"INTERPOL"`. For
  galaxies with a COSMOS-Web `real_morph_type`, the value is auto-classified
  via `classify_morphology` (no direct vocabulary mapping is attempted);
  `_morph_seed`/`_morph_type_resolved` are cached on the galaxy dict so the
  same fragment structure is reused across all bands (only magnitude varies).
- **Non-lens central components**: each `central_components` entry gets a
  pre-computed fragment from `build_light_model('field', ..., morph_type=
  str(comp['morph_type']).lower(), morph_seed=...)`, flattened into
  `lens_light_model_list`, with per-band kwargs built the same way as the
  lens/source call sites.

Both gated by `morphology.multicomponent_enabled` (no new config keys).
Verified standalone (field galaxies + non-lens centrals, mixed
multi-component `lens_light_model_list`, end-to-end lenstronomy render) and
via a 10-lens+2-non-lens pipeline run (seed 4242, `multicomponent_enabled:
true`) with no errors.

## Phase 4: native ring/merger geometry via procedural INTERPOL profiles

Replaces the thin-Sersic `ring` component and the `bulge_secondary` tail of
`post_merger` with true annular / one-sided tidal-tail surface-brightness
maps, rendered procedurally (numpy + `scipy.ndimage.gaussian_filter` for
clump noise) by the new `ring_merger_profiles.py`:

- `build_ring_interpol_kwargs(...)`: `exp(-((r-r_ring)/width)^2/2)` on an
  elliptical radius grid (axis ratio/PA from the galaxy's overall `e1`/`e2`),
  sized from the existing `r_frac=0.8` disk-radius convention.
- `build_tidal_tail_interpol_kwargs(...)`: a one-sided exponential-decay tail
  oriented from the primary toward the `bulge_secondary` component's offset,
  with length set by their separation + `R_sersic`.

Both return the same `INTERPOL` kwargs shape used by `tng_particle_light`/
`galaxygenius_stamps` (`image, center_x, center_y, phi_G, scale, magnitude`)
and are substituted into the corresponding fragment slot in
`build_light_model` when `morph_type == 'ring'`/`'post_merger'`, gated by
`morphology.native_ring_merger_geometry` (default `true`).

Verified standalone (finite/non-negative images, sensible total flux via
`magnitude2amplitude`, end-to-end lenstronomy renders) and visually: an
isolated ring component renders as a genuine annulus and an isolated
`post_merger` tail renders as a one-sided asymmetric streak, both distinct
from the old Sersic-blob approximation. A 20-lens+4-non-lens pipeline run
(seed 99, `multicomponent_enabled: true`, `--save_intermediate`) completed
with no errors.

## Phase 5: expanded local particle cutout coverage

`scripts/select_phase5_particle_cutout_candidates.py` selects new
`(snapshot, subhalo_id)` candidates from the local TNG catalog (logM
8.0-11.5, stratified across snapshots x mass bins, excluding subhalos that
already have a local `.h5` cutout via
`src.tng_galaxy_selector.local_particle_path`), for feeding to
`scripts/batch_fetch_galaxygenius_stamps.py --candidates <output>`. A first
945-candidate batch completed (944 downloaded, 1 already present, 0 failed),
raising local coverage from 820/31500 (2.6%, Phase 1/2 baseline) to
2112/31500 (~6.7%); the selection script is resumable and re-runnable for
further batches.

### Phase 5b: TNG100 high-mass extension + TNG50-1 integration

`src.tng_galaxy_selector` is now sim-parametrized (`sim="TNG100-1"|"TNG50-1"`
on every fetch/selection/path function, `SIM_FILE_PREFIX = {"TNG100-1":
"TNG_100", "TNG50-1": "TNG_50"}` controlling local `.h5`/`.json` filenames),
and `scripts/build_tng_local_catalog.py` takes `--sim`. This unblocks two
extensions:

- **TNG100-1 9.5-11.5 -> 12.5**: `select_phase6_particle_cutout_candidates.py`'s
  mass range was extended to 12.5 (new bin `11.5-12.5`); a 1908-candidate
  Phase6 batch (9.5-11.5, star-forming-biased) completed 1908/1908, and a
  dedicated 362-candidate massive batch (11.5-12.5, the rare high-mass tail
  useful for lens-galaxy morphology) completed 362/362, both 0 failed.
- **TNG50-1, new**: TNG50-1 has a ~9.7x smaller box than TNG100-1 but ~16x
  better star-particle mass resolution (e.g. ~2500 vs ~180 star particles at
  logM*~9.5, confirmed empirically). `scripts/select_tng50_lowmass_candidates.py`
  selects TNG50-1 candidates across logM 8.0-12.5. A first low-z
  (z<=1.3, 13 snapshots, 7800-subhalo) TNG50-1 local catalog was built and
  fed to `scripts/auto_tng50_pipeline.sh` (started automatically once ready):
  1405/1407 candidates downloaded (2 failed). The catalog was then extended
  to the full TNG100-like snapshot range (30 snapshots, z=0-6.49, 15869
  subhalos), covering the lensed-source/high-z-lens redshift range (COSMOS-Web
  has lenses out to z~3) -- a second 1355-candidate batch covering z up to
  6.49 is fetching in background.

Combined local catalog is now 57631 subhalos (41762 TNG100-1 + 15869 TNG50-1)
covering logM 8-12.5 and z=0-6.5 for both sims; local `.h5` coverage is 5787
(4382 TNG100-1 + 1406 TNG50-1) and climbing as the TNG50-1 high-z batch
continues.

## Phase 6: denser metallicity-dependent SED grid

`tng_particle_light._stellar_band_luminosity` now interpolates a 15-age x
4-metallicity L/M grid per JWST band (bilinear in log-age/log-metallicity),
replacing the original 5-age-bin table with a single global `Z^-0.2`
correction. Per-band metallicity exponents (`_BAND_METAL_ALPHA`) make old,
metal-rich populations dimmer in F115W/F150W and brighter in F444W
(BC03/Padova-like trend), while young populations remain ~metallicity
independent. `python-fsps` installs via pip but requires a separate
`SPS_HOME` isochrone/spectral-library download
(github.com/cconroy20/fsps, not bundled) -- out of scope for this offline
build, so this remains a parametric (but now denser, metallicity-aware)
approximation; a true FSPS/BPASS swap remains possible behind the same
`_stellar_band_luminosity(ages_gyr, metallicity, band)` interface.

## Phase 7: CAS/Gini/M20 morphology validation

`validation.compute_cas_gini_m20` (concentration, asymmetry, smoothness,
clumpiness, Gini, M20) is now computed by
`analysis/sim_obs_comparison/scripts/morphology_metrics.measure_lens_arrays`
(used by `compute_sim_lens_morphology.py`) and by
`compute_real_lens_morphology.py`'s own segmentation step, both keyed on the
central source's photutils segmentation mask. `compare_cas_gini_m20.py`
produces `analysis/sim_obs_comparison/reports/phase3_cas_gini_m20_comparison.md`
(+ histogram figure) comparing the full 435-lens COSMOS-Web real sample
against a 680-lens simulated sample (651 with valid segmentation; pooled
from existing `unified_npz` runs plus a dedicated 450-lens+50-non-lens
background generation run); all six metrics fall in their expected ranges
(Gini in [0,1], concentration > 0, M20 < 0) for both samples, with
real-vs-sim means within ~0.1-0.3 of each other for all six metrics.

Real-data multi-survey expansion (JADES/CEERS/Euclid/Roman/LSST/HSC) remains
a data-acquisition task outside this repo's current `data/real_lenses/`
holdings (COSMOS-Web only).

## Phase 8: multi-extension FITS data products

`src/fits_export.py` adds an opt-in multi-extension FITS export per sample,
gated by `output.save_fits` (default `false`), written to a new `fits/`
output subdirectory alongside the existing `unified_npz`/JPEG outputs:

- **Primary HDU**: the final per-band image cube (`image_final`, shape
  `[n_bands, H, W]`), with `NBANDS`/`BAND0..N`, `PIXSCALE`, `EXPTIME`, `MAGZP`
  header keywords.
- **`PSF`**: per-band PSF kernel cube, reusing the kernels already computed
  for `psf_arrays/{base}_psf.npz`.
- **`NOISE`**: per-band noise-sigma maps, `sqrt(bg_rms^2 + read_noise^2 +
  max(image,0)/gain)` -- an analytic CCD/IR noise-model estimate from the
  final image (gain defaults to 1.0), not a measured per-pixel propagation,
  since the detector chain applies noise in-place during rendering.
- **`SEGMENTATION`**: integer segmentation map from
  `photutils.segmentation.detect_sources` + `deblend_sources` on a
  combined-band detection image.
- **`TRUTH_CATALOG`**: one-row `BinTableHDU` flattened from the same
  per-object metadata dict written to `unified_npz`'s `metadata` JSON
  (lens/source/TNG properties, theta_E, redshifts, etc.).

Verified with a 3-lens run (`configs/tng_mode_test10.yaml`,
`output.save_fits: true`, seed 4242): all 6 output `.fits` files (incl. 4
time-delay epochs for one binary lens) open cleanly with `astropy.io.fits`,
with `PRIMARY` (4,300,300) float32, `PSF` (4,101,101) float32, `NOISE`
(4,300,300) float32 (all finite), `SEGMENTATION` (300,300) int32, and
`TRUTH_CATALOG` (1 row) whose `theta_E`/`lens_redshift` match the
corresponding `unified_npz` metadata.

## Phase 9: generative morphology (conditional VAE)

New package `src/galaxy_morphology/generative/` provides a generative
INTERPOL-source option for **lens, source, and field galaxies**, alongside
GalaxyGenius stamps and TNG particle cutouts:

- `dataset.py`: multi-band (F115W/F150W/F277W/F444W), multi-projection
  training stamp builder from TNG100-1 + TNG50-1 particle cutouts. Supports
  `--catalog2`, `--bands`, `--n-projections`, `--flip-augment` (×4 default).
- `model.py`: conditional VAE (conv encoder/decoder, latent_dim=128, PyTorch).
- `train.py`: CLI training script; best invocation:
  ```
  python -m src.galaxy_morphology.generative.train \
    --catalog /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet \
    --catalog2 /Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet \
    --bands F115W F150W F277W F444W --n-projections 2 --max-n 6000 \
    --epochs 100 --latent-dim 128 --beta 1.0 --beta-start 0.0 \
    --out src/galaxy_morphology/generative/checkpoints/cvae_v3.pt
  ```
- `inference.py`: `build_generative_interpol_kwargs(morph_type, logM,
  redshift, magnitude_ref, ...)` samples the VAE and returns INTERPOL kwargs.

### Pipeline integration (v26+)

The generative VAE can now replace all three galaxy roles in the simulator:

```yaml
tng_mode:
  sim_mode: tng_mixed          # 'tng100' | 'tng50' | 'tng_mixed'
  local_catalog_path: /Volumes/extHD/tng_local_catalog/tng100-1_local_catalog.parquet
  local_catalog_path_tng50: /Volumes/extHD/tng_local_catalog/tng50-1_local_catalog.parquet
  particle_morphology:
    enabled: true
    generative_enabled: true
    generative_force: true     # Use VAE for all roles (lens/source/field)
    generative_checkpoint: src/galaxy_morphology/generative/checkpoints/cvae_v3.pt
```

`generative_force: false` (default): VAE is fallback when no particle file.
`generative_force: true`: VAE is used for ALL galaxies regardless.

### v2 model (cvae_v2.pt): ~3,000 stamps, single-band, 50 epochs, latent_dim=64
First working model; output is smooth/symmetric (insufficient morphological
diversity — produces blob-like profiles for all morph types).

### v3 model (cvae_v3.pt): ~200,000 stamps, 4-band, 100 epochs — **current default**
- 6000 TNG50+TNG100 subhalos × 4 bands × 2 projections × 4 flips ≈ 192,000 stamps
- latent_dim=128, KL annealing (β 0→1 over 25 epochs), free-bits=1.0
- Final reconstruction loss 0.00269 (vs 0.004 for v2 — 33% improvement)
- Quality (v3 vs training): Gini 0.588 vs 0.811, half-light 1.57× of training
- Config `generative_checkpoint: src/galaxy_morphology/generative/checkpoints/cvae_v3.pt`

### Quality comparison

After training completes, run:
```
python analysis/qc/compare_generative_vs_training.py \
  --checkpoint src/galaxy_morphology/generative/checkpoints/cvae_v3.pt \
  --out analysis/qc/generative_quality_report.png
```

Metrics: half-light radius, Gini coefficient, peak SB, radial profiles,
visual stamps grid, and morph-type distribution.

### Known limitations (v3 model)
- Half-light radius ~1.57× larger than training distribution (VAE learns mean size)
- Gini 0.588 vs training 0.811 — somewhat smoother than real TNG particle renders
- `morph_type` printed as '?' for synthetic field galaxies (cosmetic — VAE receives correct type from `tng_info`)
- For research-grade morphological diversity: consider adversarial training or flow-based models

## Phase 9b: TNG50 expansion to 10,000+ local cutouts

`scripts/select_tng50_expansion_candidates.py` selects ~5,000 new TNG50-1
candidates (from 15,869-subhalo catalog, stratified by mass/snapshot) for
download via `batch_fetch_galaxygenius_stamps.py`. Combined with the existing
~8,279 h5 files (4,382 TNG100-1 + 3,897 TNG50-1), a 5,000-candidate batch
reaches ~13,000+ total cutouts. Download started and tracked at
`/tmp/tng50_expansion_download.log`.

Current status: 4,083 TNG50 h5 files on disk (from original 3,995),
download ongoing to reach ~8,800 after filtering snap 12-14 which return
HTTP 403/504 from the EU data node. The expanded TNG50 catalog is used
automatically when `tng_mode.sim_mode: tng50` or `tng_mixed` and
`local_catalog_path_tng50` is set.

## Roadmap

All originally-listed items (Phases 1-9) are implemented. Active extensions:
- v3 VAE training in progress (192k stamps, 100 epochs)
- TNG50 expansion download in progress (~5k new cutouts → 13k total)
- Quality comparison after v3 training: `analysis/qc/compare_generative_vs_training.py`
- Multi-survey real-data validation beyond COSMOS-Web pending data acquisition
coverage growth.
