# PRISM — a Multi-Telescope Physically Realistic Strong Lensing Image Simulation

A physically-realistic pipeline for generating mock strong gravitational lens observations
across five space and ground-based telescopes. Built on real COSMOS-Web data, empirical
noise models from real JWST lenses, and a complete detector-physics signal chain.
Designed for machine learning training and strong-lens detection research.

This is the public, installable Python package (`prism`). The accompanying paper draft
lives in a separate repository; input/output data (catalogs, PSFs, TNG particles,
trained ML models) lives in a separate `prism-data/` directory — see [Data](#data) below.

---

## Supported Telescopes

| Telescope | Detector | Pixel scale | Default bands | Detector type |
|-----------|----------|-------------|---------------|---------------|
| **JWST NIRCam** | H2RG HgCdTe | 0.031″/pix | F115W · F150W · F277W · F444W | IR array |
| **Roman WFI** | H4RG HgCdTe | 0.11″/pix | F106 · F129 · F158 · F184 | IR array |
| **Euclid VIS/NISP** | CCD273 (VIS) | 0.10″/pix | VIS · Y · J · H | Silicon CCD |
| **Subaru HSC** | Hamamatsu FDEPCCD | 0.168″/pix | g · r · i · z | Thick-depletion CCD |
| **LSST / Rubin** | ITL/e2v CCD | 0.200″/pix | g · r · i · z | High-resistivity CCD |

Select telescope in `configs/default_config.yaml`:
```yaml
telescope: "jwst"   # jwst | roman | euclid | subaru | lsst
```

---

## Package layout

```
src/prism/
├── core/         simulation driver, mass profiles, physical constants, fundamental plane
├── telescopes/    per-telescope filter transmission, RGB compositing, catalogs
├── morphology/    multi-component (bulge/disk/bar/ring) galaxy light models
├── lensing/       image detection/classification, time-delay modeling
├── ml/            arc detection, segmentation, environment learning, training pipelines
├── selection/     TNG subhalo selection, GalaxyGenius stamp integration
└── io/            FITS export, kappa-map output, detector chain, PSF generation
scripts/           CLI/production-run scripts
configs/           YAML run configurations
tests/             unit/integration tests
analysis/          QC, figure generation, morphology showcase tooling
container/         Singularity build files (HPC)
```

## Install

```bash
pip install -e .
```

## Data

PRISM expects a separate data directory (catalogs, PSFs, TNG catalogs/particles, trained
ML models) — not bundled in this repository. Configs under `configs/` reference an
absolute data path by default; point `--cosmos_catalog`, `local_catalog_path`, `data_dir`,
etc. at your own copy of:

```
prism-data/
├── catalogs/{cosmos,tng,euclid_q1,cowls,relations}/
├── real_lenses/
├── psf/{jwst_stpsf,jwst_v5_30mas,euclid_q1,cache}/
├── filter_throughputs/{nircam,euclid,roman,lsst,subaru_suprime}/
├── models/            trained environment/count/radius regressors
└── outputs/            pipeline run outputs
```

---

## Quick Start

### Small test run (2 lenses)
```bash
python -m prism.core.simulator \
    --config configs/default_config.yaml \
    --cosmos_catalog /path/to/prism-data/catalogs/cosmos/cosmos_web_lens_structural_properties.csv \
    --output_dir outputs/my_run \
    --n_lenses 2 --numpix 64 --seed 42
```

### 500 JWST lenses with intermediate images saved
```bash
python -m prism.core.simulator \
    --config configs/default_config.yaml \
    --cosmos_catalog /path/to/prism-data/catalogs/cosmos/cosmos_web_lens_structural_properties.csv \
    --lens_analysis_catalog /path/to/prism-data/catalogs/cosmos/lens_analysis_catalog.csv \
    --merged_field_catalog /path/to/prism-data/catalogs/cosmos/merged_lens_field_catalog.csv \
    --output_dir outputs/my_run \
    --n_lenses 500 --n_non_lenses 0 \
    --seed 42 --add_artifacts --save_intermediate --no_date_suffix
```

### Custom multi-telescope run
```yaml
# configs/default_config.yaml
telescope: "roman"
bands: ["ROMAN_F106", "ROMAN_F129", "ROMAN_F158", "ROMAN_F184"]
```

---

## Multi-source lensed systems

PRISM optionally generates systems with 1–5 lensed sources per lens (configurable
distribution in `configs/default_config.yaml` under `multi_source`), each independently
placed inside the lens caustic, with additional sources at a different redshift than
the primary having their deflection field rescaled by the distance ratio
β = D_LS/D_S rather than full multi-plane ray tracing.

---

## License

MIT
