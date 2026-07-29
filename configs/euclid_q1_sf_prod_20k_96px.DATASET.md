# Euclid Q1 Strong-Lens Simulation Sample (JELSIM / PRISM)

**Sample ID:** `euclid_q1_sf_prod_20k_96px`  
**Purpose:** Selection-function and discovery-pipeline studies for Euclid VIS/NISP  
**Generator:** JELSIM (`jwst-mock-lens-simulator`) with Euclid Q1 empirical calibration

## Contents

| Product | Path | Description |
|---------|------|-------------|
| Multi-band images | `unified_npz/` | Per system: VIS/Y/J/H stacks `(4, 96, 96)` — final, lens-only, sources-only, unlensed sources, lens+sources, field-only |
| Stacked arrays | `unified_npy/` | Intermediate+final planes stacked for ML |
| RGB previews | `jpg_rgb/` | Quick-look composites |
| κ / γ / μ maps | `kappa_maps/` | Convergence, shear, magnification (lenses) |
| Empirical PSFs used | `psf_arrays/` | Kernels assigned per system |
| **Parameter catalog** | `cosmos_lens_training_catalog.csv` | All lens parameters |
| Combined catalog | `cosmos_training_catalog_lens_and_nonlens.csv` | Lenses + non-lenses |
| Non-lens catalog | `cosmos_nonlens_training_catalog.csv` | Negatives |
| This run's config | `run_config.yaml` | Exact simulation settings |

## Sample size

- **20 000** lens systems  
- **20 000** non-lens systems  
- **4 bands:** `EUCLID_VIS`, `EUCLID_Y`, `EUCLID_J`, `EUCLID_H`  
- **96 × 96** pixels @ **0.10″/pix** (9.6″ × 9.6″ FOV, VIS native scale)  
- Exposure / ZP: 565 s, VIS AB ZP = 25.58  

## Physics & realism (Q1-calibrated)

| Quantity | Source |
|----------|--------|
| PSF | Empirical Q1 library (335 tiles × 4 bands) from Euclid Q1 cutouts |
| θ_E | Q1 SIE models (median 0.88″; 16–84% ≈ 0.61–1.30″) + 30% extended compact/high-z mix |
| Lens / source redshifts | Rojas et al. 2025 spectroscopic priors (not fixed modelling z=0.5) |
| Lens Sersic *n* | Q1 median ≈ 4.4 (bulge-dominated) |
| Shear | Q1 per-system draws (median \|γ\| ≈ 0.12) |
| VIS/Y/J/H magnitudes | Q1 MGE photometry distributions |
| Morphology | Hybrid: multicomponent Sersic **+** IllustrisTNG particle light (when cutouts available) |
| Field galaxies | COSMOS-Web number counts × FOV × environment richness (mag \< 26); TNG FoF class remaps richness only |
| Detector | Full Euclid VIS/NISP chain (IPC, BFE, Poisson, read noise, …) |

**Selection-function mixture:** 70% drawn from the Q1-detected θ_E/redshift population; 30% from an extended prior (compact θ_E and high-z lenses under-represented in Q1 discoveries).

## Catalog columns (highlights)

`lens_id`, `filename_base`, `is_lens`, `theta_E`, `lens_redshift`, `source_redshift`,  
`lens_n_sersic`, `lens_e1`, `lens_e2`, `lens_system_class`,  
`lens_mag_euclid_vis/y/j/h`, `source_mag_euclid_vis/y/j/h`,  
TNG match IDs (`tng_lens_*`, `tng_source_*`), field-galaxy summaries, flux diagnostics.

NPZ `metadata` JSON also stores band order and TNG match details.

## How to load one system

```python
import numpy as np, json, pandas as pd

cat = pd.read_csv("cosmos_lens_training_catalog.csv")
row = cat.iloc[0]
z = np.load(f"unified_npz/{row.filename_base}.npz")
img = z["image_final"]          # (4, 96, 96) = VIS, Y, J, H
meta = json.loads(str(z["metadata"]))
print(meta["bands"], meta["theta_E"])
```

## References

- Walmsley et al. 2025 (arXiv:2503.15324) — Q1 discovery engine  
- Rojas et al. 2025 (arXiv:2503.15325) — spectroscopic sample / redshifts  
- Lines et al. 2025 (arXiv:2503.15326) — compact-lens population  
- Q1 modelling products under `/Volumes/extHD/Euclid_lens_q1` (input calibration only)

## Contact / provenance

Produced with config `configs/euclid_q1_sf_prod_20k_96px.yaml` in the JELSIM repository.  
Random seed: **42** (reproducible draws given the same TNG local archive).
