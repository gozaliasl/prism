# Pair Galaxy Treatment Comparison Script

## Overview

This script generates publication-quality comparison figures showing the same lens system simulated with three different treatments of companion/pair galaxies:

1. **SIE+SIE (Binary SIE Lens)**: Both galaxies contribute as gravitational lenses using Singular Isothermal Ellipsoid profiles
2. **NFW+NFW (Binary NFW Lens)**: Both galaxies contribute using Navarro-Frenk-White mass profiles (more realistic)
3. **Shear Only**: Pair galaxy only contributes to external shear (simplified treatment)

These three approaches represent different complexity levels and computational costs:
- SIE+SIE: Fastest, simplest, standard binary lensing
- NFW+NFW: More realistic dark matter profiles, more expensive computationally
- Shear Only: Simplified approach, useful for understanding environmental effects

## Physical Motivation

**Why compare these methods?**

- **SIE+SIE**: Traditional approach, widely used in lens modeling
- **NFW+NFW**: More physically motivated, matches observed dark matter distributions
- **Shear Only**: Isolates environmental effects without explicit binary lens, useful for robustness tests

The differences manifest in:
- Arc morphology (shape and orientation of lensed images)
- Image multiplicity (2 vs 4 images)
- Magnification patterns
- Substructure sensitivity

## Usage

### Basic Usage (Compare 5 Random Lens Systems)

```bash
python analysis/figures/compare_pair_galaxy_treatments.py \
    --n-samples 5 \
    --seed 42
```

### Generate Comparisons for Specific Lens IDs

```bash
python analysis/figures/compare_pair_galaxy_treatments.py \
    --sample-ids 0 11 25 86 \
    --seed 42
```

### Customize Output

```bash
python analysis/figures/compare_pair_galaxy_treatments.py \
    --n-samples 10 \
    --output-dir analysis/figures/pair_comparisons \
    --lens-catalog data/lens_analysis_catalog.csv \
    --numpix 400 \
    --pixel-scale 0.03 \
    --seed 42
```

### Full Command-Line Options

```
--lens-catalog PATH
    Path to lens catalog CSV file
    Default: data/lens_analysis_catalog.csv

--output-dir PATH
    Output directory for comparison figures
    Default: analysis/figures/pair_comparisons

--n-samples N
    Number of lens systems to generate
    Default: 5

--sample-ids ID1 ID2 ID3 ...
    Specific sample IDs to process
    Overrides --n-samples if provided

--seed SEED
    Random seed for reproducibility
    Default: 42

--numpix N
    Image size in pixels
    Default: 300

--pixel-scale SCALE
    Pixel scale in arcsec/pixel
    Default: 0.03
```

## Output Files

The script generates figures with filenames:
```
pair_comparison_id000.png
pair_comparison_id011.png
pair_comparison_id025.png
...
```

Each figure is a 3-panel comparison showing:
- **Left panel (SIE+SIE)**: Binary lens with two SIE components
- **Middle panel (NFW+NFW)**: Binary lens with two NFW components
- **Right panel (Shear Only)**: Single lens with enhanced external shear

### Figure Details

Each panel includes:
- **Title**: Lens model type and description
- **Lens markers**: Cyan dashed circles marking lens positions
  - Larger circle: Primary lens
  - Smaller circle: Secondary lens (if applicable)
- **Model parameters box**:
  - For binary lenses: Separation (") and mass ratio
  - For shear: Shear magnitude γ
- **Lens properties** (title area):
  - Redshift: z_lens
  - Einstein radius: θ_E
  - Stellar mass: log(M*/M☉)

## Integration with Pipeline

To generate lens systems with these different pair galaxy treatments, use these configuration options:

### Option 1: SIE+SIE Binary Lens
```yaml
binary_lenses:
  enabled: true
  fraction: 0.15
  mass_profile_types:
    sie_sie: 1.0
    nfw_nfw: 0.0
```

### Option 2: NFW+NFW Binary Lens
```yaml
binary_lenses:
  enabled: true
  fraction: 0.15
  mass_profile_types:
    sie_sie: 0.0
    nfw_nfw: 1.0
```

### Option 3: Shear Only (No Binary Lens)
```yaml
binary_lenses:
  enabled: false
environment:
  types:
    galaxy_pair:
      shear_min: 0.08
      shear_max: 0.15  # Enhanced shear
```

## Example: Running Full Pipeline with Each Treatment

```bash
# Generate 1000 lenses with SIE+SIE binary treatment
python src/generate_lens_dataset.py \
    --output-dir outputs/custom_sie_sie \
    --n-lenses 1000 \
    --config configs/binary_sie_sie.yaml

# Generate 1000 lenses with NFW+NFW binary treatment
python src/generate_lens_dataset.py \
    --output-dir outputs/custom_nfw_nfw \
    --n-lenses 1000 \
    --config configs/binary_nfw_nfw.yaml

# Generate 1000 lenses with shear-only treatment
python src/generate_lens_dataset.py \
    --output-dir outputs/custom_shear_only \
    --n-lenses 1000 \
    --config configs/shear_only.yaml

# Generate comparison figures for matching lens IDs
python analysis/figures/compare_pair_galaxy_treatments.py \
    --sample-ids 0 1 2 3 4 5 \
    --output-dir analysis/figures/pair_method_comparison
```

## Key Metrics to Compare

When interpreting the comparison figures, consider:

1. **Image Multiplicity**
   - SIE+SIE and NFW+NFW: Typically 4 images
   - Shear Only: Typically 2 images

2. **Arc Morphology**
   - How much distortion in the lensed source?
   - SIE+SIE: More aligned arcs
   - NFW+NFW: Broader, more extended images
   - Shear Only: Simpler, more symmetric

3. **Magnification**
   - NFW profiles typically produce higher magnifications
   - Shear-only typically produces lower magnifications

4. **Computational Cost**
   - SIE+SIE: ~1× baseline
   - NFW+NFW: ~2-3× baseline (more complex deflections)
   - Shear Only: ~0.8× baseline (simpler)

## For Paper Discussion

Use these comparison figures to demonstrate:
- **Robustness**: How sensitive are results to pair galaxy treatment?
- **Accuracy tradeoff**: SIE faster but less realistic; NFW more realistic but slower
- **Environmental effects**: Importance of pair galaxies vs. external shear
- **Model dependence**: Whether conclusions depend on lens mass profile choice

### Suggested Figure Caption

> *"Comparison of three pair galaxy treatment methods for a representative lens system 
> (ID XXX, z=0.XX, θ_E=X.XX"). Left: Both galaxies contribute as Singular Isothermal 
> Ellipsoids (SIE+SIE). Center: Both contribute as Navarro-Frenk-White profiles 
> (NFW+NFW). Right: Pair galaxy contributes only to external shear. The different 
> treatments produce distinct arc morphologies and image multiplicities, illustrating 
> the importance of accurate pair galaxy modeling in strong lens systems."*

## Technical Details

### Image Generation

Mock lensed images are created using Gaussian profiles representing:
- **Lens light**: Central galaxy (elliptical profile)
- **Lensed images**: Source-like features at predicted image positions
- **Number of images**: Depends on lens model configuration
  - Binary lenses: 4 images
  - Single lens + shear: 2 images

### Customization

To modify the image generation:

1. **Change source properties** in `simulate_lens_images_three_modes()`:
   ```python
   source_radius = 0.15  # Larger source
   source_n = 2.0        # More concentrated
   ```

2. **Change shear strength** for Shear-Only mode:
   ```python
   shear_magnitude = rng.uniform(0.10, 0.20)  # Stronger shear
   ```

3. **Modify visual appearance**:
   - Edit colormap: change `cmap='hot'` to other matplotlib colormaps
   - Adjust stretch: modify sqrt to other stretching functions

## Related Scripts

- `analysis/figures/make_step_figure.py`: Generate full-pipeline step figures
- `src/advanced_lens_features.py`: Core binary lens generation code
- `scripts/generate_lens_dataset.py`: Main simulation pipeline

## Troubleshooting

**Issue**: "Insufficient suitable lenses found"
- Solution: Adjust Einstein radius range or increase catalog size

**Issue**: "Image appears empty or very faint"
- Solution: Check lens redshift; very high-z lenses may need adjustment
- Try increasing `source_radius` parameter

**Issue**: Comparison figures look too similar
- Solution: Check that binary separation is sufficient (>0.5 arcsec)
- Increase mass ratio to make primary/secondary more different

## Citation

If using these comparison figures in your paper, cite:
- JWST Mock Lens Simulator: [Your paper reference]
- lenstronomy: Birrer et al. 2015, 2018

---

**Last Updated**: 2026-02-15
**Compatible With**: Python 3.8+, numpy, matplotlib, pandas
