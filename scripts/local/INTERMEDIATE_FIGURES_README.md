# Intermediate Image Stage Figures

This directory contains three scripts for creating publication-quality figures showing the different stages of the lens system image generation pipeline.

## Overview

The JWST Mock Lens Simulator generates images through multiple stages:
1. **Lens Only** - Just the lens galaxy
2. **Lens + Sources** - Lens galaxy with lensed arcs/Einstein rings
3. **Sources Only** - Lensed sources without lens light (diagnostic)
4. **Field Galaxies** - Environmental contamination
5. **Final Image** - Complete realistic JWST observation

All intermediate stages include PSF convolution, realistic JWST noise, sky background, and detector artifacts.

## Scripts

### 1. `create_intermediate_stages_figure.py`
**Purpose**: Create a simple grid figure showing all 5 stages side-by-side

**Features**:
- Clean 5-panel layout
- Image normalization with log stretch
- Step labels and grid overlay
- Detailed description text
- Suitable for paper figures

**Usage**:
```bash
python scripts/local/create_intermediate_stages_figure.py <output_dir> [--lens-id <id>] [--save-path <path>]
```

**Example**:
```bash
python scripts/local/create_intermediate_stages_figure.py /Volumes/extHD/jwst-lens-similator-output/production_20250108_120000
```

**Output**: 
- Saved to `<output_dir>/intermediate_stages_figure.png`
- 150 DPI, publication quality

### 2. `create_advanced_intermediate_figure.py`
**Purpose**: Create a scientific figure with image statistics

**Features**:
- Five main image panels (top two rows)
- Statistics for each stage (max, mean, total flux)
- Color-coded statistic boxes
- More sophisticated layout
- Suitable for supplementary material

**Usage**:
```bash
python scripts/local/create_advanced_intermediate_figure.py <output_dir> [--lens-id <id>] [--save-path <path>]
```

**Example**:
```bash
python scripts/local/create_advanced_intermediate_figure.py /Volumes/extHD/jwst-lens-similator-output/ml-quick_20250108_120000 --lens-id 000005
```

**Output**:
- Saved to `<output_dir>/advanced_intermediate_figure.png`
- Includes flux statistics for each stage

### 3. `create_flow_diagram_figure.py`
**Purpose**: Create a detailed pipeline diagram showing component assembly

**Features**:
- Strategic layout showing component combination
- Explanatory text boxes
- Clear visual flow
- Ideal for presentation materials

**Usage**:
```bash
python scripts/local/create_flow_diagram_figure.py <output_dir> [--lens-id <id>] [--save-path <path>]
```

**Example**:
```bash
python scripts/local/create_flow_diagram_figure.py /Volumes/extHD/jwst-lens-similator-output/production_20250108_120000 --lens-id 000010
```

**Output**:
- Saved to `<output_dir>/flow_diagram_figure.png`
- Strategic layout with annotations

## Requirements

### Configuration
Ensure your simulation output includes intermediate images. In `configs/default_config.yaml`:

```yaml
save_intermediate_images: true  # Enable intermediate image saving
```

### Output Directory Structure
Your simulation output should have this structure:

```
output_dir/
├── jpg_rgb/
│   ├── cosmos_lens_000001.jpg                    # Final image
│   ├── intermediate_lens_only/
│   │   └── cosmos_lens_000001.jpg
│   ├── intermediate_lens_sources/
│   │   └── cosmos_lens_000001.jpg
│   ├── intermediate_sources_only/
│   │   └── cosmos_lens_000001.jpg
│   └── intermediate_field_only/
│       └── cosmos_lens_000001.jpg
└── npy/
    └── [4-band numpy arrays]
```

### Python Dependencies
```bash
pip install numpy matplotlib pillow
```

## How to Use for Your Paper

### Workflow 1: Generate Simulations with Intermediate Images

```bash
# Enable intermediate images in configs/default_config.yaml
# save_intermediate_images: true

# Run simulation
bash scripts/local/complete_workflow.sh --mode ml-quick

# This will create intermediate images during generation
```

### Workflow 2: Create Figure After Simulation

```bash
# Find your output directory
OUTPUT_DIR="/Volumes/extHD/jwst-lens-similator-output/ml-quick_20250108_120000"

# Create simple figure (recommended for paper)
python scripts/local/create_intermediate_stages_figure.py $OUTPUT_DIR

# Or create advanced figure with statistics
python scripts/local/create_advanced_intermediate_figure.py $OUTPUT_DIR

# Or create flow diagram
python scripts/local/create_flow_diagram_figure.py $OUTPUT_DIR
```

### Workflow 3: Custom Parameters

```bash
# Use a specific lens ID (default is 000001)
python scripts/local/create_intermediate_stages_figure.py $OUTPUT_DIR --lens-id 000005

# Save to custom location
python scripts/local/create_intermediate_stages_figure.py $OUTPUT_DIR \
    --save-path ./figures/my_figure.png
```

## Image Parameters

### Image Normalization
- **Stretch**: Logarithmic (log1p)
- **Range**: 1st-99th percentile
- **Format**: 8-bit RGB JPEG

### Multi-Band Composition
The RGB composites are created using:
- **Red**: F444W band
- **Green**: F277W band
- **Blue**: F115W or F150W band

This mimics near-infrared false color commonly used in astronomical papers.

### Pixel Scale
- Default: 300×300 pixels
- Physical scale: ~0.03 arcsec/pixel

## Tips for Paper Figures

### 1. **For Main Text**
Use `create_intermediate_stages_figure.py` - clean, simple, easy to understand

### 2. **For Supplementary Material**
Use `create_advanced_intermediate_figure.py` - includes statistics and detailed annotations

### 3. **For Presentations**
Use `create_flow_diagram_figure.py` - shows pipeline flow and component assembly

### 4. **Customization**
You can edit the figure dimensions:
```python
# In script: change figsize parameter
figsize=(16, 12)  # width, height in inches
```

### 5. **Output Format**
- Save as PNG (150 DPI) for initial review
- Convert to PDF for final submission:
  ```bash
  convert -density 300 figure.png figure.pdf
  ```

## Troubleshooting

### Issue: "File not found" errors
- Check that `save_intermediate_images: true` in your config
- Verify simulation completed successfully
- Check output directory path

### Issue: Black or empty images
- Verify `.jpg` files are created during simulation
- Check image file sizes (should be > 10 KB)
- Try a different `lens_id`

### Issue: Figure looks dark/washed out
- Adjust the normalization in the script
- Try different percentile values (currently 1-99)
- Increase `dpi` when saving

## Advanced: Custom Modifications

You can modify the scripts for custom layouts:

```python
# Example: Change figure size
figsize=(20, 14)

# Change DPI
plt.savefig(save_path, dpi=300)  # Higher DPI for print

# Change colormap
ax.imshow(rgb, cmap='viridis')

# Add additional annotations
ax.text(150, 150, 'Einstein Ring', color='red', fontsize=12)
```

## Questions?

See the main documentation:
- `docs/INTERMEDIATE_IMAGE_OUTPUT.md` - Technical details
- `docs/JWST_LENS_SIMULATION_BOOK.md` - Simulation overview
- `README.md` - Project overview

## Citation

If you use these figures in a publication, please cite:

> Mock lens simulation pipeline with intermediate component visualization
> Generated by: JWST Mock Lens Simulator
> Reference: [Your paper details]
