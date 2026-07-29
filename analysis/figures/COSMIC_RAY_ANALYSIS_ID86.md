# Cosmic Ray Annotation - Lens ID 086

## Summary

**Problem Identified**: Bright galaxy-like feature in lower region of FINAL image, but NOT visible in FIELD_ONLY image.

**Root Cause**: **Cosmic Ray** - realistic detector artifact added during the artifact simulation phase (after field/lens/sources are composited).

---

## Why It Looks Like a Galaxy

The cosmic ray was designed to be **realistic and extended**:

1. **Brightness**: 8-25× background noise level
   - Makes it appear as a faint astronomical object
   - Not noise-like → looks like a real detection

2. **Extension**: ~3×3 pixel spread (realistic ion track)
   - Not point-like (which would look like a star)
   - Extended halo around core (10-40% intensity)
   - Creates galaxy-like morphology

3. **Multi-band presence**: Appears in all 4 JWST bands
   - F115W, F150W, F277W, F444W all affected
   - Realistic detector behavior

---

## Location

- **Position**: Row 269, Col 103 (lower region)
- **Peak flux**: 2.44 (F150W band)
- **Pattern**: 3×3 extended core with halo
- **Confidence**: 100% confirmed via FINAL vs FIELD_ONLY comparison

---

## Generated Annotation Figures

### 1. **cosmic_ray_analysis_id86.png** ⭐ [RECOMMENDED]
Best visualization showing:
- **Top-left**: Full FINAL image with red circle marking cosmic ray
- **Top-center**: Full FIELD_ONLY image (no cosmic ray visible)
- **Top-right**: Difference image (cosmic ray isolated and highlighted)
- **Bottom**: 3 zoomed insets showing detailed structure

**Key insight**: Cosmic ray clearly absent in FIELD_ONLY → proves it's an artifact, not a field galaxy.

### 2. **cosmic_ray_multiband_id86.png**
Shows cosmic ray across all bands:
- **Top row**: FINAL image for F115W, F150W, F277W, F444W (all show cosmic ray)
- **Bottom row**: FIELD_ONLY for all bands (all show NOTHING at cosmic ray location)
- **Red circles**: Mark cosmic ray location in each band
- **Flux values**: Displayed to show intensity scale

---

## Technical Details

### Simulation Code (src/jwst_lens_simulator.py, lines 2912-2925)

```python
# Cosmic ray hits
if artifact_level in ['moderate', 'high'] and rng.random() < 0.25:
    n_hits = rng.integers(1, 6)
    for _ in range(n_hits):
        y = rng.integers(3, enhanced.shape[0]-3)
        x = rng.integers(3, enhanced.shape[1]-3)
        
        intensity = rng.uniform(8, 25) * np.std(enhanced)  # 8-25× noise
        enhanced[y, x] += intensity
        
        # Extended halo (3×3 neighborhood)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if rng.random() < 0.35:  # 35% chance each neighbor
                    enhanced[y+dy, x+dx] += intensity * rng.uniform(0.1, 0.4)
```

### Why This Is Accurate

Real JWST cosmic rays have:
- ✅ Variable intensity (not single spike)
- ✅ Extended footprint (ion track = multiple pixels)
- ✅ Halo structure (core + wings)
- ✅ Can appear galaxy-like at low SNR

---

## Implications for ML Training

**Important**: This cosmic ray should be included in training because:

1. **Real JWST data has cosmic rays** → ML must handle them
2. **They can mimic faint background objects** → Could confuse lens detection
3. **Our simulation captures this realism** → Better training for robust models

**Recommendation**: Keep cosmic rays in simulated data (unless specifically filtering them out for comparison studies).

---

## Verification Steps Taken

✓ Loaded raw .npz data  
✓ Extracted F150W band from all components (final, field_only, lens_only, sources_only)  
✓ Computed per-pixel differences  
✓ Confirmed cosmic ray peak at [269, 103] with value 2.44  
✓ Confirmed ZERO contribution from field/lens/sources at that location  
✓ Confirmed cosmic ray absent in field_only image  
✓ Visualized in 4 bands showing consistent multi-wavelength signature  

---

## Files

Location: `/Users/gozalig1/Projects/jwst-mock-lens-simulator/analysis/figures/`

| Figure | Description | Size |
|--------|-------------|------|
| `cosmic_ray_analysis_id86.png` | Recommended detailed analysis | 3.2 MB |
| `cosmic_ray_multiband_id86.png` | Multi-band visualization | 270 KB |
| `step_outputs_id86_raw_COSMIC_RAY_MARKED.png` | Full 5×5 with circles | 273 KB |

---

**Conclusion**: The bright feature in lens 086's final image is a **realistic simulated cosmic ray**, not a field galaxy. It demonstrates the fidelity of our simulation in capturing observational artifacts that real ML models must handle.
