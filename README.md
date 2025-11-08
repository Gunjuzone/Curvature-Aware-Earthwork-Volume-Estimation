# Curvature-Aware Urban Terrain Earthwork Volume Estimation

A Python implementation of curvature-aware adaptive grid refinement for accurate earthwork volume estimation from Digital Terrain Models (DTMs). This method improves upon conventional grid-based approaches by targeting high-curvature zones with recursive subdivision and spline-based surface integration.

## Overview

Traditional Digital Elevation Model of Difference (DoD) methods assume planar or linearly varying surfaces within grid cells, leading to systematic underestimation in high-curvature urban terrain. This implementation addresses that limitation through:

- **Adaptive grid refinement** driven by local surface curvature (95th percentile threshold)
- **Spline-based volume integration** capturing non-linear elevation gradients
- **Natural Neighbor Interpolation (NNI)** for pre-earthwork terrain reconstruction
- **Recursive subdivision** (up to depth 3) in steep zones while maintaining computational efficiency

The method demonstrated **20.6% error reduction** compared to standard grid-based approaches on synthetic validation tests (hemispheres, cones, pyramids).

## Citation

If you use this code in your research, please cite:
```bibtex
@article{Olagunju2025,
  title={Curvature-Aware Urban Terrain Earthwork Volume Estimation: A Multi-City Remote Sensing Framework},
  author={Sakiru Olarewaju Olagunju and Huseyin Atakan Varol and Ferhat Karaca},
  journal={Geocarto International},
  year={2025},
  note={Under Review}
}
```

## Features

-  Gaussian Laplacian curvature computation (σ=1)
-  Recursive grid subdivision (max depth = 3)
-  Bivariate cubic spline interpolation with configurable smoothing (default 0.1)
-  Quadratic surface fallback for numerical instability cases
-  Support for GeoTIFF input/output
-  Batch processing for multiple DTM pairs
-  Detailed logging with timestamps

## Requirements
```
Python >= 3.12
numpy >= 2.2.6
scipy >= 1.14.1
rasterio >= 1.4.3
matplotlib >= 3.10.1
pandas >= 2.0.0
tkinter
```

## Installation

### Clone and Install Dependencies
```bash
git clone https://github.com/Gunjuzone/curvature-aware-earthwork-volume-estimation.git
cd curvature-aware-earthwork-volume-estimation
pip install -r requirements.txt
```
## Quick Start

### Basic Usage
```python
import numpy as np
import rasterio
from curvature_aware import calculate_dtm_difference

# Load your DTMs
dtm_before = "path/to/DTM1_pre_earthwork.tif"
dtm_after = "path/to/DTM2_post_earthwork.tif"

# Compute volume
volume, area = calculate_dtm_difference(dtm_before, dtm_after)

print(f"Earthwork Volume: {volume:.2f} m³")
print(f"Earthwork Area: {area:.2f} m²")
```

### Batch Processing

The script includes a GUI file selector for batch processing multiple DTM pairs:
```bash
python Curvature_Aware.py
```

**Workflow:**
1. Select all "BEFORE" DTMs (pre-earthwork)
2. Select all "AFTER" DTMs (post-earthwork) in corresponding order
3. Results saved to `Analysis_Outputs/adaptive_volume_results.csv`

## Algorithm Parameters

### Adjustable Parameters

Edit these in `Curvature_Aware.py` if needed:
```python
# Curvature computation
sigma = 1  # Gaussian Laplacian standard deviation

# Adaptive refinement
curvature_percentile = 95  # Threshold (0-100)
max_depth = 3  # Maximum recursion depth

# Spline fitting
smoothing_factor = 0.1  # Balance noise vs. feature preservation

# Fallback conditions
min_std_dev = 1e-6  # Minimum variance for spline fitting
```

### Parameter Selection Guidance

| Parameter | Range | Recommendation |
|-----------|-------|----------------|
| `sigma` | 0.5-2.0 | 1.0 (balances noise reduction & feature preservation) |
| `curvature_percentile` | 90-97 | 95 (targets steepest 5% of terrain) |
| `max_depth` | 2-4 | 3 (balance accuracy vs. computation time) |
| `smoothing_factor` | 0.05-0.5 | 0.1 (avoids over/under-smoothing) |

## Input Data Requirements

### DTM Specifications

- **Format:** GeoTIFF (.tif)
- **Resolution:** 0.5-2.0 m recommended (tested at 0.5 m - 2 m)
- **Coordinate System:** Any projected CRS (UTM recommended)
- **Data Type:** Float32 or Float64
- **NoData Handling:** NaN values supported

### DTM Pair Requirements

- Same spatial extent and resolution
- Same coordinate reference system (CRS)
- Co-registered (aligned to same grid)



## How It Works

### 1. Elevation Differencing
```
Δh = DTM₂ (post-earthwork) - DTM₁ (pre-earthwork)
```

### 2. Curvature Computation
Gaussian Laplacian filter applied to Δh grid:
```
curvature = |∇²Δh|
threshold = 95th percentile of curvature values
```

### 3. Adaptive Refinement Decision

For each cell (i, j):
```
IF curvature[i,j] < threshold OR depth ≥ max_depth:
    → Fit spline, compute volume
ELSE:
    → Subdivide into 4 sub-cells, recurse with depth+1
```

### 4. Volume Integration

**Primary method (Spline):**
```
V_patch = ∬ S(x,y) dx dy
```
where S(x,y) is bivariate cubic spline fitted to 7×7 neighborhood

**Fallback method (Quadratic):**
```
V_patch = ∬ (ax² + by² + cxy + dx + ey + f) dx dy
```
Applied when spline fitting fails (NaN values, low variance, instability)

### 5. Total Volume
```
Total Volume = Σ V_patch,i  for all refined patches
```

## Validation & Benchmarking

The method has been validated through:

### Synthetic Terrain Tests
- **Pyramid, Cone, Hemisphere** (51×51 to 500×500 grids)
- **Error reduction:** 20.6% (hemisphere), 9.2% (cone), 6.1% (pyramid)
- **Grids tested:** 0.5 m, 1.0 m, 2.0 m resolution

### Multi-City Field Validation
- **Cities:** Astana, Almaty, Barcelona, Porto, Prague
- **Period:** 2013-2025
- **DTM validation RMSE:** 1.12-2.10 m (vs. commercial Maxar DTMs)
- **200+ earthwork sites** analyzed

### Literature Comparison
- Grohmann et al. (2020) coastal dune study replication
- **Volume difference:** 18% (acceptable for 12-year satellite vs. 9-year LiDAR comparison)

## Limitations & Future Work

### Current Limitations

1. **Manual earthwork delineation** – Boundary uncertainty 
   - *Future:* Deep learning segmentation

2. **Temporal gaps** – Misses short-duration earthworks between acquisitions
   - *Future:* Integration with daily satellite constellations

### Planned Enhancements

- [ ] Automated earthwork detection using semantic segmentation
- [ ] Support for time-series analysis (>2 DTMs)
- [ ] Docker containerization for reproducibility

## Troubleshooting

### Common Issues

**1. Memory Error with Large DTMs**
```python
# Process in tiles
tile_size = 5000  # pixels
# Implement tiling logic in your workflow
```

**2. CRS Mismatch Error**
```bash
gdalwarp -t_srs EPSG:32633 input.tif output_reprojected.tif
```

**3. Excessive NaN Warnings**

Check your earthwork delineation – ensure zones don't include large NoData areas.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/Gunjuzone/curvature-aware-earthwork-volume-estimation.git
cd curvature-aware-earthwork
pip install -r requirements.txt
pip install -r requirements-dev.txt 
```

## License

This project is licensed under the MIT License

## Acknowledgments
- Maxar
- Institue of Smart Systems and Artificial Intelligence (ISSAI)
- Annonymous Reviewers

## Contact

**Shakir Olarewaju Olagunju**  
Nazarbayev University 
Email: sakiru.olagunju@nu.edu.kz  

**Related Publication:** in progress

