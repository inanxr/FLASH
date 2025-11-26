# Utils Directory

Utility modules for data loading, metrics, ray sampling, and visualization.

## Files

- `data_loader.py` - NeRF Blender dataset loader
- `metrics.py` - PSNR, SSIM, LPIPS evaluation metrics
- `ray_utils.py` - Stratified and hierarchical sampling
- `visualization.py` - Training curves and image comparison plots

## Key Concepts

### Camera Model (`data_loader.py`)

**Focal length from camera angle**:
```
focal = 0.5 * image_size / tan(0.5 * camera_angle_x)
```

**Alpha compositing**:
- White background: `RGB = rgb * alpha + (1 - alpha)`
- Black background: `RGB = rgb * alpha`

### Metrics (`metrics.py`)

**PSNR (Peak Signal-to-Noise Ratio)**:
```
MSE = mean((pred - target)²)
PSNR = -10 * log₁₀(MSE / max_val²)
     = 10 * log₁₀(max_val² / MSE)
```
- Higher is better (30+ dB = good quality)

**SSIM (Structural Similarity)**:
```
SSIM = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))
```
Where:
- μₓ, μᵧ = mean of pred and target
- σₓ, σᵧ = variance
- σₓᵧ = covariance
- C₁, C₂ = stability constants

Range: [-1, 1], where 1 = perfect similarity

### Ray Sampling (`ray_utils.py`)

**Stratified Sampling**:
```
Divide [near, far] into N bins
t_i = bin_start + U(0, bin_width)
```
Prevents aliasing by ensuring uniform coverage.

**Hierarchical Sampling (Inverse Transform)**:
```
1. PDF: p_i = w_i / Σw_i
2. CDF: CDF_i = Σ_{j≤i} p_j
3. Sample: u ~ U(0,1)
4. Find bin where CDF(bin) ≥ u
5. Interpolate: t = bins_below + (u - CDF_below)/(CDF_above - CDF_below) * (bins_above - bins_below)
```

**Pinhole Camera Model**:
```
Pixel (i,j) → Camera direction:
  dir_x = (i - width/2) / focal
  dir_y = -(j - height/2) / focal  (flip y)
  dir_z = -1  (looking down -Z)

World direction = camera_rotation @ camera_direction
Ray origin = camera_position (same for all rays)
```

## Coordinate Systems

**Camera space**: +X right, +Y up, +Z backwards (looks down -Z)  
**World space**: Arbitrary (defined by camera-to-world matrix)

## Common Patterns

**Batch tensor shapes**:
- Images: `[H, W, 3]`
- Rays: `[num_rays, 3]`
- Samples: `[num_rays, num_samples]`
- Predictions: `[num_rays, 3]` (RGB) or `[num_rays, 1]` (density)
