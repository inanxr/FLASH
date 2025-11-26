# Models Directory

Core neural network components for Instant-NGP NeRF implementation.

## Files

- `hash_encoding.py` - Multi-resolution hash encoding
- `nerf_model.py` - Instant-NGP network architecture
- `renderer.py` - Volumetric ray marching renderer
- `occupancy_grid.py` - Empty space skipping acceleration

## Key Concepts

### Hash Encoding (`hash_encoding.py`)

**Purpose**: Replace expensive positional encoding with learnable hash tables.

**Hash Function**:
```
hash(x,y,z) = (x*p1 XOR y*p2 XOR z*p3) mod table_size
Primes: [1, 2654435761, 805459861]
```

**Trilinear Interpolation**:
- For position in voxel, get 8 corner coordinates
- Weight for corner (i,j,k): `w = w_x^i * w_y^j * w_z^k`
- Where `w_x^0 = 1-frac_x`, `w_x^1 = frac_x`

**Resolution Levels**: Geometric progression from coarse (16) to fine (512)
- Formula: `N_l = floor(N_min * b^l)` where `b = exp(log(N_max/N_min)/(L-1))`

### Volume Rendering (`renderer.py`)

**Main Equation**:
```
C(r) = Σ T_i * α_i * c_i
```

Where:
- `α_i = 1 - exp(-σ_i * δ_i)` - opacity at sample i
- `δ_i = t_{i+1} - t_i` - distance between samples
- `T_i = Π(1 - α_j)` for j < i - transmittance (accumulated transparency)

**Hierarchical Sampling**:
1. Coarse network samples uniformly along ray
2. Use density weights as PDF for fine sampling
3. Inverse transform sampling: sample where `CDF(t) >= u`

### Occupancy Grid (`occupancy_grid.py`)

**Ray-AABB Intersection**:
```python
t1 = (aabb_min - ray_o) / ray_d
t2 = (aabb_max - ray_o) / ray_d
t_min = max(min(t1, t2))  # entry point
t_max = min(max(t1, t2))  # exit point
```

**Purpose**: Skip sampling in empty regions (5-10x speedup)

## Network Architecture

```
Input: (x,y,z) position + (dx,dy,dz) direction
  ↓
Hash Encoding: 16 levels × 2 features = 32D
  ↓
Concatenate with direction: 32 + 3 = 35D
  ↓
MLP: 2 layers × 64 dims
  ↓
Output: RGB (3) + density (1)
```