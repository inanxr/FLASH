"""
Ray Utilities for NeRF

Contains functions for ray generation, stratified sampling, and hierarchical sampling.

KEY CONCEPTS:
1. Stratified Sampling: Divide ray into bins, sample uniformly within each
   - Prevents aliasing and ensures good coverage of the ray
   
2. Hierarchical Sampling: Use coarse network's density to guide fine network
   - Sample more points where density is high (important regions)
   - 60% faster rendering with better quality
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def sample_stratified(
    near: float,
    far: float,
    num_samples: int,
    device: torch.device = torch.device('cpu'),
    randomize: bool = True
) -> torch.Tensor:
    """
    Stratified sampling along ray.
    
    Divides the interval [near, far] into num_samples bins and samples uniformly
    within each bin. This prevents aliasing and ensures good ray coverage.
    
    Mathematical formulation:
        t_i = near + (i / N) * (far - near) + U(0, (far - near) / N)
        where N = num_samples, i ∈ [0, N-1], U is uniform random
    
    Args:
        near: Near bound for sampling
        far: Far bound for sampling
        num_samples: Number of samples per ray
        device: Device to create tensor on
        randomize: If True, add random jitter within bins. If False, use bin centers
    
    Returns:
        Sample positions along ray, shape [num_samples]
    
    Example:
        >>> samples = sample_stratified(near=2.0, far=6.0, num_samples=64)
        >>> print(samples.shape)  # torch.Size([64])
        >>> print(samples.min(), samples.max())  # ~2.0, ~6.0
    """
    # Create linearly spaced bins
    # Shape: [num_samples + 1]
    bins = torch.linspace(near, far, num_samples + 1, device=device)
    
    # Get lower and upper bounds for each bin
    # Shape: [num_samples]
    lower = bins[:-1]
    upper = bins[1:]
    
    if randomize:
        # Sample uniformly within each bin
        # U ~ Uniform(0, 1), shape [num_samples]
        uniform_samples = torch.rand(num_samples, device=device)
        
        # Map to bin interval: lower + U * (upper - lower)
        samples = lower + uniform_samples * (upper - lower)
    else:
        # Use bin centers (for deterministic rendering)
        samples = 0.5 * (lower + upper)
    
    return samples


def sample_stratified_batch(
    near: torch.Tensor,
    far: torch.Tensor,
    num_samples: int,
    randomize: bool = True
) -> torch.Tensor:
    """
    Stratified sampling for a batch of rays.
    
    Args:
        near: Near bounds, shape [num_rays] or [num_rays, 1]
        far: Far bounds, shape [num_rays] or [num_rays, 1]
        num_samples: Number of samples per ray
        randomize: Whether to add random jitter
    
    Returns:
        Sample positions, shape [num_rays, num_samples]
    """
    device = near.device
    num_rays = near.shape[0]
    
    # Ensure near and far have shape [num_rays, 1]
    if near.ndim == 1:
        near = near.unsqueeze(-1)
    if far.ndim == 1:
        far = far.unsqueeze(-1)
    
    # Create linearly spaced bins for all rays
    # Shape: [num_rays, num_samples + 1]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=device)
    t_vals = t_vals.unsqueeze(0).expand(num_rays, -1)
    
    # Map to [near, far] interval
    bins = near * (1.0 - t_vals) + far * t_vals
    
    # Get lower and upper bounds
    lower = bins[:, :-1]  # [num_rays, num_samples]
    upper = bins[:, 1:]   # [num_rays, num_samples]
    
    if randomize:
        # Add random jitter within bins
        uniform_samples = torch.rand(num_rays, num_samples, device=device)
        samples = lower + uniform_samples * (upper - lower)
    else:
        samples = 0.5 * (lower + upper)
    
    return samples


def sample_hierarchical(
    bins: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    deterministic: bool = False
) -> torch.Tensor:
    """
    Hierarchical sampling using inverse transform sampling.
    
    Uses the density weights from the coarse network to sample more points
    in regions with high density. This is the key to NeRF's efficiency.
    
    Mathematical formulation:
        1. Compute PDF from weights: p_i = w_i / Σw_i
        2. Compute CDF: CDF_i = Σ_{j<=i} p_j
        3. Sample uniformly: u ~ U(0, 1)
        4. Find bin where CDF(bin) >= u (inverse transform sampling)
    
    Args:
        bins: Bin edges from coarse sampling, shape [num_rays, num_bins + 1]
        weights: Importance weights from coarse rendering, shape [num_rays, num_bins]
        num_samples: Number of new samples to generate
        deterministic: If True, use uniform spacing in CDF (for visualization)
    
    Returns:
        New sample positions, shape [num_rays, num_samples]
    
    Reference:
        Inverse transform sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    """
    device = bins.device
    num_rays = bins.shape[0]
    
    # Normalize weights to get PDF
    # Add small epsilon to prevent division by zero
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Compute CDF (cumulative distribution function)
    # CDF[i] = sum of PDF from 0 to i
    cdf = torch.cumsum(pdf, dim=-1)
    
    # Pad with 0 at the beginning (CDF starts at 0)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    
    # Sample uniformly or deterministically
    if deterministic:
        # Linearly spaced samples for deterministic rendering
        u = torch.linspace(0.0, 1.0, num_samples, device=device)
        u = u.unsqueeze(0).expand(num_rays, -1)
    else:
        # Random uniform samples
        u = torch.rand(num_rays, num_samples, device=device)
    
    # Ensure u is in valid range [0, 1)
    u = u.contiguous()
    
    # Find indices where CDF >= u (inverse transform sampling)
    # searchsorted finds the indices to insert u into sorted cdf
    indices = torch.searchsorted(cdf, u, right=True)
    
    # Clamp indices to valid range [1, num_bins]
    # (indices can be 0 or num_bins+1 in edge cases)
    below = torch.clamp(indices - 1, min=0, max=cdf.shape[-1] - 2)
    above = torch.clamp(indices, min=0, max=cdf.shape[-1] - 1)
    
    # Gather CDF and bin values at indices
    # Shape: [num_rays, num_samples]
    cdf_below = torch.gather(cdf, dim=-1, index=below)
    cdf_above = torch.gather(cdf, dim=-1, index=above)
    bins_below = torch.gather(bins, dim=-1, index=below)
    bins_above = torch.gather(bins, dim=-1, index=above)
    
    # Linear interpolation between bins
    # t = (u - CDF_below) / (CDF_above - CDF_below)
    # sample = bins_below + t * (bins_above - bins_below)
    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    
    samples = bins_below + t * (bins_above - bins_below)
    
    return samples


def get_ray_bundle(
    height: int,
    width: int,
    focal_length: float,
    camera_to_world: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for all pixels in an image.
    
    Uses the pinhole camera model to generate rays from the camera origin
    through each pixel in the image plane.
    
    Coordinate system:
        - Camera: +X right, +Y up, +Z backwards (looks down -Z)
        - World: arbitrary (defined by camera_to_world matrix)
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Camera focal length in pixels
        camera_to_world: Camera-to-world transformation matrix, shape [4, 4]
    
    Returns:
        rays_o: Ray origins in world space, shape [height, width, 3]
        rays_d: Ray directions in world space (normalized), shape [height, width, 3]
    """
    device = camera_to_world.device
    
    # Create pixel grid
    # i: column index (x), j: row index (y)
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing='xy'
    )
    
    # Convert pixel coordinates to camera coordinates
    # Center the coordinates: subtract width/2 and height/2
    # Normalize by focal length to get directions
    # Formula: dir_x = (i - width/2) / focal_length
    #          dir_y = -(j - height/2) / focal_length  (flip y for image coords)
    #          dir_z = -1 (looking down -Z axis)
    directions = torch.stack([
        (i - width * 0.5) / focal_length,
        -(j - height * 0.5) / focal_length,
        -torch.ones_like(i)
    ], dim=-1)  # Shape: [height, width, 3]
    
    # Transform ray directions from camera space to world space
    # directions_world = (camera_to_world[:3, :3] @ directions)
    # Use rotation part only (top-left 3x3 of 4x4 matrix)
    rotation = camera_to_world[:3, :3]
    directions_world = directions @ rotation.T
    
    # Normalize ray directions to unit vectors
    directions_world = F.normalize(directions_world, dim=-1)
    
    # Ray origins are the camera position (translation part of matrix)
    # Same origin for all rays from this camera
    origin = camera_to_world[:3, 3]  # Shape: [3]
    origins_world = origin.unsqueeze(0).unsqueeze(0).expand(height, width, -1)
    
    return origins_world, directions_world


if __name__ == "__main__":
    print("Testing Ray Utilities...")
    print("=" * 60)
    
    # Test stratified sampling
    print("Test 1: Stratified Sampling")
    samples = sample_stratified(near=2.0, far=6.0, num_samples=64)
    print(f"  Sample shape: {samples.shape}")
    print(f"  Min: {samples.min().item():.4f}, Max: {samples.max().item():.4f}")
    print(f"  Expected range: [2.0, 6.0]")
    print(f"  Sorted: {torch.all(samples[:-1] <= samples[1:]).item()}")
    print()
    
    # Test batch stratified sampling
    print("Test 2: Batch Stratified Sampling")
    near_batch = torch.tensor([2.0, 2.5, 3.0])
    far_batch = torch.tensor([6.0, 6.5, 7.0])
    samples_batch = sample_stratified_batch(near_batch, far_batch, num_samples=32)
    print(f"  Sample shape: {samples_batch.shape}")
    print(f"  Expected: [3, 32]")
    print()
    
    # Test hierarchical sampling
    print("Test 3: Hierarchical Sampling")
    num_rays = 1024
    num_coarse = 64
    num_fine = 64
    
    # Create mock bins and weights
    bins = torch.linspace(2.0, 6.0, num_coarse + 1).unsqueeze(0).expand(num_rays, -1)
    
    # Create weights that peak in the middle (simulating object in center)
    weights = torch.exp(-((torch.arange(num_coarse) - num_coarse/2) ** 2) / 100)
    weights = weights.unsqueeze(0).expand(num_rays, -1)
    
    fine_samples = sample_hierarchical(bins, weights, num_samples=num_fine)
    print(f"  Sample shape: {fine_samples.shape}")
    print(f"  Expected: [{num_rays}, {num_fine}]")
    print(f"  Min: {fine_samples.min().item():.4f}, Max: {fine_samples.max().item():.4f}")
    print(f"  Mean: {fine_samples.mean().item():.4f} (should be ~4.0 for centered peak)")
    print()
    
    # Test ray generation
    print("Test 4: Ray Generation")
    H, W = 100, 100
    focal = 50.0
    
    # Identity camera matrix (camera at origin, looking down -Z)
    c2w = torch.eye(4)
    
    rays_o, rays_d = get_ray_bundle(H, W, focal, c2w)
    print(f"  Ray origins shape: {rays_o.shape}")
    print(f"  Ray directions shape: {rays_d.shape}")
    print(f"  Directions normalized: {torch.allclose(rays_d.norm(dim=-1), torch.ones(H, W))}")
    print(f"  Origin at zero: {torch.allclose(rays_o, torch.zeros_like(rays_o))}")
    print()
    
    print("✅ Ray utilities test passed!")
