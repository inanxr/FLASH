"""Ray utilities for stratified and hierarchical sampling."""

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
    bins = torch.linspace(near, far, num_samples + 1, device=device)
    lower = bins[:-1]
    upper = bins[1:]
    
    if randomize:
        uniform_samples = torch.rand(num_samples, device=device)
        samples = lower + uniform_samples * (upper - lower)
    else:
        samples = 0.5 * (lower + upper)
    
    return samples


def sample_stratified_batch(
    near: torch.Tensor,
    far: torch.Tensor,
    num_samples: int,
    randomize: bool = True
) -> torch.Tensor:
    device = near.device
    num_rays = near.shape[0]
    
    if near.ndim == 1:
        near = near.unsqueeze(-1)
    if far.ndim == 1:
        far = far.unsqueeze(-1)
    
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=device)
    t_vals = t_vals.unsqueeze(0).expand(num_rays, -1)
    bins = near * (1.0 - t_vals) + far * t_vals
    
    lower = bins[:, :-1]
    upper = bins[:, 1:]
    
    if randomize:
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
    device = bins.device
    num_rays = bins.shape[0]
    
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    
    if deterministic:
        u = torch.linspace(0.0, 1.0, num_samples, device=device)
        u = u.unsqueeze(0).expand(num_rays, -1)
    else:
        u = torch.rand(num_rays, num_samples, device=device)
    
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True)
    
    below = torch.clamp(indices - 1, min=0, max=cdf.shape[-1] - 2)
    above = torch.clamp(indices, min=0, max=cdf.shape[-1] - 1)
    
    cdf_below = torch.gather(cdf, dim=-1, index=below)
    cdf_above = torch.gather(cdf, dim=-1, index=above)
    bins_below = torch.gather(bins, dim=-1, index=below)
    bins_above = torch.gather(bins, dim=-1, index=above)
    
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
    device = camera_to_world.device
    
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing='xy'
    )
    
    directions = torch.stack([
        (i - width * 0.5) / focal_length,
        -(j - height * 0.5) / focal_length,
        -torch.ones_like(i)
    ], dim=-1)
    
    rotation = camera_to_world[:3, :3]
    directions_world = directions @ rotation.T
    directions_world = F.normalize(directions_world, dim=-1)
    
    origin = camera_to_world[:3, 3]
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
