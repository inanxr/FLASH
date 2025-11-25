"""
Volumetric Renderer for NeRF

Implements the core volume rendering equation that makes NeRF work.

VOLUME RENDERING EQUATION:
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt

Where:
  r(t) = o + t·d       is the ray (origin o, direction d)
  σ(r(t))              is volume density at point r(t)
  c(r(t), d)           is RGB color at point r(t) viewed from direction d
  T(t) = exp(-∫₀ᵗ σ(r(s)) ds) is transmittance (accumulated transparency)

DISCRETE APPROXIMATION:
Ĉ(r) = Σᵢ Tᵢ · αᵢ · cᵢ

Where:
  αᵢ = 1 - exp(-σᵢ·δᵢ)     is opacity at sample i
  δᵢ = tᵢ₊₁ - tᵢ            is distance between samples
  Tᵢ = Πⱼ₌₁ⁱ⁻¹ (1 - αⱼ)   is accumulated transmittance

This is alpha compositing from computer graphics!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import torch.nn as nn

class VolumetricRenderer:
    """
    Renders RGB images from NeRF using volumetric ray marching.
    ...
    """
    
    def __init__(
        self,
        near: float = 2.0,
        far: float = 6.0,
        num_coarse_samples: int = 32,
        num_fine_samples: int = 64,
        use_viewdirs: bool = True,
        white_background: bool = False,
        occupancy_grid: Optional['OccupancyGrid'] = None
    ):
        self.near = near
        self.far = far
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.use_viewdirs = use_viewdirs
        self.white_background = white_background
        self.occupancy_grid = occupancy_grid
    
    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        model_coarse: nn.Module,
        model_fine: Optional[nn.Module] = None,
        randomize: bool = True,
        return_extras: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Render rays using volumetric integration.
        ...
        """
        from utils.ray_utils import sample_stratified_batch, sample_hierarchical
        
        device = rays_o.device
        num_rays = rays_o.shape[0]
        
        # Ensure ray directions are normalized
        rays_d = F.normalize(rays_d, dim=-1)
        
        # ===== COARSE NETWORK =====
        # Stratified sampling along rays
        if self.occupancy_grid is not None:
            # Use occupancy grid for empty space skipping
            t_coarse = self.occupancy_grid.sample_occupied_regions(
                rays_o, rays_d,
                near=torch.full((num_rays,), self.near, device=device),
                far=torch.full((num_rays,), self.far, device=device),
                num_samples=self.num_coarse_samples,
                randomize=randomize
            )
        else:
            # Standard uniform sampling
            t_coarse = sample_stratified_batch(
                near=torch.full((num_rays,), self.near, device=device),
                far=torch.full((num_rays,), self.far, device=device),
                num_samples=self.num_coarse_samples,
                randomize=randomize
            )  # Shape: [num_rays, num_coarse_samples]
        
        # Render with coarse network
        rgb_coarse, depth_coarse, weights_coarse, _ = self._render_samples(
            rays_o, rays_d, t_coarse, model_coarse
        )
        
        # ===== FINE NETWORK (HIERARCHICAL SAMPLING) =====
        if model_fine is not None:
            # Hierarchical sampling: sample more points where density is high
            # Prepare bins for hierarchical sampling
            # bins are the edges, so we need to get midpoints
            t_mid = 0.5 * (t_coarse[:, :-1] + t_coarse[:, 1:])  # [num_rays, num_coarse_samples-1]
            
            # Pad to match weights shape
            bins = torch.cat([
                t_coarse[:, :1],
                t_mid,
                t_coarse[:, -1:]
            ], dim=-1)  # [num_rays, num_coarse_samples]
            
            # Detach weights to stop gradients (hierarchical sampling is not differentiable)
            weights_for_sampling = weights_coarse.detach()
            
            # Sample new points using inverse transform sampling
            t_fine = sample_hierarchical(
                bins=bins,
                weights=weights_for_sampling,
                num_samples=self.num_fine_samples,
                deterministic=not randomize
            )  # [num_rays, num_fine_samples]
            
            # Combine coarse and fine samples, then sort
            t_combined = torch.cat([t_coarse, t_fine], dim=-1)  # [num_rays, num_coarse + num_fine]
            t_combined, _ = torch.sort(t_combined, dim=-1)
            
            # Render with fine network using all samples
            rgb_fine, depth_fine, weights_fine, acc_fine = self._render_samples(
                rays_o, rays_d, t_combined, model_fine
            )
            
            # Return fine network outputs
            rgb = rgb_fine
            depth = depth_fine
        else:
            # No fine network, return coarse outputs
            rgb = rgb_coarse
            depth = depth_coarse
            _, _, _, acc_coarse = self._render_samples(
                rays_o, rays_d, t_coarse, model_coarse
            )
        
        # Prepare extras dictionary
        extras = {}
        if return_extras:
            extras['rgb_coarse'] = rgb_coarse
            extras['depth_coarse'] = depth_coarse
            extras['weights_coarse'] = weights_coarse
            if model_fine is not None:
                extras['t_samples'] = t_combined
                extras['weights_fine'] = weights_fine
                extras['acc_fine'] = acc_fine
            else:
                extras['t_samples'] = t_coarse
                extras['acc_coarse'] = acc_coarse
        
        return rgb, depth, extras
    
    def _render_samples(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        t_samples: torch.Tensor,
        model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render samples along rays using volume rendering.
        ...
        """
        num_rays, num_samples = t_samples.shape
        
        # Compute 3D positions along rays
        # r(t) = o + t * d
        # Shape: [num_rays, num_samples, 3]
        positions = rays_o.unsqueeze(1) + t_samples.unsqueeze(2) * rays_d.unsqueeze(1)
        
        # Expand viewing directions to match samples
        # Shape: [num_rays, num_samples, 3]
        if self.use_viewdirs:
            directions = rays_d.unsqueeze(1).expand(-1, num_samples, -1)
        else:
            directions = None
        
        # Flatten for network query
        positions_flat = positions.reshape(-1, 3)
        if directions is not None:
            directions_flat = directions.reshape(-1, 3)
        else:
            directions_flat = None
        
        # Query NeRF network
        # rgb_flat: [num_rays * num_samples, 3]
        # density_flat: [num_rays * num_samples, 1]
        rgb_flat, density_flat = model(positions_flat, directions_flat)
        
        # Reshape back to ray samples
        rgb = rgb_flat.reshape(num_rays, num_samples, 3)
        density = density_flat.reshape(num_rays, num_samples)
        
        # Compute distances between adjacent samples
        # δᵢ = tᵢ₊₁ - tᵢ
        # Shape: [num_rays, num_samples-1]
        dists = t_samples[:, 1:] - t_samples[:, :-1]
        
        # Append large value for last distance (ray goes to infinity)
        dists = torch.cat([
            dists,
            torch.full((num_rays, 1), 1e10, device=dists.device)
        ], dim=-1)  # [num_rays, num_samples]
        
        # Compute opacity: αᵢ = 1 - exp(-σᵢ · δᵢ)
        # This is the probability that light terminates at sample i
        alpha = 1.0 - torch.exp(-density * dists)
        
        # Compute transmittance: Tᵢ = Πⱼ₌₁ⁱ⁻¹ (1 - αⱼ)
        # This is the probability that light reaches sample i
        # We compute this using cumulative product: Tᵢ = cumprod(1 - α₀, 1 - α₁, ..., 1 - αᵢ₋₁)
        # Add 1.0 at the beginning (T₀ = 1, no attenuation initially)
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones((num_rays, 1), device=alpha.device),
                1.0 - alpha + 1e-10  # Add epsilon for numerical stability
            ], dim=-1),
            dim=-1
        )[:, :-1]  # Remove last element to match shape [num_rays, num_samples]
        
        # Compute weights: wᵢ = Tᵢ · αᵢ
        # These are the contribution of each sample to the final color
        weights = transmittance * alpha
        
        # Integrate RGB: Ĉ(r) = Σᵢ wᵢ · cᵢ
        rgb_integrated = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # [num_rays, 3]
        
        # Compute expected depth: E[t] = Σᵢ wᵢ · tᵢ
        depth = torch.sum(weights * t_samples, dim=-1)  # [num_rays]
        
        # Compute accumulated opacity (for compositing)
        # acc = Σᵢ wᵢ = 1 - T_final
        acc = torch.sum(weights, dim=-1)  # [num_rays]
        
        # Composite onto background
        if self.white_background:
            # White background: RGB = RGB_integrated + (1 - acc) * 1.0
            rgb_final = rgb_integrated + (1.0 - acc.unsqueeze(-1))
        else:
            # Black background: RGB = RGB_integrated
            rgb_final = rgb_integrated
        
        return rgb_final, depth, weights, acc
    
    def render_image(
        self,
        height: int,
        width: int,
        focal_length: float,
        camera_to_world: torch.Tensor,
        model_coarse: nn.Module,
        model_fine: Optional[nn.Module] = None,
        chunk_size: int = 1024,
        randomize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a full image from a camera pose.
        ...
        """
        from utils.ray_utils import get_ray_bundle
        
        # Generate rays for all pixels
        rays_o, rays_d = get_ray_bundle(height, width, focal_length, camera_to_world)
        
        # Flatten rays for batch processing
        rays_o_flat = rays_o.reshape(-1, 3)  # [H*W, 3]
        rays_d_flat = rays_d.reshape(-1, 3)  # [H*W, 3]
        
        # Render in chunks to avoid OOM
        num_rays = rays_o_flat.shape[0]
        rgb_chunks = []
        depth_chunks = []
        
        for i in range(0, num_rays, chunk_size):
            # Get chunk of rays
            rays_o_chunk = rays_o_flat[i:i+chunk_size]
            rays_d_chunk = rays_d_flat[i:i+chunk_size]
            
            # Render chunk
            with torch.no_grad():
                rgb_chunk, depth_chunk, _ = self.render_rays(
                    rays_o_chunk,
                    rays_d_chunk,
                    model_coarse,
                    model_fine,
                    randomize=randomize,
                    return_extras=False
                )
            
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)
        
        # Concatenate chunks
        rgb_flat = torch.cat(rgb_chunks, dim=0)
        depth_flat = torch.cat(depth_chunks, dim=0)
        
        # Reshape to image dimensions
        rgb_image = rgb_flat.reshape(height, width, 3)
        depth_image = depth_flat.reshape(height, width)
        
        return rgb_image, depth_image


if __name__ == "__main__":
    print("Testing Volumetric Renderer...")
    print("=" * 60)
    
    from .nerf_model import InstantNGPNeRF
    
    # Create models
    model_coarse = InstantNGPNeRF(
        num_levels=8,
        log2_hashmap_size=14,
        hidden_dim=32
    )
    
    model_fine = InstantNGPNeRF(
        num_levels=8,
        log2_hashmap_size=14,
        hidden_dim=32
    )
    
    # Create renderer
    renderer = VolumetricRenderer(
        near=2.0,
        far=6.0,
        num_coarse_samples=16,
        num_fine_samples=16
    )
    
    print("Test 1: Render Rays (Coarse Only)")
    num_rays = 128
    rays_o = torch.zeros(num_rays, 3)  # All rays from origin
    rays_d = F.normalize(torch.randn(num_rays, 3), dim=-1)  # Random directions
    
    rgb, depth, extras = renderer.render_rays(
        rays_o, rays_d, model_coarse, model_fine=None
    )
    
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print()
    
    print("Test 2: Render Rays (Coarse + Fine)")
    rgb, depth, extras = renderer.render_rays(
        rays_o, rays_d, model_coarse, model_fine
    )
    
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Has coarse outputs: {'rgb_coarse' in extras}")
    print(f"  Has fine outputs: {'weights_fine' in extras}")
    print()
    
    print("Test 3: Render Full Image")
    H, W = 50, 50
    focal = 25.0
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0., 0., 4.])  # Move camera back
    
    rgb_img, depth_img = renderer.render_image(
        H, W, focal, c2w, model_coarse, model_fine, chunk_size=256
    )
    
    print(f"  RGB image shape: {rgb_img.shape}")
    print(f"  Depth image shape: {depth_img.shape}")
    print(f"  Expected: [{H}, {W}, 3] and [{H}, {W}]")
    print()
    
    print("✅ Volumetric renderer test passed!")
