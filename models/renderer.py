"""Volumetric renderer for NeRF."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class VolumetricRenderer:
    
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
        from utils.ray_utils import sample_stratified_batch, sample_hierarchical
        
        device = rays_o.device
        num_rays = rays_o.shape[0]
        rays_d = F.normalize(rays_d, dim=-1)
        
        if self.occupancy_grid is not None:
            t_coarse = self.occupancy_grid.sample_occupied_regions(
                rays_o, rays_d,
                near=torch.full((num_rays,), self.near, device=device),
                far=torch.full((num_rays,), self.far, device=device),
                num_samples=self.num_coarse_samples,
                randomize=randomize
            )
        else:
            t_coarse = sample_stratified_batch(
                near=torch.full((num_rays,), self.near, device=device),
                far=torch.full((num_rays,), self.far, device=device),
                num_samples=self.num_coarse_samples,
                randomize=randomize
            )
        
        rgb_coarse, depth_coarse, weights_coarse, _ = self._render_samples(
            rays_o, rays_d, t_coarse, model_coarse
        )
        
        if model_fine is not None:
            t_mid = 0.5 * (t_coarse[:, :-1] + t_coarse[:, 1:])
            bins = torch.cat([t_coarse[:, :1], t_mid, t_coarse[:, -1:]], dim=-1)
            weights_for_sampling = weights_coarse.detach()
            
            t_fine = sample_hierarchical(
                bins=bins,
                weights=weights_for_sampling,
                num_samples=self.num_fine_samples,
                deterministic=not randomize
            )
            
            t_combined = torch.cat([t_coarse, t_fine], dim=-1)
            t_combined, _ = torch.sort(t_combined, dim=-1)
            
            rgb_fine, depth_fine, weights_fine, acc_fine = self._render_samples(
                rays_o, rays_d, t_combined, model_fine
            )
            
            rgb = rgb_fine
            depth = depth_fine
        else:
            rgb = rgb_coarse
            depth = depth_coarse
            _, _, _, acc_coarse = self._render_samples(
                rays_o, rays_d, t_coarse, model_coarse
            )
        
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
        num_rays, num_samples = t_samples.shape
        
        positions = rays_o.unsqueeze(1) + t_samples.unsqueeze(2) * rays_d.unsqueeze(1)
        
        if self.use_viewdirs:
            directions = rays_d.unsqueeze(1).expand(-1, num_samples, -1)
        else:
            directions = None
        
        positions_flat = positions.reshape(-1, 3)
        if directions is not None:
            directions_flat = directions.reshape(-1, 3)
        else:
            directions_flat = None
        
        rgb_flat, density_flat = model(positions_flat, directions_flat)
        
        rgb = rgb_flat.reshape(num_rays, num_samples, 3)
        density = density_flat.reshape(num_rays, num_samples)
        
        dists = t_samples[:, 1:] - t_samples[:, :-1]
        dists = torch.cat([dists, torch.full((num_rays, 1), 1e10, device=dists.device)], dim=-1)
        
        alpha = 1.0 - torch.exp(-density * dists)
        
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones((num_rays, 1), device=alpha.device),
                1.0 - alpha + 1e-10
            ], dim=-1),
            dim=-1
        )[:, :-1]
        
        weights = transmittance * alpha
        rgb_integrated = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        depth = torch.sum(weights * t_samples, dim=-1)
        acc = torch.sum(weights, dim=-1)
        
        if self.white_background:
            rgb_final = rgb_integrated + (1.0 - acc.unsqueeze(-1))
        else:
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
        from utils.ray_utils import get_ray_bundle
        
        rays_o, rays_d = get_ray_bundle(height, width, focal_length, camera_to_world)
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        num_rays = rays_o_flat.shape[0]
        rgb_chunks = []
        depth_chunks = []
        
        for i in range(0, num_rays, chunk_size):
            rays_o_chunk = rays_o_flat[i:i+chunk_size]
            rays_d_chunk = rays_d_flat[i:i+chunk_size]
            
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
        
        rgb_flat = torch.cat(rgb_chunks, dim=0)
        depth_flat = torch.cat(depth_chunks, dim=0)
        
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
