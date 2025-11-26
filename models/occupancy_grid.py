"""Occupancy grid for empty space skipping in NeRF rendering."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class OccupancyGrid:
    
    def __init__(
        self,
        resolution: int = 128,
        aabb_min: list = [-1.5, -1.5, -1.5],
        aabb_max: list = [1.5, 1.5, 1.5],
        density_threshold: float = 0.01
    ):
        self.resolution = resolution
        self.aabb_min = torch.tensor(aabb_min, dtype=torch.float32)
        self.aabb_max = torch.tensor(aabb_max, dtype=torch.float32)
        self.density_threshold = density_threshold
        
        self.grid = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
        self.voxel_size = (self.aabb_max - self.aabb_min) / resolution
        
        print(f"Initialized OccupancyGrid:")
        print(f"  Resolution: {resolution}³ = {resolution**3:,} voxels")
        print(f"  Bounds: {aabb_min} to {aabb_max}")
        print(f"  Voxel size: {self.voxel_size.tolist()}")
    
    def world_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        normalized = (positions - self.aabb_min) / (self.aabb_max - self.aabb_min)
        grid_coords = normalized * self.resolution
        grid_indices = torch.clamp(grid_coords, 0, self.resolution - 1).long()
        return grid_indices
    
    def grid_to_world(self, grid_indices: torch.Tensor) -> torch.Tensor:
        normalized = (grid_indices.float() + 0.5) / self.resolution
        positions = self.aabb_min + normalized * (self.aabb_max - self.aabb_min)
        return positions
    
    @torch.no_grad()
    def update_from_density(
        self,
        model: nn.Module,
        threshold: Optional[float] = None,
        batch_size: int = 8192
    ):
        if threshold is None:
            threshold = self.density_threshold
        
        device = next(model.parameters()).device
        self.grid = self.grid.to(device)
        self.aabb_min = self.aabb_min.to(device)
        self.aabb_max = self.aabb_max.to(device)
        
        res = self.resolution
        x = torch.linspace(0, res - 1, res, device=device)
        y = torch.linspace(0, res - 1, res, device=device)
        z = torch.linspace(0, res - 1, res, device=device)
        
        grid_coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        grid_coords = grid_coords.reshape(-1, 3)
        world_positions = self.grid_to_world(grid_coords)
        
        num_points = world_positions.shape[0]
        densities = []
        
        for i in range(0, num_points, batch_size):
            batch = world_positions[i:i+batch_size]
            _, density = model(batch, None)
            densities.append(density)
        
        densities = torch.cat(densities, dim=0).squeeze(-1)
        occupied = densities > threshold
        self.grid = occupied.reshape(res, res, res)
        
        num_occupied = self.grid.sum().item()
        total_voxels = res ** 3
        occupancy_rate = num_occupied / total_voxels * 100
        
        print(f"Updated OccupancyGrid:")
        print(f"  Occupied voxels: {num_occupied:,} / {total_voxels:,} ({occupancy_rate:.1f}%)")
        print(f"  Density threshold: {threshold}")
    
    def ray_aabb_intersection(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = rays_o.device
        aabb_min = self.aabb_min.to(device)
        aabb_max = self.aabb_max.to(device)
        
        t1 = (aabb_min - rays_o) / (rays_d + 1e-8)
        t2 = (aabb_max - rays_o) / (rays_d + 1e-8)
        
        t_min = torch.min(t1, t2).max(dim=-1)[0]
        t_max = torch.max(t1, t2).min(dim=-1)[0]
        t_min = torch.clamp(t_min, min=0.0)
        
        return t_min, t_max
    
    def is_occupied(self, positions: torch.Tensor) -> torch.Tensor:
        device = positions.device
        self.grid = self.grid.to(device)
        self.aabb_min = self.aabb_min.to(device)
        self.aabb_max = self.aabb_max.to(device)
        
        grid_indices = self.world_to_grid(positions)
        occupied = self.grid[
            grid_indices[:, 0],
            grid_indices[:, 1],
            grid_indices[:, 2]
        ]
        
        return occupied
    
    def sample_occupied_regions(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        num_samples: int,
        randomize: bool = True
    ) -> torch.Tensor:
        device = rays_o.device
        num_rays = rays_o.shape[0]
        
        t_min, t_max = self.ray_aabb_intersection(rays_o, rays_d)
        
        if isinstance(near, torch.Tensor):
            t_min = torch.maximum(t_min, near)
            t_max = torch.minimum(t_max, far)
        else:
            t_min = torch.maximum(t_min, torch.full_like(t_min, near))
            t_max = torch.minimum(t_max, torch.full_like(t_max, far))
        
        t_samples = torch.linspace(0, 1, num_samples, device=device).unsqueeze(0)
        t_samples = t_min.unsqueeze(-1) + t_samples * (t_max - t_min).unsqueeze(-1)
        
        if randomize:
            bin_size = (t_max - t_min).unsqueeze(-1) / num_samples
            jitter = torch.rand_like(t_samples) * bin_size
            t_samples = t_samples + jitter
        
        return t_samples


if __name__ == "__main__":
    print("Testing OccupancyGrid...")
    print("=" * 60)
    
    # Create grid
    grid = OccupancyGrid(resolution=64)
    
    print("\nTest 1: World to Grid Conversion")
    positions = torch.tensor([
        [0.0, 0.0, 0.0],      # Center
        [-1.5, -1.5, -1.5],   # Min corner
        [1.5, 1.5, 1.5],      # Max corner
    ])
    grid_indices = grid.world_to_grid(positions)
    print(f"  World: {positions.tolist()}")
    print(f"  Grid:  {grid_indices.tolist()}")
    print()
    
    print("Test 2: Ray-AABB Intersection")
    rays_o = torch.tensor([[0.0, 0.0, -3.0]])  # Camera at z=-3
    rays_d = F.normalize(torch.tensor([[0.0, 0.0, 1.0]]), dim=-1)  # Looking forward
    t_min, t_max = grid.ray_aabb_intersection(rays_o, rays_d)
    print(f"  Ray origin: {rays_o.tolist()}")
    print(f"  Ray direction: {rays_d.tolist()}")
    print(f"  Entry distance: {t_min.item():.2f}")
    print(f"  Exit distance: {t_max.item():.2f}")
    print()
    
    print("Test 3: Occupancy Check")
    # Mark some voxels as occupied
    grid.grid[30:34, 30:34, 30:34] = True
    test_pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    occupied = grid.is_occupied(test_pos)
    print(f"  Positions: {test_pos.tolist()}")
    print(f"  Occupied: {occupied.tolist()}")
    print()
    
    print("✅ OccupancyGrid test passed!")
