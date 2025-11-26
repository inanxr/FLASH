"""Multi-resolution hash encoding for Instant-NGP."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class HashEncoding(nn.Module):
    
    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        bounding_box: Tuple[list, list] = ([-1, -1, -1], [1, 1, 1]),
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        self.register_buffer('bbox_min', torch.tensor(bounding_box[0], dtype=torch.float32))
        self.register_buffer('bbox_max', torch.tensor(bounding_box[1], dtype=torch.float32))
        
        self.resolutions = []
        for i in range(num_levels):
            resolution = int(np.floor(
                base_resolution * np.exp(
                    i * np.log(finest_resolution / base_resolution) / (num_levels - 1)
                )
            ))
            self.resolutions.append(resolution)
        
        print(f"\n📐 Hash Encoding: {self.num_levels * self.features_per_level}D")
        print(f"   {num_levels} levels  |  {features_per_level} features/level")
        print(f"   Table: 2^{log2_hashmap_size} ({2**log2_hashmap_size:,} entries)")
        print(f"   Resolution: {base_resolution} → {finest_resolution}")
        
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, features_per_level)
            for _ in range(num_levels)
        ])
        
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
    
    def hash_function(self, coords: torch.Tensor) -> torch.Tensor:
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=torch.int64)
        coords = coords.long()
        
        hashed = coords[..., 0] * primes[0]
        hashed = hashed ^ (coords[..., 1] * primes[1])
        hashed = hashed ^ (coords[..., 2] * primes[2])
        indices = hashed % self.hashmap_size
        
        return indices
    
    def get_corner_coords(
        self, 
        positions: torch.Tensor, 
        resolution: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = positions.shape[0]
        device = positions.device
        
        normalized = (positions - self.bbox_min) / (self.bbox_max - self.bbox_min)
        scaled = normalized * resolution
        floor_coords = torch.floor(scaled)
        frac = scaled - floor_coords
        
        corners_offset = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], device=device, dtype=torch.float32)
        
        corner_coords = floor_coords.unsqueeze(1) + corners_offset.unsqueeze(0)
        corner_coords = torch.clamp(corner_coords, 0, resolution)
        
        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], dim=1)
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], dim=1)
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], dim=1)
        
        corner_weights = (
            wx[:, corners_offset[:, 0].long()] *
            wy[:, corners_offset[:, 1].long()] *
            wz[:, corners_offset[:, 2].long()]
        )
        
        return corner_coords, corner_weights
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() != 2 or positions.shape[1] != 3:
            raise ValueError(f"Expected positions shape [N, 3], got {positions.shape}")
        
        if not torch.isfinite(positions).all():
            raise ValueError("Positions contain NaN or Inf values")
        
        outside = (positions < self.bbox_min).any(dim=1) | (positions > self.bbox_max).any(dim=1)
        if outside.any():
            print(f"Warning: {outside.sum()} positions outside bounding box (will be clamped)")
        
        N = positions.shape[0]
        encoded_features = []
        
        for level_idx, resolution in enumerate(self.resolutions):
            corner_coords, corner_weights = self.get_corner_coords(positions, resolution)
            hash_indices = self.hash_function(corner_coords)
            features = self.hash_tables[level_idx](hash_indices)
            interpolated = torch.sum(features * corner_weights.unsqueeze(-1), dim=1)
            encoded_features.append(interpolated)
        
        encoded = torch.cat(encoded_features, dim=-1)
        
        if not torch.isfinite(encoded).all():
            raise RuntimeError("Hash encoding produced NaN or Inf (check hash table initialization)")
        
        return encoded
    
    def get_output_dim(self) -> int:
        return self.num_levels * self.features_per_level


# ============================================================
# UNIT TESTS
# ============================================================

def test_hash_encoding():
    """Test hash encoding module."""
    print("=" * 60)
    print("TESTING HASH ENCODING")
    print("=" * 60)
    
    # Create encoder
    encoder = HashEncoding(
        num_levels=16,
        features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512
    )
    
    print(f"\nTest 1: Forward Pass")
    # Test forward pass
    positions = torch.randn(100, 3) * 0.5  # Random positions in [-0.5, 0.5]
    features = encoder(positions)
    
    print(f"  Input shape:  {positions.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Expected:     torch.Size([100, 32])")
    assert features.shape == (100, 32), f"Wrong output shape: {features.shape}"
    assert torch.isfinite(features).all(), "Output contains NaN or Inf"
    print("  ✓ Forward pass works")
    
    print(f"\nTest 2: Hash Function Distribution")
    # Test hash distribution
    test_coords = torch.randint(0, 512, (1000, 8, 3))
    indices = encoder.hash_function(test_coords)
    
    unique_indices = torch.unique(indices).numel()
    collision_rate = 1 - (unique_indices / (1000 * 8))
    
    print(f"  Total hashes:    {1000 * 8}")
    print(f"  Unique indices:  {unique_indices}")
    print(f"  Collision rate:  {collision_rate:.2%}")
    print(f"  Table size:      {encoder.hashmap_size:,}")
    print("  ✓ Hash function works (collisions are OK and expected)")
    
    print(f"\nTest 3: Positions Outside Bounding Box")
    # Test positions outside bounds (should clamp with warning)
    outside_positions = torch.tensor([
        [-2.0, -2.0, -2.0],  # Way outside
        [0.0, 0.0, 0.0],     # Inside
        [2.0, 2.0, 2.0],     # Way outside
    ])
    
    features_outside = encoder(outside_positions)
    
    print(f"  Input positions: {outside_positions.tolist()}")
    print(f"  Output shape:    {features_outside.shape}")
    assert torch.isfinite(features_outside).all(), "Failed on outside positions"
    print("  ✓ Handles outside positions (with warning)")
    
    print(f"\nTest 4: Gradient Flow")
    # Test gradients flow through hash tables
    positions = torch.randn(10, 3, requires_grad=True)
    features = encoder(positions)
    loss = features.sum()
    loss.backward()
    
    # Check hash table gradients
    has_gradients = all(
        table.weight.grad is not None and (table.weight.grad.abs().sum() > 0)
        for table in encoder.hash_tables
    )
    
    print(f"  Hash tables have gradients: {has_gradients}")
    print(f"  Sample gradient magnitude:  {encoder.hash_tables[0].weight.grad.abs().mean():.2e}")
    assert has_gradients, "Gradients not flowing through hash tables"
    print("  ✓ Gradients flow correctly")
    
    print("\n" + "=" * 60)
    print("✅ ALL HASH ENCODING TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when executed directly
    test_hash_encoding()
    
    print("\n" + "=" * 60)
    print("EXAMPLE USAGE")
    print("=" * 60)
    
    # Example usage
    encoder = HashEncoding()
    
    # Encode some positions
    positions = torch.randn(1024, 3) * 0.8  # 1024 random points
    encoded = encoder(positions)
    
    print(f"\nEncoded {positions.shape[0]} positions")
    print(f"  Output shape: {encoded.shape}")
    print(f"  Output range: [{encoded.min():.4f}, {encoded.max():.4f}]")
    print(f"  Memory usage: {encoded.element_size() * encoded.nelement() / 1024:.2f} KB")
