"""
Multi-resolution Hash Encoding from Instant-NGP

Replaces fixed positional encoding with LEARNABLE multi-resolution hash tables.
This is the key innovation that makes Instant-NGP extremely fast and efficient.

Paper: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
Authors: Müller et al. (NVIDIA), SIGGRAPH 2022

KEY CONCEPTS:
1. Multiple resolution levels (coarse to fine grids)
2. Hash function maps 3D position to table index  
3. Trilinear interpolation for smooth features
4. Learnable hash tables trained via backpropagation

EXAMPLE:
    >>> encoder = HashEncoding(num_levels=16, features_per_level=2)
    >>> positions = torch.randn(100, 3)  # 100 random 3D points
    >>> features = encoder(positions)     # Encode to 32D (16×2)
    >>> print(features.shape)  # torch.Size([100, 32])
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class HashEncoding(nn.Module):
    """
    Multi-resolution hash encoding for spatial features.
    
    For each input position (x,y,z):
    1. For each of 16 resolution levels:
       - Find 8 surrounding grid points (corners of voxel)
       - Hash each corner to get table index
       - Look up features from learnable hash table
       - Trilinearly interpolate
    2. Concatenate all 16 levels → output features
    
    Args:
        num_levels: Number of resolution levels (default: 16)
        features_per_level: Feature dimension per level (default: 2)
        log2_hashmap_size: Hash table size as log2 (default: 19 → 512K entries)
        base_resolution: Coarsest grid resolution (default: 16)
        finest_resolution: Finest grid resolution (default: 512)
        bounding_box: Scene bounds as [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    """
    
    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,  # 2^19 = 524,288 entries (~8MB for 8GB laptop)
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
        
        # Bounding box (scene bounds)
        self.register_buffer('bbox_min', torch.tensor(bounding_box[0], dtype=torch.float32))
        self.register_buffer('bbox_max', torch.tensor(bounding_box[1], dtype=torch.float32))
        
        # Calculate resolution for each level (geometric progression)
        # Formula: N_l = floor(N_min * b^l) where b = exp((log(N_max/N_min))/(L-1))
        self.resolutions = []
        for i in range(num_levels):
            resolution = int(np.floor(
                base_resolution * np.exp(
                    i * np.log(finest_resolution / base_resolution) / (num_levels - 1)
                )
            ))
            self.resolutions.append(resolution)
        
        print(f"Hash Encoding - Resolutions: {self.resolutions[:4]}...{self.resolutions[-4:]}")
        print(f"Hash Encoding - Table size: {self.hashmap_size:,} entries ({self.hashmap_size * 4 / 1024 / 1024:.1f} MB)")
        
        # Create learnable hash tables for each level
        # Each table: [hashmap_size, features_per_level]
        # Total memory: num_levels × hashmap_size × features_per_level × 4 bytes
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, features_per_level)
            for _ in range(num_levels)
        ])
        
        # Initialize with small uniform distribution (important for stability)
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
        
        print(f"Hash Encoding - Output dim: {self.get_output_dim()}D")
    
    def hash_function(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Hash 3D integer coordinates to table indices using prime number hashing.
        
        Formula: hash(x,y,z) = (x*p1 XOR y*p2 XOR z*p3) mod table_size
        
        Hash collisions (multiple positions → same index) are INTENTIONAL and OK!
        The network learns to distinguish them using viewing direction.
        
        Args:
            coords: [N, 8, 3] integer grid coordinates
            
        Returns:
            indices: [N, 8] hash table indices in range [0, hashmap_size)
        """
        # Large prime numbers from Instant-NGP paper
        # These help distribute positions evenly across hash table
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=torch.int64)
        
        # Ensure coords are integers
        coords = coords.long()
        
        # XOR-based hash: combine x, y, z coordinates
        # XOR is fast and provides good distribution
        hashed = coords[..., 0] * primes[0]
        hashed = hashed ^ (coords[..., 1] * primes[1])
        hashed = hashed ^ (coords[..., 2] * primes[2])
        
        # Modulo to get valid indices in [0, hashmap_size)
        indices = hashed % self.hashmap_size
        
        return indices
    
    def get_corner_coords(
        self, 
        positions: torch.Tensor, 
        resolution: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get 8 corner grid coordinates and trilinear interpolation weights.
        
        For each position, finds the surrounding voxel and computes:
        - Corner coordinates (8 corners of the voxel)
        - Interpolation weights (based on distance to corners)
        
        Args:
            positions: [N, 3] positions in [bbox_min, bbox_max]
            resolution: Grid resolution for this level
            
        Returns:
            corner_coords: [N, 8, 3] integer grid coordinates
            corner_weights: [N, 8] trilinear interpolation weights
        """
        N = positions.shape[0]
        device = positions.device
        
        # Normalize positions to [0, resolution]
        # This maps world coordinates to grid coordinates
        normalized = (positions - self.bbox_min) / (self.bbox_max - self.bbox_min)
        scaled = normalized * resolution
        
        # Get floor (bottom-left-back corner of voxel)
        floor_coords = torch.floor(scaled)
        
        # Get fractional part for interpolation [0, 1]
        # This tells us how far between grid points we are
        frac = scaled - floor_coords  # [N, 3]
        
        # Generate 8 corners of the voxel
        # Binary pattern: (0,0,0), (0,0,1), (0,1,0), ..., (1,1,1)
        corners_offset = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], device=device, dtype=torch.float32)  # [8, 3]
        
        # Add offsets to floor to get all 8 corners
        corner_coords = floor_coords.unsqueeze(1) + corners_offset.unsqueeze(0)  # [N, 8, 3]
        
        # Clamp to valid grid range [0, resolution]
        # Positions outside bounds will clamp to boundary
        corner_coords = torch.clamp(corner_coords, 0, resolution)
        
        # === TRILINEAR INTERPOLATION WEIGHTS ===
        # For smooth feature lookup (critical for quality!)
        #
        # Weight for each corner depends on distance:
        # - Close corner: high weight
        # - Far corner: low weight
        #
        # For corner (i,j,k):
        #   w = w_x^i * w_y^j * w_z^k
        # where w_x^0 = 1-frac_x, w_x^1 = frac_x
        
        # Weights along each axis
        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], dim=1)  # [N, 2]
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], dim=1)  # [N, 2]
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], dim=1)  # [N, 2]
        
        # Compute weight for each of the 8 corners
        # corners_offset tells us which weight to use (0 or 1) for each axis
        corner_weights = (
            wx[:, corners_offset[:, 0].long()] *
            wy[:, corners_offset[:, 1].long()] *
            wz[:, corners_offset[:, 2].long()]
        )  # [N, 8]
        
        return corner_coords, corner_weights
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions using multi-resolution hash encoding.
        
        Algorithm:
        1. For each resolution level:
           a. Find 8 corner coordinates
           b. Hash corners to get table indices
           c. Look up features from hash table
           d. Trilinearly interpolate
        2. Concatenate all levels
        
        Args:
            positions: [N, 3] 3D positions in world space
            
        Returns:
            encoded: [N, num_levels * features_per_level] encoded features
        """
        # Input validation
        if positions.dim() != 2 or positions.shape[1] != 3:
            raise ValueError(f"Expected positions shape [N, 3], got {positions.shape}")
        
        # Check for NaN/Inf (common source of training issues)
        if not torch.isfinite(positions).all():
            raise ValueError("Positions contain NaN or Inf values")
        
        # Warn if positions are outside bounding box (will be clamped)
        outside = (positions < self.bbox_min).any(dim=1) | (positions > self.bbox_max).any(dim=1)
        if outside.any():
            print(f"Warning: {outside.sum()} positions outside bounding box (will be clamped)")
        
        N = positions.shape[0]
        encoded_features = []
        
        # Process each resolution level
        for level_idx, resolution in enumerate(self.resolutions):
            # Get 8 corner coordinates and interpolation weights
            corner_coords, corner_weights = self.get_corner_coords(positions, resolution)
            
            # Hash corner coordinates to get table indices
            hash_indices = self.hash_function(corner_coords)  # [N, 8]
            
            # Look up features from hash table
            # hash_tables[level_idx] is an Embedding layer
            features = self.hash_tables[level_idx](hash_indices)  # [N, 8, features_per_level]
            
            # Trilinear interpolation: weighted sum of 8 corners
            # This makes features smooth (no blocky artifacts)
            interpolated = torch.sum(
                features * corner_weights.unsqueeze(-1),  # [N, 8, features_per_level]
                dim=1
            )  # [N, features_per_level]
            
            encoded_features.append(interpolated)
        
        # Concatenate all levels
        # Coarse levels (early) provide large-scale features
        # Fine levels (later) provide detail
        encoded = torch.cat(encoded_features, dim=-1)  # [N, num_levels * features_per_level]
        
        # Sanity check output
        if not torch.isfinite(encoded).all():
            raise RuntimeError("Hash encoding produced NaN or Inf (check hash table initialization)")
        
        return encoded
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
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
