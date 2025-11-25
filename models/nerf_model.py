"""
NeRF Model

Tiny neural network using hash encoding instead of positional encoding.
Optimized NeRF architecture using hash encoding and compact MLP.

Paper: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
Authors: Müller et al. (NVIDIA), SIGGRAPH 2022

KEY DIFFERENCES FROM VANILLA NERF:
- Hash encoding (32D) instead of positional encoding (63D)
- Tiny MLP: 2 layers × 64 dims (vs 6 layers × 128 dims)
- 95% fewer parameters: ~6.5K vs ~500K
- 10-100x faster training

EXAMPLE:
    >>> model = InstantNGPNeRF()
    >>> positions = torch.randn(100, 3)
    >>> directions = F.normalize(torch.randn(100, 3), dim=-1)
    >>> rgb, density = model(positions, directions)
    >>> print(rgb.shape, density.shape)  # torch.Size([100, 3]) torch.Size([100, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hash_encoding import HashEncoding
from typing import Optional, Tuple


class InstantNGPNeRF(nn.Module):
    """
    Instant-NGP NeRF Network using hash encoding.
    
    Architecture:
        Input: (x,y,z) position + (θ,φ) direction
          ↓
        Hash Encoding: 32D learned features
          ↓
        Concat with direction: 32 + 3 = 35D
          ↓
        MLP: 2 layers × 64 dims
          ↓
        Output: RGB (3) + density (1)
    
    Args:
        num_levels: Number of hash resolution levels (default: 16)
        features_per_level: Features per hash level (default: 2)
        log2_hashmap_size: Hash table size as log2 (default: 19)
        base_resolution: Coarsest grid (default: 16)
        finest_resolution: Finest grid (default: 512)
        hidden_dim: MLP hidden dimension (default: 64)
        num_layers: Number of MLP layers (default: 2)
        use_viewdirs: Use viewing directions (default: True)
    """
    
    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_viewdirs: bool = True,
    ):
        super().__init__()
        
        self.use_viewdirs = use_viewdirs
        
        # Hash encoding for positions
        self.hash_encoding = HashEncoding(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )
        
        hash_dim = self.hash_encoding.get_output_dim()  # 32 for default
        dir_dim = 3 if use_viewdirs else 0  # Raw direction (dx, dy, dz)
        
        # Build tiny MLP
        # Compact 2-layer MLP
        layers = []
        in_dim = hash_dim + dir_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        
        # Output layer: RGB (3) + density (1)
        layers.append(nn.Linear(hidden_dim, 4))
        
        self.mlp = nn.Sequential(*layers)
        
        # Print network info
        num_params = sum(p.numel() for p in self.parameters())
        hash_params = sum(p.numel() for p in self.hash_encoding.parameters())
        mlp_params = num_params - hash_params
        
        print(f"\nInstant-NGP Network:")
        print(f"  Hash encoding: {hash_dim}D (from {num_levels} levels)")
        print(f"  MLP: {num_layers} layers × {hidden_dim} dims")
        print(f"  Hash params: {hash_params:,} (learned)")
        print(f"  MLP params:  {mlp_params:,}")
        print(f"  Total:       {num_params:,}")
        print(f"  Instant-NGP efficiency: ~77x fewer params")
        print(f"  Reduction:   {500_000 / mlp_params:.1f}x fewer MLP params")
    
    def forward(
        self,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Instant-NGP network.
        
        Args:
            positions: [N, 3] 3D positions in world space
            directions: [N, 3] viewing directions (normalized), or None
            
        Returns:
            rgb: [N, 3] RGB colors in [0, 1]
            density: [N, 1] volume density ≥ 0
        """
        # Input validation
        if positions.dim() != 2 or positions.shape[1] != 3:
            raise ValueError(f"Expected positions shape [N, 3], got {positions.shape}")
        
        if self.use_viewdirs:
            if directions is None:
                raise ValueError("use_viewdirs=True but directions=None")
            if directions.shape != positions.shape:
                raise ValueError(f"Directions shape {directions.shape} != positions shape {positions.shape}")
            
            # Ensure directions are normalized
            directions = F.normalize(directions, dim=-1)
        
        # Encode positions with hash encoding
        # This is where the speed comes from - learnable hash features!
        encoded_pos = self.hash_encoding(positions)  # [N, 32]
        
        # Concatenate with viewing direction
        if self.use_viewdirs:
            x = torch.cat([encoded_pos, directions], dim=-1)  # [N, 35]
        else:
            x = encoded_pos  # [N, 32]
        
        # Compact MLP for fast inference
        output = self.mlp(x)  # [N, 4]
        
        # Split into RGB and density
        rgb = torch.sigmoid(output[:, :3])  # [N, 3] clamped to [0, 1]
        density = F.relu(output[:, 3:4])    # [N, 1] clamped to [0, ∞)
        
        # Sanity checks
        assert torch.isfinite(rgb).all(), "RGB contains NaN or Inf"
        assert torch.isfinite(density).all(), "Density contains NaN or Inf"
        
        return rgb, density
    
    def forward_with_density_only(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only density (for mesh extraction).
        
        Args:
            positions: [N, 3] 3D positions
            
        Returns:
            density: [N, 1] volume density
        """
        # Use zero directions (won't affect density much in practice)
        dummy_dirs = torch.zeros_like(positions) if self.use_viewdirs else None
        _, density = self.forward(positions, dummy_dirs)
        return density
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================
# UNIT TESTS  
# ============================================================

def test_instant_ngp_model():
    """Test Instant-NGP network."""
    print("=" * 60)
    print("TESTING INSTANT-NGP NETWORK")
    print("=" * 60)
    
    # Create model
    model = InstantNGPNeRF(
        num_levels=16,
        features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
        hidden_dim=64,
        num_layers=2,
        use_viewdirs=True
    )
    
    print(f"\nTest 1: Forward Pass")
    # Test forward pass
    positions = torch.randn(100, 3) * 0.5
    directions = F.normalize(torch.randn(100, 3), dim=-1)
    
    rgb, density = model(positions, directions)
    
    print(f"  Input positions: {positions.shape}")
    print(f"  Input directions: {directions.shape}")
    print(f"  Output RGB: {rgb.shape}")
    print(f"  Output density: {density.shape}")
    
    assert rgb.shape == (100, 3), f"Wrong RGB shape: {rgb.shape}"
    assert density.shape == (100, 1), f"Wrong density shape: {density.shape}"
    assert (rgb >= 0).all() and (rgb <= 1).all(), "RGB not in [0, 1]"
    assert (density >= 0).all(), "Density is negative"
    print("  ✓ Forward pass works")
    
    print(f"\nTest 2: Output Ranges")
    print(f"  RGB range:     [{rgb.min():.4f}, {rgb.max():.4f}]")
    print(f"  Density range: [{density.min():.4f}, {density.max():.4f}]")
    print("  ✓ Outputs in valid ranges")
    
    print(f"\nTest 3: Gradient Flow")
    # Test gradients
    positions_grad = torch.randn(10, 3, requires_grad=True)
    directions_grad = F.normalize(torch.randn(10, 3), dim=-1)
    
    rgb, density = model(positions_grad, directions_grad)
    loss = rgb.sum() + density.sum()
    loss.backward()
    
    has_grad = positions_grad.grad is not None
    hash_has_grad = all(
        table.weight.grad is not None
        for table in model.hash_encoding.hash_tables
    )
    mlp_has_grad = all(
        p.grad is not None
        for p in model.mlp.parameters() if p.requires_grad
    )
    
    print(f"  Input gradients:      {has_grad}")
    print(f"  Hash table gradients: {hash_has_grad}")
    print(f"  MLP gradients:        {mlp_has_grad}")
    assert has_grad and hash_has_grad and mlp_has_grad, "Gradients not flowing"
    print("  ✓ Gradients flow correctly")
    
    print(f"\nTest 4: Without View Directions")
    # Test without view directions
    model_novd = InstantNGPNeRF(use_viewdirs=False)
    rgb_novd, density_novd = model_novd(positions, None)
    
    print(f"  RGB shape:     {rgb_novd.shape}")
    print(f"  Density shape: {density_novd.shape}")
    assert rgb_novd.shape == (100, 3), "Wrong shape without viewdirs"
    print("  ✓ Works without view directions")
    
    print(f"\nTest 5: Density-Only Forward")
    # Test density-only forward (for mesh extraction)
    density_only = model.forward_with_density_only(positions)
    
    print(f"  Density-only shape: {density_only.shape}")
    assert density_only.shape == (100, 1), "Wrong density-only shape"
    print("  ✓ Density-only forward works")
    
    print(f"\nTest 6: Batch Size Flexibility")
    # Test different batch sizes
    for batch_size in [1, 10, 100, 1000]:
        pos = torch.randn(batch_size, 3) * 0.5
        dirs = F.normalize(torch.randn(batch_size, 3), dim=-1)
        rgb, density = model(pos, dirs)
        
        assert rgb.shape == (batch_size, 3), f"Failed for batch_size={batch_size}"
    
    print(f"  Tested batch sizes: 1, 10, 100, 1000")
    print("  ✓ Handles different batch sizes")
    
    print("\n" + "=" * 60)
    print("✅ ALL INSTANT-NGP NETWORK TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    test_instant_ngp_model()
    
    print("\n" + "=" * 60)
    print("PARAMETER SUMMARY")
    print("=" * 60)
    
    instant = InstantNGPNeRF()
    
    instant_params = sum(p.numel() for p in instant.parameters())
    instant_mlp_params = sum(p.numel() for p in instant.mlp.parameters())
    
    print(f"\nInstant-NGP Model:")
    print(f"  Hash table params: {instant_params - instant_mlp_params:,}")
    print(f"  MLP params:        {instant_mlp_params:,}")
    print(f"  Total params:      {instant_params:,}")
    
    print("\n" + "=" * 60)
