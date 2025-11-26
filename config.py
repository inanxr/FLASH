"""Instant-NGP configuration."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class InstantNGPConfig:
    
    num_levels: int = 20
    features_per_level: int = 4
    log2_hashmap_size: int = 21
    base_resolution: int = 16
    finest_resolution: int = 512
    
    hidden_dim: int = 64
    num_layers: int = 2
    use_viewdirs: bool = True
    
    batch_size: int = 16384
    num_iterations: int = 5000
    learning_rate_hash: float = 1e-2
    learning_rate_mlp: float = 5e-4
    lr_decay_steps: int = 1000
    lr_decay_rate: float = 0.95
    
    data_dir: str = 'data/nerf_synthetic/lego'
    img_size: int = 400
    white_background: bool = True
    precompute_rays: bool = False
    max_train_images: Optional[int] = None
    num_val_images: int = 4
    
    num_coarse_samples: int = 128
    num_fine_samples: int = 0
    near: float = 2.0
    far: float = 6.0
    chunk_size: int = 4096
    
    use_occupancy_grid: bool = True
    occupancy_resolution: int = 128
    occupancy_threshold: float = 0.01
    update_grid_every: int = 500
    
    experiment_name: str = 'instant_ngp_lego'
    log_dir: str = 'logs'
            'features_per_level': self.features_per_level,
            'log2_hashmap_size': self.log2_hashmap_size,
            'base_resolution': self.base_resolution,
            'finest_resolution': self.finest_resolution,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'use_viewdirs': self.use_viewdirs,
            'batch_size': self.batch_size,
            'num_iterations': self.num_iterations,
            'learning_rate_hash': self.learning_rate_hash,
            'learning_rate_mlp': self.learning_rate_mlp,
            'data_dir': self.data_dir,
            'img_size': self.img_size,
            'experiment_name': self.experiment_name,
        }


def get_instant_ngp_config() -> InstantNGPConfig:
    return InstantNGPConfig()


def get_quick_test_config() -> InstantNGPConfig:
    config = InstantNGPConfig()
    config.img_size = 100
    config.batch_size = 1024
    config.num_iterations = 1000
    config.log2_hashmap_size = 17
    config.experiment_name = 'instant_ngp_quick_test'
    config.validate_every = 500
    return config


if __name__ == "__main__":
    print("Instant-NGP Configuration Examples\n")
    
    print("DEFAULT CONFIG (8GB Laptop Optimized):")
    print("=" * 60)
    config = get_instant_ngp_config()
    config.print()
    
    print("\n\nQUICK TEST CONFIG (For Debugging):")
    print("=" * 60)
    quick = get_quick_test_config()
    quick.print()