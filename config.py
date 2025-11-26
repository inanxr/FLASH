"""Instant-NGP configuration."""

from dataclasses import dataclass, field
from typing import Optional
import torch
import os

@dataclass
class InstantNGPConfig:
    # --- Model Architecture ---
    num_levels: int = 16          # Reduced from 20 to save memory
    features_per_level: int = 2   # Reduced from 4 to save memory
    log2_hashmap_size: int = 19   # Reduced from 21 (2M) to 19 (500k parameters)
    base_resolution: int = 16
    finest_resolution: int = 512
    hidden_dim: int = 64
    num_layers: int = 2
    use_viewdirs: bool = True
    
    # --- Training Hyperparameters ---
    # CRITICAL FIX for 6GB VRAM: Reduced batch size from 16384 to 4096
    batch_size: int = 4096        
    num_iterations: int = 10000   # Increased iterations to compensate for smaller batch
    learning_rate_hash: float = 1e-2
    learning_rate_mlp: float = 1e-3
    lr_decay_steps: int = 2000
    lr_decay_rate: float = 0.95
    
    # --- Data & Rendering ---
    data_dir: str = 'data/nerf_synthetic/lego'
    img_size: int = 400
    white_background: bool = True
    precompute_rays: bool = False
    max_train_images: Optional[int] = None
    num_val_images: int = 4
    num_coarse_samples: int = 64  # Reduced samples for speed/memory
    num_fine_samples: int = 0
    near: float = 2.0
    far: float = 6.0
    chunk_size: int = 2048        # Lower chunk size for validation rendering
    
    # --- Occupancy Grid ---
    use_occupancy_grid: bool = True
    occupancy_resolution: int = 128
    occupancy_threshold: float = 0.01
    update_grid_every: int = 500
    
    # --- System & Logging ---
    experiment_name: str = 'instant_ngp_lego_6gb'
    log_dir: str = 'logs'
    checkpoint_dir: str = 'checkpoints'
    output_dir: str = 'outputs'
    log_every: int = 10
    validate_every: int = 1000
    save_checkpoint_every: int = 1000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    use_mixed_precision: bool = True
    resume_checkpoint: Optional[str] = None

    def __post_init__(self):
        """Creates necessary directories and checks hardware status."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.device == 'cpu':
            print("⚠️  Running on CPU (no GPU detected)")
            print("   Training time: ~25 min  |  With GPU: ~10-15 min")

    def print(self):
        """Prints the configuration in a human-readable, sectioned format."""
        print(f"\nConfiguration: {self.experiment_name} (6GB VRAM Optimized)")
        print("-" * 50)
        
        print("🧠 Model Architecture:")
        print(f"   • Hash Table: 2^{self.log2_hashmap_size} entries, {self.num_levels} levels")
        print(f"   • Resolution: {self.base_resolution} -> {self.finest_resolution}")
        print(f"   • MLP: {self.num_layers} layers, {self.hidden_dim} hidden units")
        
        print("\n🏋️ Training:")
        print(f"   • Iterations: {self.num_iterations}")
        print(f"   • Batch Size: {self.batch_size} (Optimized for 6GB)")
        print(f"   • Learning Rates: Hash={self.learning_rate_hash}, MLP={self.learning_rate_mlp}")
        print(f"   • Precision: {'Mixed (FP16)' if self.use_mixed_precision else 'FP32'}")
        
        print("\n🖼️  Rendering:")
        print(f"   • Image Size: {self.img_size}x{self.img_size}")
        print(f"   • Samples: {self.num_coarse_samples} (coarse)")
        
        print("\n⚙️  System:")
        print(f"   • Device: {self.device}")
        print(f"   • Output Dir: {self.output_dir}")
        print("-" * 50)


def get_instant_ngp_config() -> InstantNGPConfig:
    return InstantNGPConfig()


def get_quick_test_config() -> InstantNGPConfig:
    config = InstantNGPConfig()
    config.img_size = 100
    config.batch_size = 2048
    config.num_iterations = 1000
    config.log2_hashmap_size = 16
    config.experiment_name = 'instant_ngp_quick_test'
    config.validate_every = 500
    return config


if __name__ == "__main__":
    print("Instant-NGP Configuration Examples\n")
    config = get_instant_ngp_config()
    config.print()