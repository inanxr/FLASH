"""Training script for Instant-NGP NeRF."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import argparse
from typing import Dict, Tuple
from tqdm import tqdm

# Import NeRF components
from models.nerf_model import InstantNGPNeRF
from models.renderer import VolumetricRenderer
from models.occupancy_grid import OccupancyGrid
from utils.data_loader import NeRFDataset
from utils.metrics import compute_psnr, compute_ssim, compute_mse
from config import InstantNGPConfig, get_instant_ngp_config, get_quick_test_config


class InstantNGPTrainer:
    
    def __init__(self, config: InstantNGPConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        print("\n🚀 Initializing Instant-NGP...")
        self.model = InstantNGPNeRF(
            num_levels=config.num_levels,
            features_per_level=config.features_per_level,
            log2_hashmap_size=config.log2_hashmap_size,
            base_resolution=config.base_resolution,
            finest_resolution=config.finest_resolution,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            use_viewdirs=config.use_viewdirs
        ).to(self.device)
        
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            print("⚡ Optimizing with torch.compile()...")
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"   Note: Compilation skipped ({str(e)[:50]}...)")
        
        occupancy_grid = None
        if config.use_occupancy_grid:
            print("📦 Setting up occupancy grid...")
            occupancy_grid = OccupancyGrid(
                resolution=config.occupancy_resolution,
                aabb_min=[-1.5, -1.5, -1.5],
                aabb_max=[1.5, 1.5, 1.5],
                density_threshold=config.occupancy_threshold
            )
        
        self.renderer = VolumetricRenderer(
            near=config.near,
            far=config.far,
            num_coarse_samples=config.num_coarse_samples,
            num_fine_samples=config.num_fine_samples,
            use_viewdirs=config.use_viewdirs,
            white_background=config.white_background,
            occupancy_grid=occupancy_grid
        )
        
        self.optimizer = optim.Adam([
            {
                'params': self.model.hash_encoding.parameters(),
                'lr': config.learning_rate_hash,
                'name': 'hash_encoding'
            },
            {
                'params': self.model.mlp.parameters(),
                'lr': config.learning_rate_mlp,
                'name': 'mlp'
            }
        ])
        
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=config.lr_decay_rate
        )
        
        print("📂 Loading datasets...")
        self.train_dataset = NeRFDataset(
            config.data_dir,
            split='train',
            img_size=config.img_size,
            white_background=config.white_background,
            precompute_rays=config.precompute_rays,
            max_images=config.max_train_images
        )
        
        try:
            self.val_dataset = NeRFDataset(
                config.data_dir,
                split='val',
                img_size=config.img_size,
                white_background=config.white_background,
                precompute_rays=False,
                max_images=config.num_val_images
            )
        except FileNotFoundError:
            print("   No val set found, using test set...")
            try:
                self.val_dataset = NeRFDataset(
                    config.data_dir,
                    split='test',
                    img_size=config.img_size,
                    white_background=config.white_background,
                    precompute_rays=False,
                    max_images=config.num_val_images
                )
            except FileNotFoundError:
                print("   Warning: No val/test data found")
                self.val_dataset = None
        
        log_path = os.path.join(config.log_dir, config.experiment_name)
        self.writer = SummaryWriter(log_path)
        print(f"   Logs → {log_path}\n")
        
        self.iteration = 0
        self.best_psnr = 0.0
        self.training_start_time = None
        
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
    
    def train(self):
        config = self.config
        
        print("\n" + "="*60)
        print("🎯 Starting Training")
        print("="*60)
        print(f"   Iterations: {config.num_iterations:,}  |  Batch: {config.batch_size} rays")
        print(f"   Device: {self.device}  |  Time: {'~25 min' if self.device.type == 'cpu' else '~3 min'}")
        print()
        
        self.training_start_time = time.time()
        pbar = tqdm(range(self.iteration, config.num_iterations), desc="Training")
        
        for iteration in pbar:
            self.iteration = iteration
            
            metrics = self.train_step()
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'lr_hash': f"{self.get_lr('hash'):.2e}",
                'lr_mlp': f"{self.get_lr('mlp'):.2e}"
            })
            
            if iteration % config.log_every == 0:
                self.log_metrics(metrics, iteration)
            
            if iteration == 500 and self.val_dataset is not None:
                print(f"\n📸  Quick check @ {iteration} iters...")
                self.render_validation_image(iteration)
                print("   → Check outputs/ to verify progress\n")
            
            if iteration % config.validate_every == 0 and iteration > 0:
                if self.val_dataset is not None:
                    val_metrics = self.validate()
                    self.log_validation(val_metrics, iteration)
            
            if iteration % config.save_checkpoint_every == 0 and iteration > 0:
                self.save_checkpoint(iteration)
            
            if (self.renderer.occupancy_grid is not None and 
                iteration % config.update_grid_every == 0 and 
                iteration > 0):
                print(f"\n🔄 Updating occupancy grid @ {iteration}...")
                self.renderer.occupancy_grid.update_from_density(
                    self.model,
                    threshold=config.occupancy_threshold
                )
            
            if iteration % config.lr_decay_steps == 0 and iteration > 0:
                self.scheduler.step()
            
            if self.device.type == 'cuda' and iteration % 100 == 0:
                torch.cuda.empty_cache()
        
        self.save_checkpoint(config.num_iterations, is_final=True)
        
        elapsed = time.time() - self.training_start_time
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        print(f"   Time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
        print(f"  Best PSNR: {self.best_psnr:.2f} dB")
        print(f"   Checkpoint: {os.path.join(config.checkpoint_dir, f'{config.experiment_name}_final.pth')}")
        print("="*60)
        print()
        
        self.writer.close()
    
    def train_step(self) -> Dict[str, float]:
        self.model.train()
        
        rays_o, rays_d, target_rgb = self.sample_rays_batch()
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)
        target_rgb = target_rgb.to(self.device)
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                rgb_pred, depth, _  = self.renderer.render_rays(
                    rays_o, rays_d,
                    self.model,
                    self.model,
                    randomize=True,
                    return_extras=False
                )
                
                loss = compute_mse(rgb_pred, target_rgb)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            rgb_pred, depth, _ = self.renderer.render_rays(
                rays_o, rays_d,
                self.model,
                self.model,
                randomize=True,
                return_extras=False)
            
            loss = compute_mse(rgb_pred, target_rgb)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    print(f"⚠️  NaN/Inf gradient in {name}")
                    has_nan_grad = True
            
            if has_nan_grad:
                print("⚠️  Training instability detected! Skipping this step.")
                print("   TIP: Try reducing learning rate (--lr_hash 5e-3)")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
        
        with torch.no_grad():
            psnr = compute_psnr(rgb_pred, target_rgb)
        
        return {
            'loss': loss.item(),
            'psnr': psnr.item()
        }
    
    def sample_rays_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.precompute_rays:
            rays_o, rays_d, rgb = self.train_dataset[0]
            num_rays = rays_o.shape[0]
            indices = torch.randint(0, num_rays, (self.config.batch_size,))
            return rays_o[indices], rays_d[indices], rgb[indices]
        else:
            # Sample from random image
            num_images = len(self.train_dataset)
            img_idx = np.random.randint(0, num_images)
            rays_o, rays_d, rgb = self.train_dataset[img_idx]
            num_rays = rays_o.shape[0]
            indices = torch.randint(0, num_rays, (self.config.batch_size,))
            return rays_o[indices], rays_d[indices], rgb[indices]
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        
        print(f"\n🔍 Validation at iteration {self.iteration}...")
        
        metrics_list = []
        
        with torch.no_grad():
            for i in range(len(self.val_dataset)):
                rays_o, rays_d, target_rgb = self.val_dataset[i]
                
                # Render in chunks
                num_rays = rays_o.shape[0]
                rgb_chunks = []
                
                for j in range(0, num_rays, self.config.chunk_size):
                    rays_o_chunk = rays_o[j:j+self.config.chunk_size].to(self.device)
                    rays_d_chunk = rays_d[j:j+self.config.chunk_size].to(self.device)
                    
                    rgb_chunk, _, _ = self.renderer.render_rays(
                        rays_o_chunk, rays_d_chunk,
                        self.model, self.model,
                        randomize=False,
                        return_extras=False
                    )
                    
                    rgb_chunks.append(rgb_chunk.cpu())
                
                rgb_pred = torch.cat(rgb_chunks, dim=0)
                
                # Compute metrics
                psnr = compute_psnr(rgb_pred, target_rgb).item()
                ssim = compute_ssim(
                    rgb_pred.reshape(self.config.img_size, self.config.img_size, 3),
                    target_rgb.reshape(self.config.img_size, self.config.img_size, 3)
                ).item()
                
                metrics_list.append({'psnr': psnr, 'ssim': ssim})
                print(f"  Val image {i}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
                
                # Save first validation image
                if i == 0:
                    self.save_rendered_image(rgb_pred, f'{self.config.experiment_name}_val_{self.iteration:06d}')
        
        avg_metrics = {
            'psnr': np.mean([m['psnr'] for m in metrics_list]),
            'ssim': np.mean([m['ssim'] for m in metrics_list])
        }
        
        # Track best PSNR
        if avg_metrics['psnr'] > self.best_psnr:
            self.best_psnr = avg_metrics['psnr']
            self.save_checkpoint(self.iteration, is_best=True)
        
        return avg_metrics
    
    def render_validation_image(self, iteration: int):
        """Render a single validation image (for early checking)."""
        self.model.eval()
        
        with torch.no_grad():
            if self.val_dataset is not None and len(self.val_dataset) > 0:
                rays_o, rays_d, target_rgb = self.val_dataset[0]
                
                # Render in chunks
                num_rays = rays_o.shape[0]
                rgb_chunks = []
                
                for j in range(0, num_rays, self.config.chunk_size):
                    rays_o_chunk = rays_o[j:j+self.config.chunk_size].to(self.device)
                    rays_d_chunk = rays_d[j:j+self.config.chunk_size].to(self.device)
                    
                    rgb_chunk, _, _ = self.renderer.render_rays(
                        rays_o_chunk, rays_d_chunk,
                        self.model, self.model,
                        randomize=False,
                        return_extras=False
                    )
                    
                    rgb_chunks.append(rgb_chunk.cpu())
                
                rgb_pred = torch.cat(rgb_chunks, dim=0)
                self.save_rendered_image(rgb_pred, f'{self.config.experiment_name}_early_check_{iteration:06d}')
    
    def save_rendered_image(self, rgb: torch.Tensor, filename: str):
        """Save rendered image to disk."""
        img = rgb.reshape(self.config.img_size, self.config.img_size, 3)
        img_np = (img.numpy() * 255).astype(np.uint8)
        
        from PIL import Image
        output_path = os.path.join(self.config.output_dir, f'{filename}.png')
        Image.fromarray(img_np).save(output_path)
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, iteration)
        
        self.writer.add_scalar('train/lr_hash', self.get_lr('hash'), iteration)
        self.writer.add_scalar('train/lr_mlp', self.get_lr('mlp'), iteration)
    
    def log_validation(self, metrics: Dict[str, float], iteration: int):
        """Log validation metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, iteration)
    
    def get_lr(self, param_group_name: str = 'hash') -> float:
        """Get current learning rate for a parameter group."""
        for group in self.optimizer.param_groups:
            if group.get('name') == f'{param_group_name}_encoding' or group.get('name') == param_group_name:
                return group['lr']
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, iteration: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'best_psnr': self.best_psnr,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        if is_final:
            path = os.path.join(self.config.checkpoint_dir, f'{self.config.experiment_name}_final.pth')
        elif is_best:
            path = os.path.join(self.config.checkpoint_dir, f'{self.config.experiment_name}_best.pth')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'{self.config.experiment_name}_{iteration:06d}.pth')
        
        torch.save(checkpoint, path)
        print(f"  💾 Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}...")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.iteration = checkpoint['iteration']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        
        print(f"  Resumed from iteration {self.iteration}")
        print(f"  Best PSNR: {self.best_psnr:.2f} dB")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Instant-NGP NeRF')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--lr_hash', type=float, default=None, help='Learning rate for hash encoding')
    parser.add_argument('--lr_mlp', type=float, default=None, help='Learning rate for MLP')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.quick_test:
        config = get_quick_test_config()
        print("Using quick test configuration (1000 iterations)")
    else:
        config = get_instant_ngp_config()
        print("Using Instant-NGP configuration (5000 iterations)")
    
    # Override with command line args
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.resume:
        config.resume_checkpoint = args.resume
    if args.lr_hash:
        config.learning_rate_hash = args.lr_hash
    if args.lr_mlp:
        config.learning_rate_mlp = args.lr_mlp
    
    # Print configuration
    config.print()
    
    # Create trainer
    trainer = InstantNGPTrainer(config)
    
    # Load checkpoint if resuming
    if config.resume_checkpoint:
        trainer.load_checkpoint(config.resume_checkpoint)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
