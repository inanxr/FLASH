"""
Training Worker for FLASH Studio

Background thread that runs NeRF training without freezing the UI.
Emits signals for progress updates and preview images.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import FLASH code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainingWorker(QThread):
    """Background worker for NeRF training."""
    
    # Signals
    progress_update = pyqtSignal(int, float, float)  # iteration, loss, psnr
    preview_update = pyqtSignal(np.ndarray)  # preview image
    status_update = pyqtSignal(str)  # status message
    training_complete = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, config_dict):
        """
        Initialize training worker.
        
        Args:
            config_dict: Dictionary with training configuration
        """
        super().__init__()
        self.config_dict = config_dict
        self.should_stop = False
        self.should_pause = False
    
    def run(self):
        """Run training in background thread."""
        try:
            # Import FLASH training code
            from train import InstantNGPTrainer
            from config import InstantNGPConfig
            
            # Create config from dict
            config = InstantNGPConfig()
            
            # Apply user settings
            config.data_dir = f"data/nerf_example_data/nerf_synthetic/{self.config_dict['dataset']}"
            config.num_iterations = self.config_dict['iterations']
            config.batch_size = self.config_dict['batch_size']
            config.num_levels = self.config_dict['num_levels']
            config.features_per_level = self.config_dict['features_per_level']
            config.num_coarse_samples = self.config_dict['num_samples']
            config.img_size = self.config_dict['img_size']
            
            # Quick test mode
            if self.config_dict.get('quick_test', False):
                config.num_iterations = 1000
                config.img_size = 100
                config.batch_size = 1024
            
            # Status update
            self.status_update.emit("Initializing trainer...")
            
            # Create trainer
            trainer = InstantNGPTrainer(config)
            
            # Override train loop to emit signals
            self._train_with_updates(trainer, config)
            
            self.training_complete.emit()
            
        except Exception as e:
            self.training_error.emit(str(e))
    
    def _train_with_updates(self, trainer, config):
        """Modified training loop with UI updates."""
        from tqdm import tqdm
        import time
        
        # Training loop
        for iteration in range(config.num_iterations):
            # Check if should stop
            if self.should_stop:
                self.status_update.emit("Training stopped by user")
                break
            
            # Check if should pause
            while self.should_pause and not self.should_stop:
                time.sleep(0.1)
            
            # Training step
            trainer.iteration = iteration
            metrics = trainer.train_step()
            
            # Emit progress every 10 iterations
            if iteration % 10 == 0:
                self.progress_update.emit(
                    iteration,
                    metrics['loss'],
                    metrics['psnr']
                )
            
            # Emit preview every 100 iterations
            if iteration % 100 == 0 and iteration > 0:
                try:
                    preview = self._render_preview(trainer)
                    self.preview_update.emit(preview)
                except Exception as e:
                    print(f"Preview render failed: {e}")
            
            # Status update every 500 iterations
            if iteration % 500 == 0:
                elapsed = time.time() - trainer.training_start_time
                eta = (elapsed / (iteration + 1)) * (config.num_iterations - iteration - 1)
                self.status_update.emit(
                    f"Iter {iteration}/{config.num_iterations} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"PSNR: {metrics['psnr']:.2f} dB | "
                    f"ETA: {eta/60:.1f}m"
                )
    
    def _render_preview(self, trainer):
        """Render a preview image."""
        import torch
        
        # Get a validation image
        if trainer.val_dataset is not None and len(trainer.val_dataset) > 0:
            rays_o, rays_d, target_rgb = trainer.val_dataset[0]
            
            # Render with current model
            with torch.no_grad():
                rays_o = rays_o[:4096].to(trainer.device)  # Limit rays for speed
                rays_d = rays_d[:4096].to(trainer.device)
                
                rgb_pred, _, _ = trainer.renderer.render_rays(
                    rays_o, rays_d,
                    trainer.model,
                    trainer.model,
                    randomize=False,
                    return_extras=False
                )
            
            # Convert to numpy image
            img_size = trainer.config.img_size
            rgb_pred = rgb_pred.cpu().numpy()
            
            # Reshape to image
            if rgb_pred.shape[0] == img_size * img_size:
                image = rgb_pred.reshape(img_size, img_size, 3)
            else:
                # Pad if needed
                total_pixels = img_size * img_size
                if rgb_pred.shape[0] < total_pixels:
                    padded = np.zeros((total_pixels, 3))
                    padded[:rgb_pred.shape[0]] = rgb_pred
                    rgb_pred = padded
                image = rgb_pred[:total_pixels].reshape(img_size, img_size, 3)
            
            # Convert to uint8
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            return image
        else:
            # Return black image if no validation data
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def stop(self):
        """Stop training."""
        self.should_stop = True
    
    def pause(self):
        """Pause training."""
        self.should_pause = True
    
    def resume(self):
        """Resume training."""
        self.should_pause = False
