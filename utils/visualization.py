"""Visualization utilities for training curves and image comparisons."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import os


def plot_training_curves(
    iterations: List[int],
    losses: List[float],
    psnrs: List[float],
    output_path: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(iterations, losses, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.semilogy()
    
    ax2.plot(iterations, psnrs, linewidth=2, color='green')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Image Quality (PSNR)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_path}")


def plot_comparison(
    gt_image: np.ndarray,
    pred_image: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    output_path: str = 'comparison.png',
    psnr: Optional[float] = None,
    ssim: Optional[float] = None
):
    if depth_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(gt_image)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    title = 'Prediction'
    if psnr is not None:
        title += f'\nPSNR: {psnr:.2f} dB'
    if ssim is not None:
        title += f', SSIM: {ssim:.4f}'
    axes[1].imshow(pred_image)
    axes[1].set_title(title)
    axes[1].axis('off')
    
    if depth_map is not None:
        depth_vis = axes[2].imshow(depth_map, cmap='viridis')
        axes[2].set_title('Depth Map')
        axes[2].axis('off')
        plt.colorbar(depth_vis, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {output_path}")


def visualize_depth_map(
    depth_map: np.ndarray,
    output_path: str = 'depth.png',
    cmap: str = 'viridis'
):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(depth_map, cmap=cmap)
    plt.colorbar(im, label='Depth')
    plt.title('Depth Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Depth map saved to: {output_path}")


if __name__ == "__main__":
    print("Testing Visualization Utilities...")
    print("=" * 60)
    
    print("Test 1: Training Curves")
    iterations = list(range(0, 10000, 100))
    losses = [0.1 * np.exp(-i/5000) + 0.001 for i in iterations]
    psnrs = [20 + 10 * (1 - np.exp(-i/5000)) for i in iterations]
    
    plot_training_curves(iterations, losses, psnrs, 'test_curves.png')
    print()
    
    print("Test 2: Image Comparison")
    gt = np.random.rand(100, 100, 3)
    pred = gt + np.random.randn(100, 100, 3) * 0.05
    pred = np.clip(pred, 0, 1)
    depth = np.random.rand(100, 100)
    
    plot_comparison(gt, pred, depth, 'test_comparison.png', psnr=28.5, ssim=0.92)
    print()
    
    print("Test 3: Depth Visualization")
    visualize_depth_map(depth, 'test_depth.png')
    print()
    
    print("✅ Visualization test passed!")
    print("Generated test images: test_curves.png, test_comparison.png, test_depth.png")
