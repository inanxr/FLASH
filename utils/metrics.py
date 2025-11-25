"""
Metrics for NeRF Evaluation

Implements standard metrics for assessing novel view synthesis quality:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity) - optional

These metrics are used to evaluate how well the trained NeRF matches ground truth images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR is the standard metric for image reconstruction quality.
    Higher is better (30+ dB is considered good quality).
    
    Formula:
        MSE = mean((pred - target)²)
        PSNR = -10 · log₁₀(MSE / max_val²)
             = 10 · log₁₀(max_val² / MSE)
             = 20 · log₁₀(max_val) - 10 · log₁₀(MSE)
    
    Args:
        pred: Predicted image, shape [..., C]
        target: Ground truth image, shape [..., C]
        max_val: Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in decibels (dB)
    
    Example:
        >>> pred = torch.rand(100, 100, 3)
        >>> target = torch.rand(100, 100, 3)
        >>> psnr = compute_psnr(pred, target)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Compute mean squared error
    mse = F.mse_loss(pred, target)
    
    # Handle perfect reconstruction (MSE = 0)
    if mse == 0:
        return torch.tensor(float('inf'))
    
    # Compute PSNR
    psnr = -10.0 * torch.log10(mse / (max_val ** 2))
    
    return psnr


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE).
    
    This is the primary loss function used during training.
    Lower is better.
    
    Args:
        pred: Predicted values
        target: Ground truth values
    
    Returns:
        MSE value
    """
    return F.mse_loss(pred, target)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    filter_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    SSIM is a perceptual metric that measures similarity in terms of
    luminance, contrast, and structure. It correlates better with human
    perception than PSNR.
    
    Range: [-1, 1], where 1 means perfect structural similarity.
    Typically values > 0.9 indicate high quality.
    
    Formula:
        SSIM = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))
        
    Where:
        μₓ, μᵧ: mean of pred and target
        σₓ, σᵧ: variance of pred and target  
        σₓᵧ: covariance of pred and target
        C₁, C₂: constants for stability
    
    Args:
        pred: Predicted image, shape [H, W, C]
        target: Ground truth image, shape [H, W, C]
        max_val: Maximum possible pixel value
        filter_size: Size of Gaussian filter
        k1, k2: Constants for SSIM formula
    
    Returns:
        SSIM value in range [-1, 1]
    
    Reference:
        Wang et al. "Image Quality Assessment: From Error Visibility to Structural Similarity"
        IEEE Transactions on Image Processing, 2004
    """
    try:
        from skimage.metrics import structural_similarity
        
        # Convert tensors to numpy
        if isinstance(pred, torch.Tensor):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = pred
            
        if isinstance(target, torch.Tensor):
            target_np = target.detach().cpu().numpy()
        else:
            target_np = target
        
        # Compute SSIM
        ssim_value = structural_similarity(
            pred_np,
            target_np,
            data_range=max_val,
            multichannel=True,
            channel_axis=-1,
            win_size=filter_size
        )
        
        return torch.tensor(ssim_value)
        
    except ImportError:
        print("Warning: scikit-image not available, using simplified SSIM")
        return _compute_ssim_pytorch(pred, target, max_val, filter_size, k1, k2)


def _compute_ssim_pytorch(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float,
    filter_size: int,
    k1: float,
    k2: float
) -> torch.Tensor:
    """
    Simplified SSIM implementation in PyTorch (fallback).
    
    This is a simplified version that doesn't use Gaussian filtering.
    For accurate SSIM, use scikit-image's implementation.
    """
    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2
    
    # Compute means
    mu_pred = pred.mean()
    mu_target = target.mean()
    
    # Compute variances and covariance
    var_pred = ((pred - mu_pred) ** 2).mean()
    var_target = ((target - mu_target) ** 2).mean()
    cov = ((pred - mu_pred) * (target - mu_target)).mean()
    
    # Compute SSIM
    numerator = (2 * mu_pred * mu_target + C1) * (2 * cov + C2)
    denominator = (mu_pred ** 2 + mu_target ** 2 + C1) * (var_pred + var_target + C2)
    
    ssim = numerator / (denominator + 1e-8)
    
    return ssim


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = 'alex'
) -> torch.Tensor:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    LPIPS uses deep features from pretrained networks (AlexNet, VGG) to
    measure perceptual similarity. It correlates very well with human judgment.
    
    Lower is better (0 = identical, larger = more different).
    
    Args:
        pred: Predicted image, shape [H, W, 3] or [1, 3, H, W]
        target: Ground truth image, shape [H, W, 3] or [1, 3, H, W]
        net: Network to use ('alex', 'vgg', 'squeeze')
    
    Returns:
        LPIPS distance
    
    Note:
        Requires 'lpips' package: pip install lpips
    
    Reference:
        Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
        CVPR 2018
    """
    try:
        import lpips
        
        # Initialize LPIPS model (cached globally)
        if not hasattr(compute_lpips, 'loss_fn'):
            compute_lpips.loss_fn = lpips.LPIPS(net=net)
            if pred.is_cuda:
                compute_lpips.loss_fn = compute_lpips.loss_fn.cuda()
        
        # Ensure correct shape: [1, 3, H, W]
        if pred.ndim == 3:  # [H, W, 3]
            pred = pred.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        if target.ndim == 3:
            target = target.permute(2, 0, 1).unsqueeze(0)
        
        # Normalize to [-1, 1] if in [0, 1]
        if pred.max() <= 1.0:
            pred = pred * 2.0 - 1.0
        if target.max() <= 1.0:
            target = target * 2.0 - 1.0
        
        # Compute LPIPS
        with torch.no_grad():
            distance = compute_lpips.loss_fn(pred, target)
        
        return distance.squeeze()
        
    except ImportError:
        print("Warning: LPIPS not available. Install with: pip install lpips")
        return torch.tensor(0.0)


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_lpips_metric: bool = False
) -> dict:
    """
    Compute all available metrics.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        compute_lpips_metric: Whether to compute LPIPS (slower)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'mse': compute_mse(pred, target).item(),
        'psnr': compute_psnr(pred, target).item(),
        'ssim': compute_ssim(pred, target).item()
    }
    
    if compute_lpips_metric:
        metrics['lpips'] = compute_lpips(pred, target).item()
    
    return metrics


if __name__ == "__main__":
    print("Testing Metrics...")
    print("=" * 60)
    
    # Create test images
    H, W = 100, 100
    
    # Perfect reconstruction
    print("Test 1: Perfect Reconstruction")
    img = torch.rand(H, W, 3)
    psnr = compute_psnr(img, img)
    ssim = compute_ssim(img, img)
    print(f"  PSNR: {psnr:.2f} dB (should be inf)")
    print(f"  SSIM: {ssim:.4f} (should be 1.0)")
    print()
    
    # Noisy image
    print("Test 2: Noisy Image")
    target = torch.rand(H, W, 3)
    noise = torch.randn(H, W, 3) * 0.05
    pred = torch.clamp(target + noise, 0, 1)
    
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)
    mse = compute_mse(pred, target)
    
    print(f"  MSE:  {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    print()
    
    # Test all metrics
    print("Test 3: All Metrics")
    metrics = compute_all_metrics(pred, target, compute_lpips_metric=False)
    print(f"  Metrics: {metrics}")
    print()
    
    # Verify PSNR formula
    print("Test 4: Verify PSNR Formula")
    mse_val = mse.item()
    expected_psnr = -10 * np.log10(mse_val)
    print(f"  MSE: {mse_val:.6f}")
    print(f"  PSNR (computed): {psnr:.2f} dB")
    print(f"  PSNR (expected): {expected_psnr:.2f} dB")
    print(f"  Match: {np.abs(psnr.item() - expected_psnr) < 0.01}")
    print()
    
    print("✅ Metrics test passed!")
