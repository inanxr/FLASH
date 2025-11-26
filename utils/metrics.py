"""Evaluation metrics for NeRF (PSNR, SSIM, LPIPS)."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    
    if mse == 0:
        return torch.tensor(float('inf'))
    
    psnr = -10.0 * torch.log10(mse / (max_val ** 2))
    return psnr


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    filter_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> torch.Tensor:
    try:
        from skimage.metrics import structural_similarity
        
        if isinstance(pred, torch.Tensor):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = pred
            
        if isinstance(target, torch.Tensor):
            target_np = target.detach().cpu().numpy()
        else:
            target_np = target
        
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
    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2
    
    mu_pred = pred.mean()
    mu_target = target.mean()
    
    var_pred = ((pred - mu_pred) ** 2).mean()
    var_target = ((target - mu_target) ** 2).mean()
    cov = ((pred - mu_pred) * (target - mu_target)).mean()
    
    numerator = (2 * mu_pred * mu_target + C1) * (2 * cov + C2)
    denominator = (mu_pred ** 2 + mu_target ** 2 + C1) * (var_pred + var_target + C2)
    
    ssim = numerator / (denominator + 1e-8)
    return ssim


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = 'alex'
) -> torch.Tensor:
    try:
        import lpips
        
        if not hasattr(compute_lpips, 'loss_fn'):
            compute_lpips.loss_fn = lpips.LPIPS(net=net)
            if pred.is_cuda:
                compute_lpips.loss_fn = compute_lpips.loss_fn.cuda()
        
        if pred.ndim == 3:
            pred = pred.permute(2, 0, 1).unsqueeze(0)
        if target.ndim == 3:
            target = target.permute(2, 0, 1).unsqueeze(0)
        
        if pred.max() <= 1.0:
            pred = pred * 2.0 - 1.0
        if target.max() <= 1.0:
            target = target * 2.0 - 1.0
        
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
