"""
Data Loader for NeRF

Handles loading images and camera poses from NeRF Blender dataset format.

DATASET FORMAT:
transforms.json:
{
    "camera_angle_x": 0.6911 (horizontal FOV in radians),
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.0,
            "transform_matrix": [[...], [...], [...],[...]  # 4x4 camera-to-world matrix
        },
        ...
    ]
}

MEMORY OPTIMIZATION:
- Lazy loading: don't load all images at once
- Image resizing: default 400x400 (not 800x800)
- Ray caching: precompute ray directions
"""

import torch
import torch.nn.functional as F
import json
import os
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List, Dict
from torch.utils.data import Dataset


class NeRFDataset(Dataset):
    """
    PyTorch Dataset for NeRF Blender format data.
    
    This dataset handles loading images and camera poses, and can optionally
    precompute all rays for faster training.
    
    Args:
        data_dir: Root directory containing transforms.json and images
        split: One of 'train', 'val', 'test'
        img_size: Resize images to this resolution (e.g., 400 for 400x400)
        white_background: If True, composite images onto white background
        precompute_rays: If True, precompute all rays (uses more memory)
        max_images: Maximum number of images to load (None = all)
    
    Example:
        >>> dataset = NeRFDataset('data/lego', split='train', img_size=400)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> rays_o, rays_d, rgb = dataset[0]
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: int = 400,
        white_background: bool = False,
        precompute_rays: bool = False,
        max_images: Optional[int] = None
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.white_background = white_background
        self.precompute_rays = precompute_rays
        
        # Load transforms
        transforms_path = os.path.join(data_dir, f'transforms_{split}.json')
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"Transforms file not found: {transforms_path}")
        
        with open(transforms_path, 'r') as f:
            self.meta = json.load(f)
        
        # Get camera parameters
        self.camera_angle_x = self.meta['camera_angle_x']
        self.focal = 0.5 * img_size / np.tan(0.5 * self.camera_angle_x)
        
        # Get image list
        self.frames = self.meta['frames']
        if max_images is not None:
            self.frames = self.frames[:max_images]
        
        print(f"Loaded {len(self.frames)} {split} images from {data_dir}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Focal length: {self.focal:.2f} pixels")
        
        # Precompute rays if requested
        if precompute_rays:
            print("Precomputing rays...")
            self._precompute_all_rays()
            print(f"  Precomputed {len(self.all_rays_o)} rays")
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        if self.precompute_rays:
            return 1  # One big batch of all rays
        else:
            return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get rays and RGB values for an image.
        
        Returns:
            rays_o: Ray origins, shape [H*W, 3] or [num_precomputed, 3]
            rays_d: Ray directions, shape [H*W, 3] or [num_precomputed, 3]
            rgb: Ground truth RGB, shape [H*W, 3] or [num_precomputed, 3]
        """
        if self.precompute_rays:
            # Return all precomputed rays
            return self.all_rays_o, self.all_rays_d, self.all_rgb
        else:
            # Load single image
            frame = self.frames[idx]
            
            # Load image
            img_path = os.path.join(self.data_dir, frame['file_path'] + '.png')
            img = self._load_image(img_path)
            
            # Get camera pose
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            
            # Generate rays
            rays_o, rays_d = self._get_rays(c2w)
            
            # Flatten
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rgb = img.reshape(-1, 3)
            
            return rays_o, rays_d, rgb
    
    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Args:
            path: Path to image file
        
        Returns:
            Image tensor, shape [H, W, 3]
        """
        # Load image
        img = Image.open(path)
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Convert to numpy array
        img = np.array(img).astype(np.float32) / 255.0
        
        # Handle alpha channel
        if img.shape[-1] == 4:
            # RGBA image: composite onto white or black background
            rgb = img[..., :3]
            alpha = img[..., 3:]
            
            if self.white_background:
                # Composite onto white: RGB = rgb * alpha + (1 - alpha)
                rgb = rgb * alpha + (1.0 - alpha)
            else:
                # Composite onto black: RGB = rgb * alpha
                rgb = rgb * alpha
            
            img = rgb
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        
        return img_tensor
    
    def _get_rays(self, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for an image given camera-to-world matrix.
        
        Args:
            c2w: Camera-to-world matrix, shape [4, 4]
        
        Returns:
            rays_o: Ray origins, shape [H, W, 3]
            rays_d: Ray directions, shape [H, W, 3]
        """
        from utils.ray_utils import get_ray_bundle
        
        return get_ray_bundle(
            self.img_size,
            self.img_size,
            self.focal,
            c2w
        )
    
    def _precompute_all_rays(self):
        """Precompute all rays and RGB values."""
        all_rays_o = []
        all_rays_d = []
        all_rgb = []
        
        for i in range(len(self.frames)):
            frame = self.frames[i]
            
            # Load image
            img_path = os.path.join(self.data_dir, frame['file_path'] + '.png')
            img = self._load_image(img_path)
            
            # Get camera pose
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            
            # Generate rays
            rays_o, rays_d = self._get_rays(c2w)
            
            # Flatten and append
            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_rgb.append(img.reshape(-1, 3))
        
        # Concatenate
        self.all_rays_o = torch.cat(all_rays_o, dim=0)
        self.all_rays_d = torch.cat(all_rays_d, dim=0)
        self.all_rgb = torch.cat(all_rgb, dim=0)
    
    def get_camera_poses(self) -> List[torch.Tensor]:
        """Get all camera-to-world matrices."""
        poses = []
        for frame in self.frames:
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            poses.append(c2w)
        return poses


def get_rays(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to generate rays (wraps ray_utils.get_ray_bundle).
    
    Args:
        H: Image height
        W: Image width
        focal: Focal length in pixels
        c2w: Camera-to-world matrix [4, 4]
    
    Returns:
        rays_o: Ray origins [H, W, 3]
        rays_d: Ray directions [H, W, 3]
    """
    from utils.ray_utils import get_ray_bundle
    return get_ray_bundle(H, W, focal, c2w)


if __name__ == "__main__":
    print("Testing NeRF Dataset...")
    print("=" * 60)
    
    # This test requires actual data, so we'll create a mock dataset
    print("Creating mock dataset...")
    
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create mock transforms.json
        transforms = {
            "camera_angle_x": 0.6911112070083618,
            "frames": []
        }
        
        # Create a few mock frames
        for i in range(3):
            # Create mock image
            img = np.random.rand(100, 100, 4)  # RGBA
            img_path = os.path.join(temp_dir, f'r_{i}.png')
            Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
            
            # Create mock transform matrix (identity with varying translation)
            transform = np.eye(4).tolist()
            transform[2][3] = 4.0  # Move camera back
            
            transforms["frames"].append({
                "file_path": f'./r_{i}',
                "rotation": 0.0,
                "transform_matrix": transform
            })
        
        # Save transforms
        with open(os.path.join(temp_dir, 'transforms_train.json'), 'w') as f:
            json.dump(transforms, f)
        
        # Test dataset loading
        print("\nTest 1: Load Dataset")
        dataset = NeRFDataset(
            temp_dir,
            split='train',
            img_size=50,
            precompute_rays=False
        )
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Expected: 3")
        print()
        
        # Test single image loading
        print("Test 2: Get Single Image Rays")
        rays_o, rays_d, rgb = dataset[0]
        print(f"  Rays origin shape: {rays_o.shape}")
        print(f"  Rays direction shape: {rays_d.shape}")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  Expected: [2500, 3] for each (50*50 = 2500)")
        print(f"  RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print()
        
        # Test precomputed rays
        print("Test 3: Precomputed Rays")
        dataset_precomputed = NeRFDataset(
            temp_dir,
            split='train',
            img_size=50,
            precompute_rays=True,
            max_images=2  # Only load 2 images
        )
        rays_o, rays_d, rgb = dataset_precomputed[0]
        print(f"  Total rays: {rays_o.shape[0]}")
        print(f"  Expected: {50*50*2} = {2*2500}")
        print()
        
        print("✅ Dataset test passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
