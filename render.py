"""
Novel View Rendering for NeRF

Render novel views from trained NeRF model and create videos.

Usage:
    # Render spiral video:
    python render.py --checkpoint checkpoints/model.pth --create_video
    
    # Render 360° video:
    python render.py --checkpoint checkpoints/model.pth --render_path 360
    
    # Render test views:
    python render.py --checkpoint checkpoints/model.pth --render_test
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image
import imageio

from models.nerf_model import InstantNGPNeRF
from models.renderer import VolumetricRenderer
from utils.data_loader import NeRFDataset


def generate_360_path(num_views=60, radius=2.5, height=0.0):
    """
    Generate 360° circular camera path.
    
    Args:
        num_views: Number of camera positions
        radius: Circle radius
        height: Camera height (z coordinate)
    
    Returns:
        poses: [num_views, 4, 4] camera-to-world matrices
    """
    poses = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Camera position on circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        
        # Camera looking at origin
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 0, 1])
        
        # Right vector
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        # Recalculate up
        up = np.cross(forward, right)
        
        # Build pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = [x, y, z]
        
        poses.append(pose)
    
    return np.stack(poses)


def generate_spiral_path(num_views=120, radius=2.0, height_range=0.5, n_spirals=2):
    """
    Generate spiral camera path (vertical movement + rotation).
    
    Args:
        num_views: Number of camera positions
        radius: Spiral radius
        height_range: Vertical range
        n_spirals: Number of complete spirals
    
    Returns:
        poses: [num_views, 4, 4] camera-to-world matrices
    """
    poses = []
    
    for i in range(num_views):
        t = i / num_views
        
        # Angle increases for rotation
        angle = 2 * np.pi * n_spirals * t
        
        # Height oscillates
        z = height_range * np.sin(2 * np.pi * n_spirals * t)
        
        # Position on spiral
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Camera looking at origin
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 0, 1])
        
        # Right vector
        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Recalculate up
        up = np.cross(forward, right)
        
        # Build pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = [x, y, z]
        
        poses.append(pose)
    
    return np.stack(poses)


def render_novel_views(
    checkpoint_path,
    poses,
    img_size=400,
    device='cuda',
    output_dir='renders'
):
    """
    Render novel views from camera poses.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        poses: [N, 4, 4] camera-to-world matrices
        img_size: Image resolution
        device: 'cuda' or 'cpu'
        output_dir: Output directory
    
    Returns:
        images: List of rendered RGB images
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    model = InstantNGPNeRF().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize renderer
    renderer = VolumetricRenderer(
        near=2.0,
        far=6.0,
        num_coarse_samples=32,
        num_fine_samples=64
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    images = []
    
    print(f"Rendering {len(poses)} views...")
    
    with torch.no_grad():
        for idx, pose in enumerate(tqdm(poses)):
            # Get rays for this camera
            pose_tensor = torch.from_numpy(pose).float().to(device)
            
            # Generate rays
            from utils.ray_utils import get_ray_bundle
            rays_o, rays_d = get_ray_bundle(
                img_size, img_size,
                focal_length=img_size * 0.7,  # Approximate focal length
                camera_to_world=pose_tensor
            )
            
            # Flatten rays
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)
            
            # Render in chunks
            chunk_size = 4096
            rgb_chunks = []
            
            for i in range(0, rays_o_flat.shape[0], chunk_size):
                rays_o_chunk = rays_o_flat[i:i+chunk_size]
                rays_d_chunk = rays_d_flat[i:i+chunk_size]
                
                rgb, _, _ = renderer.render_rays(
                    rays_o_chunk, rays_d_chunk,
                    model, model,
                    randomize=False,
                    return_extras=False
                )
                
                rgb_chunks.append(rgb.cpu())
            
            # Concatenate and reshape
            rgb_image = torch.cat(rgb_chunks, dim=0).reshape(img_size, img_size, 3)
            rgb_np = (rgb_image.numpy() * 255).astype(np.uint8)
            
            # Save image
            img_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
            Image.fromarray(rgb_np).save(img_path)
            
            images.append(rgb_np)
    
    print(f"Rendered images saved to: {output_dir}")
    return images


def create_video(image_dir, output_path='renders/video.mp4', fps=30):
    """Create video from rendered images."""
    import glob
    
    image_files = sorted(glob.glob(os.path.join(image_dir, 'frame_*.png')))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"Creating video from {len(image_files)} images...")
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    for img_path in tqdm(image_files):
        img = imageio.imread(img_path)
        writer.append_data(img)
    
    writer.close()
    
    print(f"Video saved to: {output_path}")


def render_test_views(checkpoint_path, data_dir, device='cuda', output_dir='test_renders'):
    """Render test set views for evaluation."""
    # Load test dataset
    try:
        test_dataset = NeRFDataset(data_dir, split='test', img_size=400)
    except:
        print("No test set found!")
        return
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = InstantNGPNeRF().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    renderer = VolumetricRenderer(
        near=2.0, far=6.0,
        num_coarse_samples=32,
        num_fine_samples=64
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rendering {len(test_dataset)} test views...")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            rays_o, rays_d, _ = test_dataset[i]
            
            # Render in chunks
            chunk_size = 4096
            rgb_chunks = []
            
            for j in range(0, rays_o.shape[0], chunk_size):
                rays_o_chunk = rays_o[j:j+chunk_size].to(device)
                rays_d_chunk = rays_d[j:j+chunk_size].to(device)
                
                rgb, _, _ = renderer.render_rays(
                    rays_o_chunk, rays_d_chunk,
                    model, model,
                    randomize=False,
                    return_extras=False
                )
                
                rgb_chunks.append(rgb.cpu())
            
            rgb_image = torch.cat(rgb_chunks, dim=0).reshape(400, 400, 3)
            rgb_np = (rgb_image.numpy() * 255).astype(np.uint8)
            
            Image.fromarray(rgb_np).save(
                os.path.join(output_dir, f'test_{i:03d}.png')
            )
    
    print(f"Test renders saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Render NeRF novel views')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--render_path', type=str, default='360', choices=['360', 'spiral'], help='Camera path type')
    parser.add_argument('--num_views', type=int, default=60, help='Number of views to render')
    parser.add_argument('--img_size', type=int, default=400, help='Image resolution')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='renders', help='Output directory')
    parser.add_argument('--create_video', action='store_true', help='Create video from renders')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--render_test', action='store_true', help='Render test views instead')
    parser.add_argument('--data_dir', type=str, default='data/nerf_synthetic/lego', help='Dataset directory (for test rendering)')
    
    args = parser.parse_args()
    
    # Auto-detect CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    if args.render_test:
        # Render test set
        render_test_views(args.checkpoint, args.data_dir, args.device, args.output_dir)
    else:
        # Generate camera path
        if args.render_path == '360':
            print(f"Generating 360° camera path with {args.num_views} views...")
            poses = generate_360_path(args.num_views, radius=2.5, height=0.0)
        else:  # spiral
            print(f"Generating spiral camera path with {args.num_views} views...")
            poses = generate_spiral_path(args.num_views, radius=1.8, height_range=0.4)
        
        # Render novel views
        images = render_novel_views(
            args.checkpoint,
            poses,
            img_size=args.img_size,
            device=args.device,
            output_dir=args.output_dir
        )
        
        # Create video
        if args.create_video:
            video_path = os.path.join(args.output_dir, 'video.mp4')
            create_video(args.output_dir, video_path, fps=args.fps)


if __name__ == "__main__":
    main()
