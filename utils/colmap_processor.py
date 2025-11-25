"""
COLMAP Processor for FLASH Studio

Automatically processes user photos into NeRF-ready datasets.
Handles feature extraction, matching, SfM, and transforms.json generation.
"""

import os
import json
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ColmapProcessor:
    """Process images with COLMAP and convert to NeRF format."""
    
    def __init__(self):
        self.colmap_exe = self._find_colmap()
    
    def _find_colmap(self) -> Optional[str]:
        """Find COLMAP executable."""
        # Try pycolmap first
        try:
            import pycolmap
            return "pycolmap"
        except ImportError:
            pass
        
        # Try system COLMAP
        if shutil.which("colmap"):
            return "colmap"
        
        return None
    
    def process_images(
        self, 
        image_folder: str, 
        output_dir: str,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Process images with COLMAP and create NeRF dataset.
        
        Args:
            image_folder: Path to folder containing images
            output_dir: Where to save processed dataset
            progress_callback: Function to report progress
            
        Returns:
            (success, message)
        """
        try:
            if not self.colmap_exe:
                return False, "COLMAP not found. Install with: pip install pycolmap"
            
            # Create output structure
            os.makedirs(output_dir, exist_ok=True)
            images_dir = os.path.join(output_dir, "images")
            sparse_dir = os.path.join(output_dir, "sparse")
            database_path = os.path.join(output_dir, "database.db")
            
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(sparse_dir, exist_ok=True)
            
            # Copy images
            if progress_callback:
                progress_callback("Copying images...")
            
            image_files = [
                f for f in os.listdir(image_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if len(image_files) < 10:
                return False, f"Need at least 10 images, found {len(image_files)}"
            
            for img in image_files:
                shutil.copy2(
                    os.path.join(image_folder, img),
                    os.path.join(images_dir, img)
                )
            
            # Use pycolmap if available
            if self.colmap_exe == "pycolmap":
                return self._process_with_pycolmap(
                    images_dir, sparse_dir, database_path, output_dir, progress_callback
                )
            else:
                return self._process_with_colmap_cli(
                    images_dir, sparse_dir, database_path, output_dir, progress_callback
                )
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _process_with_pycolmap(
        self, images_dir, sparse_dir, database_path, output_dir, progress_callback
    ):
        """Process using pycolmap library."""
        import pycolmap
        
        if progress_callback:
            progress_callback("Extracting features...")
        
        # Feature extraction
        pycolmap.extract_features(database_path, images_dir)
        
        if progress_callback:
            progress_callback("Matching features...")
        
        # Feature matching
        pycolmap.match_exhaustive(database_path)
        
        if progress_callback:
            progress_callback("Running Structure-from-Motion...")
        
        # Mapper (SfM)
        maps = pycolmap.incremental_mapping(
            database_path, images_dir, sparse_dir
        )
        
        if not maps:
            return False, "COLMAP mapping failed - try taking more photos with better overlap"
        
        # Export to NeRF format
        if progress_callback:
            progress_callback("Converting to NeRF format...")
        
        # Get the reconstruction
        reconstruction = maps[0]  # Use first reconstruction
        
        # Convert to transforms.json
        success = self._export_transforms(
            reconstruction, output_dir, images_dir
        )
        
        if success:
            return True, f"Successfully processed {len(reconstruction.images)} images"
        else:
            return False, "Failed to export transforms.json"
    
    def _process_with_colmap_cli(
        self, images_dir, sparse_dir, database_path, output_dir, progress_callback
    ):
        """Process using COLMAP command-line interface."""
        # Feature extraction
        if progress_callback:
            progress_callback("Extracting features...")
        
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", images_dir
        ], check=True)
        
        # Feature matching
        if progress_callback:
            progress_callback("Matching features...")
        
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", database_path
        ], check=True)
        
        # Mapper
        if progress_callback:
            progress_callback("Running Structure-from-Motion...")
        
        subprocess.run([
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", images_dir,
            "--output_path", sparse_dir
        ], check=True)
        
        # Convert to text
        model_path = os.path.join(sparse_dir, "0")
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", model_path,
            "--output_path", model_path,
            "--output_type", "TXT"
        ], check=True)
        
        # Export transforms
        if progress_callback:
            progress_callback("Converting to NeRF format...")
        
        success = self._export_transforms_from_text(model_path, output_dir, images_dir)
        
        if success:
            return True, "Successfully processed dataset"
        else:
            return False, "Failed to export transforms.json"
    
    def _export_transforms(self, reconstruction, output_dir, images_dir):
        """Export pycolmap reconstruction to transforms.json."""
        try:
            import pycolmap
            
            frames = []
            
            # Get all images
            for image_id, image in reconstruction.images.items():
                # Get camera
                camera = reconstruction.cameras[image.camera_id]
                
                # Get transformation matrix
                rotation = image.rotation_matrix()
                translation = image.tvec
                
                # Convert to NeRF format (camera-to-world)
                c2w = np.eye(4)
                c2w[:3, :3] = rotation.T
                c2w[:3, 3] = -rotation.T @ translation
                
                # NeRF coordinate system adjustment
                c2w = c2w @ np.diag([1, -1, -1, 1])
                
                frame = {
                    "file_path": f"images/{image.name}",
                    "transform_matrix": c2w.tolist()
                }
                frames.append(frame)
            
            # Create transforms.json
            transforms = {
                "camera_angle_x": 2 * np.arctan(camera.width / (2 * camera.focal_length_x)),
                "frames": frames
            }
            
            # Save
            with open(os.path.join(output_dir, "transforms_train.json"), 'w') as f:
                json.dump(transforms, f, indent=2)
            
            # Create val set (use last 10%)
            val_count = max(1, len(frames) // 10)
            transforms_val = {
                "camera_angle_x": transforms["camera_angle_x"],
                "frames": frames[-val_count:]
            }
            
            with open(os.path.join(output_dir, "transforms_val.json"), 'w') as f:
                json.dump(transforms_val, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def _export_transforms_from_text(self, model_path, output_dir, images_dir):
        """Export from COLMAP text format."""
        # This is a simplified version - full implementation would parse
        # cameras.txt and images.txt files
        # For now, return False to indicate pycolmap is preferred
        return False


# Standalone script for command-line usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python colmap_processor.py <image_folder> <output_dir>")
        sys.exit(1)
    
    processor = ColmapProcessor()
    
    def progress(msg):
        print(f"[COLMAP] {msg}")
    
    success, message = processor.process_images(
        sys.argv[1], sys.argv[2], progress
    )
    
    if success:
        print(f"✓ {message}")
    else:
        print(f"✗ {message}")
        sys.exit(1)
