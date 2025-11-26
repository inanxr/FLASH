# Data Directory

Dataset management and custom dataset guidelines for FLASH NeRF.

## Quick Start

**Download official NeRF dataset:**
```bash
python data/download_dataset.py --scene lego
```

**Available scenes**: lego, ship, drums, ficus, hotdog, materials, mic, chair

**Train on downloaded data:**
```bash
python train.py --data_dir data/nerf_synthetic/lego
```

## Dataset Format (NeRF Blender)

Expected directory structure:
```
your_scene/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/
│   ├── r_0.png
│   ├── r_1.png
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Creating Custom Datasets

### 1. Data Collection

**Capture requirements:**
- 50-100 images from different viewpoints
- Consistent lighting (avoid shadows/reflections)
- Object centered in frame
- White/uniformly colored background recommended
- 360° coverage for best results

**Camera tips:**
- Use fixed focal length (no zoom)
- Keep camera settings constant (aperture, ISO, white balance)
- Use tripod or stable mount
- Capture at least 2-3 images per 10° rotation

### 2. Format Images

**Image specifications:**
- Format: PNG with alpha channel (RGBA) or JPG
- Resolution: 400×400 to 800×800 (square recommended)
- Naming: `r_0.png`, `r_1.png`, etc.

**Background removal** (if needed):
```python
from rembg import remove
from PIL import Image

img = Image.open('input.jpg')
output = remove(img)  # Returns RGBA with transparent background
output.save('r_0.png')
```

### 3. Create transforms.json

**Minimal structure:**
```json
{
  "camera_angle_x": 0.6911112070083618,
  "frames": [
    {
      "file_path": "./train/r_0",
      "rotation": 0.0,
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 4.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

**Camera angle calculation:**
```python
import math

# For 50mm focal length on 35mm sensor (assumes sensor_width = 36mm)
focal_mm = 50
sensor_width = 36
camera_angle_x = 2 * math.atan(sensor_width / (2 * focal_mm))
# Result: ~0.6911 radians (~39.6°)
```

### 4. Generate Camera Poses

#### Option A: COLMAP (Automated - Recommended)

Use COLMAP for automatic camera pose estimation:

```bash
# 1. Install COLMAP
# https://github.com/colmap/colmap

# 2. Run COLMAP structure-from-motion
colmap automatic_reconstructor \
    --image_path images/ \
    --workspace_path colmap_output/

# 3. Convert COLMAP output to NeRF format
# Use tools like: https://github.com/NVlabs/instant-ngp#preparing-new-nerf-datasets
```

#### Option B: Manual Poses (Simple Turntable)

For turntable captures with fixed camera distance:

```python
import json
import math

def create_turntable_transforms(num_images, radius=4.0, height=0.0):
    frames = []
    camera_angle_x = 0.6911112070083618  # ~50mm lens
    
    for i in range(num_images):
        angle = (i / num_images) * 2 * math.pi
        
        # Camera position on circle
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = height
        
        # Look at center (0,0,0)
        transform_matrix = [
            [-math.sin(angle), 0.0, math.cos(angle), x],
            [0.0, 1.0, 0.0, y],
            [-math.cos(angle), 0.0, -math.sin(angle), z],
            [0.0, 0.0, 0.0, 1.0]
        ]
        
        frames.append({
            "file_path": f"./train/r_{i}",
            "rotation": angle,
            "transform_matrix": transform_matrix
        })
    
    return {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }

# Generate for 80 images
data = create_turntable_transforms(80)
with open('transforms_train.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### 5. Training Configuration

**For custom datasets, adjust:**

```python
# In config.py or command line
--img_size 400              # Match your image resolution
--white_background True     # If using white background
--num_iterations 30000      # Increase for complex scenes
--batch_size 8192           # Decrease if out of memory
```

## Troubleshooting Custom Datasets

| Issue | Solution |
|-------|----------|
| Blurry results | Increase `num_iterations` or `finest_resolution` |
| Out of memory | Reduce `batch_size` or `img_size` |
| Floating artifacts | Add more viewpoints, ensure 360° coverage |
| Wrong scale | Adjust camera radius in transform matrices |
| Dark/bright images | Check `white_background` setting |
| Training diverges | Reduce learning rate, check camera poses |

## Advanced: Multi-Scale Training

For large/detailed objects:
```python
# Stage 1: Low resolution (fast)
python train.py --img_size 200 --num_iterations 10000

# Stage 2: Full resolution (fine details)
python train.py --img_size 800 --num_iterations 20000 --checkpoint stage1.pth
```

## Dataset Best Practices

1. **Start simple**: Test with official datasets first
2. **Validate poses**: Visualize camera positions before training
3. **Check coverage**: Ensure no large gaps in viewing angles
4. **Consistent lighting**: Avoid changing light conditions
5. **Clean backgrounds**: Simpler backgrounds = better results
6. **Adequate samples**: 50+ images minimum, 100+ recommended
