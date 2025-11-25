# FLASH - Fast Learning for Accurate Scene Hashing

**A high-performance NeRF implementation with native Windows desktop app**

Turn your photos into 3D models with one click!

## Features

✨ **Native Windows Desktop App** - Beautiful PyQt6 interface with glassmorphism design  
📸 **Custom Dataset Support** - Load your own photos, automatic COLMAP processing  
⚡ **Instant-NGP** - Fast hash-encoded NeRF training (2-5 min on GPU)  
🎨 **Live Monitoring** - Real-time loss graphs and preview rendering  
🔧 **Easy to Use** - No command line needed, just click and train!

## Quick Start

```bash
# Clone
git clone https://github.com/inanxr/FLASH.git
cd FLASH

# Install dependencies
pip install -r requirements.txt

# Launch FLASH Studio
python studio.py
```

## Turn Photos into 3D

1. **Take 20-100 photos** walking around your subject
2. **Click "Load Custom Dataset"** in FLASH Studio
3. **Select your photos folder**
4. **Wait for COLMAP processing** (automatic!)
5. **Click "Start Training"** and watch the magic happen! 🎉

## Interface

**FLASH Studio** provides:
- Dataset loading (built-in + custom)
- Training parameter controls (sliders for iterations, batch size, etc.)
- Live progress monitoring (loss/PSNR graphs)
- Real-time preview rendering
- Background processing (UI stays responsive)

## Architecture

**Instant-NGP Implementation:**
- Multi-resolution hash encoding (20 levels)
- Compact MLP (2 layers, 64 hidden units)
- Occupancy grid acceleration
- Mixed precision training (FP16)
- PyTorch 2.0+ compile support

**Desktop App:**
- PyQt6 with minimal glassmorphism design
- Background workers (QThread) for training
- COLMAP integration for photo processing
- Font Awesome icons (qtawesome)

## Requirements

- Python 3.8+
- CUDA GPU recommended (CPU supported but slow)
- 4GB RAM minimum for training
- Windows 10/11 (Linux/Mac supported for CLI)

## Project Structure

```
FLASH/
├── studio.py              # Desktop app entry point
├── train.py               # CLI training script
├── config.py              # Configuration
├── models/                # NeRF models
│   ├── hash_encoding.py
│   ├── nerf_model.py
│   ├── renderer.py
│   └── occupancy_grid.py
├── ui/                    # Desktop app UI
│   ├── main_window.py
│   └── training_tab.py
├── workers/               # Background processing
│   ├── training_worker.py
│   └── colmap_worker.py
└── utils/                 # Utilities
    ├── colmap_processor.py
    ├── data_loader.py
    └── ray_utils.py
```

## Performance

**Training Speed:**
- GPU (RTX 3060+): 2-5 minutes for 5000 iterations
- CPU: 20-30 minutes for 5000 iterations

**Quality:**
- PSNR: 28-32 dB on synthetic scenes
- High-quality renders at 400x400 resolution

## CLI Usage

If you prefer command line:

```bash
# Train on built-in dataset
python train.py --data_dir data/nerf_example_data/nerf_synthetic/lego

# Quick test
python train.py --quick_test

# Custom dataset
python train.py --data_dir data/custom/my_photos
```

## Tips for Best Photos

✅ **DO:**
- Take 50-100 photos for best quality
- Walk in complete circle around subject
- Keep consistent lighting
- 70% overlap between views
- Multiple heights (low, medium, high)

❌ **DON'T:**
- Rush the photography
- Change lighting mid-shoot
- Use blurry photos
- Skip areas of the object

## Credits

**Based on:**
- [Instant-NGP](https://github.com/NVlabs/instant-ngp) by NVIDIA
- [NeRF](https://www.matthewtancik.com/nerf) by Mildenhall et al.

**Built with:**
- PyTorch
- PyQt6
- COLMAP (pycolmap)
- qtawesome

## License

MIT License - See LICENSE file

---

**Happy 3D scanning!** 📸 → 🎬 → 🎯
