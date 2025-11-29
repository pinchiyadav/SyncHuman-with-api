# SyncHuman Quick Start Guide

Get your first 3D human model in 5 minutes!

## Prerequisites

- Linux machine with NVIDIA GPU (40GB+ VRAM)
- Conda installed (Miniconda or Anaconda)
- Internet connection for downloading models

## Automatic Setup (Recommended)

### 1. Run the Automated Setup Script

```bash
# Make sure you're in the SyncHuman directory
cd /path/to/SyncHuman

# Run the setup script (takes ~45 minutes including model downloads)
bash setup.sh
```

The script will:
- ✓ Create a conda environment with Python 3.10
- ✓ Install PyTorch with CUDA 12.1 support
- ✓ Install all dependencies
- ✓ Download model checkpoints (~8.5GB)
- ✓ Verify the installation

### 2. Activate the Environment

```bash
# RECOMMENDED: Source conda directly
source /opt/conda/bin/activate SyncHuman

# Alternative: Using conda command (if conda init worked)
conda activate SyncHuman

# Alternative: Using the environment script
source env.sh
```

**Note:** If `conda activate` doesn't work, use `source /opt/conda/bin/activate SyncHuman` instead.

### 3. Run Your First Inference

```bash
# Set attention backend
export ATTN_BACKEND=xformers

# Run Stage 1 inference (generates multi-view predictions)
# Edit inference_OneStage.py to change input image path
python inference_OneStage.py

# Output files will be in: outputs/OneStage/
# - color_0.png through color_4.png (multi-view colors)
# - normal_0.png through normal_4.png (multi-view normals)
```

Processing time: ~1.5-2 minutes on NVIDIA A40

## Using the Web API

### Start the API Server

```bash
export ATTN_BACKEND=xformers
python api_server.py
```

Server will start at: `http://localhost:8000`

### Test with cURL

```bash
# Generate 3D model from image file
curl -X POST http://localhost:8000/generate \
  -F "image=@path/to/image.png" \
  -F "stage=stage1" \
  --output result.glb

# Or use an image URL
curl -X POST http://localhost:8000/generate \
  -F "image_url=https://example.com/image.png" \
  -F "stage=both" \
  --output result.glb
```

### Interactive Web UI

Visit http://localhost:8000 in your browser for an interactive interface!

## Input Image Requirements

For best results:
- **Format**: PNG or JPG with transparent background (RGBA PNG)
- **Size**: 512x512 to 1024x1024 pixels
- **Subject**: Human figure in clear view
- **Quality**: Well-lit, no occlusion of body parts
- **Background**: Should be removed (transparency or simple background)

### Remove Background Quickly

Use online tools or Python:
```python
from rembg import remove
from PIL import Image

image = Image.open("input.jpg")
result = remove(image)
result.save("output_rgba.png")
```

## Example Usage

### Python Script

```python
import torch
from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline

# Load pipeline
pipeline = SyncHumanOneStagePipeline.from_pretrained('./ckpts/OneStage')
pipeline.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()

# Run inference
pipeline.run(
    image_path='./examples/input_rgba.png',
    save_path='./outputs/OneStage'
)

# Outputs are in ./outputs/OneStage/
```

### Command Line

```bash
# Edit the image path in the script
nano inference_OneStage.py
# Change: image_path='./examples/input_rgba.png'

# Run
python inference_OneStage.py
```

## Understanding the Output

### Stage 1 Output

- **color_N.png** (N=0-4): Multi-view color predictions from 5 different angles
- **normal_N.png** (N=0-4): Multi-view normal map predictions
- **coordinates.npz**: Sparse 3D coordinate structure
- **input.png**: Preprocessed input image

### Stage 2 Output (If Available)

- **output.glb**: Final 3D model in glTF/GLB format (ready for 3D viewers)
- **output_mesh.ply**: Mesh geometry in PLY format
- **output_gs.ply**: Gaussian splatting point cloud

## Quality Parameters

### For Speed (Quick Preview)
```bash
# In inference script or API
stage1_steps=30  # Fewer diffusion steps
```
**Time**: ~1 minute | **Quality**: Low

### Balanced (Default)
```bash
stage1_steps=50
```
**Time**: ~1.5 minutes | **Quality**: Good

### For Quality (Best Results)
```bash
stage1_steps=75
```
**Time**: ~2.5 minutes | **Quality**: High

## Troubleshooting

### "No module named 'xformers'"
```bash
pip install xformers
```

### "Out of CUDA memory"
- Reduce `stage1_steps` to 30
- Close other GPU applications
- Try float16 precision

### "No module named 'spconv'"
```bash
pip install spconv-cu121  # For CUDA 12.1
```

### "Import error with stage2"
Stage 2 has additional dependencies. Stage 1 works independently!

## What's Next?

- View results in 3D viewers:
  - [Three.js](https://threejs.org/editor/)
  - [Babylon.js](https://playground.babylonjs.com/)
  - [Sketchfab](https://sketchfab.com/)

- Integrate into your application via the API

- Fine-tune for specific domains with custom datasets

## Performance Benchmarks

| GPU | Stage 1 Time | Memory Used |
|-----|-------------|------------|
| NVIDIA A40 (46GB) | ~1.5 min | ~40GB |
| NVIDIA H100 (80GB) | ~1.2 min | ~42GB |
| NVIDIA A100 (40GB) | ~2 min | ~39GB |

## Documentation

- Full setup guide: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- API documentation: [API_USAGE.md](API_USAGE.md)
- Original README: [README.md](README.md)

## Support

- GitHub Issues: https://github.com/xishuxishu/SyncHuman/issues
- Paper: https://arxiv.org/pdf/2510.07723
- Project Page: https://xishuxishu.github.io/SyncHuman.github.io/

## Citation

```bibtex
@article{chen2025synchuman,
  title={SyncHuman: Synchronizing 2D and 3D Diffusion Models for Single-view Human Reconstruction},
  author={Chen, Wenyue and Li, Peng and Zheng, Wangguandong and Zhao, Chengfeng and Li, Mengfei and Zhu, Yaolong and Dou, Zhiyang and Wang, Ronggang and Liu, Yuan},
  journal={arXiv preprint arXiv:2510.07723},
  year={2025}
}
```

---

**Tested on**: Ubuntu 20.04+, Python 3.10, PyTorch 2.1.1, CUDA 12.1
**Last Updated**: November 2025
