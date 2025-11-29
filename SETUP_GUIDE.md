# SyncHuman Setup and Installation Guide

> **Official Implementation of:**
> "SyncHuman: Synchronizing 2D and 3D Diffusion Models for Single-view Human Reconstruction"
>
> Authors: Wenyue Chen et al. (Peking University, HKUST, Southeast University, University of Hong Kong)
> arXiv: 2510.07723

## Quick Start Summary

This guide provides step-by-step instructions to set up SyncHuman on a fresh Linux machine. All instructions follow the official recommendations from the SyncHuman and TRELLIS developers.

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with ≥40GB VRAM (tested on H800 with 46GB, A40 with 46GB)
- **CUDA**: 12.1 or compatible version
- **Python**: 3.10
- **Conda**: Miniconda or Anaconda

### Estimated Setup Time
- First-time installation: ~30-45 minutes
- Model downloads: ~20-30 minutes (depending on internet speed)
- Inference (Stage 1): ~2 minutes on A40 GPU

## Step-by-Step Installation

### 1. Create Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n SyncHuman python=3.10 -y
conda activate SyncHuman
```

### 2. Install PyTorch with CUDA 12.1

```bash
# Install PyTorch 2.1.1+ with CUDA 12.1 support
conda install pytorch::pytorch torchvision pytorch::pytorch-cuda=12.1 -c pytorch -c nvidia -y

# If the above fails, try installing with pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Core Dependencies

```bash
# Install core Python packages
pip install accelerate safetensors==0.4.5 diffusers==0.29.1 transformers==4.36.0

# Install attention optimization libraries
pip install xformers  # Alternative to flash-attn (flash-attn requires compilation)

# Install 3D and image processing
pip install trimesh open3d omegaconf imageio imageio-ffmpeg rembg
pip install plyfile scikit-image scipy scikit-learn pyyaml

# Install sparse tensor support
pip install spconv-cu121  # For CUDA 12.1

# Install additional utilities
pip install onnxruntime onnx einops pyvista PyMeshFix igraph pillow opencv-python pydantic
pip install utils3d xatlas ninja easydict peft moviepy
```

### 4. Install NVIDIA-specific Libraries (For Stage 2)

```bash
# nvdiffrast - Differentiable rasterization
cd /tmp
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install -e .

# kaolin - Optional, for advanced mesh operations
# Note: Kaolin requires PyTorch <= 2.8.0. If you have issues, skip this.
# cd /tmp && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git
# cd kaolin && pip install -e .
```

### 5. Clone and Setup SyncHuman

```bash
# Clone the official SyncHuman repository
git clone https://github.com/xishuxishu/SyncHuman.git
cd SyncHuman

# Download pretrained model checkpoints
python download.py

# Verify checkpoints are downloaded
ls -la ckpts/OneStage
ls -la ckpts/SecondStage
```

### 6. Verify Installation

```bash
# Test Stage 1 pipeline loading
ATTN_BACKEND=xformers python test_stage1.py
```

## Running Inference

### Stage 1: Multi-view Generation

```bash
# Set attention backend to xformers (for compatibility)
export ATTN_BACKEND=xformers

# Run inference with custom image
python inference_OneStage.py
# Edit inference_OneStage.py to change image_path
```

**Expected Output:**
- Color maps: `outputs/OneStage/color_*.png`
- Normal maps: `outputs/OneStage/normal_*.png`
- Sparse structure: `outputs/OneStage/coordinates.npz`
- Processing time: 1.5-2 minutes on A40 GPU

### Stage 2: Final 3D Model Generation (Optional)

```bash
export ATTN_BACKEND=xformers
python inference_SecondStage.py
```

**Expected Output:**
- 3D Model: `outputs/SecondStage/output.glb`
- Supporting files (mesh, texture) in outputs/SecondStage/
- Processing time: 2-3 minutes on A40 GPU

## Using the FastAPI Server

### Start the API Server

```bash
export ATTN_BACKEND=xformers
python api_server.py
# Server runs on http://0.0.0.0:8000
```

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Generate 3D Model:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/image.png" \
  -F "stage=both" \
  -F "stage1_steps=50" \
  -F "stage2_steps=25" \
  -F "download=true" \
  --output output.glb
```

**Parameters:**
- `image`: Input image file (RGBA PNG recommended, background should be transparent)
- `image_url`: Alternatively, provide a URL to image
- `stage`: "both", "stage1", or "stage2"
- `stage1_steps`: Diffusion steps for Stage 1 (default: 50, higher = better quality but slower)
- `stage2_steps`: Sampler steps for Stage 2 (default: 25)
- `download`: Return GLB file as response if true

## Environment Variables

### Attention Backend Selection

```bash
# Use xformers (recommended for most GPUs)
export ATTN_BACKEND=xformers

# Use PyTorch scaled dot-product attention (fallback)
export ATTN_BACKEND=sdpa

# Use flash-attn (if installed, requires compilation)
export ATTN_BACKEND=flash_attn

# Use naive attention (slowest, most compatible)
export ATTN_BACKEND=naive
```

### Sparse Attention Backend

```bash
# Enable sparse attention logging
export ATTN_DEBUG=1

# Sparse attention algorithm selection
export SPARSE_ATTN_BACKEND=flash_attn
```

## Troubleshooting

### Issue: "No module named 'flash_attn'"
**Solution:** Use xformers instead
```bash
export ATTN_BACKEND=xformers
```

### Issue: "No module named 'spconv'"
**Solution:** Install spconv for your CUDA version
```bash
pip install spconv-cu121  # For CUDA 12.1
pip install spconv-cu124  # For CUDA 12.4
```

### Issue: Kaolin installation fails
**Solution:** This is optional for Stage 1. Skip if Stage 1 works.
Kaolin requires PyTorch <= 2.8.0, incompatible with latest PyTorch. Use the pre-exported models instead.

### Issue: Out of memory errors
**Solution:** Reduce model precision or batch size
```python
# In your code, use float16 for Stage 1
pipeline = SyncHumanOneStagePipeline.from_pretrained(
    './ckpts/OneStage',
    dtype=torch.float16  # Use half precision
)
```

### Issue: CUDA out of memory during Stage 2
**Solution:** Clear GPU cache and reduce batch size
```python
torch.cuda.empty_cache()
```

## Model Checkpoints

The following models are downloaded automatically:

| Model | Size | Purpose |
|-------|------|---------|
| OneStage | ~3.5GB | 2D-3D cross-space diffusion |
| SecondStage | ~2.1GB | Final mesh/gaussian generation |
| DINOv2 | ~1.3GB | Vision feature extraction |
| CLIP | ~1.7GB | Text/image encoding |

**Total download size:** ~8.5GB

## Tips for Best Quality

1. **Prepare input images well:**
   - Use clear, well-lit photos of humans
   - Remove backgrounds (transparent RGBA PNG recommended)
   - Avoid occlusion of body parts
   - Image resolution: 512x512 to 1024x1024 recommended

2. **Use recommended parameters:**
   ```bash
   # For high quality (slower):
   stage1_steps=75
   stage2_steps=50

   # For balanced quality/speed (default):
   stage1_steps=50
   stage2_steps=25

   # For quick preview (faster):
   stage1_steps=30
   stage2_steps=15
   ```

3. **GPU Memory Management:**
   - Minimum: 40GB VRAM
   - Recommended: 48GB+ for comfortable operation
   - Clear cache between runs: `torch.cuda.empty_cache()`

## Advanced Configuration

### Custom Pipeline Parameters

```python
from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline

pipeline = SyncHumanOneStagePipeline.from_pretrained(
    './ckpts/OneStage',
    dtype=torch.float16,  # Use half precision
    device='cuda',
    num_views=5,  # Number of view predictions
    mv_img_wh=(768, 768),  # Output resolution
    mv_bg_color='white'  # Background color for views
)

# Enable attention optimizations
pipeline.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()

# Run inference
pipeline.run(
    image_path='./test_image.png',
    save_path='./outputs/OneStage',
)
```

### Custom Sampling Strategy (Stage 2)

```python
from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline

pipeline = SyncHumanTwoStagePipeline.from_pretrained('./ckpts/SecondStage')
pipeline.cuda()

# Custom parameters
pipeline.run(
    image_path='./outputs/OneStage',
    outpath='./outputs/SecondStage',
    num_steps=50,  # Increase for better quality
    guidance_scale=7.5,  # Increase for more adherence to input
)
```

## Performance Metrics

**Tested on NVIDIA A40 (46GB VRAM):**

| Stage | Task | Time | Memory |
|-------|------|------|--------|
| 1 | Pipeline Loading | ~2 min | ~32GB |
| 1 | Inference (50 steps) | ~1.5 min | ~40GB |
| 2 | Pipeline Loading | ~30s | ~28GB |
| 2 | Inference (25 steps) | ~2 min | ~42GB |
| Total | End-to-end | ~6.5 min | ~42GB |

## References

- **SyncHuman Paper:** https://arxiv.org/pdf/2510.07723
- **Project Page:** https://xishuxishu.github.io/SyncHuman.github.io/
- **GitHub:** https://github.com/xishuxishu/SyncHuman
- **Model Weights:** https://huggingface.co/xishushu/SyncHuman
- **TRELLIS (Base framework):** https://github.com/microsoft/TRELLIS
- **PSHuman (Reference):** https://github.com/pengHTYX/PSHuman

## Citation

If you use SyncHuman in your research, please cite:

```bibtex
@article{chen2025synchuman,
  title={SyncHuman: Synchronizing 2D and 3D Diffusion Models for Single-view Human Reconstruction},
  author={Chen, Wenyue and Li, Peng and Zheng, Wangguandong and Zhao, Chengfeng and Li, Mengfei and Zhu, Yaolong and Dou, Zhiyang and Wang, Ronggang and Liu, Yuan},
  journal={arXiv preprint arXiv:2510.07723},
  year={2025}
}
```

## License

This project includes code from multiple sources with different licenses. Please check individual module licenses before using in commercial applications.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review GitHub issues: https://github.com/xishuxishu/SyncHuman/issues
3. Check TRELLIS documentation: https://github.com/microsoft/TRELLIS

---

**Last Updated:** November 2025
**Tested Configuration:** Python 3.10, PyTorch 2.5.1, CUDA 12.1, NVIDIA A40 (46GB)
