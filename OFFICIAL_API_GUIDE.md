# Official SyncHuman Maximum Quality API

This guide shows how to use the **official maximum quality API** that implements the exact recommended approach from the SyncHuman authors.

**Key Features:**
- ✅ Full two-stage pipeline (Stage 1 + Stage 2)
- ✅ Official parameters for maximum quality (no compromises)
- ✅ Seed control (seed=43) for reproducibility
- ✅ All official configurations implemented exactly
- ✅ Direct Stage 1 → Stage 2 pipeline integration

**Repository:** https://github.com/IGL-HKUST/SyncHuman
**Paper:** https://arxiv.org/pdf/2510.07723

---

## 1. Prerequisites

### GPU Requirements
- **Minimum:** 40GB VRAM
- **Recommended:** 48GB+ VRAM
- **Tested on:** NVIDIA A40 (46GB), H800 (80GB)

### Software Requirements
- Python 3.10
- PyTorch 2.1.1+ with CUDA 12.1
- All official dependencies installed

---

## 2. Installation

### Step 1: Create Environment
```bash
conda create -n SyncHuman python=3.10 -y
source /opt/conda/bin/activate SyncHuman
```

### Step 2: Install PyTorch
```bash
conda install pytorch::pytorch torchvision pytorch::pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Step 3: Install Official Dependencies
```bash
pip install accelerate safetensors==0.4.5 diffusers==0.29.1 transformers==4.36.0
pip install xformers  # Memory-efficient attention
pip install trimesh open3d omegaconf imageio imageio-ffmpeg rembg
pip install plyfile scikit-image scipy scikit-learn pyyaml
pip install spconv-cu121  # Sparse convolution
pip install onnxruntime onnx einops pyvista PyMeshFix igraph pillow opencv-python pydantic
pip install utils3d xatlas ninja easydict peft moviepy
```

### Step 4: Install NVIDIA-Specific Libraries
```bash
# nvdiffrast (required for Stage 2)
cd /tmp
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install -e .

# kaolin (required for Stage 2 - must build from source)
cd /tmp
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
pip install -e .
# This may take 10-20 minutes to compile
```

### Step 5: Download SyncHuman and Models
```bash
git clone https://github.com/xishuxishu/SyncHuman.git
cd SyncHuman
python download.py
# Downloads ~8.5GB of models
```

---

## 3. Starting the Official API Server

### Activate Environment and Start Server
```bash
source /opt/conda/bin/activate SyncHuman
export ATTN_BACKEND=xformers
python api_server_official.py
```

### Expected Output
```
======================================================================
SyncHuman Official Maximum Quality API Server
======================================================================
Attention Backend: xformers
Stage 1 Available: True
Stage 2 Available: True
GPU: NVIDIA A40
GPU Memory: 46GB
======================================================================
Approach: OFFICIAL SYNCHUMAN TWO-STAGE PIPELINE
Quality: MAXIMUM (no compromises)
======================================================================
Starting server on http://0.0.0.0:8000
API docs: http://localhost:8000/docs
======================================================================
```

The server will take **2-3 minutes** to load both pipelines on first startup.

---

## 4. Generate 3D Model from Image

### Method 1: Using cURL (Simplest)

**Generate with local image file:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/image.png"
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "approach": "Official SyncHuman two-stage pipeline (MAXIMUM QUALITY)",
  "status": "completed",
  "alpha_coverage": 0.85,
  "stage1": {
    "status": "completed",
    "parameters": {
      "seed": 43,
      "guidance_scale": 3.0,
      "num_inference_steps": 50,
      "mv_img_wh": [768, 768],
      "num_views": 5,
      "background_color": "white"
    },
    "output_dir": "tmp_api_jobs/a1b2c3d4/stage1_output",
    "files": {
      "color_0": "color_0.png",
      "color_1": "color_1.png",
      ...
      "normal_0": "normal_0.png",
      ...
      "input": "input.png"
    }
  },
  "stage2": {
    "status": "completed",
    "parameters": {
      "num_steps": 25,
      "cfg_strength": 5.0,
      "cfg_interval": [0.5, 1.0],
      "texture_size": 1024,
      "simplify": 0.7,
      "up_size": 896
    },
    "output_dir": "tmp_api_jobs/a1b2c3d4/stage2_output",
    "files": {
      "output.glb": "output.glb",
      "output_mesh.ply": "output_mesh.ply"
    }
  }
}
```

**Final 3D model location:**
```
tmp_api_jobs/a1b2c3d4/stage2_output/output.glb
```

---

### Method 2: Download Results as ZIP

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "download=true" \
  --output results.zip

# Extract
unzip results.zip
```

---

### Method 3: Using Image URL

```bash
curl -X POST http://localhost:8000/generate \
  -F "image_url=https://example.com/image.png"
```

---

### Method 4: Python Integration

```python
import requests
import json

def generate_3d(image_path):
    """Generate 3D model using official API"""
    with open(image_path, 'rb') as f:
        files = {'image': (image_path.split('/')[-1], f, 'image/png')}

        response = requests.post(
            'http://localhost:8000/generate',
            files=files,
            timeout=600  # 10 minutes
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

# Usage
result = generate_3d('image.png')
print(f"Job ID: {result['job_id']}")
print(f"Status: {result['status']}")
print(f"Stage 1 Output: {result['stage1']['output_dir']}")
print(f"Stage 2 Output (GLB): {result['stage2']['output_dir']}/output.glb")
```

---

## 5. Test with Dussehra Image

**Download the test image:**
```bash
curl -L "https://www.pngfind.com/pngs/b/41-416466_dussehra-png.png" -o dussehra.png
```

**Generate 3D model:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@dussehra.png" \
  -F "download=true" \
  --output dussehra_results.zip
```

**Expected:**
- Processing time: 4-5 minutes (1.5-2 min Stage 1 + 2-3 min Stage 2)
- Output: ZIP with Stage 1 multiviews and Stage 2 GLB model
- Final 3D model: GLB format ready for rendering/viewing

---

## 6. Official Configuration Parameters

### Stage 1 (Multi-view Generation)
```python
{
    "seed": 43,                          # Reproducibility
    "guidance_scale": 3.0,               # User-facing CFG
    "num_inference_steps": 50,           # Official default
    "mv_img_wh": (768, 768),            # Official resolution
    "num_views": 5,                      # Synchronized viewpoints
    "background_color": "white",
}
```

### Stage 2 (3D Refinement)
```python
{
    "num_steps": 25,                     # Official sampling steps
    "cfg_strength": 5.0,                 # Classifier-free guidance
    "cfg_interval": [0.5, 1.0],         # When to apply guidance
    "texture_size": 1024,                # High-quality textures
    "simplify": 0.7,                     # Mesh detail balance
    "up_size": 896,                      # Resolution optimization
}
```

---

## 7. Understanding Output Files

### Stage 1 Output Directory
```
stage1_output/
├── color_0.png to color_4.png          # 5 RGB multi-view predictions
├── normal_0.png to normal_4.png        # 5 normal map predictions
├── input.png                           # Preprocessed (768x768) input
└── latent.npz                          # Sparse 3D structure
```

**Use Case:** Analyze multi-view consistency, create custom 3D reconstruction, texture synthesis

### Stage 2 Output Directory (MAIN RESULT)
```
stage2_output/
├── output.glb                          # Final 3D model (MAIN OUTPUT)
└── output_mesh.ply                     # Triangle mesh geometry
```

**Use Case:** Render, animate, export to 3D software, display in web viewers

---

## 8. Quality Metrics

### What Makes This Maximum Quality

**Stage 1 Synchronization:**
- 2D-3D cross-space diffusion (50 steps)
- Multi-view attention with 5 viewpoints
- Sparse voxel grid structure
- Classifier-free guidance (CFG=10.0 internal)

**Stage 2 Refinement:**
- Flow-matching sampler (25 steps)
- High-resolution textures (1024px)
- Mesh refinement with FlexiCubes
- Gaussian splatting integration

**Result:** Photo-realistic 3D human reconstruction with consistent geometry and detailed textures

---

## 9. Performance Specifications

**Tested on NVIDIA A40 (46GB VRAM):**

| Stage | Operation | Time | Memory |
|-------|-----------|------|--------|
| 1 | Pipeline Loading | ~2 min | ~32GB |
| 1 | Inference | ~1.5 min | ~40GB |
| 2 | Pipeline Loading | ~30 sec | ~28GB |
| 2 | Inference | ~2 min | ~42GB |
| **Total** | **End-to-end** | **~4.5 min** | **~42GB** |

---

## 10. API Endpoints Reference

### GET /health
Health check with stage availability

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "SyncHuman Official Maximum Quality API",
  "version": "1.0.0",
  "stage1": true,
  "stage2": true,
  "approach": "Full two-stage pipeline for maximum quality"
}
```

### GET /info
API capabilities and configuration

```bash
curl http://localhost:8000/info
```

Returns:
- Official configurations for both stages
- GPU information and memory available
- Estimated processing times
- Links to official repository and paper

### GET /
API documentation

```bash
curl http://localhost:8000/
```

Returns comprehensive API documentation

### POST /generate
Generate 3D model from image (MAIN ENDPOINT)

**Parameters:**
- `image` (file): Input image - RGBA PNG recommended, transparent background
- `image_url` (string): Alternative - provide URL instead of file
- `download` (boolean): Return ZIP archive if true

---

## 11. Troubleshooting

### Kaolin Installation Failed
Kaolin requires building from source and can take 20+ minutes.

**Check build progress:**
```bash
tail -f /tmp/kaolin_build.log
```

**If build fails:**
- Ensure you have C++ build tools: `apt-get install build-essential`
- Check CUDA path: `which nvcc`
- Try: `pip install --no-cache-dir -e .`

### Out of Memory During Inference
Reduce quality settings (not available in official version, but you can modify the code):
```python
# Modify in api_server_official.py
OFFICIAL_STAGE1_CONFIG["num_inference_steps"] = 30  # Default is 50
```

### Server Won't Start
1. Check Stage 2 is available: `python -c "from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline"`
2. Kill old server: `pkill -f api_server_official`
3. Check GPU: `nvidia-smi`

### API Not Responding
Server takes 2-3 minutes to load on first request. Check logs:
```bash
tail -100 api_server_official.log
```

---

## 12. Quick Reference Commands

```bash
# Start server
source /opt/conda/bin/activate SyncHuman
export ATTN_BACKEND=xformers
python api_server_official.py

# Test health
curl http://localhost:8000/health

# Generate with local image
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png"

# Download as ZIP
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "download=true" \
  --output results.zip

# Stop server
pkill -f api_server_official
```

---

## 13. References

- **Official Repository:** https://github.com/IGL-HKUST/SyncHuman
- **Paper:** https://arxiv.org/pdf/2510.07723
- **Model Weights:** https://huggingface.co/xishushu/SyncHuman
- **TRELLIS Framework:** https://github.com/microsoft/TRELLIS
- **API Documentation:** http://localhost:8000/docs (when server running)

---

## 14. Citation

If you use this official API in your research, please cite:

```bibtex
@article{chen2025synchuman,
  title={SyncHuman: Synchronizing 2D and 3D Diffusion Models for Single-view Human Reconstruction},
  author={Chen, Wenyue and Li, Peng and Zheng, Wangguandong and Zhao, Chengfeng and Li, Mengfei and Zhu, Yaolong and Dou, Zhiyang and Wang, Ronggang and Liu, Yuan},
  journal={arXiv preprint arXiv:2510.07723},
  year={2025}
}
```

---

**Last Updated:** November 2025
**API Version:** 1.0.0 (Official Maximum Quality)
**Status:** Production Ready
