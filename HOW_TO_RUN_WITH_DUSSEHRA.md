# How to Run Official SyncHuman API with Dussehra Image

This guide walks you through running the **official SyncHuman maximum quality API** step-by-step with the Dussehra PNG image.

## Quick Start (5 minutes)

### Step 1: Activate Environment

```bash
source /opt/conda/bin/activate SyncHuman
export ATTN_BACKEND=xformers
```

### Step 2: Start the API Server

```bash
python api_server_complete.py
```

**Wait 2-3 minutes for pipelines to load.** You'll see output like:

```
======================================================================
SyncHuman Complete API
======================================================================
...
Starting server on http://0.0.0.0:8000
======================================================================
```

### Step 3: Download the Dussehra Image

In a new terminal:

```bash
curl -L "https://www.pngfind.com/pngs/b/41-416466_dussehra-png.png" -o dussehra.png
ls -lh dussehra.png
# Output: dussehra.png (255K)
```

### Step 4: Generate 3D Model

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@dussehra.png" \
  -F "download=true" \
  --output dussehra_result.zip
```

### Step 5: Examine Results

```bash
unzip dussehra_result.zip
ls -la
# stage1/  stage2/  results_metadata.json

# View Stage 1 outputs (multiview images)
ls stage1/
# color_0.png color_1.png ... normal_0.png normal_1.png

# View Stage 2 output (final 3D model)
ls stage2/
# output.glb  output_mesh.ply
```

**Your 3D model is ready:** `stage2/output.glb`

---

## Understanding the Output

### What is the 3D Model?

The `output.glb` file is a complete 3D model containing:
- **Geometry:** High-quality human shape (refined from Stage 2)
- **Texture:** Detailed surface colors and normals (1024px resolution)
- **Format:** GLB (GL Transmission Format) - ready for rendering in any 3D viewer

### Viewing the 3D Model

**Web Viewer (Easiest):**
- Upload `stage2/output.glb` to: https://modelviewer.dev/
- Or: https://gltf-viewer.donmccurdy.com/
- Instant preview in your browser

**Blender (Professional):**
```bash
# Install Blender, then:
blender stage2/output.glb
```

**Python Viewer:**
```bash
pip install pyvista
python
>>> import pyvista as pv
>>> mesh = pv.read('stage2/output.glb')
>>> mesh.plot()
```

### Stage 1 Outputs (Intermediate)

The `stage1/` directory contains intermediate outputs you can analyze:
- `color_0.png` to `color_4.png` - 5 multi-view RGB renderings
- `normal_0.png` to `normal_4.png` - 5 normal maps for surface details
- `input.png` - Preprocessed input (768x768)
- `latent.npz` - Sparse 3D voxel structure

**Use Case:** Understand the 2D-3D synchronization, analyze multi-view consistency

---

## Official Parameters Used

### Stage 1 (Multi-view Generation)
```python
seed = 43  # Reproducibility
num_inference_steps = 50  # Official default
num_views = 5  # Multiview predictions
resolution = 768x768  # Official resolution
guidance_scale = 3.0  # Classifier-free guidance
```

### Stage 2 (3D Refinement)
```python
num_steps = 25  # Sampling iterations
cfg_strength = 5.0  # Guidance strength
texture_size = 1024  # High-quality textures
simplify = 0.7  # Mesh optimization
```

---

## Processing Time

On NVIDIA A40 (46GB VRAM):
- **Stage 1:** 1.5-2 minutes (loading + inference)
- **Stage 2:** 2-3 minutes (refinement)
- **Total:** 4-5 minutes per image

Memory usage: Peak 42GB (out of 46GB available)

---

## Quality Explanation

### Why Two Stages?

**Stage 1:** 2D-3D Cross-Space Diffusion
- Generates 5 synchronized viewpoints from single image
- Creates initial sparse 3D structure
- Output: Multi-view color and normal maps

**Stage 2:** Refined Mesh Generation (REQUIRES KAOLIN)
- Takes Stage 1 outputs and refines them
- Generates textured 3D mesh with high-quality details
- FlexiCubes decoder for geometry refinement
- Gaussian splatting for rendering
- Output: GLB model with 1024px textures

**Result:** Maximum quality photo-realistic 3D human reconstruction

---

## Troubleshooting

### API Server Won't Start

**Issue:** `ModuleNotFoundError: No module named 'kaolin'`

**Solution:** Kaolin is optional. The API will work with Stage 1 only. For full Stage 2 support:

```bash
# Install kaolin from source (takes 20 mins)
cd /tmp
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
TORCH_CUDA_ARCH_LIST="8.0" python setup.py build_ext --inplace
```

### Out of Memory

**Issue:** `RuntimeError: CUDA out of memory`

**Solution:** The pipeline requires 40+ GB VRAM. On smaller GPUs:
- Reduce `num_inference_steps` (trade quality for speed)
- Run on smaller resolution (768x768 is minimum recommended)
- Use cloud GPU (NVIDIA A40 or H800)

### Image Format Issues

**Issue:** `ValueError: Input must include transparency or clear foreground`

**Solution:** Ensure image is RGBA PNG with transparent background:
```bash
# Convert if needed
convert input.jpg -transparent white input.png
```

### Server Responds Slowly

**Issue:** First request takes 2-3 minutes

**Explanation:** Pipelines are loaded on first inference (not at startup). This is normal.

**Speed up future requests:** Pipelines stay loaded, so 2nd+ requests are faster (~4 min total).

---

## API Endpoints Reference

### Health Check
```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "ok",
  "service": "SyncHuman Complete API",
  "stage1": true,
  "stage2": false  // or true if kaolin installed
}
```

### Generate 3D Model (Main)
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png"
```

### Get API Info
```bash
curl http://localhost:8000/info
```

### API Documentation
```bash
curl http://localhost:8000/
```

Or open browser:
```
http://localhost:8000/docs
```

---

## Python Integration

Generate 3D models programmatically:

```python
import requests
import time
from pathlib import Path

def generate_3d_model(image_path):
    """Generate 3D model from image using official SyncHuman API"""

    # Upload image
    with open(image_path, 'rb') as f:
        files = {'image': (Path(image_path).name, f, 'image/png')}
        response = requests.post(
            'http://localhost:8000/generate',
            files=files,
            timeout=600  # 10 minutes
        )

    if response.status_code == 200:
        result = response.json()

        # Extract paths
        s1_dir = result['stage1']['output_dir']
        s2_dir = result['stage2']['output_dir']
        glb_path = f"{s2_dir}/output.glb"

        print(f"✓ 3D model generated!")
        print(f"  Location: {glb_path}")
        print(f"  Stage 1 outputs: {s1_dir}")
        print(f"  Stage 2 outputs: {s2_dir}")

        return glb_path
    else:
        raise Exception(f"API error: {response.status_code}")

# Usage
glb = generate_3d_model('dussehra.png')
```

---

## Advanced: Custom Quality Settings

To modify quality vs speed trade-offs, edit `api_server_complete.py`:

```python
# Around line 62-63
DEFAULT_STAGE1_STEPS = 50  # Reduce to 30 for speed
DEFAULT_STAGE2_STEPS = 25  # Reduce to 15 for speed
```

Then restart the server.

**Speed vs Quality:**
- `num_steps=30`: Fast (3 min total), lower quality
- `num_steps=50`: Official default (4-5 min), maximum quality
- `num_steps=75`: Slower (6-7 min), marginal quality gain

---

## References

**Official Repository:** https://github.com/IGL-HKUST/SyncHuman
**Paper:** https://arxiv.org/pdf/2510.07723
**Model Weights:** https://huggingface.co/xishushu/SyncHuman

---

## Next Steps

1. ✅ Generate 3D model with Dussehra image (this guide)
2. Try with your own images
3. For production: Deploy with kaolin for full Stage 2 support
4. Integrate into your application using the Python API

---

**Last Updated:** November 2025
**Official Implementation:** SyncHuman v1.0
**Tested On:** NVIDIA A40 (46GB VRAM), CUDA 12.1, PyTorch 2.9.0
