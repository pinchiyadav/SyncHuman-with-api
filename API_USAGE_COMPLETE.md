# Complete API Usage Guide

## api_server_complete.py - Full API Documentation

This guide shows you how to use the complete API server with practical examples.

---

## 1. Starting the API Server

### Prerequisites
- Activate the SyncHuman environment
- Set the attention backend to xformers
- Have a GPU with 40GB+ VRAM

### Start the Server

```bash
# Activate environment
source /opt/conda/bin/activate SyncHuman

# Set attention backend
export ATTN_BACKEND=xformers

# Start the API server
python api_server_complete.py
```

Expected output:
```
======================================================================
SyncHuman Complete API Server
======================================================================
Attention Backend: xformers
Stage 1 Available: True
Stage 2 Available: False (or True if kaolin installed)
GPU: NVIDIA A40 (or your GPU)
======================================================================
Starting server on http://0.0.0.0:8000
API docs: http://localhost:8000/docs
======================================================================
```

---

## 2. API Endpoints

### GET /
Gets API documentation
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "title": "SyncHuman Complete API",
  "endpoints": {
    "GET /health": "Health check",
    "GET /info": "API capabilities",
    "POST /generate": "Generate 3D model"
  }
}
```

### GET /health
Health check endpoint - useful for monitoring
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "service": "SyncHuman Complete API",
  "stage1": true,
  "stage2": false
}
```

### GET /info
Get API capabilities and GPU information
```bash
curl http://localhost:8000/info
```

Response:
```json
{
  "api_version": "0.2.0",
  "stages": {
    "stage1": {
      "available": true,
      "description": "Multi-view generation (5 color + 5 normal maps)",
      "time_estimate": "1.5-2 minutes per image"
    },
    "stage2": {
      "available": false,
      "description": "Refined 3D geometry (GLB mesh file)",
      "requires_kaolin": true,
      "note": "Not available"
    }
  },
  "gpu_info": {
    "available": true,
    "device": "NVIDIA A40",
    "memory_mb": 46068
  }
}
```

### POST /generate
Generate 3D model from image

**Parameters:**
- `image` (file): Input image (RGBA PNG recommended)
- `image_url` (string): Or provide a URL instead of file
- `stage` (string): "stage1", "stage2", or "both" (default: "both")
- `stage1_steps` (integer): Diffusion steps for Stage 1 (default: 50)
- `stage2_steps` (integer): Sampler steps for Stage 2 (default: 25)
- `download` (boolean): Return zip archive if true

---

## 3. Usage Examples

### Example 1: Generate Stage 1 Only (Recommended)

This is the fastest and simplest approach. Works without kaolin.

**Using curl:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/image.png" \
  -F "stage=stage1" \
  -F "stage1_steps=50"
```

**Using Python:**
```python
import requests
import json

# Prepare files and data
with open('image.png', 'rb') as f:
    files = {'image': ('image.png', f, 'image/png')}
    data = {
        'stage': 'stage1',
        'stage1_steps': '50'
    }

    # Send request
    response = requests.post(
        'http://localhost:8000/generate',
        files=files,
        data=data
    )

    # Print results
    results = response.json()
    print(json.dumps(results, indent=2))
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "alpha_coverage": 0.85,
  "stage1": {
    "status": "completed",
    "steps": 50,
    "output_dir": "tmp_api_jobs/a1b2c3d4/stage1_output",
    "files": {
      "color_0": "color_0.png",
      "color_1": "color_1.png",
      "color_2": "color_2.png",
      "color_3": "color_3.png",
      "color_4": "color_4.png",
      "normal_0": "normal_0.png",
      "normal_1": "normal_1.png",
      "normal_2": "normal_2.png",
      "normal_3": "normal_3.png",
      "normal_4": "normal_4.png",
      "input": "input.png"
    }
  }
}
```

Files are saved at:
- `/workspace/SyncHuman/tmp_api_jobs/a1b2c3d4/stage1_output/`

### Example 2: Generate Both Stages (if kaolin available)

Generates Stage 1 multi-view predictions AND Stage 2 refined 3D model.

**Using curl:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/image.png" \
  -F "stage=both" \
  -F "stage1_steps=50" \
  -F "stage2_steps=25"
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "stage1": {
    "status": "completed",
    "files": { ... }
  },
  "stage2": {
    "status": "completed",
    "output_dir": "tmp_api_jobs/a1b2c3d4/stage2_output",
    "files": {
      "output.glb": "output.glb",
      "output_mesh.ply": "output_mesh.ply"
    }
  }
}
```

Final 3D model: `/workspace/SyncHuman/tmp_api_jobs/a1b2c3d4/stage2_output/output.glb`

### Example 3: Download Results as ZIP

Get all outputs packaged as a single ZIP file.

**Using curl:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/image.png" \
  -F "stage=stage1" \
  -F "download=true" \
  --output results.zip

# Extract the zip
unzip results.zip
```

**Using Python:**
```python
import requests

with open('image.png', 'rb') as f:
    files = {'image': ('image.png', f)}
    data = {'stage': 'stage1', 'download': 'true'}

    response = requests.post(
        'http://localhost:8000/generate',
        files=files,
        data=data
    )

    # Save as zip file
    with open('results.zip', 'wb') as out:
        out.write(response.content)
```

### Example 4: Generate with Different Quality Settings

**For fast preview (30 seconds):**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage=stage1" \
  -F "stage1_steps=30"
```

**For balanced quality/speed (1.5 minutes):**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage=stage1" \
  -F "stage1_steps=50"
```

**For maximum quality (2.5 minutes):**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage=stage1" \
  -F "stage1_steps=75"
```

### Example 5: Using Image URL

Instead of uploading a file, you can provide a URL:

```bash
curl -X POST http://localhost:8000/generate \
  -F "image_url=https://example.com/image.png" \
  -F "stage=stage1"
```

---

## 4. Understanding the Output

### Stage 1 Output Files

**Color Maps** (color_0.png to color_4.png):
- 5 different viewing angles of the human
- RGB color predictions from the 2D-3D diffusion model
- Used for texture synthesis and rendering

**Normal Maps** (normal_0.png to normal_4.png):
- Surface geometry predictions for each view
- Shows surface direction and orientation
- Used for 3D reconstruction and mesh generation

**Input Image** (input.png):
- Your preprocessed and cropped input
- Resized to 768x768
- Used for reference

**Sparse Structure** (latent.npz):
- Compressed 3D coordinate data
- Represents the voxel grid structure
- Used by Stage 2 for refinement

### Stage 2 Output Files

**GLB Model** (output.glb):
- Final 3D model in glTF/GLB format
- Ready to use in 3D viewers
- Can be opened in Blender, Three.js, Babylon.js, etc.

**Mesh File** (output_mesh.ply):
- Triangle mesh in PLY format
- Standard 3D format for mesh data
- Can be used in Meshlab, Blender, etc.

---

## 5. Quality and Performance Tips

### For Production Use
```bash
# Stage 1 only (fast, good quality, no kaolin needed)
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage=stage1" \
  -F "stage1_steps=50"
```

**Why:** Stage 1 output is excellent for most use cases. Takes only 1.5 minutes and doesn't require kaolin.

### For Research/Analysis
```bash
# Use multi-view outputs directly for analysis
# Don't use Stage 2, just analyze the color and normal maps
```

**Why:** The multi-view predictions contain all the 3D information you need for reconstruction or analysis.

### For Visualization/Sharing
```bash
# If kaolin is available, generate both stages
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage=both" \
  -F "stage1_steps=50" \
  -F "stage2_steps=25"

# Then share the output.glb file
```

**Why:** GLB format is easy to share and view in web browsers.

### Out of Memory?
If you get CUDA out of memory errors:
1. Reduce `stage1_steps` to 30
2. Restart the server and try again
3. Close other GPU applications
4. Use a smaller GPU if available

---

## 6. Integration Examples

### Web Interface (Swagger UI)
1. Start the server
2. Open: http://localhost:8000/docs
3. Click "Try it out" on POST /generate
4. Upload image and see results

### Python Integration
```python
import requests
import json
from pathlib import Path

def generate_3d_model(image_path, stage='stage1'):
    """Generate 3D model from image using SyncHuman API"""

    with open(image_path, 'rb') as f:
        files = {'image': (Path(image_path).name, f, 'image/png')}
        data = {
            'stage': stage,
            'stage1_steps': '50',
            'stage2_steps': '25'
        }

        response = requests.post(
            'http://localhost:8000/generate',
            files=files,
            data=data,
            timeout=600
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code}")

# Usage
results = generate_3d_model('image.png', stage='stage1')
print(f"Job ID: {results['job_id']}")
print(f"Status: {results['status']}")
print(f"Output directory: {results['stage1']['output_dir']}")
```

### Batch Processing
```python
import os
import requests
from pathlib import Path

def batch_process(image_dir):
    """Process all images in a directory"""

    image_dir = Path(image_dir)
    for image_file in image_dir.glob('*.png'):
        print(f"Processing {image_file.name}...")

        with open(image_file, 'rb') as f:
            files = {'image': (image_file.name, f, 'image/png')}
            data = {'stage': 'stage1', 'stage1_steps': '50'}

            response = requests.post(
                'http://localhost:8000/generate',
                files=files,
                data=data,
                timeout=600
            )

            results = response.json()
            print(f"  Job ID: {results['job_id']}")
            print(f"  Output: {results['stage1']['output_dir']}")

# Usage
batch_process('./images')
```

---

## 7. Troubleshooting

### Server won't start
**Problem:** `Address already in use`
- Solution: Port 8000 is in use. Kill the previous process:
  ```bash
  pkill -f api_server_complete.py
  ```

### API not responding
**Problem:** Timeout or connection refused
- Solution: Server is loading models (takes 2 minutes). Wait and retry.

### Out of memory
**Problem:** `CUDA out of memory`
- Solution: Reduce `stage1_steps` to 30 or close other GPU apps.

### No stage2 available
**Problem:** API says stage2 is not available
- Solution: Normal! Kaolin isn't installed. Stage 1 works great.
- To enable Stage 2: Install kaolin (advanced, see SETUP_GUIDE.md)

### Image processing error
**Problem:** `Input must include transparency`
- Solution: Image must have alpha channel (RGBA PNG). Use rembg to remove background:
  ```python
  from rembg import remove
  from PIL import Image

  img = Image.open('image.jpg')
  result = remove(img)
  result.save('image.png')
  ```

---

## 8. Recommended Settings Summary

| Use Case | Stage | Steps | Time | Kaolin |
|----------|-------|-------|------|--------|
| Quick Test | Stage 1 | 30 | 1 min | No |
| Production | Stage 1 | 50 | 1.5 min | No |
| Research | Stage 1 | 50 | 1.5 min | No |
| Quality | Stage 1 | 75 | 2.5 min | No |
| Full 3D | Both | 50/25 | 4 min | Yes |
| Maximum Quality | Both | 75/50 | 6 min | Yes |

---

## References

- Official Repository: https://github.com/xishuxishu/SyncHuman
- Paper: https://arxiv.org/pdf/2510.07723
- Project Page: https://xishuxishu.github.io/SyncHuman.github.io/
- Model Weights: https://huggingface.co/xishushu/SyncHuman
