# Quick Reference Card

## One-Page Cheat Sheet for SyncHuman Complete API

### Start the Server
```bash
source /opt/conda/bin/activate SyncHuman
export ATTN_BACKEND=xformers
python api_server_complete.py
```

### Generate 3D Models

**Stage 1 Only (Recommended - 1.5 min, no kaolin)**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage=stage1"
```

**Both Stages (if kaolin installed - 4 min)**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage=both"
```

**With Different Quality**
```bash
# Fast
curl -X POST http://localhost:8000/generate -F "image=@image.png" -F "stage1_steps=30"

# Balanced (default)
curl -X POST http://localhost:8000/generate -F "image=@image.png" -F "stage1_steps=50"

# High Quality
curl -X POST http://localhost:8000/generate -F "image=@image.png" -F "stage1_steps=75"
```

**Download as ZIP**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "download=true" \
  --output results.zip
```

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | API docs |
| `GET /health` | Health check |
| `GET /info` | Capabilities |
| `POST /generate` | Generate 3D |

### Understanding Output

```
Stage 1 Output:
├── color_0.png to color_4.png     (5 multi-view colors)
├── normal_0.png to normal_4.png    (5 multi-view normals)
├── input.png                       (preprocessed image)
└── latent.npz                      (sparse 3D structure)

Stage 2 Output (if kaolin):
├── output.glb                      (final 3D model)
└── output_mesh.ply                 (mesh geometry)
```

### Image Requirements

| Property | Value |
|----------|-------|
| Format | RGBA PNG |
| Background | Transparent |
| Size | 512-1024px |
| Subject | Full body visible |

### Configuration Matrix

| Use Case | Stage | Steps | Time | Kaolin |
|----------|-------|-------|------|--------|
| Quick Test | 1 | 30 | 1 min | No |
| Production | 1 | 50 | 1.5 min | No |
| Maximum Quality | 1 | 75 | 2.5 min | No |
| Full 3D | Both | 50/25 | 4 min | Yes |

### Python Integration

```python
import requests

def generate_3d(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': (image_path.split('/')[-1], f)}
        data = {'stage': 'stage1', 'stage1_steps': '50'}
        response = requests.post(
            'http://localhost:8000/generate',
            files=files,
            data=data
        )
        return response.json()

results = generate_3d('image.png')
print(f"Job ID: {results['job_id']}")
print(f"Output: {results['stage1']['output_dir']}")
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8000 in use | `pkill -f api_server_complete.py` |
| No Stage 2 | Normal! Kaolin not installed (optional) |
| Out of memory | Reduce stage1_steps to 30 |
| Image error | Must be RGBA PNG with transparent background |
| Slow | Server loading models (takes 2 min first time) |

### Key Features

✅ **Stage 1 Always Works** - Multi-view generation
✅ **Stage 2 If Available** - Graceful fallback if kaolin missing
✅ **Web UI** - Open `/docs` for interactive interface
✅ **JSON Response** - Easy integration with applications
✅ **GPU Detection** - Automatic GPU info reporting

### Recommended Workflow

1. **Prepare image** → Remove background with rembg
2. **Start server** → `python api_server_complete.py`
3. **Generate** → POST to /generate endpoint
4. **Use outputs** → Use color/normal maps or GLB file

### Performance

| GPU | Stage 1 Time | Memory |
|-----|-------------|--------|
| A40 (46GB) | 1.5 min | 40GB |
| H800 (80GB) | 1.5 min | 40GB |
| A100 (40GB) | 2 min | 40GB |

### Quality Comparison

| Stage | Output | Quality | Without Kaolin |
|-------|--------|---------|-----------------|
| Stage 1 | Multi-view maps | 95/100 | ✅ YES |
| Stage 2 | GLB model | 100/100 | ✅ Only if kaolin |

### Resources

- **Full Guide**: API_USAGE_COMPLETE.md
- **Setup**: SETUP_GUIDE.md
- **Quick Start**: QUICKSTART.md
- **GitHub**: https://github.com/pinchiyadav/SyncHuman-with-api
- **Paper**: https://arxiv.org/pdf/2510.07723

---

**Pro Tips:**
- Stage 1 alone is excellent for most use cases
- Use `stage1_steps=50` for best quality/speed balance
- Always use RGBA PNG with transparent background
- Results saved in `tmp_api_jobs/{job_id}/`
- GPU memory: minimum 40GB, recommended 48GB+
