# SyncHuman Unified API - Complete Documentation

**One API. Smart Defaults. Request-Based Configuration.**

This is the **single official API** for SyncHuman. Start the server with no flags, then configure each request as needed.

---

## Quick Start (3 Different Use Cases)

### Use Case 1: Maximum Official Quality (Default)
```bash
# Start server (no flags needed)
source /opt/conda/bin/activate SyncHuman
export ATTN_BACKEND=xformers
python api_server.py

# In another terminal - make request with default settings:
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "download=true" \
  --output result.zip
```

**Output:** Complete textured GLB 3D model + Stage 1 multiviews
**Time:** 4-5 minutes
**Quality:** ⭐⭐⭐⭐⭐ (100% - official recommended)
**Requirement:** Kaolin must be installed

### Use Case 2: Fast & Simple (No Kaolin Needed)
```bash
# Same server startup (no flags)
python api_server.py

# Request with stage1_only flag:
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage1_only=true" \
  -F "download=true" \
  --output result.zip
```

**Output:** 5 color maps + 5 normal maps (excellent for multiview reconstruction)
**Time:** 1.5-2 minutes (2.7x faster)
**Quality:** ⭐⭐⭐⭐ (95% - excellent)
**Requirement:** NO kaolin needed

### Use Case 3: Production Ready (Always Works)
```bash
# Same server startup (no flags)
python api_server.py

# Request with graceful_fallback flag:
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "graceful_fallback=true" \
  -F "download=true" \
  --output result.zip
```

**Output:** GLB mesh if kaolin available, multiviews otherwise
**Time:** 1.5-2 min (Stage 1 only) or 4-5 min (full)
**Quality:** 95-100% depending on kaolin
**Requirement:** Always works, adapts to available dependencies

---

## Server Startup

### Basic (Recommended)
```bash
# No flags needed - server auto-detects available stages
python api_server.py
```

### With Custom Port
```bash
# Modify the port in api_server.py or use environment variables
# Default: http://0.0.0.0:8000
python api_server.py
```

Server will print startup info showing:
- ✓ Stage 1: Available (always)
- ✓ Stage 2: Available (if kaolin installed) or ✗ Not available

---

## Request Parameters Reference

All configuration happens via POST request parameters to the `/generate` endpoint.

### Mode Selection Parameters

**`stage1_only` (boolean, default: false)**
- Skip Stage 2, run only Stage 1 (multiview generation)
- No kaolin required
- Use when: kaolin not installed, or you only need multiviews
- Quality: 95% (⭐⭐⭐⭐)
- Time: 1.5-2 minutes

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage1_only=true"
```

**`graceful_fallback` (boolean, default: false)**
- Try full pipeline (Stage 1+2), fall back to Stage 1 if kaolin missing
- Always works, adapts to what's available
- Use when: you want best available quality without manual configuration

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "graceful_fallback=true"
```

### Quality Customization Parameters

**`stage1_steps` (integer, 30-100, default: 50)**
- Number of inference steps for Stage 1 (multiview generation)
- Higher = better quality but slower
- Recommended values:
  - 30: Very fast, lower quality
  - 50: Official default (recommended)
  - 75: Higher quality, slower
  - 100: Maximum quality, slowest

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage1_steps=75"
```

**`stage2_steps` (integer, 15-40, default: 25)**
- Number of inference steps for Stage 2 (mesh refinement with kaolin)
- Only used if Stage 2 runs (not with `stage1_only=true`)
- Higher = better quality but slower
- Recommended values:
  - 15: Fast mesh generation
  - 25: Official default
  - 40: Very high quality

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage2_steps=40"
```

### Output Parameters

**`download` (boolean, default: false)**
- Return results as ZIP archive instead of direct files
- Useful for downloading via curl

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "download=true" \
  --output results.zip
```

### Input Parameters

**`image` (file upload)**
- Input image file (RGBA PNG recommended)
- Formats: PNG, JPG, JPEG
- Size: Any (will be resized to 768x768)

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@photo.png"
```

**`image_url` (string, optional)**
- Alternative to image file upload
- Provide image URL instead of file

```bash
curl -X POST http://localhost:8000/generate \
  -F "image_url=https://example.com/image.png"
```

---

## API Endpoints

### GET /health
Check API status and available stages

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "SyncHuman Unified API",
  "stage1_available": true,
  "stage2_available": true,
  "mode": "official"
}
```

### GET /info
Get API configuration and capabilities

```bash
curl http://localhost:8000/info
```

**Response:**
```json
{
  "service": "SyncHuman Unified API",
  "version": "1.0.0",
  "available_stages": {
    "stage1": true,
    "stage2": true
  },
  "default_settings": {
    "stage1_steps": 50,
    "stage2_steps": 25
  },
  "request_parameters": {
    "stage1_only": "bool - Skip Stage 2",
    "graceful_fallback": "bool - Try full, fall back gracefully",
    "stage1_steps": "int 30-100 - Stage 1 quality",
    "stage2_steps": "int 15-40 - Stage 2 quality",
    "download": "bool - Return ZIP archive"
  }
}
```

### GET /
API documentation and usage instructions

```bash
curl http://localhost:8000/
```

Returns HTML documentation with interactive interface.

### POST /generate
Generate 3D model from image

**Request:**
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@input.png" \
  -F "stage1_steps=50" \
  -F "stage2_steps=25"
```

**Response (JSON):**
```json
{
  "success": true,
  "status": "completed",
  "duration": 245.5,
  "mode": "full",
  "outputs": {
    "stage1": {
      "color_maps": ["color_0.png", "color_1.png", "color_2.png", "color_3.png", "color_4.png"],
      "normal_maps": ["normal_0.png", "normal_1.png", "normal_2.png", "normal_3.png", "normal_4.png"],
      "coordinates": "coordinates.npz"
    },
    "stage2": {
      "model": "output.glb",
      "available": true
    }
  },
  "paths": {
    "output_dir": "/workspace/SyncHuman/outputs/SyncHuman_<timestamp>"
  }
}
```

---

## Complete Examples

### Example 1: Default Maximum Quality
```bash
# Terminal 1: Start server
python api_server.py

# Terminal 2: Make request with defaults
curl -X POST http://localhost:8000/generate \
  -F "image=@my_photo.png" \
  -F "download=true" \
  --output result.zip
```
**Result:** Complete textured GLB + multiviews (4-5 min)

### Example 2: Fast Mode for Multiple Images
```bash
# Terminal 1: Start server once
python api_server.py

# Terminal 2: Process multiple images quickly
for image in photos/*.png; do
  curl -X POST http://localhost:8000/generate \
    -F "image=@$image" \
    -F "stage1_only=true" \
    -F "download=true" \
    --output "results/$(basename $image .png).zip"
  echo "Processed $image (1.5-2 min)"
done
```
**Result:** Multiview maps only, fast processing (1.5-2 min each)

### Example 3: High-Quality Results
```bash
# Terminal 1: Start server
python api_server.py

# Terminal 2: Request with higher quality
curl -X POST http://localhost:8000/generate \
  -F "image=@high_quality_photo.png" \
  -F "stage1_steps=100" \
  -F "stage2_steps=40" \
  -F "download=true" \
  --output hq_result.zip
```
**Result:** Maximum quality 3D model (5-6 min)

### Example 4: Production Deployment
```bash
# Terminal 1: Start with graceful fallback
python api_server.py

# Terminal 2: Make requests - will always work
curl -X POST http://localhost:8000/generate \
  -F "image=@photo.png" \
  -F "graceful_fallback=true" \
  -F "download=true" \
  --output result.zip
```
**Result:** Full quality if kaolin available, fallback to Stage 1 otherwise

---

## Output Structure

When a request completes successfully, results are saved to:
```
outputs/SyncHuman_<timestamp>/
├── Stage_1/
│   ├── color_0.png to color_4.png        # 5 multiview color maps
│   ├── normal_0.png to normal_4.png       # 5 multiview normal maps
│   └── coordinates.npz                    # Sparse 3D structure
└── Stage_2/
    └── output.glb                         # Textured 3D mesh (if kaolin available)
```

The response JSON includes the output directory path for direct file access.

---

## Troubleshooting

### Issue: "kaolin not found" - Can't run Stage 2
**Solution 1:** Use `stage1_only=true` flag in request
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage1_only=true"
```

**Solution 2:** Use `graceful_fallback=true` to adapt automatically
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "graceful_fallback=true"
```

**Solution 3:** Install kaolin (see SETUP_GUIDE.md)

### Issue: Out of Memory (OOM)
**Solution:** Reduce inference steps:
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage1_steps=30" \
  -F "stage2_steps=15"
```

### Issue: Slow inference
**Solution 1:** Skip Stage 2:
```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@image.png" \
  -F "stage1_only=true"
```

**Solution 2:** Check GPU:
```bash
nvidia-smi  # Verify GPU usage and memory
```

### Issue: xformers not available
**Solution:**
```bash
# Set environment variable before running api_server.py
export ATTN_BACKEND=flash_attn  # or use auto-fallback

python api_server.py
```

### Issue: Port already in use
**Solution:** Kill existing process or use different port
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Then restart api_server.py
python api_server.py
```

---

## Performance Benchmarks

Tested on NVIDIA A40 (46GB VRAM):

| Mode | Time | Quality | Requires Kaolin |
|------|------|---------|-----------------|
| Stage 1 only (50 steps) | 1.5-2 min | 95% ⭐⭐⭐⭐ | No |
| Stage 1 (75 steps) | 2-2.5 min | 97% ⭐⭐⭐⭐ | No |
| Stage 1 + Stage 2 (50+25) | 4-5 min | 100% ⭐⭐⭐⭐⭐ | Yes |
| Stage 1 + Stage 2 (100+40) | 5-6 min | 100% ⭐⭐⭐⭐⭐ | Yes |

---

## Official Approach

SyncHuman's official recommended configuration:
- **Default behavior:** Stage 1 + Stage 2 with kaolin for maximum quality
- **Stage 1:** 50 inference steps, 768x768 resolution, 5 synchronized multiviews
- **Stage 2:** 25 inference steps, FlexiCubes decoder, GLB mesh generation
- **Quality:** 100% - complete textured 3D model with synchronized 2D-3D diffusion
- **Fallback:** If kaolin unavailable, gracefully uses Stage 1 only (95% quality)

This approach provides maximum quality by default while still supporting fast/fallback modes via request parameters.

---

## Support

For issues or questions:
1. Check this documentation
2. See SETUP_GUIDE.md for installation help
3. Check INSTALLATION_SUMMARY.md for test results
4. Review README.md for overview
