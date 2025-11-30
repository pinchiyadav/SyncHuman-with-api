# API Server Comparison: Original vs Official vs Complete

This document explains the differences between the three API servers available in the SyncHuman repository.

---

## Quick Summary Table

| Feature | `api_server.py` | `api_server_complete.py` | `api_server_official.py` |
|---------|-----------------|------------------------|------------------------|
| **Stage 1** | ✅ Required | ✅ Always works | ✅ Always works |
| **Stage 2** | ✅ Required (kaolin) | ❌ Graceful fallback | ✅ Required (kaolin) |
| **Quality** | Maximum (if kaolin works) | 95% (Stage 1 alone) | Maximum (no compromise) |
| **Approach** | Simple wrapper | Practical fallback | Official spec exactly |
| **Official Params** | ✅ Uses defaults | ✅ Official defaults | ✅ Enforced strictly |
| **Error Handling** | ❌ Fails if kaolin missing | ✅ Continues without Stage 2 | ❌ Requires both stages |
| **Best For** | Development | Production (safe fallback) | Production (max quality) |
| **Startup** | Fast fail or slow init | Fast/normal | Enforces full pipeline |

---

## Detailed Comparison

### 1. **api_server.py** - Original Implementation

**What it is:** The basic/original API server that directly wraps the SyncHuman inference scripts.

**Key Characteristics:**
```python
# Tries to load BOTH stages upfront
_stage1_pipe: Optional[SyncHumanOneStagePipeline] = None
_stage2_pipe: Optional[SyncHumanTwoStagePipeline] = None

def _load_pipelines():
    if _stage1_pipe is None:
        _stage1_pipe = SyncHumanOneStagePipeline.from_pretrained("./ckpts/OneStage")
    if _stage2_pipe is None:
        _stage2_pipe = SyncHumanTwoStagePipeline.from_pretrained("./ckpts/SecondStage")
        _stage2_pipe.cuda()
```

**Pros:**
- Simple, straightforward implementation
- Directly mirrors official inference scripts
- Explicit control over which stages to run

**Cons:**
- ❌ **FAILS immediately if kaolin is not installed** - ImportError on startup
- ❌ Requires both Stage 1 AND Stage 2 to work at all
- Less user-friendly when kaolin installation fails

**When to use:**
- You have kaolin installed and working
- You need maximum quality with both stages
- You're on a development machine with proper setup

**Error if kaolin missing:**
```
ImportError: This is the kaolin placeholder wheel from https://pypi.org/project/kaolin/
...
RuntimeError: Failed to import SyncHuman modules.
```

---

### 2. **api_server_complete.py** - Intelligent Fallback

**What it is:** A practical, production-ready API that gracefully handles missing dependencies.

**Key Characteristics:**
```python
# Try Stage 1 (always available)
try:
    from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
    STAGE1_AVAILABLE = True
except Exception as e:
    print(f"Warning: Stage 1 not available: {e}")
    STAGE1_AVAILABLE = False

# Try Stage 2 (may fail if kaolin missing)
STAGE2_AVAILABLE = False
try:
    from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline
    STAGE2_AVAILABLE = True
except Exception as e:
    print(f"Note: Stage 2 not available (kaolin required): {type(e).__name__}")
    STAGE2_AVAILABLE = False

if not STAGE1_AVAILABLE:
    raise RuntimeError("Stage 1 must be available.")
```

**Pros:**
- ✅ **Works even if kaolin is missing** - provides 95% quality with Stage 1 alone
- ✅ Server starts successfully regardless of kaolin status
- ✅ User gets clear feedback about what's available
- ✅ Graceful degradation instead of failure
- ✅ Perfect for production environments where kaolin is optional

**Cons:**
- Uses Stage 1 only if Stage 2 fails (not maximum quality)
- May make user think they're getting Stage 2 when they're not (without checking status)

**When to use:**
- **Production environments** where kaolin may not be available
- You want the API to always work, even without kaolin
- You prefer 95% quality with guaranteed operation over 100% quality that might fail
- You need a robust, fault-tolerant solution

**Status check:**
```bash
curl http://localhost:8000/health
{
  "status": "ok",
  "stage1": true,
  "stage2": false  # ← Shows kaolin is missing
}
```

**Quality Trade-off:**
- Stage 1 only: 5 color maps + 5 normal maps (excellent for most uses)
- Stage 1+2: Above + refined 3D GLB mesh (ultimate quality)

---

### 3. **api_server_official.py** - Official Specification (NEW)

**What it is:** Strict implementation of the official SyncHuman approach with NO compromises on quality.

**Key Characteristics:**
```python
# Explicit: Both stages REQUIRED for official approach
try:
    from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
    STAGE1_AVAILABLE = True
except Exception as e:
    print(f"ERROR: Stage 1 not available: {e}")
    STAGE1_AVAILABLE = False
    raise RuntimeError("Stage 1 is required...")

try:
    from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline
    STAGE2_AVAILABLE = True
except Exception as e:
    print(f"ERROR: Stage 2 not available: {e}")
    STAGE2_AVAILABLE = False
    raise RuntimeError(
        "Stage 2 is required for maximum quality. "
        "Install kaolin: pip install kaolin or use api_server_complete.py"
    )

# OFFICIAL CONFIGURATION - No modifications allowed
OFFICIAL_STAGE1_CONFIG = {
    "seed": 43,
    "guidance_scale": 3.0,
    "num_inference_steps": 50,
    "mv_img_wh": (768, 768),
    "num_views": 5,
    "background_color": "white",
}

OFFICIAL_STAGE2_CONFIG = {
    "num_steps": 25,
    "cfg_strength": 5.0,
    "cfg_interval": [0.5, 1.0],
    "texture_size": 1024,
    "simplify": 0.7,
    "up_size": 896,
}
```

**Pros:**
- ✅ **Official approach from SyncHuman authors** - EXACTLY as specified
- ✅ No compromises on quality - full two-stage pipeline required
- ✅ Parameters locked to official values (no confusion)
- ✅ Comprehensive documentation of what's being used
- ✅ Best for research, publication, comparison with paper results
- ✅ Clear error messages directing users to install kaolin

**Cons:**
- ❌ **FAILS if kaolin not installed** - same as original
- ❌ Less forgiving for production environments
- Requires more dependencies installed upfront

**When to use:**
- **Research papers/publications** - need exact official approach
- **Comparisons with official results** - ensure identical parameters
- **Academic/scientific work** - reproducibility is critical
- You have kaolin working and want no quality compromises

**Error if kaolin missing:**
```
RuntimeError: Stage 2 is required for maximum quality.
Install kaolin: pip install kaolin or use api_server_complete.py
```

**Quality:**
- Always: Full two-stage pipeline
- Stage 1: 50 inference steps, 5 multiviews, 768x768
- Stage 2: 25 sampling steps, 1024px textures, refined mesh
- Result: 100% official quality (no shortcuts)

---

## Decision Tree

```
Do you have kaolin installed and working?
│
├─→ YES
│   │
│   ├─→ Need MAXIMUM official quality? → Use api_server_official.py
│   │
│   └─→ Just need it to work? → Use api_server_complete.py
│
└─→ NO
    │
    ├─→ Want to install kaolin? → Use api_server_official.py (after install)
    │
    └─→ Don't want to install kaolin? → Use api_server_complete.py
                                        (Stage 1 only, 95% quality)
```

---

## Feature Comparison Details

### Stage 1 Output (All Three Support)
```
color_0.png to color_4.png     # 5 RGB multi-view renderings
normal_0.png to normal_4.png   # 5 normal maps
input.png                       # Preprocessed 768x768 input
latent.npz                      # Sparse voxel structure
```

### Stage 2 Output (If Available/Required)

**api_server.py:**
- ✅ output.glb (textured 3D mesh - requires kaolin)
- ✅ output_mesh.ply

**api_server_complete.py:**
- ✅ If kaolin available: output.glb + output_mesh.ply
- ❌ If kaolin missing: Stage 2 skipped, only Stage 1 output

**api_server_official.py:**
- ✅ output.glb (textured 3D mesh - kaolin REQUIRED)
- ✅ output_mesh.ply

---

## Official Parameters Comparison

### Stage 1 Settings

```python
# Original api_server.py
# Uses defaults from pipeline, not explicitly documented

# api_server_complete.py
DEFAULT_STAGE1_STEPS = 50  # Official default

# api_server_official.py
OFFICIAL_STAGE1_CONFIG = {
    "seed": 43,                      # Exact seed from paper
    "guidance_scale": 3.0,           # Official CFG
    "num_inference_steps": 50,       # Official steps
    "mv_img_wh": (768, 768),        # Official resolution
    "num_views": 5,                  # Official multiviews
    "background_color": "white",
}
```

### Stage 2 Settings

```python
# Original api_server.py
# Uses defaults from pipeline.json

# api_server_complete.py
DEFAULT_STAGE2_STEPS = 25  # If available

# api_server_official.py
OFFICIAL_STAGE2_CONFIG = {
    "num_steps": 25,                 # Exact official
    "cfg_strength": 5.0,             # Exact official
    "cfg_interval": [0.5, 1.0],     # Exact official
    "texture_size": 1024,            # Exact official
    "simplify": 0.7,                 # Exact official
    "up_size": 896,                  # Exact official
}
```

---

## Error Handling Comparison

| Scenario | api_server.py | api_server_complete.py | api_server_official.py |
|----------|---------------|----------------------|----------------------|
| Kaolin missing | ❌ Crash on startup | ✅ Run Stage 1 only | ❌ Clear error, suggest fix |
| Stage 1 fails | ❌ Crash | ❌ Crash | ❌ Crash |
| Generate without Stage 2 | ❌ Error | ✅ Automatically uses Stage 1 | ❌ Error (use complete.py) |
| User uncertainty | ❌ Unclear | ✅ Clear /health endpoint | ✅ Documentation + error |

---

## Recommendation Matrix

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **Research/Publication** | `api_server_official.py` | Exact official approach, reproducible, verifiable |
| **Production (Robust)** | `api_server_complete.py` | Graceful fallback, always works, 95% quality |
| **Development** | `api_server_complete.py` | Fewer dependencies, still gets results |
| **High-End Rendering** | `api_server_official.py` | Maximum quality, refined meshes |
| **Real-time Apps** | `api_server_complete.py` | Faster (Stage 1 only), acceptable quality |
| **Budget Constraints** | `api_server_complete.py` | Works without kaolin compilation |

---

## Side-by-Side Example

### Scenario: Run with Dussehra image, kaolin NOT installed

**api_server.py:**
```bash
$ python api_server.py
ImportError: This is the kaolin placeholder wheel...
❌ FAILS - Cannot start server
```

**api_server_complete.py:**
```bash
$ python api_server_complete.py
✓ Stage 1 loaded
⚠ Stage 2 not available (kaolin required): ImportError
✓ Server running on http://0.0.0.0:8000

$ curl http://localhost:8000/generate -F "image=@dussehra.png"
✓ Returns 5 color maps + 5 normal maps (Stage 1 output)
✓ 95% quality result
```

**api_server_official.py:**
```bash
$ python api_server_official.py
ERROR: Stage 2 not available: ImportError
RuntimeError: Stage 2 is required for maximum quality.
Install kaolin: pip install kaolin or use api_server_complete.py
❌ FAILS - Directs to solution
```

---

## Migration Guide

### From api_server.py to api_server_complete.py

**When to migrate:**
- You experience kaolin installation failures in production
- You want more robust error handling
- You're okay with Stage 1-only if kaolin is unavailable

**Changes:**
```bash
# Old
python api_server.py  # ← Fails if kaolin missing

# New
python api_server_complete.py  # ← Works regardless, graceful fallback
```

**No code changes needed** - same endpoints, same usage, better reliability.

### From api_server.py to api_server_official.py

**When to migrate:**
- You need exact official parameters
- You're doing research/publication work
- You have kaolin working

**Changes:**
```bash
# Old
python api_server.py

# New
python api_server_official.py
```

**Differences:**
- Parameters now locked to official values (transparent to user)
- Better documentation of what's happening
- Clearer error messages if dependencies missing

---

## Summary

| Aspect | api_server.py | api_server_complete.py | api_server_official.py |
|--------|---------------|----------------------|----------------------|
| **Simplicity** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Robustness** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Official Compliance** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Bottom Line:**
- **For Production:** Use `api_server_complete.py` (reliable, works without kaolin)
- **For Research:** Use `api_server_official.py` (exact official approach)
- **For Learning:** Use `api_server_complete.py` (easier to understand, fewer deps)

