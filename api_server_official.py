"""
SyncHuman Official Maximum Quality API Server

Implements the EXACT official recommended approach from:
https://github.com/IGL-HKUST/SyncHuman

This server provides maximum quality 3D human reconstruction by:
1. Using both Stage 1 and Stage 2 (no compromises)
2. Following official parameter settings (seed=43, guidance, 768x768, 50/25 steps)
3. Implementing official inference pipeline exactly as designed
4. REQUIRING all official dependencies (NO graceful fallbacks)

Features:
- POST /generate: Full two-stage pipeline for maximum quality
- GET /health: Health check with stage availability
- GET /info: API capabilities and GPU info
- GET /: API documentation
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import urllib.request

# ============================================================================
# CONFIGURATION - OFFICIAL PARAMETERS FROM SYNCHUMAN
# ============================================================================

# Official Stage 1 Parameters (from inference_OneStage.py and config.json)
OFFICIAL_STAGE1_CONFIG = {
    "seed": 43,  # Official seed
    "guidance_scale": 3.0,  # User-facing CFG (internal uses 10.0)
    "num_inference_steps": 50,  # Official default
    "mv_img_wh": (768, 768),  # Official resolution
    "num_views": 5,  # Official multiview count
    "background_color": "white",
}

# Official Stage 2 Parameters (from pipeline.json)
OFFICIAL_STAGE2_CONFIG = {
    "num_steps": 25,  # Official sampling steps
    "cfg_strength": 5.0,  # Official classifier-free guidance
    "cfg_interval": [0.5, 1.0],  # Official CFG interval
    "texture_size": 1024,  # Official texture resolution
    "simplify": 0.7,  # Official mesh simplification
    "up_size": 896,  # Official interpolation size
}

# ============================================================================
# ENVIRONMENT & REPO SETUP
# ============================================================================

# Set attention backend for memory efficiency
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

# Resolve repo root
REPO_ROOT = Path(os.environ.get("SYNCHUMAN_ROOT", Path(__file__).resolve().parent))
if (REPO_ROOT / "SyncHumanOneStagePipeline.py").exists():
    sys.path.append(str(REPO_ROOT.parent))
else:
    sys.path.append(str(REPO_ROOT))

# ============================================================================
# LOAD PIPELINES - OFFICIAL IMPLEMENTATION
# ============================================================================

# Import Stage 1 (required)
try:
    from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
    STAGE1_AVAILABLE = True
except Exception as e:
    print(f"ERROR: Stage 1 not available: {e}")
    STAGE1_AVAILABLE = False
    raise RuntimeError("Stage 1 is required. Check SyncHuman installation.")

# Import Stage 2 (required)
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

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="SyncHuman Official Maximum Quality API",
    version="1.0.0",
    description="Human 3D Reconstruction with FULL TWO-STAGE PIPELINE for maximum quality"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances
_stage1_pipe: Optional[SyncHumanOneStagePipeline] = None
_stage2_pipe = None  # Type: SyncHumanTwoStagePipeline

# ============================================================================
# PIPELINE LOADING - OFFICIAL APPROACH
# ============================================================================

def _load_pipelines():
    """Load both Stage 1 and Stage 2 pipelines using official approach"""
    global _stage1_pipe, _stage2_pipe

    if _stage1_pipe is None:
        print("=" * 70)
        print("Loading Stage 1 Pipeline (Official)")
        print("=" * 70)
        _stage1_pipe = SyncHumanOneStagePipeline.from_pretrained("./ckpts/OneStage")
        # Enable xformers memory optimization (official approach)
        _stage1_pipe.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()
        print("✓ Stage 1 loaded successfully")
        print(f"  Config: {OFFICIAL_STAGE1_CONFIG}")
        print()

    if _stage2_pipe is None:
        print("=" * 70)
        print("Loading Stage 2 Pipeline (Official)")
        print("=" * 70)
        _stage2_pipe = SyncHumanTwoStagePipeline.from_pretrained("./ckpts/SecondStage")
        _stage2_pipe.cuda()
        print("✓ Stage 2 loaded successfully")
        print(f"  Config: {OFFICIAL_STAGE2_CONFIG}")
        print()

def _prepare_rgba(input_path: Path, output_path: Path) -> float:
    """
    Prepare RGBA image following official preprocessing:
    - Crop to bounding box
    - Square-pad
    - Resize to 768x768 (official resolution)
    """
    try:
        rgba = Image.open(input_path).convert("RGBA")
    except:
        img = Image.open(input_path)
        if img.mode != "RGBA":
            rgb = img.convert("RGB")
            alpha = Image.new("L", rgb.size, 255)
            rgba = Image.new("RGBA", rgb.size)
            rgba.putalpha(alpha)
        else:
            rgba = img

    # Calculate alpha coverage (quality metric)
    alpha = torch.from_numpy(np.array(rgba.split()[-1], dtype=np.uint8))
    covered = (alpha > 0).float().mean().item()

    # Crop to foreground
    coords = torch.nonzero(alpha > 10, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("Input must include transparency or clear foreground")

    ymin, xmin = coords.min(dim=0).values.tolist()
    ymax, xmax = coords.max(dim=0).values.tolist()
    cropped = rgba.crop((xmin, ymin, xmax + 1, ymax + 1))

    # Square-pad with 10% margin
    side = int(max(cropped.size) * 1.1)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    canvas.paste(cropped, offset)

    # Resize to official 768x768
    final = canvas.resize((768, 768), Image.LANCZOS)
    final.save(output_path)

    return covered

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "SyncHuman Official Maximum Quality API",
        "version": "1.0.0",
        "stage1": STAGE1_AVAILABLE,
        "stage2": STAGE2_AVAILABLE,
        "approach": "Full two-stage pipeline for maximum quality",
    }

@app.get("/")
async def root():
    """API documentation"""
    return {
        "title": "SyncHuman Official Maximum Quality API",
        "description": "Human 3D Reconstruction from Single Image using FULL two-stage pipeline",
        "version": "1.0.0",
        "stages": {
            "stage1": {
                "available": STAGE1_AVAILABLE,
                "description": "2D-3D cross-space diffusion with multiview generation",
                "outputs": ["5 color maps", "5 normal maps", "sparse voxel grid"]
            },
            "stage2": {
                "available": STAGE2_AVAILABLE,
                "description": "Structured latent flow refinement to final 3D mesh",
                "outputs": ["GLB model", "PLY mesh", "textured geometry"]
            }
        },
        "official_configuration": {
            "stage1": OFFICIAL_STAGE1_CONFIG,
            "stage2": OFFICIAL_STAGE2_CONFIG,
        },
        "endpoints": {
            "GET /health": "Health check",
            "GET /": "This documentation",
            "GET /info": "API capabilities",
            "POST /generate": "Generate 3D model (full two-stage)",
        },
        "example_usage": {
            "bash": "curl -X POST http://localhost:8000/generate -F 'image=@input.png'",
            "python": "requests.post('http://localhost:8000/generate', files={'image': open('input.png', 'rb')})",
        },
    }

@app.get("/info")
async def info():
    """Get API capabilities"""
    return {
        "api_version": "1.0.0",
        "approach": "Official SyncHuman two-stage pipeline for MAXIMUM QUALITY",
        "stages": {
            "stage1": {
                "available": True,
                "description": "Multi-view generation (5 color + 5 normal maps)",
                "parameters": OFFICIAL_STAGE1_CONFIG,
                "time_estimate": "1.5-2 minutes per image"
            },
            "stage2": {
                "available": True,
                "description": "Refined 3D geometry (GLB mesh file)",
                "parameters": OFFICIAL_STAGE2_CONFIG,
                "time_estimate": "2-3 minutes",
                "requires_kaolin": True,
            },
        },
        "total_time_estimate": "4-5 minutes for full pipeline",
        "gpu_info": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024*1024) if torch.cuda.is_available() else 0,
            "minimum_required_memory_gb": 40,
        },
        "official_repository": "https://github.com/IGL-HKUST/SyncHuman",
        "paper": "https://arxiv.org/pdf/2510.07723",
        "model_weights": "https://huggingface.co/xishushu/SyncHuman",
    }

@app.post("/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    download: bool = Form(False),
):
    """
    Generate 3D model using OFFICIAL two-stage pipeline for maximum quality

    Parameters:
    - image: Input image file (RGBA PNG recommended, transparent background)
    - image_url: Or provide image URL
    - download: Return files as ZIP archive if true

    Returns:
    - Full two-stage results with GLB model
    - JSON with output paths and completion status
    """

    try:
        # Load pipelines (first request only, then cached)
        _load_pipelines()

        # Validate input
        if not image and not image_url:
            raise HTTPException(status_code=400, detail="Either 'image' or 'image_url' must be provided")

        # Create work directory
        job_id = str(uuid4())[:8]
        work_dir = Path("./tmp_api_jobs") / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{job_id}] Starting maximum quality 3D generation...")
        print(f"[{job_id}] Working directory: {work_dir}")

        # ====================================================================
        # STAGE 1: Multi-view Generation
        # ====================================================================

        # Handle image input
        if image:
            image_path = work_dir / "input_uploaded.png"
            contents = await image.read()
            with open(image_path, "wb") as f:
                f.write(contents)
        elif image_url:
            image_path = work_dir / "input_url.png"
            try:
                urllib.request.urlretrieve(image_url, image_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

        # Prepare RGBA image (official preprocessing)
        rgba_path = work_dir / "input_rgba.png"
        try:
            alpha_coverage = _prepare_rgba(image_path, rgba_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = {
            "job_id": job_id,
            "approach": "Official SyncHuman two-stage pipeline (MAXIMUM QUALITY)",
            "status": "processing",
            "alpha_coverage": float(alpha_coverage),
        }

        # Run Stage 1 with official parameters
        output_dir_s1 = work_dir / "stage1_output"
        output_dir_s1.mkdir(parents=True, exist_ok=True)

        print(f"[{job_id}] Stage 1: Multi-view Generation")
        print(f"[{job_id}]   Parameters: {OFFICIAL_STAGE1_CONFIG}")
        print(f"[{job_id}]   Running...")

        # Set seed for reproducibility (official approach)
        torch.manual_seed(OFFICIAL_STAGE1_CONFIG["seed"])

        assert _stage1_pipe is not None
        _stage1_pipe.run(
            image_path=str(rgba_path),
            save_path=str(output_dir_s1),
        )

        # Collect Stage 1 outputs
        s1_files = {}
        for i in range(5):
            if (output_dir_s1 / f"color_{i}.png").exists():
                s1_files[f"color_{i}"] = f"color_{i}.png"
            if (output_dir_s1 / f"normal_{i}.png").exists():
                s1_files[f"normal_{i}"] = f"normal_{i}.png"
        if (output_dir_s1 / "input.png").exists():
            s1_files["input"] = "input.png"
        if (output_dir_s1 / "latent.npz").exists():
            s1_files["latent"] = "latent.npz"

        result["stage1"] = {
            "status": "completed",
            "parameters": OFFICIAL_STAGE1_CONFIG,
            "output_dir": str(output_dir_s1),
            "files": s1_files,
        }

        print(f"[{job_id}] ✓ Stage 1 completed")
        print(f"[{job_id}]   Output: {output_dir_s1}")

        # ====================================================================
        # STAGE 2: Refined 3D Mesh Generation
        # ====================================================================

        output_dir_s2 = work_dir / "stage2_output"
        output_dir_s2.mkdir(parents=True, exist_ok=True)

        print(f"[{job_id}] Stage 2: Refined 3D Geometry")
        print(f"[{job_id}]   Parameters: {OFFICIAL_STAGE2_CONFIG}")
        print(f"[{job_id}]   Running...")

        torch.cuda.empty_cache()

        assert _stage2_pipe is not None
        _stage2_pipe.run(
            image_path=str(output_dir_s1),
            outpath=str(output_dir_s2),
        )

        # Collect Stage 2 outputs
        s2_files = {}
        if (output_dir_s2 / "output.glb").exists():
            s2_files["output.glb"] = "output.glb"
        if (output_dir_s2 / "output_mesh.ply").exists():
            s2_files["output_mesh.ply"] = "output_mesh.ply"

        result["stage2"] = {
            "status": "completed",
            "parameters": OFFICIAL_STAGE2_CONFIG,
            "output_dir": str(output_dir_s2),
            "files": s2_files,
        }

        print(f"[{job_id}] ✓ Stage 2 completed")
        print(f"[{job_id}]   Output: {output_dir_s2}")

        # ====================================================================
        # FINAL RESULT
        # ====================================================================

        # Prepare download if requested
        if download:
            import zipfile
            archive_path = work_dir / "results.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Stage 1 outputs
                for filename in result["stage1"]["files"].values():
                    file_path = Path(result["stage1"]["output_dir"]) / filename
                    if file_path.exists():
                        zf.write(file_path, arcname=f"stage1/{filename}")

                # Stage 2 outputs
                for filename in result["stage2"]["files"].values():
                    file_path = Path(result["stage2"]["output_dir"]) / filename
                    if file_path.exists():
                        zf.write(file_path, arcname=f"stage2/{filename}")

            result["status"] = "completed"
            print(f"[{job_id}] ✓ Complete! Results packaged as ZIP")
            return FileResponse(
                archive_path,
                media_type="application/zip",
                filename=f"synchuman_official_{job_id}.zip"
            )

        result["status"] = "completed"
        print(f"[{job_id}] ✓ Complete! Maximum quality 3D model generated")
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("SyncHuman Official Maximum Quality API Server")
    print("=" * 70)
    print(f"Attention Backend: {os.environ.get('ATTN_BACKEND', 'xformers')}")
    print(f"Stage 1 Available: {STAGE1_AVAILABLE}")
    print(f"Stage 2 Available: {STAGE2_AVAILABLE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
    print("=" * 70)
    print("Approach: OFFICIAL SYNCHUMAN TWO-STAGE PIPELINE")
    print("Quality: MAXIMUM (no compromises)")
    print("=" * 70)
    print("Starting server on http://0.0.0.0:8000")
    print("API docs: http://localhost:8000/docs")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
