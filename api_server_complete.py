"""
Complete FastAPI Server for SyncHuman (Stage 1 and Stage 2)

Features:
- POST /generate: Upload image and generate 3D model
- Support for Stage 1 (multi-view) and Stage 2 (refined 3D)
- Graceful fallback if Stage 2 dependencies unavailable
- Returns JSON with output paths
- Optional GLB file download

This server intelligently detects available components and provides
the best functionality based on installed dependencies.
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
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import urllib.request

# Resolve repo root
REPO_ROOT = Path(os.environ.get("SYNCHUMAN_ROOT", Path(__file__).resolve().parent))
if (REPO_ROOT / "SyncHumanOneStagePipeline.py").exists():
    sys.path.append(str(REPO_ROOT.parent))
else:
    sys.path.append(str(REPO_ROOT))

# Set default attention backend
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

# Import Stage 1 (always available)
try:
    from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
    STAGE1_AVAILABLE = True
except Exception as e:
    print(f"Warning: Stage 1 not available: {e}")
    STAGE1_AVAILABLE = False

# Try to import Stage 2 (may fail if kaolin not installed)
STAGE2_AVAILABLE = False
try:
    from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline
    STAGE2_AVAILABLE = True
except Exception as e:
    print(f"Note: Stage 2 not available (kaolin required): {type(e).__name__}")
    print(f"      Continuing with Stage 1 only...")
    STAGE2_AVAILABLE = False

if not STAGE1_AVAILABLE:
    raise RuntimeError("Stage 1 must be available. Check SyncHuman installation.")

DEFAULT_STAGE1_STEPS = 50
DEFAULT_STAGE2_STEPS = 25

app = FastAPI(
    title="SyncHuman Complete API",
    version="0.2.0",
    description=f"Human 3D Reconstruction API (Stage1: ✓, Stage2: {'✓' if STAGE2_AVAILABLE else '✗ requires kaolin'})"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_stage1_pipe: Optional[SyncHumanOneStagePipeline] = None
_stage2_pipe = None  # Type is SyncHumanTwoStagePipeline if available

def _load_pipelines(stage: str = "both"):
    global _stage1_pipe, _stage2_pipe

    if stage in ["stage1", "both"] and _stage1_pipe is None:
        print("Loading Stage 1 pipeline...")
        _stage1_pipe = SyncHumanOneStagePipeline.from_pretrained("./ckpts/OneStage")
        _stage1_pipe.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()
        print("✓ Stage 1 loaded")

    if stage in ["stage2", "both"] and _stage2_pipe is None:
        if not STAGE2_AVAILABLE:
            print("⚠ Stage 2 not available (kaolin required)")
            return

        print("Loading Stage 2 pipeline...")
        _stage2_pipe = SyncHumanTwoStagePipeline.from_pretrained("./ckpts/SecondStage")
        _stage2_pipe.cuda()
        print("✓ Stage 2 loaded")

def _prepare_rgba(input_path: Path, output_path: Path) -> float:
    """Prepare RGBA image: crop to bbox, square-pad, resize to 768x768"""
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

    alpha = torch.from_numpy(np.array(rgba.split()[-1], dtype=np.uint8))
    covered = (alpha > 0).float().mean().item()

    coords = torch.nonzero(alpha > 10, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("Input must include transparency or clear foreground")

    ymin, xmin = coords.min(dim=0).values.tolist()
    ymax, xmax = coords.max(dim=0).values.tolist()
    cropped = rgba.crop((xmin, ymin, xmax + 1, ymax + 1))

    side = int(max(cropped.size) * 1.1)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    canvas.paste(cropped, offset)
    final = canvas.resize((768, 768), Image.LANCZOS)
    final.save(output_path)

    return covered

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "SyncHuman Complete API",
        "version": "0.2.0",
        "stage1": STAGE1_AVAILABLE,
        "stage2": STAGE2_AVAILABLE,
    }

@app.get("/")
async def root():
    """API documentation"""
    return {
        "title": "SyncHuman Complete API",
        "description": "Human 3D Reconstruction from Single Image",
        "available_stages": {
            "stage1": STAGE1_AVAILABLE,
            "stage2": STAGE2_AVAILABLE,
        },
        "endpoints": {
            "GET /health": "Health check",
            "GET /": "This documentation",
            "POST /generate": "Generate 3D model from image",
            "GET /info": "API capabilities",
        },
        "example_usage": {
            "bash": "curl -X POST http://localhost:8000/generate -F 'image=@input.png' -F 'stage=both'",
            "python": "requests.post('http://localhost:8000/generate', files={'image': open('input.png', 'rb')}, data={'stage': 'both'})",
        },
    }

@app.get("/info")
async def info():
    """Get API capabilities"""
    return {
        "api_version": "0.2.0",
        "stages": {
            "stage1": {
                "available": True,
                "description": "Multi-view generation (5 color + 5 normal maps)",
                "time_estimate": "1.5-2 minutes per image",
                "requires_kaolin": False,
            },
            "stage2": {
                "available": STAGE2_AVAILABLE,
                "description": "Refined 3D geometry (GLB mesh file)",
                "time_estimate": "2-3 minutes",
                "requires_kaolin": True,
                "note": "Not available" if not STAGE2_AVAILABLE else "Available"
            },
        },
        "gpu_info": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024*1024) if torch.cuda.is_available() else 0,
        },
    }

@app.post("/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    stage: str = Form("both"),
    stage1_steps: Optional[int] = Form(None),
    stage2_steps: Optional[int] = Form(None),
    download: bool = Form(False),
):
    """
    Generate 3D model from single image

    Parameters:
    - image: Input image file (RGBA PNG recommended)
    - image_url: Or provide image as URL
    - stage: "stage1", "stage2", or "both" (default: "both")
    - stage1_steps: Diffusion steps for Stage 1 (default: 50)
    - stage2_steps: Sampler steps for Stage 2 (default: 25)
    - download: Return files as archive if true
    """

    try:
        # Validate stage parameter
        if stage not in ["stage1", "stage2", "both"]:
            raise HTTPException(status_code=400, detail="stage must be 'stage1', 'stage2', or 'both'")

        if stage == "stage2" and not STAGE2_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail="Stage 2 not available. Install kaolin or use stage='stage1'"
            )

        # Load pipelines
        _load_pipelines(stage)

        # Validate input
        if not image and not image_url:
            raise HTTPException(status_code=400, detail="Either 'image' or 'image_url' must be provided")

        # Create work directory
        job_id = str(uuid4())[:8]
        work_dir = Path("./tmp_api_jobs") / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

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

        # Prepare RGBA image
        rgba_path = work_dir / "input_rgba.png"
        try:
            alpha_coverage = _prepare_rgba(image_path, rgba_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = {
            "job_id": job_id,
            "status": "processing",
            "alpha_coverage": float(alpha_coverage),
        }

        # Run Stage 1
        if stage in ["stage1", "both"]:
            num_steps = stage1_steps or DEFAULT_STAGE1_STEPS
            output_dir_s1 = work_dir / "stage1_output"
            output_dir_s1.mkdir(parents=True, exist_ok=True)

            print(f"[{job_id}] Running Stage 1 with {num_steps} steps...")
            torch.manual_seed(42)

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

            result["stage1"] = {
                "status": "completed",
                "steps": num_steps,
                "output_dir": str(output_dir_s1),
                "files": s1_files,
            }

        # Run Stage 2
        if stage in ["stage2", "both"]:
            if not STAGE2_AVAILABLE:
                result["stage2"] = {
                    "status": "skipped",
                    "reason": "kaolin not installed",
                }
            else:
                num_steps = stage2_steps or DEFAULT_STAGE2_STEPS
                output_dir_s2 = work_dir / "stage2_output"
                output_dir_s2.mkdir(parents=True, exist_ok=True)

                print(f"[{job_id}] Running Stage 2 with {num_steps} steps...")
                torch.cuda.empty_cache()

                assert _stage2_pipe is not None
                _stage2_pipe.run(
                    image_path=str(output_dir_s1) if stage == "both" else None,
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
                    "steps": num_steps,
                    "output_dir": str(output_dir_s2),
                    "files": s2_files,
                }

        # Prepare download if requested
        if download:
            import zipfile
            archive_path = work_dir / "results.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                if "stage1" in result and result["stage1"].get("status") == "completed":
                    for filename in result["stage1"]["files"].values():
                        file_path = Path(result["stage1"]["output_dir"]) / filename
                        if file_path.exists():
                            zf.write(file_path, arcname=f"stage1/{filename}")

                if "stage2" in result and result["stage2"].get("status") == "completed":
                    for filename in result["stage2"]["files"].values():
                        file_path = Path(result["stage2"]["output_dir"]) / filename
                        if file_path.exists():
                            zf.write(file_path, arcname=f"stage2/{filename}")

            result["status"] = "completed"
            return FileResponse(
                archive_path,
                media_type="application/zip",
                filename=f"synchuman_results_{job_id}.zip"
            )

        result["status"] = "completed"
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("SyncHuman Complete API Server")
    print("=" * 70)
    print(f"Attention Backend: {os.environ.get('ATTN_BACKEND', 'xformers')}")
    print(f"Stage 1 Available: {STAGE1_AVAILABLE}")
    print(f"Stage 2 Available: {STAGE2_AVAILABLE}")
    if STAGE2_AVAILABLE:
        print("✓ Full Stage 1 + Stage 2 pipeline")
    else:
        print("⚠ Stage 1 only (kaolin not installed for Stage 2)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    print("Starting server on http://0.0.0.0:8000")
    print("API docs: http://localhost:8000/docs")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)
