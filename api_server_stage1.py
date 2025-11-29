"""
FastAPI Server for SyncHuman Stage 1 (Multi-view Generation)

This is a Stage 1-only API variant that doesn't require kaolin or Stage 2 dependencies.
Use this for quick 3D multi-view generation without the final mesh.

Features:
- POST /generate: Upload image and get multi-view color and normal predictions
- Returns JSON with file paths
- Can optionally stream the result archive
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
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Resolve repo root
REPO_ROOT = Path(os.environ.get("SYNCHUMAN_ROOT", Path(__file__).resolve().parent))
if (REPO_ROOT / "SyncHumanOneStagePipeline.py").exists():
    sys.path.append(str(REPO_ROOT.parent))
else:
    sys.path.append(str(REPO_ROOT))

try:
    from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
    from SyncHuman.utils.inference_utils import save_coords_to_npz, save_images, add_margin
except Exception as exc:
    raise RuntimeError("Failed to import SyncHuman modules. Set SYNCHUMAN_ROOT to the repo path.") from exc

DEFAULT_STAGE1_STEPS = 50

# Set default attention backend to xformers
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

app = FastAPI(
    title="SyncHuman Stage 1 API",
    version="0.1.0",
    description="Multi-view Human 3D Generation from Single Image"
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

def _load_pipeline():
    global _stage1_pipe
    if _stage1_pipe is None:
        print("Loading Stage 1 pipeline...")
        _stage1_pipe = SyncHumanOneStagePipeline.from_pretrained("./ckpts/OneStage")
        _stage1_pipe.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()
        print("✓ Pipeline loaded")

def _prepare_rgba(input_path: Path, output_path: Path) -> float:
    """
    Prepare RGBA image: crop to bbox, square-pad, resize to 768x768
    """
    try:
        rgba = Image.open(input_path).convert("RGBA")
    except:
        # If conversion fails, assume it's already RGBA or convert from RGB
        img = Image.open(input_path)
        if img.mode != "RGBA":
            # Convert RGB to RGBA with opaque alpha
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
        raise ValueError("Input must include transparency or clear foreground; none detected.")

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
        "service": "SyncHuman Stage 1 API",
        "version": "0.1.0"
    }

@app.post("/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    stage1_steps: Optional[int] = Form(None),
    download: bool = Form(False),
):
    """
    Generate multi-view 3D predictions from a single image

    Parameters:
    - image: Input image file (RGBA PNG recommended)
    - image_url: Or provide image as URL
    - stage1_steps: Number of diffusion steps (default: 50)
    - download: Return archive as file if true
    """

    try:
        # Load pipeline
        _load_pipeline()
        assert _stage1_pipe is not None

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
            import urllib.request
            image_path = work_dir / "input_url.png"
            try:
                urllib.request.urlretrieve(image_url, image_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")

        # Prepare RGBA image
        rgba_path = work_dir / "input_rgba.png"
        try:
            alpha_coverage = _prepare_rgba(image_path, rgba_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Run Stage 1 inference
        num_steps = stage1_steps or DEFAULT_STAGE1_STEPS
        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(42)

        print(f"[{job_id}] Running Stage 1 inference with {num_steps} steps...")
        _stage1_pipe.run(
            image_path=str(rgba_path),
            save_path=str(output_dir),
        )

        # Collect output files
        output_files = {}
        if (output_dir / "input.png").exists():
            output_files["input"] = "input.png"

        for i in range(5):
            if (output_dir / f"color_{i}.png").exists():
                output_files[f"color_{i}"] = f"color_{i}.png"
            if (output_dir / f"normal_{i}.png").exists():
                output_files[f"normal_{i}"] = f"normal_{i}.png"

        if (output_dir / "coordinates.npz").exists():
            output_files["coordinates"] = "coordinates.npz"

        # Prepare response
        result = {
            "status": "success",
            "job_id": job_id,
            "alpha_coverage": float(alpha_coverage),
            "num_steps": num_steps,
            "output_dir": str(output_dir),
            "files": output_files,
        }

        # If download requested, create archive
        if download:
            import zipfile
            archive_path = work_dir / "results.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for filename in output_files.values():
                    file_path = output_dir / filename
                    if file_path.exists():
                        zf.write(file_path, arcname=filename)

            # Return as file
            return FileResponse(
                archive_path,
                media_type="application/zip",
                filename=f"synchuman_results_{job_id}.zip"
            )

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """API documentation"""
    return {
        "title": "SyncHuman Stage 1 API",
        "description": "Multi-view Human 3D Generation from Single Image (Stage 1 Only)",
        "endpoints": {
            "GET /health": "Health check",
            "GET /": "This documentation",
            "POST /generate": "Generate multi-view predictions from image",
        },
        "example_usage": {
            "bash": "curl -X POST http://localhost:8000/generate -F 'image=@input.png' -F 'stage1_steps=50'",
            "python": "requests.post('http://localhost:8000/generate', files={'image': open('input.png', 'rb')}, data={'stage1_steps': 50})",
        },
        "notes": [
            "Input image should be RGBA PNG with transparent background",
            "Recommended size: 512x512 to 1024x1024",
            "Processing time: 1-2 minutes per image on A40 GPU",
            "Output includes 5 multi-view color and normal predictions",
        ]
    }

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("SyncHuman Stage 1 API Server")
    print("=" * 60)
    print(f"Attention Backend: {os.environ.get('ATTN_BACKEND', 'xformers')}")
    print("Starting server on http://0.0.0.0:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
