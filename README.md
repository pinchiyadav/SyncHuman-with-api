<p align="center">
  <img src="assets/icon.png" alt="SyncHuman" width="60%>
</p>

<div align="center">

# Synchronizing 2D and 3D Generative Models for Single-view Human Reconstruction

</div>

<div align="center">

[Wenyue Chen](#)<sup>1</sup>, [Peng Li](https://penghtyx.github.io/yuki-lipeng/)<sup>2</sup>, [Wangguandong Zheng](https://wangguandongzheng.github.io/)<sup>3</sup>, [Chengfeng Zhao](https://afterjourney00.github.io/)<sup>2</sup>, [Mengfei Li](#)<sup>2</sup>, [Yaolong Zhu](#)<sup>1</sup>, [Zhiyang Dou](https://frank-zy-dou.github.io/)<sup>4</sup>, [Ronggang Wang](https://scholar.google.com/citations?user=CEEvb64AAAAJ&hl)<sup>1</sup>, [Yuan Liu](https://liuyuan-pal.github.io/)<sup>2</sup>

<sup>1</sup> Peking University
<sup>2</sup> The Hong Kong University of Science and Technology 
<sup>3</sup> Southeast University
<sup>4</sup> The University of Hong Kong

</div>

>  **Official code of SyncHuman: Synchronizing 2D and 3D Generative Models for Single-view Human Reconstruction**

<div align="center">
<a href='https://arxiv.org/pdf/2510.07723'><img src='https://img.shields.io/badge/arXiv-2510.07723-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://xishuxishu.github.io/SyncHuman.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/xishushu/SyncHuman"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>


## ⚡ Quick Start

### Automatic Setup (Recommended)
For a fully automated setup on a fresh machine, use our setup script:

```bash
bash setup.sh
```

This will:
- ✓ Create conda environment with Python 3.10
- ✓ Install PyTorch with CUDA 12.1 support
- ✓ Install all dependencies
- ✓ Download model checkpoints (~8.5GB)
- ✓ Verify installation

**Setup time:** ~1 hour (mostly model download)

### Quick Test
```bash
# Activate environment (RECOMMENDED METHOD)
source /opt/conda/bin/activate SyncHuman

# Or if that doesn't work, try:
# conda activate SyncHuman

# Run Stage 1 inference
export ATTN_BACKEND=xformers
python inference_OneStage.py

# Or use the unified API
python api_server.py
# Visit http://localhost:8000
```

## 🎯 Unified API - One Server, Request-Based Configuration

The **single official API** with flexible per-request configuration:

```bash
# Start server (no flags needed)
python api_server.py

# Then make requests with flags (examples below)
```

**Usage Examples (pass flags in curl request):**

```bash
# 1. Maximum Quality (default - Stage 1 + Stage 2)
curl -X POST http://localhost:8000/generate \
  -F "image=@input.png"
# Output: Complete textured GLB model (4-5 min)

# 2. Fast Mode (Stage 1 only, no kaolin)
curl -X POST http://localhost:8000/generate \
  -F "image=@input.png" \
  -F "stage1_only=true"
# Output: Multiview maps (1.5-2 min)

# 3. Production Safe (graceful fallback)
curl -X POST http://localhost:8000/generate \
  -F "image=@input.png" \
  -F "graceful_fallback=true"
# Output: Full if kaolin available, Stage 1 otherwise

# 4. Custom Quality
curl -X POST http://localhost:8000/generate \
  -F "image=@input.png" \
  -F "stage1_steps=75" \
  -F "stage2_steps=35"
# Output: Adjust steps for quality/speed tradeoff
```

**Full API documentation:** [API.md](API.md) - Complete reference with all request parameters and troubleshooting

## 📚 Documentation

- **[API.md](API.md)** - Complete API reference with all commands, examples, and troubleshooting
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation and configuration
- **[INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md)** - Verification and performance metrics

## 🚀 Inference

### Stage 1: Multi-view Generation
```bash
export ATTN_BACKEND=xformers
python inference_OneStage.py
```

**Output (in `outputs/OneStage/`):**
- `color_0.png` to `color_4.png` - 5 multi-view color predictions
- `normal_0.png` to `normal_4.png` - 5 multi-view normal predictions
- `coordinates.npz` - Sparse 3D structure
- Processing time: ~1.5-2 minutes on A40 GPU

### Stage 2: Final Geometry (Optional)
```bash
export ATTN_BACKEND=xformers
python inference_SecondStage.py
```

**Output:**
- `outputs/SecondStage/output.glb` - Final 3D model
- Note: Requires kaolin (see SETUP_GUIDE.md for installation)

## 🌐 Web API

Start the unified API server (no flags needed):
```bash
export ATTN_BACKEND=xformers
python api_server.py
```

Then make requests with desired configuration:
```bash
# Visit http://localhost:8000 for interactive UI

# Or use curl with request-based flags:
curl -X POST http://localhost:8000/generate \
  -F "image=@input.png" \
  -F "stage1_only=true"
```

See [API.md](API.md) for all request parameters and examples.

## 📁 Project Structure

```
SyncHuman/
├── README.md                    # This file
├── API.md                       # Complete API reference
├── SETUP_GUIDE.md              # Detailed setup guide
├── INSTALLATION_SUMMARY.md     # Installation verification
├── setup.sh                    # Automated setup script
├── api_server.py               # ✓ Unified API server (one server, all modes)
├── inference_OneStage.py       # Stage 1 inference script
├── inference_SecondStage.py    # Stage 2 inference script
├── test_inference.py           # Test Stage 1
├── test_api.py                 # Test API server
├── ckpts/                      # Model checkpoints
│   ├── OneStage/              # Stage 1 models
│   └── SecondStage/           # Stage 2 models
├── SyncHuman/                  # Main package
├── examples/                   # Example images
└── outputs/                    # Inference results
```

## ✓ Tested & Verified

This installation has been tested on:
- **GPU:** NVIDIA A40 (46GB VRAM)
- **OS:** Linux (Ubuntu 20.04+)
- **CUDA:** 12.1
- **Python:** 3.10
- **PyTorch:** 2.5.1

**All tests passing:**
- ✓ Stage 1 inference
- ✓ API endpoint
- ✓ Multi-image batch processing
- ✓ GPU memory management

See [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md) for detailed test results.


## Ack
Our code is based on these wonderful works:
* **[TRELLIS](https://github.com/microsoft/TRELLIS)**
* **[PSHuman](https://github.com/pengHTYX/PSHuman)**



## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{chen2025synchuman,
  title={SyncHuman: Synchronizing 2D and 3D Diffusion Models for Single-view Human Reconstruction},
  author={Wenyue Chen, Peng Li, Wangguandong Zheng, Chengfeng Zhao, Mengfei Li, Yaolong Zhu, Zhiyang Dou, Ronggang Wang, Yuan Liu},
  journal={arXiv preprint arXiv:2510.07723},
  year={2025}
}
```
