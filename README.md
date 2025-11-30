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

## 🎯 Unified API - One Server, Three Modes

The **single official API** with flexible configuration:

```bash
# Default: Maximum official quality (Stage 1 + Stage 2 with kaolin)
python api_server.py

# Fast mode: No kaolin needed (Stage 1 only, 1.5-2 min)
python api_server.py --stage1-only

# Production: Always works (falls back gracefully)
python api_server.py --graceful-fallback

# Custom quality (adjust for your needs)
python api_server.py --stage1-steps=75 --stage2-steps=35
```

**Full API documentation:** [API.md](API.md) - Complete reference with examples, flags, and troubleshooting

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

Start the API server:
```bash
export ATTN_BACKEND=xformers
python api_server_stage1.py
```

Then use it:
```bash
# Upload image and get results
curl -X POST http://localhost:8000/generate \
  -F "image=@input.png" \
  -F "stage1_steps=50"

# Visit http://localhost:8000 for interactive UI
```

## 📁 Project Structure

```
SyncHuman/
├── QUICKSTART.md                 # Quick start guide
├── SETUP_GUIDE.md               # Comprehensive setup guide
├── INSTALLATION_SUMMARY.md      # Verification report
├── setup.sh                     # Automated setup script
├── env.sh                       # Environment activation
├── api_server_stage1.py         # Stage 1-only API server (✓ recommended)
├── api_server.py                # Original API server
├── inference_OneStage.py        # Stage 1 inference
├── inference_SecondStage.py     # Stage 2 inference
├── test_inference.py            # Stage 1 test script
├── test_api.py                  # API test script
├── ckpts/                       # Model checkpoints
│   ├── OneStage/               # Stage 1 models
│   └── SecondStage/            # Stage 2 models
├── SyncHuman/                   # Main package
├── examples/                    # Example images
└── outputs/                     # Inference results
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
