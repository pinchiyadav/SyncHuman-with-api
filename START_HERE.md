# 🚀 SyncHuman - START HERE

Welcome to SyncHuman! This document will guide you through all available resources and get you up and running quickly.

## What is SyncHuman?

SyncHuman is a state-of-the-art system for **reconstructing 3D human bodies from single 2D photographs**. It synchronizes 2D and 3D diffusion models to generate high-quality multi-view predictions and 3D geometry.

**Paper:** [SyncHuman: Synchronizing 2D and 3D Diffusion Models for Single-view Human Reconstruction](https://arxiv.org/pdf/2510.07723)

---

## 🎯 Quick Decision Tree

### If you want to...

**Get up and running in 5 minutes:**
→ Read [QUICKSTART.md](QUICKSTART.md)

**Set up on a new machine automatically:**
→ Run `bash setup.sh`

**Install manually with detailed explanations:**
→ Read [SETUP_GUIDE.md](SETUP_GUIDE.md)

**Check what's been tested and verified:**
→ Read [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md)

**Use the Web API:**
→ Run `python api_server_stage1.py` then visit http://localhost:8000

**Understand the project structure:**
→ Read [README.md](README.md)

**Debug installation issues:**
→ See "Troubleshooting" section in [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## 📚 Documentation Overview

### Quick Start Documents (5-15 minutes)

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [QUICKSTART.md](QUICKSTART.md) | Get your first 3D model in 5 minutes | 5 min |
| [README.md](README.md) | Project overview and quick links | 5 min |

### Setup & Installation (30-60 minutes)

| Document | Purpose | Read Time |
|----------|---------|-----------|
| `bash setup.sh` | Fully automated setup script | 60 min (automatic) |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Manual step-by-step installation with explanations | 30 min |
| [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md) | Verification report of what works | 10 min |

### Usage Guides

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [API_USAGE.md](API_USAGE.md) | Using the REST API endpoints | 10 min |

---

## ⚡ Three Ways to Get Started

### Option 1: Automatic Setup (Recommended for new users)

```bash
# One command does everything!
bash setup.sh

# Then run inference
conda activate SyncHuman
export ATTN_BACKEND=xformers
python inference_OneStage.py
```

**Time:** ~60 minutes (mostly automated)
**Difficulty:** ⭐ Very Easy

---

### Option 2: Quick Setup (For experienced users)

```bash
# Follow the Quick Setup section in QUICKSTART.md
conda create -n SyncHuman python=3.10 -y
conda activate SyncHuman

# Install dependencies (see QUICKSTART.md)
pip install [dependencies]

# Download models and run
python download.py
python inference_OneStage.py
```

**Time:** ~30 minutes
**Difficulty:** ⭐⭐ Moderate

---

### Option 3: Detailed Manual Setup

Follow [SETUP_GUIDE.md](SETUP_GUIDE.md) step-by-step for complete control and understanding.

**Time:** ~45 minutes
**Difficulty:** ⭐⭐⭐ Detailed but thorough

---

## 🧪 Testing After Setup

### Test Stage 1 Inference
```bash
conda activate SyncHuman
export ATTN_BACKEND=xformers
python test_inference.py
```

### Test API Server
```bash
conda activate SyncHuman
python api_server_stage1.py &
# In another terminal:
python test_api.py
```

---

## 🎛️ Common Tasks

### Run Inference on Your Image
1. Prepare input image (PNG with transparent background recommended)
2. Edit `inference_OneStage.py` and change `image_path`
3. Run: `python inference_OneStage.py`
4. Check outputs in `outputs/OneStage/`

### Use the Web API
```bash
python api_server_stage1.py
# Visit http://localhost:8000
# Or use curl:
curl -X POST http://localhost:8000/generate \
  -F "image=@your_image.png"
```

### Generate High-Quality Results
See "Tips for Best Quality" in [SETUP_GUIDE.md](SETUP_GUIDE.md)

### Troubleshoot Installation
See "Troubleshooting" section in [SETUP_GUIDE.md](SETUP_GUIDE.md) or [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md)

---

## 📋 System Requirements

**Minimum (for Stage 1):**
- GPU: NVIDIA with 40GB VRAM
- OS: Linux (Ubuntu 20.04+)
- RAM: 32GB
- Disk: 20GB

**Recommended (for best performance):**
- GPU: NVIDIA H100 or A100 (48GB+)
- OS: Linux (Ubuntu 22.04)
- RAM: 64GB
- Disk: 50GB

---

## 🔍 Key Features

✅ **Stage 1 - Multi-view Generation:**
- Input: Single 2D image
- Output: 5 multi-view color predictions + 5 normal maps
- Time: ~1.5-2 minutes on A40 GPU
- Quality: High

✅ **Stage 2 - Geometry Refinement:** (Optional)
- Input: Stage 1 outputs
- Output: Final GLB 3D model or point clouds
- Time: ~2-3 minutes
- Quality: Very high
- Note: Requires kaolin

✅ **Web API:**
- REST endpoints for easy integration
- File upload or URL input
- JSON response with outputs
- Docker-ready

---

## 🐛 Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| "No module named X" | See SETUP_GUIDE.md Troubleshooting |
| Out of CUDA memory | Reduce `stage1_steps` to 30 |
| GPU not found | Check nvidia-smi and CUDA installation |
| API server won't start | Check port 8000 is available |
| Slow inference | GPU might be loading, be patient |

---

## 📖 Full Documentation Index

### Guides
- [QUICKSTART.md](QUICKSTART.md) - Quick start
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Comprehensive setup
- [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md) - Verification report
- [README.md](README.md) - Original project README
- [API_USAGE.md](API_USAGE.md) - API documentation
- [START_HERE.md](START_HERE.md) - This file

### Automation
- [setup.sh](setup.sh) - Automated setup script
- [env.sh](env.sh) - Environment activation script

### Code
- [inference_OneStage.py](inference_OneStage.py) - Stage 1 inference
- [inference_SecondStage.py](inference_SecondStage.py) - Stage 2 inference
- [api_server_stage1.py](api_server_stage1.py) - API server (Stage 1 only)
- [api_server.py](api_server.py) - API server (full, requires kaolin)

### Testing
- [test_inference.py](test_inference.py) - Stage 1 test
- [test_api.py](test_api.py) - API test
- [test_stage2.py](test_stage2.py) - Stage 2 test

---

## ✓ Verification Checklist

After setup, verify everything works:

- [ ] Conda environment created: `conda activate SyncHuman`
- [ ] PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- [ ] Models downloaded: `ls ckpts/OneStage/`
- [ ] Test inference: `python test_inference.py` (should show "✓ completed")
- [ ] API works: `python api_server_stage1.py` then `curl http://localhost:8000/health`

---

## 🚀 Next Steps

1. **Choose your setup method** (Automatic/Quick/Detailed)
2. **Follow the corresponding guide**
3. **Run the tests** to verify everything works
4. **Process your first image**
5. **Explore advanced features** in [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## 💡 Tips & Tricks

- **Faster inference:** Set `stage1_steps=30` in inference script
- **Better quality:** Set `stage1_steps=75`
- **Memory issues:** Close other GPU applications, reduce steps
- **API batch processing:** Submit multiple requests sequentially
- **Headless mode:** Use `nohup python api_server_stage1.py > server.log 2>&1 &`

---

## 📞 Support & Resources

- **GitHub Issues:** https://github.com/xishuxishu/SyncHuman/issues
- **Project Page:** https://xishuxishu.github.io/SyncHuman.github.io/
- **Paper:** https://arxiv.org/pdf/2510.07723
- **Model Weights:** https://huggingface.co/xishushu/SyncHuman

---

## 📄 Citation

If you use SyncHuman in your research, please cite:

```bibtex
@article{chen2025synchuman,
  title={SyncHuman: Synchronizing 2D and 3D Diffusion Models for Single-view Human Reconstruction},
  author={Chen, Wenyue and Li, Peng and Zheng, Wangguandong and Zhao, Chengfeng and Li, Mengfei and Zhu, Yaolong and Dou, Zhiyang and Wang, Ronggang and Liu, Yuan},
  journal={arXiv preprint arXiv:2510.07723},
  year={2025}
}
```

---

## 🎓 Learning Path

```
START_HERE
    ↓
QUICKSTART (5 min) → Automatic Setup (60 min)
    ↓
Run first inference
    ↓
SETUP_GUIDE (reference as needed)
    ↓
Explore API
    ↓
Advanced features
```

---

## ✨ What You Get

After completing setup, you'll have:

✓ Working SyncHuman environment
✓ Models downloaded and verified
✓ Ability to process images to 3D
✓ REST API for integration
✓ All documentation
✓ Test scripts for validation
✓ Automated setup script for future machines

---

## 🎯 Goals

- **Easiest possible setup** ← You are here!
- **No dependency conflicts**
- **Clear documentation**
- **Working out of the box**
- **Easy to troubleshoot**

---

**Ready to begin? Pick a guide above and get started!**

Questions? Check [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md) or [SETUP_GUIDE.md](SETUP_GUIDE.md) troubleshooting section.

---

*Last Updated: November 29, 2025*
*Version: 1.0 (Tested & Verified)*
*Status: ✓ Production Ready*
