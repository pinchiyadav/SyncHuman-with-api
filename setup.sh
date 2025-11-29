#!/bin/bash

################################################################################
# SyncHuman Automated Setup Script
#
# Installs SyncHuman and all dependencies on a fresh Linux machine
# Tested on: Ubuntu 20.04+, Python 3.10, CUDA 12.1
#
# Usage:
#   bash setup.sh                    # Full automatic installation
#   bash setup.sh --skip-models      # Skip downloading large models
#   bash setup.sh --env-only         # Only create environment
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="SyncHuman"
PYTHON_VERSION="3.10"
SKIP_MODELS=false
ENV_ONLY=false

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}SyncHuman Setup Script${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --env-only)
            ENV_ONLY=true
            SKIP_MODELS=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check for conda
echo -e "${YELLOW}Checking for Conda...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed. Please install Miniconda or Anaconda first.${NC}"
    echo "Visit: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html"
    exit 1
fi
echo -e "${GREEN}✓ Conda found: $(conda --version)${NC}"
echo ""

# Check for NVIDIA GPU
echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA GPU drivers not found.${NC}"
    echo "Please install NVIDIA drivers and CUDA toolkit."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo -e "${GREEN}✓ GPU found: $GPU_NAME ($GPU_MEMORY)${NC}"
echo ""

# Check VRAM requirement
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | grep -oP '\d+')
if [ "$VRAM" -lt 40000 ]; then
    echo -e "${YELLOW}Warning: GPU has ${VRAM}MB VRAM. Recommended minimum is 40GB.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Step 1: Create conda environment
echo -e "${BLUE}[1/7]${NC} ${YELLOW}Creating Conda Environment...${NC}"
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Removing...${NC}"
    conda remove -n "$ENV_NAME" --all -y > /dev/null 2>&1 || true
fi

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
echo -e "${GREEN}✓ Conda environment created${NC}"
echo ""

# Step 2: Install PyTorch
echo -e "${BLUE}[2/7]${NC} ${YELLOW}Installing PyTorch 2.1.1 with CUDA 12.1...${NC}"
conda run -n "$ENV_NAME" conda install pytorch::pytorch torchvision pytorch::pytorch-cuda=12.1 -c pytorch -c nvidia -y > /dev/null 2>&1
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Step 3: Install core dependencies
echo -e "${BLUE}[3/7]${NC} ${YELLOW}Installing Core Dependencies...${NC}"
conda run -n "$ENV_NAME" pip install -q \
    accelerate safetensors==0.4.5 diffusers==0.29.1 transformers==4.36.0 \
    xformers \
    trimesh open3d omegaconf imageio imageio-ffmpeg rembg \
    plyfile scikit-image scipy scikit-learn pyyaml \
    spconv-cu121 \
    onnxruntime onnx einops pyvista PyMeshFix igraph pillow opencv-python pydantic \
    utils3d xatlas ninja easydict peft moviepy

echo -e "${GREEN}✓ Core dependencies installed${NC}"
echo ""

# Step 4: Install NVIDIA libraries
echo -e "${BLUE}[4/7]${NC} ${YELLOW}Installing NVIDIA Libraries (nvdiffrast)...${NC}"
if [ ! -d "/tmp/nvdiffrast" ]; then
    cd /tmp
    git clone https://github.com/NVlabs/nvdiffrast.git > /dev/null 2>&1
    cd nvdiffrast
    conda run -n "$ENV_NAME" pip install -e . -q > /dev/null 2>&1
    cd /workspace/SyncHuman
else
    cd /tmp/nvdiffrast
    conda run -n "$ENV_NAME" pip install -e . -q > /dev/null 2>&1
fi
echo -e "${GREEN}✓ nvdiffrast installed${NC}"
echo ""

# Step 5: Download model checkpoints
if [ "$ENV_ONLY" = true ]; then
    echo -e "${YELLOW}Skipping model download (--env-only flag)${NC}"
    echo ""
elif [ "$SKIP_MODELS" = true ]; then
    echo -e "${YELLOW}Skipping model download (--skip-models flag)${NC}"
    echo ""
else
    echo -e "${BLUE}[5/7]${NC} ${YELLOW}Downloading Model Checkpoints (~8.5GB)...${NC}"
    echo "This may take 20-30 minutes..."
    conda run -n "$ENV_NAME" python download.py > /dev/null 2>&1
    echo -e "${GREEN}✓ Models downloaded${NC}"
    echo ""
fi

# Step 6: Set environment variables
echo -e "${BLUE}[6/7]${NC} ${YELLOW}Creating Environment Script...${NC}"
cat > /workspace/SyncHuman/env.sh << 'ENVEOF'
#!/bin/bash
# SyncHuman Environment Activation Script
conda activate SyncHuman
export ATTN_BACKEND=xformers
echo "✓ SyncHuman environment activated"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Attention Backend: $ATTN_BACKEND"
ENVEOF

chmod +x /workspace/SyncHuman/env.sh
echo -e "${GREEN}✓ Environment script created at env.sh${NC}"
echo ""

# Step 7: Verification
echo -e "${BLUE}[7/7]${NC} ${YELLOW}Verifying Installation...${NC}"
echo "Checking Python environment..."
PYTHON_CHECK=$(conda run -n "$ENV_NAME" python --version 2>&1)
echo -e "${GREEN}  ✓ $PYTHON_CHECK${NC}"

echo "Checking PyTorch..."
TORCH_CHECK=$(conda run -n "$ENV_NAME" python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1)
echo -e "${GREEN}  ✓ $TORCH_CHECK${NC}"

echo "Checking xformers..."
XFORM_CHECK=$(conda run -n "$ENV_NAME" python -c "import xformers; print('xformers available')" 2>&1)
echo -e "${GREEN}  ✓ $XFORM_CHECK${NC}"

echo "Checking CUDA..."
CUDA_CHECK=$(conda run -n "$ENV_NAME" python -c "import torch; print(f'CUDA {torch.version.cuda}')" 2>&1)
echo -e "${GREEN}  ✓ $CUDA_CHECK${NC}"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "To activate the environment, run:"
echo -e "  ${BLUE}conda activate $ENV_NAME${NC}"
echo ""
echo "Or use the environment script:"
echo -e "  ${BLUE}source env.sh${NC}"
echo ""
echo "To run Stage 1 inference:"
echo -e "  ${BLUE}export ATTN_BACKEND=xformers${NC}"
echo -e "  ${BLUE}python inference_OneStage.py${NC}"
echo ""
echo "To start the API server:"
echo -e "  ${BLUE}export ATTN_BACKEND=xformers${NC}"
echo -e "  ${BLUE}python api_server.py${NC}"
echo ""
echo "For more information, see SETUP_GUIDE.md"
echo ""
