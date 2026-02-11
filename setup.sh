#!/bin/bash

# Inflight Selfie Generator - Setup Script
# Automates the installation and configuration process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Banner
echo -e "${BLUE}"
cat << "EOF"
  ___        __ _ _       _     _     ____       _  __ _
 |_ _|_ __  / _| (_) __ _| |__ | |_  / ___|  ___| |/ _(_) ___
  | || '_ \| |_| | |/ _` | '_ \| __| \___ \ / _ \ | |_| |/ _ \
  | || | | |  _| | | (_| | | | | |_   ___) |  __/ |  _| |  __/
 |___|_| |_|_| |_|_|\__, |_| |_|\__| |____/ \___|_|_| |_|\___|
                    |___/
  ____                           _
 / ___| ___ _ __   ___ _ __ __ _| |_ ___  _ __
| |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
| |_| |  __/ | | |  __/ | | (_| | || (_) | |
 \____|\___|_| |_|\___|_|  \__,_|\__\___/|_|

EOF
echo -e "${NC}"

echo "Inflight Selfie Generator - Setup Script"
echo "=========================================="
echo ""

# Check Python version
print_step "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Not running in a virtual environment"
    echo "It's recommended to use a virtual environment"
    read -p "Create and activate virtual environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"

        echo ""
        print_warning "Please activate the virtual environment and run this script again:"
        echo "  source venv/bin/activate  # Linux/Mac"
        echo "  venv\\Scripts\\activate     # Windows"
        exit 0
    fi
fi

# Check GPU
print_step "Checking for GPU..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_success "GPU detected: $GPU_NAME"
    USE_GPU=true
else
    print_warning "No GPU detected - will run in CPU mode (slower)"
    USE_GPU=false
fi

# Install dependencies
print_step "Installing dependencies..."

cd backend

if [ "$USE_GPU" = true ]; then
    print_step "Installing with GPU support..."
    pip install -q -r requirements.txt
else
    print_step "Installing for CPU mode..."
    # Replace onnxruntime-gpu with onnxruntime
    cat requirements.txt | sed 's/onnxruntime-gpu/onnxruntime/g' | pip install -q -r /dev/stdin
fi

print_success "Dependencies installed"

# Create directories
print_step "Creating directory structure..."
mkdir -p models
mkdir -p ../dataset/images
mkdir -p ../dataset/face_refs
print_success "Directories created"

# Download models
print_step "Downloading models..."
cd ..
python3 scripts/download_models.py --models-dir backend/models

print_success "Base models downloaded"

# Create .env file
if [ ! -f backend/.env ]; then
    print_step "Creating .env file..."
    cp backend/.env.example backend/.env

    if [ "$USE_GPU" = true ]; then
        sed -i.bak 's/DEVICE=cuda/DEVICE=cuda/' backend/.env
    else
        sed -i.bak 's/DEVICE=cuda/DEVICE=cpu/' backend/.env
    fi

    rm -f backend/.env.bak
    print_success ".env file created"
else
    print_warning ".env file already exists, skipping"
fi

# Check for fine-tuned models
print_step "Checking for fine-tuned models..."
if [ -d "backend/models/scene_planner_lora" ]; then
    print_success "Fine-tuned scene planner found"
else
    print_warning "Fine-tuned scene planner not found"
    echo ""
    echo "To get the fine-tuned models:"
    echo "  1. Open notebooks/Inflight_Selfie_Training.ipynb in Google Colab"
    echo "  2. Run all cells to train the model (~40 minutes)"
    echo "  3. Download inflight_selfie_models.zip"
    echo "  4. Extract to backend/models/"
    echo ""
    echo "The system will work with base TinyLlama, but scene planning"
    echo "will be less optimized without fine-tuning."
    echo ""
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Train the scene planner (if not done):"
echo "   - Upload notebooks/Inflight_Selfie_Training.ipynb to Google Colab"
echo "   - Enable T4 GPU and run all cells"
echo "   - Download and extract models to backend/models/"
echo ""
echo "2. Start the server:"
echo "   cd backend"
echo "   uvicorn server:app --host 0.0.0.0 --port 8000"
echo ""
echo "3. Test the API:"
echo "   - Visit http://localhost:8000/docs"
echo "   - Or run: python3 scripts/test_api.py"
echo ""
echo "For more information:"
echo "   - Quick Start: QUICKSTART.md"
echo "   - Full README: README.md"
echo "   - API Docs: http://localhost:8000/docs (when server is running)"
echo ""

if [ "$USE_GPU" = false ]; then
    print_warning "Running in CPU mode - generation will be slow (5-10 min per image)"
    echo "For better performance, use a system with NVIDIA GPU"
    echo ""
fi

echo "=========================================="
echo -e "${BLUE}Happy generating!${NC} ✈️"
echo "=========================================="
