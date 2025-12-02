#!/bin/bash
# =============================================================================
# AutoReject Environment Setup Script
# =============================================================================
# This script sets up a Python virtual environment with all dependencies
# for running AutoReject with GPU acceleration.
#
# Usage:
#   ./scripts/setup_environment.sh [--compute-canada] [--cuda] [--mps]
#
# Options:
#   --compute-canada  : Setup for Compute Canada HPC clusters (Narval, Beluga, etc.)
#   --cuda            : Install CUDA-enabled PyTorch (default on Compute Canada)
#   --mps             : Install MPS-enabled PyTorch for Apple Silicon (default on macOS)
#   --cpu-only        : Install CPU-only PyTorch
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# =============================================================================
# Parse arguments
# =============================================================================
COMPUTE_CANADA=false
TORCH_BACKEND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --compute-canada)
            COMPUTE_CANADA=true
            shift
            ;;
        --cuda)
            TORCH_BACKEND="cuda"
            shift
            ;;
        --mps)
            TORCH_BACKEND="mps"
            shift
            ;;
        --cpu-only)
            TORCH_BACKEND="cpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect backend if not specified
if [ -z "$TORCH_BACKEND" ]; then
    if [ "$COMPUTE_CANADA" = true ]; then
        TORCH_BACKEND="cuda"
    elif [[ "$(uname)" == "Darwin" ]]; then
        # macOS - check for Apple Silicon
        if [[ "$(uname -m)" == "arm64" ]]; then
            TORCH_BACKEND="mps"
        else
            TORCH_BACKEND="cpu"
        fi
    elif command -v nvidia-smi &> /dev/null; then
        TORCH_BACKEND="cuda"
    else
        TORCH_BACKEND="cpu"
    fi
fi

print_step "Detected/Selected backend: $TORCH_BACKEND"

# =============================================================================
# Get script directory (works even when script is sourced)
# =============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"
print_step "Working directory: $PROJECT_ROOT"

# =============================================================================
# Compute Canada Setup
# =============================================================================
if [ "$COMPUTE_CANADA" = true ]; then
    print_step "Setting up for Compute Canada..."
    
    # Load required modules
    print_step "Loading modules..."
    module purge
    module load StdEnv/2023
    module load python/3.11
    module load scipy-stack/2024a
    
    if [ "$TORCH_BACKEND" = "cuda" ]; then
        module load cuda/12.2
        print_success "Loaded CUDA module"
    fi
    
    print_success "Modules loaded"
    
    # Create virtual environment
    VENV_DIR="$PROJECT_ROOT/.venv"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_step "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        print_step "Creating virtual environment..."
        python -m venv --system-site-packages "$VENV_DIR"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_step "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install PyTorch for CUDA
    if [ "$TORCH_BACKEND" = "cuda" ]; then
        print_step "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
        print_success "PyTorch with CUDA installed"
    else
        print_step "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
        print_success "PyTorch (CPU) installed"
    fi

# =============================================================================
# Local Setup (macOS / Linux)
# =============================================================================
else
    print_step "Setting up local environment..."
    
    # Check Python version
    PYTHON_CMD=""
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        print_error "Python 3 not found. Please install Python 3.10 or higher."
        exit 1
    fi
    
    print_step "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
    
    # Create virtual environment
    VENV_DIR="$PROJECT_ROOT/.venv"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_step "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        print_step "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_step "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install PyTorch based on backend
    case $TORCH_BACKEND in
        mps)
            print_step "Installing PyTorch with MPS support (Apple Silicon)..."
            pip install torch torchvision torchaudio --quiet
            print_success "PyTorch with MPS installed"
            ;;
        cuda)
            print_step "Installing PyTorch with CUDA support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
            print_success "PyTorch with CUDA installed"
            ;;
        cpu)
            print_step "Installing PyTorch (CPU only)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
            print_success "PyTorch (CPU) installed"
            ;;
    esac
fi

# =============================================================================
# Install Common Dependencies
# =============================================================================
print_step "Installing MNE-Python and dependencies..."
pip install mne --quiet
print_success "MNE-Python installed"

print_step "Installing development dependencies..."
pip install pytest pytest-cov joblib scikit-learn matplotlib --quiet
print_success "Development dependencies installed"

# Install autoreject in development mode
print_step "Installing autoreject in development mode..."
pip install -e "$PROJECT_ROOT" --quiet
print_success "autoreject installed"

# =============================================================================
# Verify Installation
# =============================================================================
print_step "Verifying installation..."

python -c "
import torch
import mne
import autoreject

print(f'  PyTorch version: {torch.__version__}')
print(f'  MNE version: {mne.__version__}')
print(f'  AutoReject version: {autoreject.__version__}')

# Check GPU availability
if torch.cuda.is_available():
    print(f'  CUDA available: Yes ({torch.cuda.get_device_name(0)})')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  MPS available: Yes (Apple Silicon)')
else:
    print(f'  GPU: Not available (CPU only)')
"

print_success "Installation complete!"

# =============================================================================
# Print Usage Instructions
# =============================================================================
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run with GPU acceleration:"
echo "  export AUTOREJECT_BACKEND=torch"
echo ""
echo "To run benchmarks:"
echo "  python benchmarks/run_single.py --config standard_64ch"
echo ""
if [ "$COMPUTE_CANADA" = true ]; then
    echo "For batch jobs, add these to your SLURM script:"
    echo "  module load StdEnv/2023 python/3.11 scipy-stack/2024a cuda/12.2"
    echo "  source $VENV_DIR/bin/activate"
    echo "  export AUTOREJECT_BACKEND=torch"
    echo ""
fi
