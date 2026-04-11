#!/bin/bash
# ======================================================================
# Environment setup for AutoReject GPU Benchmark on Narval (A100)
#
# Run this ONCE on a login node to create the virtualenv and install deps.
# Login nodes have internet access; compute nodes do not.
#
# Usage:
#   bash benchmarks/narval/setup_env.sh
# ======================================================================

set -euo pipefail

ENV_NAME="autoreject-bench"
ENV_DIR="$HOME/envs/$ENV_NAME"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== AutoReject Benchmark — Narval Environment Setup ==="
echo "Repo: $REPO_DIR"
echo "Env:  $ENV_DIR"
echo ""

# --- Step 1: Load modules ---
echo "[1/5] Loading modules..."
module purge
module load StdEnv/2023 python/3.12 scipy-stack cuda/12.2
echo "  Loaded: $(module list 2>&1 | tail -1)"

# --- Step 2: Create virtualenv ---
if [[ -d "$ENV_DIR" ]]; then
    echo "[2/5] Virtualenv already exists at $ENV_DIR, skipping creation."
else
    echo "[2/5] Creating virtualenv with system site-packages..."
    python -m venv --system-site-packages "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
echo "  Python: $(python --version) at $(which python)"

# --- Step 3: Install GPU packages (CC wheels) ---
echo "[3/5] Installing GPU packages from CC wheels..."
pip install --no-index --upgrade pip
pip install --no-index torch cupy
echo "  torch $(python -c 'import torch; print(torch.__version__)')"
echo "  cupy  $(python -c 'import cupy; print(cupy.__version__)')"

# --- Step 4: Install autoreject in editable mode ---
echo "[4/5] Installing autoreject (editable) + benchmark deps..."
cd "$REPO_DIR"

# openneuro-py is not on CC wheels — download if not cached
if ! pip show openneuro-py &>/dev/null; then
    echo "  Downloading openneuro-py wheel (login node has internet)..."
    pip download --no-deps openneuro-py -d "$HOME/wheels/"
    pip install --no-index --find-links "$HOME/wheels/" openneuro-py
fi

# Install remaining deps that CC provides + the package itself
pip install --no-index psutil memory-profiler
pip install -e "." 2>/dev/null || pip install -e "." --no-build-isolation

# --- Step 5: Verify ---
echo "[5/5] Verifying installation..."
python -c "
import torch
import cupy
import mne
import autoreject

print('torch:      ', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('cupy:       ', cupy.__version__)
print('mne:        ', mne.__version__)
print('autoreject: ', autoreject.__version__)

if torch.cuda.is_available():
    print('GPU:        ', torch.cuda.get_device_name(0))
    print('VRAM:       ', torch.cuda.get_device_properties(0).total_memory // (1024**3), 'GB')
else:
    print('WARNING: CUDA not available (expected on login node, will work on GPU nodes)')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with:"
echo "  module load StdEnv/2023 python/3.12 scipy-stack cuda/12.2"
echo "  source $ENV_DIR/bin/activate"
