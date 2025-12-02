#!/bin/bash
#SBATCH --job-name=autoreject-benchmark
#SBATCH --account=def-YOUR_ACCOUNT  # Replace with your allocation
#SBATCH --time=02:00:00             # 2 hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# =============================================================================
# AutoReject Benchmark Job for Compute Canada
# =============================================================================
# 
# Usage:
#   sbatch scripts/slurm_example.sh
#
# Before first run:
#   1. Run setup: ./scripts/setup_environment.sh --compute-canada
#   2. Edit this file and replace def-YOUR_ACCOUNT with your allocation
#
# =============================================================================

# Exit on error
set -e

# Load modules
module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack/2024a
module load cuda/12.2

# Activate virtual environment
source .venv/bin/activate

# Enable GPU backend
export AUTOREJECT_BACKEND=torch

# Print environment info
echo "=============================================="
echo "  AutoReject Benchmark Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Verify PyTorch sees the GPU
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Starting benchmark..."
echo ""

# Run benchmark
# Modify this line to run your specific benchmark
python benchmarks/run_single.py --config highdensity_128ch

echo ""
echo "=============================================="
echo "  Benchmark Complete!"
echo "=============================================="
