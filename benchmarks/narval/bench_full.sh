#!/bin/bash
# ======================================================================
# SLURM: Full benchmark — real_data + scaling + variance suites
#
# Replicates the complete MPS benchmark (benchmark_mps_20260410_1510.md)
# on CUDA/A100. Runs all suites with all available backends.
#
# The orchestrator is idempotent: existing results are skipped.
# Safe to re-submit if interrupted.
#
# Usage:
#   sbatch benchmarks/narval/bench_full.sh
# ======================================================================
#SBATCH --account=def-gdumas85
#SBATCH --job-name=ar-bench-full
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/bench-full-%j.out
#SBATCH --mail-user=remy.bhatt@umontreal.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# --- Environment ---
module --force purge
module load StdEnv/2023 python/3.12 scipy-stack cuda/12.2
source "$HOME/envs/autoreject-bench/bin/activate"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# --- GPU diagnostics ---
echo "=== GPU Info ==="
nvidia-smi
python -c "
import torch, cupy
print(f'torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB')
print(f'cupy {cupy.__version__} | devices: {cupy.cuda.runtime.getDeviceCount()}')
"
echo ""

# --- Suite 1: Real datasets ---
echo "=== [1/3] Running real_data suite ==="
python -m benchmarks.run --suite real_data
echo ""

# --- Suite 2: Scaling analysis ---
echo "=== [2/3] Running scaling suite ==="
python -m benchmarks.run --suite scaling
echo ""

# --- Suite 3: Variance / reproducibility (15 seeds) ---
echo "=== [3/3] Running variance suite ==="
python -m benchmarks.run --suite variance
echo ""

# --- Generate report ---
echo "=== Generating report ==="
python -m benchmarks.report

echo ""
echo "=== Full benchmark complete ==="
echo "Results:  $SLURM_SUBMIT_DIR/benchmarks/results/"
echo "Report:   $SLURM_SUBMIT_DIR/benchmarks/reports/"
