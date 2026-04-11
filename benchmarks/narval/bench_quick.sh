#!/bin/bash
# ======================================================================
# SLURM: Smoke test — quick validation that CUDA benchmark works
#
# Usage:
#   sbatch benchmarks/narval/bench_quick.sh
# ======================================================================
#SBATCH --account=def-gdumas85
#SBATCH --job-name=ar-bench-quick
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/bench-quick-%j.out
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

# --- Run smoke test ---
echo "=== Running quick benchmark suite ==="
python -m benchmarks.run --suite quick

# --- Generate report ---
echo ""
echo "=== Generating report ==="
python -m benchmarks.report

echo ""
echo "=== Smoke test complete ==="
