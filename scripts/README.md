# AutoReject Scripts

This directory contains utility scripts for setting up and running AutoReject.

## Environment Setup

### Local Installation (macOS / Linux)

```bash
# For Apple Silicon (MPS)
./scripts/setup_environment.sh --mps

# For NVIDIA GPU (CUDA)
./scripts/setup_environment.sh --cuda

# For CPU only
./scripts/setup_environment.sh --cpu-only
```

### Compute Canada (Narval, Beluga, Cedar, Graham)

```bash
# On a compute node or login node
./scripts/setup_environment.sh --compute-canada

# This will:
# 1. Load required modules (python, scipy-stack, cuda)
# 2. Create a virtual environment with --system-site-packages
# 3. Install PyTorch with CUDA support
# 4. Install MNE and autoreject
```

## Activating the Environment

After setup, activate the environment:

```bash
source .venv/bin/activate
```

To use GPU acceleration:

```bash
export AUTOREJECT_BACKEND=torch
```

## Running Benchmarks

```bash
# Single benchmark
python benchmarks/run_single.py --config standard_64ch

# All benchmarks
python benchmarks/run_all.py
```

## SLURM Job Example (Compute Canada)

See `scripts/slurm_example.sh` for an example batch job script.

```bash
sbatch scripts/slurm_example.sh
```
