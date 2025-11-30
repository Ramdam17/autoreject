# AutoReject GPU Benchmark Suite

Comprehensive benchmarking infrastructure for measuring GPU acceleration performance.

## Quick Start

```bash
# 1. Run all benchmarks
python run_all.py

# 2. Generate report and figures
python generate_report.py
```

## Components

### `config.yaml`
Configuration file defining benchmark parameters:
- **channels**: [32, 64, 128, 256]
- **sfreq**: [250, 500, 1000] Hz
- **duration**: [5, 10, 20] seconds
- **n_cv**: [5, 10] cross-validation folds
- **artifact_rate**: [10, 30, 50] %

Predefined configurations:
- `scaling_*`: Test scalability across dataset sizes
- `production_*`: Real-world scenarios
- `quick_test`: Fast validation

### `run_single.py`
Run a single benchmark configuration:

```bash
# Run by config name
python run_single.py --config scaling_64ch_150ep

# Run with custom parameters
python run_single.py --n-channels 128 --n-epochs 300 --sfreq 500

# Force overwrite existing results
python run_single.py --config production_128ch --overwrite
```

### `run_all.py` (Orchestrator)
Run all configurations with smart skipping:

```bash
# Run all (skip existing)
python run_all.py

# Dry run (show what would be run)
python run_all.py --dry-run

# Run 2 configs in parallel
python run_all.py --parallel 2

# Only run configs matching pattern
python run_all.py --filter "scaling"

# Force rerun everything
python run_all.py --overwrite
```

### `generate_report.py`
Generate figures and markdown report:

```bash
# Generate PNG figures
python generate_report.py

# Generate PDF figures
python generate_report.py --format pdf
```

## Output

### `results/`
JSON files with benchmark results:
- Execution times (CPU and GPU)
- Memory usage
- Configuration details
- Output verification

### `figures/`
Generated plots:
- `speedup_vs_channels.png` - Bar chart of speedup by channel count
- `speedup_vs_epochs.png` - Bar chart of speedup by epoch count
- `timing_comparison.png` - Side-by-side CPU vs GPU timing
- `memory_usage.png` - Memory comparison
- `speedup_heatmap.png` - Heatmap of speedup across configurations
- `benchmark_report.md` - Complete markdown report

### `logs/`
Detailed execution logs for debugging.

## Typical Results

| Configuration | Channels | Epochs | CPU Time | GPU Time | Speedup |
|---------------|----------|--------|----------|----------|---------|
| quick_test    | 32       | 50     | ~5s      | ~3s      | ~1.7x   |
| production_64 | 64       | 200    | ~45s     | ~6s      | ~7.5x   |
| production_128| 128      | 300    | ~180s    | ~11s     | ~17x    |

Speedup scales with data size - larger datasets benefit more from GPU acceleration.

## Requirements

- Python 3.9+
- PyTorch with MPS (Mac) or CUDA support
- MNE-Python
- autoreject (this package)
- matplotlib (for report generation)
- pyyaml
- psutil (optional, for detailed memory tracking)
