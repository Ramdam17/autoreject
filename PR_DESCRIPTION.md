# Pull Request: GPU Acceleration for AutoReject

## Summary

This PR adds optional GPU acceleration to AutoReject via PyTorch, providing significant speedups (10x+) on large EEG/MEG datasets while maintaining full backward compatibility with the existing CPU implementation.

## Motivation

AutoReject's cross-validation based artifact rejection can be computationally expensive on high-density EEG (128-256 channels) or long recordings. GPU acceleration addresses this bottleneck by offloading array operations to graphics hardware.

## Key Features

### 1. Backend Abstraction Layer (`autoreject/backends.py`)

A new module providing a unified interface for compute operations across different backends:

- **NumPy** (default): Reference CPU implementation, always available
- **PyTorch**: GPU acceleration via CUDA (Linux/Windows) or MPS (Apple Silicon)

The abstraction ensures all backends produce numerically consistent results.

### 2. GPU-Accelerated Interpolation (`autoreject/gpu_interpolation.py`)

Reimplementation of spherical spline interpolation optimized for GPU:

- Batched Legendre polynomial computation
- Vectorized interpolation matrix construction
- Matches MNE's interpolation output within floating-point tolerance

### 3. GPU Pipeline (`autoreject/gpu_pipeline.py`)

End-to-end GPU-accelerated processing pipeline:

- Data stays on GPU between operations (minimizes CPU↔GPU transfers)
- Batched epoch processing
- Optimized cross-validation scoring

### 4. Automatic Hardware Detection

The system automatically detects available hardware:

```python
from autoreject.backends import detect_hardware
hw = detect_hardware()
# {'cpu': True, 'cuda': False, 'mps': True, ...}
```

## Usage

### Default Behavior (No Change Required)

Existing code continues to work without modification. GPU is **not** enabled by default.

### Enabling GPU Acceleration

Set the `AUTOREJECT_BACKEND` environment variable:

```bash
export AUTOREJECT_BACKEND=torch
python my_script.py
```

Or in Python:

```python
import os
os.environ['AUTOREJECT_BACKEND'] = 'torch'

from autoreject import AutoReject
ar = AutoReject()
ar.fit(epochs)  # Uses GPU if available
```

### Installation

```bash
pip install autoreject[gpu]  # Installs PyTorch
```

## Performance

Benchmarks on synthetic data (64 channels, 200 epochs, 500 samples):

| Backend | Time | Speedup |
|---------|------|---------|
| NumPy (CPU) | 22.0s | 1.0x |
| PyTorch (MPS) | 2.0s | **11.0x** |

Performance gains scale with dataset size. Larger datasets benefit more from GPU acceleration.

## Numerical Parity

- CPU (NumPy) and GPU (PyTorch) produce functionally equivalent results
- Minor differences (<0.1%) due to floating-point precision are expected and acceptable
- Reproducibility is guaranteed when using `random_state` parameter

## Files Added/Modified

### New Files

| File | Description |
|------|-------------|
| `autoreject/backends.py` | Backend abstraction layer (~1100 lines) |
| `autoreject/gpu_interpolation.py` | GPU-accelerated interpolation (~1050 lines) |
| `autoreject/gpu_pipeline.py` | GPU processing pipeline (~1350 lines) |
| `autoreject/tests/test_backends.py` | Backend tests (~700 lines) |
| `autoreject/tests/test_gpu_interpolation.py` | GPU interpolation tests (~250 lines) |

### Modified Files

| File | Changes |
|------|---------|
| `autoreject/autoreject.py` | Backend integration for core operations |
| `autoreject/ransac.py` | Backend integration for RANSAC |
| `autoreject/__init__.py` | Export new functions |
| `pyproject.toml` | Added `[gpu]` optional dependency |
| `README.rst` | GPU acceleration section |
| `doc/whats_new.rst` | Changelog entry |

## Testing

All existing tests pass. New tests added for:

- Backend consistency (all backends produce same results)
- GPU interpolation accuracy (matches MNE reference)
- Hardware detection
- DeviceArray wrapper functionality

```
68 passed, 1 skipped
```

## Backward Compatibility

- **100% backward compatible**: No API changes to existing classes/functions
- **Opt-in GPU**: Users must explicitly enable GPU via environment variable
- **Graceful fallback**: If PyTorch unavailable, falls back to NumPy silently

## Dependencies

### Required (unchanged)
- mne >= 1.5.0
- numpy >= 1.21.2
- scipy >= 1.7.1
- scikit-learn >= 1.0.0

### Optional (new)
- torch >= 2.0 (for GPU acceleration)

## Future Work

- Automatic backend selection based on dataset size
- Memory-efficient processing for very large datasets

## How to Test

```bash
# Install with GPU support
pip install -e .[gpu]

# Run tests
AUTOREJECT_BACKEND=numpy pytest autoreject/tests/

# Quick GPU benchmark
export AUTOREJECT_BACKEND=torch
python -c "
from autoreject import AutoReject
from autoreject.backends import get_backend
print(f'Backend: {get_backend().name}, Device: {get_backend().device}')
"
```

## Related Issues

- Addresses performance concerns for high-density EEG processing
- Enables practical use of AutoReject on large datasets

---

*This PR maintains the philosophy of AutoReject: automatic, reproducible artifact rejection. GPU acceleration is an optional enhancement that does not change the algorithm, only its execution speed.*
