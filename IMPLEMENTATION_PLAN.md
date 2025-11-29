# Performance Optimization Implementation Plan

**Branch:** `feature/parallel-gpu-acceleration`  
**Goal:** Drastically improve computation time via CPU parallelization (Numba) and optional GPU acceleration (PyTorch MPS/CUDA, JAX), while maintaining 100% backward compatibility.

**⚠️ DELETE THIS FILE BEFORE MERGING ⚠️**

---

## Status Summary

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Retrocompatibility test infrastructure (13 tests) |
| 2 | ✅ Complete | Backend abstraction layer (33 tests, 4 skipped for JAX) |
| 3 | ✅ Complete | CPU parallelization integration |
| 4 | ✅ Complete | GPU acceleration via PyTorch MPS |
| 5 | ✅ Complete | Benchmarking scripts |

---

## Phase 1: Retrocompatibility Test Infrastructure ✅

Secure the existing behavior before making any changes.

### Tasks

- [x] **1.1** Create `autoreject/tests/references/` directory
- [x] **1.2** Create `tools/generate_references.py` script
  - Generate deterministic test epochs (seed=42)
  - Save reference outputs for:
    - `_vote_bad_epochs` → labels, bad_sensor_counts
    - `_compute_thresholds` → threshold dict
    - `Ransac` → correlations, bad_chs_
    - `_interpolate_bad_epochs` → interpolated data sample
- [x] **1.3** Run `generate_references.py` to create `.npz` files
- [x] **1.4** Create `autoreject/tests/test_retrocompat.py`
  - `TestVoteBadEpochsRetrocompat`
  - `TestComputeThresholdsRetrocompat`
  - `TestRansacRetrocompat`
  - `TestInterpolateEpochsRetrocompat`
  - `TestLocalRejectCVRetrocompat`
- [x] **1.5** Add `@pytest.mark.retrocompat` marker to `conftest.py`
- [x] **1.6** Run tests to confirm all pass with current implementation

---

## Phase 2: Backend Abstraction Layer ✅

Create the infrastructure for multiple compute backends.

### Tasks

- [x] **2.1** Create `autoreject/backends.py`
  - `detect_hardware()` → dict of available accelerators
  - `get_backend(prefer=None)` → returns best backend instance
  - Support `AUTOREJECT_BACKEND` environment variable
- [x] **2.2** Implement `NumpyBackend` (baseline, always available)
  - `ptp(data, axis)` → peak-to-peak
  - `median(data, axis)` → median
  - `correlation(x, y)` → correlation coefficient
  - `to_numpy(arr)` → convert to numpy array
- [x] **2.3** Implement `NumbaBackend` (CPU parallel)
  - Graceful fallback if numba not installed
  - `@jit(nopython=True, parallel=True)` decorators
  - `prange` for parallel loops
- [x] **2.4** Implement `TorchBackend` (CUDA/MPS)
  - Auto-detect device (cuda > mps > cpu)
  - Handle CPU↔GPU transfers
  - **Fixed:** Use float32 on MPS (Apple Silicon doesn't support float64)
- [x] **2.5** Implement `JaxBackend` (CUDA/TPU)
  - JIT compilation with `@jax.jit`
  - Device placement
- [x] **2.6** Add unit tests for backends in `autoreject/tests/test_backends.py`
- [x] **2.7** Export `detect_hardware`, `get_backend` in `autoreject/__init__.py`

---

## Phase 3: CPU Parallelization with Numba ✅

Optimize the main computational bottlenecks.

### Tasks

- [x] **3.1** Optimize `_vote_bad_epochs` in `autoreject.py`
  - Use backend for peak-to-peak computation
- [x] **3.2** Add `n_jobs` parameter to `_run_local_reject_cv`
  - Parallel epoch interpolation via joblib
- [x] **3.3** Parallelize `_interpolate_bad_epochs` in `autoreject.py`
  - Added `_interpolate_single_epoch` helper
  - Parallel epoch processing with `joblib.Parallel`
- [x] **3.4** Optimize RANSAC in `ransac.py`
  - Use backend for median and correlation computation
- [x] **3.5** Run retrocompatibility tests → all 13 pass
- [x] **3.6** Run backend tests → 33 pass, 4 skipped (JAX not installed)

---

## Phase 4: GPU Acceleration ✅

Add optional GPU support via PyTorch and JAX.

### Tasks

- [x] **4.1** Backend selection via `AUTOREJECT_BACKEND` environment variable
  - Default: auto-detect best available
  - Options: `'numpy'`, `'numba'`, `'torch'`, `'jax'`
- [x] **4.2** Integrate backends into `_vote_bad_epochs`
- [x] **4.3** Integrate backends into RANSAC correlation computation
- [x] **4.4** Run retrocompatibility tests → all pass

---

## Phase 5: Benchmarking ✅

### Tasks

- [x] **5.1** Create `tools/benchmark.py` script
  - Benchmark `_vote_bad_epochs`
  - Benchmark RANSAC
  - Benchmark interpolation with different n_jobs values
- [x] **5.2** Update `pyproject.toml` with optional dependencies

### Benchmark Results (Apple Silicon M3 Pro, 16 cores)

**Small data (30 epochs, 32 channels):**
- NumPy fastest due to JIT/GPU overhead
- Parallel interpolation has too much fork overhead

**Larger data (100 epochs, 128 channels):**
- Numba ~= NumPy for RANSAC (1.01x speedup)
- Parallel interpolation with n_jobs=4: ~20% speedup potential

---

## Future Work

### Remaining Optimizations

1. **Keep data on GPU throughout pipeline**
   - Avoid CPU↔GPU transfers for each operation
   - Significant gains expected for torch/jax backends

2. **Add explicit `backend` parameter to AutoReject/Ransac**
   - Currently uses environment variable
   - API improvement for explicit control

3. **Parallel cross-validation folds**
   - Currently only interpolation is parallel within CV
   - Could parallelize across folds

4. **Numba-optimized interpolation matrix computation**
   - `_make_interpolation_matrix` is a bottleneck

### CI/CD Updates

- [ ] Add matrix entry for Numba-less testing
- [ ] Add Apple Silicon runner for MPS testing
- [ ] Add benchmark comparison in CI

### Documentation

- [ ] Update README with installation instructions
- [ ] Document AUTOREJECT_BACKEND environment variable
- [ ] Add performance expectations section

---

## Quick Reference

### Running Tests

```bash
# All tests
pytest autoreject/tests/

# Retrocompatibility tests only
pytest autoreject/tests/test_retrocompat.py -v

# Backend tests
pytest autoreject/tests/test_backends.py -v

# Force specific backend
AUTOREJECT_BACKEND=numpy pytest autoreject/tests/test_retrocompat.py
```

### Running Benchmarks

```bash
# Quick benchmark
python tools/benchmark.py --n-epochs 30 --n-channels 32

# Realistic benchmark
python tools/benchmark.py --n-epochs 100 --n-channels 128

# Specific backend only
python tools/benchmark.py --backend numpy --n-epochs 50
```

### Installing Optional Dependencies

```bash
# CPU parallelization (Numba)
pip install -e ".[parallel]"

# GPU (PyTorch - all platforms)
pip install -e ".[gpu]"

# All acceleration packages
pip install -e ".[accel]"
```

---

## Summary

This implementation provides:

1. **Multi-backend architecture** supporting NumPy (baseline), Numba (CPU parallel), PyTorch (CUDA/MPS), and JAX (CUDA/TPU)

2. **100% backward compatibility** verified by 13 retrocompatibility tests against reference data

3. **Parallel epoch interpolation** via joblib with configurable n_jobs

4. **Automatic hardware detection** and backend selection

5. **Environment variable control** (`AUTOREJECT_BACKEND`) for explicit backend selection

The main performance gains come from:
- Parallel interpolation on multi-core CPUs (~20% for larger datasets)
- JIT compilation via Numba (marginal for current operations)
- GPU acceleration potential (requires keeping data on GPU)

**Note:** GPU backends show overhead for small operations due to data transfer costs. Real gains will come from keeping data on GPU throughout the pipeline (future work).
