# Performance Optimization Implementation Plan

**Branch:** `feature/parallel-gpu-acceleration`  
**Goal:** Drastically improve computation time via CPU parallelization (Numba) and optional GPU acceleration (PyTorch MPS/CUDA, JAX), while maintaining 100% backward compatibility.

**⚠️ DELETE THIS FILE BEFORE MERGING ⚠️**

---

## Phase 1: Retrocompatibility Test Infrastructure

Secure the existing behavior before making any changes.

### Tasks

- [ ] **1.1** Create `autoreject/tests/references/` directory
- [ ] **1.2** Create `tools/generate_references.py` script
  - Generate deterministic test epochs (seed=42)
  - Save reference outputs for:
    - `_vote_bad_epochs` → labels, bad_sensor_counts
    - `_compute_thresholds` → threshold dict
    - `Ransac` → correlations, bad_chs_
    - `_interpolate_bad_epochs` → interpolated data sample
- [ ] **1.3** Run `generate_references.py` to create `.npz` files
- [ ] **1.4** Create `autoreject/tests/test_retrocompat.py`
  - `TestVoteBadEpochsRetrocompat`
  - `TestComputeThresholdsRetrocompat`
  - `TestRansacRetrocompat`
  - `TestInterpolateEpochsRetrocompat`
- [ ] **1.5** Add `@pytest.mark.retrocompat` marker to `conftest.py`
- [ ] **1.6** Run tests to confirm all pass with current implementation

---

## Phase 2: Backend Abstraction Layer

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
- [x] **2.5** Implement `JaxBackend` (CUDA/TPU)
  - JIT compilation with `@jax.jit`
  - Device placement
- [x] **2.6** Add unit tests for backends in `autoreject/tests/test_backends.py`
- [x] **2.7** Export `detect_hardware`, `get_backend` in `autoreject/__init__.py`

---

## Phase 3: CPU Parallelization with Numba

Optimize the main computational bottlenecks.

### Tasks

- [ ] **3.1** Optimize `_vote_bad_epochs` in `autoreject.py`
  - Add `_compute_deltas_parallel()` with `@jit(parallel=True)`
  - Use backend for peak-to-peak computation
- [ ] **3.2** Parallelize `_run_local_reject_cv` in `autoreject.py`
  - Flatten triple loop (n_interpolate × consensus × cv_folds)
  - Use `joblib.Parallel` with `n_jobs` parameter
- [ ] **3.3** Optimize `_interpolate_bad_epochs` in `autoreject.py`
  - Parallelize epoch processing with `joblib.Parallel`
- [ ] **3.4** Optimize RANSAC in `ransac.py`
  - Accelerate correlation computation
  - Batch interpolation matrix operations
- [ ] **3.5** Run retrocompatibility tests → all must pass
- [ ] **3.6** Run existing test suite → all must pass

---

## Phase 4: GPU Acceleration

Add optional GPU support via PyTorch and JAX.

### Tasks

- [ ] **4.1** Add `backend` parameter to `AutoReject.__init__()`
  - Default: `'auto'` (best available)
  - Options: `'auto'`, `'numpy'`, `'numba'`, `'torch'`, `'jax'`
- [ ] **4.2** Add `backend` parameter to `Ransac.__init__()`
- [ ] **4.3** Integrate backends into `_vote_bad_epochs`
- [ ] **4.4** Integrate backends into `_compute_thresholds`
- [ ] **4.5** Integrate backends into RANSAC correlation computation
- [ ] **4.6** Run retrocompatibility tests → all must pass
- [ ] **4.7** Run existing test suite → all must pass

---

## Phase 5: Dependencies and Configuration

Update project configuration.

### Tasks

- [x] **5.1** Update `pyproject.toml` with optional dependencies
  ```toml
  parallel = ["numba>=0.57,<1.0"]
  gpu = ["torch>=2.0"]
  gpu-cuda = ["jax[cuda12]"]
  benchmark = ["pytest-benchmark", "psutil", "memory_profiler"]
  ```
- [ ] **5.2** Update `conftest.py` with CLI options
  - `--run-benchmarks` flag
  - `--benchmark-scale=tiny|small|medium|large`
  - Benchmark fixtures with configurable data sizes
- [ ] **5.3** Create `autoreject/tests/test_benchmarks.py`
  - Benchmark `_vote_bad_epochs`
  - Benchmark `_compute_thresholds`
  - Benchmark `_run_local_reject_cv`
  - Benchmark `Ransac.fit`

---

## Phase 6: CI/CD Updates

Update GitHub Actions for comprehensive testing.

### Tasks

- [ ] **6.1** Update `.github/workflows/test.yml`
  - Add matrix entry: test without Numba (fallback validation)
  - Add `macos-14` runner (Apple Silicon M1 for MPS testing)
- [ ] **6.2** Add optional benchmark job
  - Runs on PRs with `--run-benchmarks --benchmark-scale=small`
  - Compare against baseline, alert if >50% slower
- [ ] **6.3** Run full CI locally to validate

---

## Phase 7: Documentation

Update user-facing documentation.

### Tasks

- [ ] **7.1** Update `README.rst`
  - Installation section for optional dependencies
  - `pip install autoreject[parallel]`
  - `pip install autoreject[gpu]`
- [ ] **7.2** Document `backend` parameter in docstrings
- [ ] **7.3** Document `AUTOREJECT_BACKEND` environment variable
- [ ] **7.4** Add performance expectations section
  - Expected speedups by configuration
  - Hardware recommendations

---

## Phase 8: Final Validation and Cleanup

### Tasks

- [ ] **8.1** Run full test suite with all backends
- [ ] **8.2** Run benchmarks and document results
- [ ] **8.3** Review all changes for code quality
- [ ] **8.4** Update `whats_new.rst` with changelog entry
- [ ] **8.5** Delete this file (`IMPLEMENTATION_PLAN.md`)
- [ ] **8.6** Create PR with comprehensive description

---

## Quick Reference

### Running Tests

```bash
# Standard tests
pytest autoreject/tests/

# Retrocompatibility tests only
pytest autoreject/tests/test_retrocompat.py -v

# With benchmarks (small scale)
pytest autoreject/tests/ --run-benchmarks --benchmark-scale=small

# With specific backend
AUTOREJECT_BACKEND=numba pytest autoreject/tests/
```

### Installing Optional Dependencies

```bash
# CPU parallelization
pip install -e ".[parallel]"

# GPU (PyTorch - works on all platforms)
pip install -e ".[gpu]"

# GPU (JAX with CUDA - Linux/Windows only)
pip install -e ".[gpu-cuda]"

# Development with benchmarks
pip install -e ".[test,benchmark]"
```

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Retrocompat Tests | ✅ Complete | 13 tests passing |
| 2. Backend Layer | ✅ Complete | 28 tests (9 skipped for optional deps) |
| 3. CPU Parallelization | ⬜ Not started | |
| 4. GPU Acceleration | ⬜ Not started | |
| 5. Dependencies | 🔄 In progress | pyproject.toml updated |
| 6. CI/CD | ⬜ Not started | |
| 7. Documentation | ⬜ Not started | |
| 8. Final Validation | ⬜ Not started | |

**Legend:** ⬜ Not started | 🔄 In progress | ✅ Complete | ❌ Blocked
