"""Single-config benchmark executor.

Runs one benchmark configuration against one backend, measuring:
- Total wall time for ``AutoReject.fit()``
- Peak GPU memory
- Output extraction (thresholds, consensus, n_interpolate, loss)

The runner sets the appropriate environment variables and constructs
AutoReject with the right parameters for each backend.

References
----------
.. [1] Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A.
       (2017). Autoreject: Automated artifact rejection for MEG and EEG data.
       NeuroImage, 159, 417-429. doi:10.1016/j.neuroimage.2017.06.030
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hardware import reset_gpu_memory, get_peak_gpu_mb, sync_device
from .registry import BackendSpec

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    config_name: str
    backend_name: str
    seed: int

    # Timing
    elapsed_ms: float = 0.0
    warmup_ms: float = 0.0

    # Memory
    peak_gpu_mb: float = float("nan")

    # Outputs
    threshes: dict = field(default_factory=dict)
    consensus: dict = field(default_factory=dict)
    n_interpolate: dict = field(default_factory=dict)
    loss: dict = field(default_factory=dict)

    # Metadata
    n_epochs: int = 0
    n_channels: int = 0
    n_times: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "config_name": self.config_name,
            "backend_name": self.backend_name,
            "seed": self.seed,
            "elapsed_ms": self.elapsed_ms,
            "warmup_ms": self.warmup_ms,
            "peak_gpu_mb": self.peak_gpu_mb,
            "threshes": {k: float(v) for k, v in self.threshes.items()},
            "consensus": {k: float(v) for k, v in self.consensus.items()},
            "n_interpolate": {
                k: int(v) for k, v in self.n_interpolate.items()
            },
            "n_epochs": self.n_epochs,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "error": self.error,
        }


def run_single(config: dict, backend_spec: BackendSpec, epochs: Any,
               seed: int = 42, warmup_runs: int = 1,
               timing_runs: int = 3) -> BenchmarkResult:
    """Run a single benchmark: one config × one backend.

    Parameters
    ----------
    config : dict
        Benchmark configuration.
    backend_spec : BackendSpec
        Backend to use.
    epochs : mne.Epochs
        Pre-loaded data.
    seed : int
        Random state for AutoReject.
    warmup_runs : int
        Discard first N runs (JIT warmup).
    timing_runs : int
        Number of timed runs (take median).

    Returns
    -------
    BenchmarkResult
    """
    config_name = config["name"]
    backend_name = backend_spec.name

    result = BenchmarkResult(
        config_name=config_name,
        backend_name=backend_name,
        seed=seed,
        n_epochs=len(epochs),
        n_channels=len(epochs.ch_names),
        n_times=epochs.get_data().shape[-1],
    )

    device = backend_spec.resolved_device

    try:
        # Set environment
        os.environ["AUTOREJECT_BACKEND"] = backend_spec.env_backend

        # Resolve presets
        presets = _resolve_presets(config)
        n_interpolate = presets["n_interpolate"]
        consensus = presets["consensus"]
        cv_folds = config.get("cv_folds", 10)

        # Build AutoReject kwargs
        ar_kwargs = dict(
            n_interpolate=np.array(n_interpolate),
            consensus=np.array(consensus),
            cv=cv_folds,
            random_state=seed,
            verbose=False,
        )

        # Warmup
        for i in range(warmup_runs):
            _clear_caches()
            ar = _make_autoreject(ar_kwargs, backend_spec)
            sync_device(device)
            t0 = time.perf_counter()
            ar.fit(epochs)
            sync_device(device)
            result.warmup_ms = (time.perf_counter() - t0) * 1000

        # Timed runs
        times = []
        for i in range(timing_runs):
            _clear_caches()
            reset_gpu_memory(device)

            ar = _make_autoreject(ar_kwargs, backend_spec)
            sync_device(device)
            t0 = time.perf_counter()
            ar.fit(epochs)
            sync_device(device)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        result.elapsed_ms = float(np.median(times))
        result.peak_gpu_mb = get_peak_gpu_mb(device)

        # Extract outputs from last run
        result.threshes = dict(ar.threshes_)
        result.consensus = dict(ar.consensus_)
        result.n_interpolate = dict(ar.n_interpolate_)

    except Exception as e:
        import traceback
        logger.error("  %s/%s failed: %s", config_name, backend_name, e)
        logger.error("  Traceback:\n%s", traceback.format_exc())
        result.error = str(e)

    finally:
        os.environ.pop("AUTOREJECT_BACKEND", None)

    return result


def _make_autoreject(ar_kwargs: dict, backend_spec: BackendSpec) -> Any:
    """Create an AutoReject instance configured for the given backend.

    Routes to the appropriate code path based on backend spec:
    - use_argmin: sets thresh_method='gpu_argmin' for on-device exact argmin
    - use_kernel: enables Metal/CUDA fused kernels + batched scoring
    - device: explicitly forced so AutoReject doesn't fall back to CPU
      based on dataset-size heuristics during benchmarking
    """
    from autoreject import AutoReject

    kwargs = dict(ar_kwargs)

    if backend_spec.use_argmin:
        kwargs["thresh_method"] = "gpu_argmin"

    if backend_spec.use_kernel:
        kwargs["use_kernel"] = True

    # Force the resolved device so size-based heuristics don't override us
    # (e.g. n_epochs < 50 threshold would fall back to CPU otherwise)
    if backend_spec.name != "numpy_cpu":
        kwargs["device"] = backend_spec.resolved_device

    return AutoReject(**kwargs)


def _resolve_presets(config: dict) -> dict[str, list]:
    """Resolve preset names (light/medium/aggressive) to actual values."""
    # Default presets
    presets_map = {
        "n_interpolate": {
            "light": [1, 4],
            "medium": [1, 4, 8],
            "aggressive": [1, 2, 4, 8, 12, 16],
        },
        "consensus": {
            "light": [0.1, 0.3, 0.5],
            "standard": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    }

    result = {}
    for key in ("n_interpolate", "consensus"):
        val = config.get(key, "medium" if key == "n_interpolate" else "standard")
        if isinstance(val, str) and val in presets_map[key]:
            result[key] = presets_map[key][val]
        elif isinstance(val, list):
            result[key] = val
        else:
            result[key] = presets_map[key].get(
                "medium" if key == "n_interpolate" else "standard"
            )

    return result


def _clear_caches() -> None:
    """Clear backend and interpolation caches between runs."""
    try:
        from autoreject.backends import get_backend
        get_backend.cache_clear()
    except Exception:
        pass

    try:
        from autoreject.gpu_interpolation import _LOOCV_INTERP_CACHE
        _LOOCV_INTERP_CACHE.clear()
    except Exception:
        pass
