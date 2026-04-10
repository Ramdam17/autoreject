"""GPU-aware timer with synchronization.

Handles MPS/CUDA sync for accurate GPU timing. Extracted and generalized
from ``autoreject/benchmarks/profile_pipeline.py``.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Generator

import numpy as np


class GPUTimer:
    """Timer that synchronizes GPU before measuring.

    Parameters
    ----------
    device : str or None
        ``'mps'``, ``'cuda'``, or ``None`` (CPU-only).
    """

    def __init__(self, device: str | None = None):
        self.device = device
        self._torch = None
        self._records = OrderedDict()

        if device is not None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                pass

    def sync(self) -> None:
        """Synchronize GPU to ensure accurate timing."""
        if self._torch is None:
            return
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        elif self.device == "mps":
            self._torch.mps.synchronize()

    @contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        """Context manager to time an operation with GPU sync."""
        self.sync()
        start = time.perf_counter()
        yield
        self.sync()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if name not in self._records:
            self._records[name] = []
        self._records[name].append(elapsed_ms)

    def get_results(self) -> dict[str, float]:
        """Return {name: median_ms} for all recorded operations."""
        return {
            name: float(np.median(times))
            for name, times in self._records.items()
        }

    def get_last(self, name: str) -> float:
        """Return the last recorded time for an operation."""
        if name in self._records and self._records[name]:
            return self._records[name][-1]
        return float("nan")

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._records.clear()
