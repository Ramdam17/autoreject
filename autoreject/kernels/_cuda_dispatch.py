"""Shared CUDA dispatch logic for autoreject kernels.

Uses CuPy RawKernel for inline CUDA source. All kernels use float64
for exact precision matching the CPU reference.

Adapted from HyPyP's ``_cuda_dispatch.py`` [1]_.

References
----------
.. [1] Ramadour, R. HyPyP GPU acceleration — CUDA dispatch module.
       hypyp/sync/kernels/_cuda_dispatch.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def dispatch_cuda_kernel(kernel: Any, grid_size: tuple[int, ...],
                         block_size: tuple[int, ...],
                         args: tuple) -> None:
    """Dispatch a CUDA kernel via CuPy.

    Parameters
    ----------
    kernel : cp.RawKernel
        Compiled CUDA kernel.
    grid_size : tuple of int
        Grid dimensions.
    block_size : tuple of int
        Block dimensions.
    args : tuple
        Kernel arguments.
    """
    kernel(grid_size, block_size, args)


def to_cupy(arr: np.ndarray, dtype: Any = None) -> Any:
    """Convert numpy array to CuPy array on GPU.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    dtype : dtype or None
        Target dtype. Defaults to float64.

    Returns
    -------
    cp.ndarray
        Array on GPU.
    """
    if dtype is None:
        dtype = cp.float64
    return cp.asarray(np.ascontiguousarray(arr), dtype=dtype)


def from_cupy(arr: Any) -> np.ndarray:
    """Convert CuPy array back to numpy.

    Parameters
    ----------
    arr : cp.ndarray
        GPU array.

    Returns
    -------
    np.ndarray
        Array on CPU.
    """
    return cp.asnumpy(arr)
