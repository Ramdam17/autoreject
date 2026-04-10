"""Shared CUDA dispatch logic for autoreject kernels.

Uses CuPy RawKernel for inline CUDA source. All kernels use float64
for exact precision matching the CPU reference.

Adapted from HyPyP's _cuda_dispatch.py.
"""

import numpy as np

from . import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def dispatch_cuda_kernel(kernel, grid_size, block_size, args):
    """Dispatch a CUDA kernel via CuPy.

    Parameters
    ----------
    kernel : cp.RawKernel
        Compiled CUDA kernel.
    grid_size : tuple
        Grid dimensions.
    block_size : tuple
        Block dimensions.
    args : tuple
        Kernel arguments.
    """
    kernel(grid_size, block_size, args)


def to_cupy(arr, dtype=None):
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
    """
    if dtype is None:
        dtype = cp.float64
    return cp.asarray(np.ascontiguousarray(arr), dtype=dtype)


def from_cupy(arr):
    """Convert CuPy array back to numpy.

    Parameters
    ----------
    arr : cp.ndarray

    Returns
    -------
    np.ndarray
    """
    return cp.asnumpy(arr)
