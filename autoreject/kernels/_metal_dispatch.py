"""Shared Metal dispatch logic for autoreject kernels.

Adapted from HyPyP's ``_metal_dispatch.py`` [1]_. Unlike HyPyP's pairwise
dispatch (1 thread per channel pair), autoreject kernels use threadgroup-level
parallelism where each threadgroup handles one (channel, threshold) pair
and threads within the group cooperate on the time dimension.

References
----------
.. [1] Ramadour, R. HyPyP GPU acceleration — Metal dispatch module.
       hypyp/sync/kernels/_metal_dispatch.py
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from . import METAL_AVAILABLE

if METAL_AVAILABLE:
    import Metal


def make_const_buffer(device: Any, value: int | float,
                      fmt: str = 'I') -> Any:
    """Create a Metal buffer containing a single constant.

    Parameters
    ----------
    device : Metal.MTLDevice
        The Metal device.
    value : int or float
        The value to store.
    fmt : str
        struct format character. 'I' for uint32, 'f' for float32.

    Returns
    -------
    Metal buffer
    """
    data = struct.pack(fmt, value)
    return device.newBufferWithBytes_length_options_(
        data, len(data), Metal.MTLResourceStorageModeShared
    )


def make_buffer_from_numpy(device: Any, arr: np.ndarray) -> Any:
    """Create a Metal buffer from a numpy array.

    Parameters
    ----------
    device : Metal.MTLDevice
        The Metal device.
    arr : np.ndarray
        Must be contiguous.

    Returns
    -------
    Metal buffer
    """
    arr = np.ascontiguousarray(arr)
    return device.newBufferWithBytes_length_options_(
        arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared
    )


def make_output_buffer(device: Any, n_bytes: int) -> Any:
    """Create an empty Metal output buffer.

    Parameters
    ----------
    device : Metal.MTLDevice
        The Metal device.
    n_bytes : int
        Size in bytes.

    Returns
    -------
    Metal buffer
    """
    return device.newBufferWithLength_options_(
        n_bytes, Metal.MTLResourceStorageModeShared
    )


def read_buffer_float32(buf: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Read a Metal buffer back as a numpy float32 array.

    Parameters
    ----------
    buf : Metal buffer
    shape : tuple
        Output shape.

    Returns
    -------
    np.ndarray, float32
    """
    n_bytes = int(np.prod(shape)) * 4
    ptr = buf.contents()
    membuf = ptr.as_buffer(n_bytes)
    return np.frombuffer(membuf, dtype=np.float32).copy().reshape(shape)


def dispatch_threadgroups(device: Any, pipeline: Any,
                          buffers: list[tuple[Any, int]],
                          grid_size: tuple[int, int, int],
                          threadgroup_size: int = 256) -> None:
    """Dispatch a Metal compute kernel with threadgroup-level parallelism.

    Unlike HyPyP's flat thread dispatch, this dispatches threadgroups where
    threads within each group cooperate via shared memory.

    Parameters
    ----------
    device : Metal.MTLDevice
    pipeline : Metal compute pipeline state
    buffers : list of (buffer, index) tuples
        Each element is (Metal buffer, buffer index).
    grid_size : tuple of (int, int, int)
        Number of threadgroups in each dimension.
    threadgroup_size : int
        Threads per threadgroup (default: 256).
    """
    queue = device.newCommandQueue()
    cmd_buffer = queue.commandBuffer()
    encoder = cmd_buffer.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline)
    for buf, idx in buffers:
        encoder.setBuffer_offset_atIndex_(buf, 0, idx)

    # Dispatch threadgroups (not flat threads)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSize(*grid_size),
        Metal.MTLSize(threadgroup_size, 1, 1),
    )
    encoder.endEncoding()

    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()


def compile_metal_function(source: str,
                           function_name: str) -> tuple[Any, Any]:
    """Compile a Metal shader source and return (device, pipeline).

    Parameters
    ----------
    source : str
        Metal shader source code.
    function_name : str
        Name of the kernel function to compile.

    Returns
    -------
    device : Metal.MTLDevice
    pipeline : Metal compute pipeline state

    Raises
    ------
    RuntimeError
        If compilation fails.
    """
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device found")

    options = Metal.MTLCompileOptions.new()
    library, error = device.newLibraryWithSource_options_error_(
        source, options, None
    )
    if library is None:
        raise RuntimeError(f"Metal compilation failed: {error}")

    fn = library.newFunctionWithName_(function_name)
    if fn is None:
        raise RuntimeError(
            f"Function '{function_name}' not found in compiled library"
        )

    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        fn, None
    )
    if pipeline is None:
        raise RuntimeError(f"Pipeline creation failed: {error}")

    return device, pipeline
