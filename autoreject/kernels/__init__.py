"""Custom GPU kernels for autoreject.

Provides Metal (Apple Silicon) and CUDA (NVIDIA) implementations
for operations where fusing multiple PyTorch ops into a single kernel
pass eliminates large intermediate tensors and reduces memory traffic.

Current kernels
---------------
thresh_cv : Fused threshold-CV kernel
    Replaces: PTP comparison → boolean mask → BMM masked mean → RMSE
    Eliminates: O(n_train × n_ch × n_thresh) boolean tensor
                O(n_ch × n_times × n_thresh) masked sum tensor
"""

# Metal availability (Apple Silicon via PyObjC)
try:
    import Metal as _Metal  # noqa: F401
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

# CUDA availability (NVIDIA via CuPy)
try:
    import cupy as _cp  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

__all__ = ["METAL_AVAILABLE", "CUPY_AVAILABLE"]
