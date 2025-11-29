"""Backend abstraction for compute operations.

This module provides a unified interface for array operations across different
compute backends (NumPy, Numba, PyTorch, JAX). It enables:

- Automatic hardware detection (CPU, CUDA, MPS)
- Graceful fallback when optional dependencies are not installed
- User-configurable backend selection via environment variable

Usage
-----
>>> from autoreject.backends import get_backend, detect_hardware
>>> 
>>> # Auto-detect best available backend
>>> backend = get_backend()
>>> result = backend.ptp(data, axis=-1)
>>> 
>>> # Force a specific backend
>>> backend = get_backend(prefer='numba')
>>> 
>>> # Check available hardware
>>> hw = detect_hardware()
>>> print(hw)  # {'cpu': True, 'cuda': False, 'mps': True, ...}

Environment Variables
---------------------
AUTOREJECT_BACKEND : str
    Override automatic backend selection. Valid values:
    'numpy', 'numba', 'torch', 'jax'
"""

# Authors: autoreject contributors

import os
import warnings
from functools import lru_cache

import numpy as np


__all__ = ['detect_hardware', 'get_backend', 'get_backend_names']


# =============================================================================
# Hardware Detection
# =============================================================================

@lru_cache(maxsize=1)
def detect_hardware():
    """Detect available hardware acceleration.
    
    Returns
    -------
    available : dict
        Dictionary with keys for each hardware type and boolean values
        indicating availability. Keys include:
        - 'cpu': Always True
        - 'cuda': True if NVIDIA GPU available via PyTorch or JAX
        - 'mps': True if Apple Silicon GPU available via PyTorch
        - 'cuda_device': Name of CUDA device (if available)
        - 'mps_device': 'Apple Silicon' (if available)
    
    Examples
    --------
    >>> hw = detect_hardware()
    >>> if hw.get('mps'):
    ...     print("Apple Silicon GPU available")
    """
    available = {'cpu': True}
    
    # Check for NVIDIA GPU via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            available['cuda'] = True
            available['cuda_device'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except Exception:
        pass  # CUDA initialization errors
    
    # Check for Apple Silicon GPU via PyTorch MPS
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available['mps'] = True
            available['mps_device'] = 'Apple Silicon'
    except ImportError:
        pass
    except Exception:
        pass
    
    # Check for JAX GPU support
    try:
        import jax
        devices = jax.devices()
        for d in devices:
            if d.platform == 'gpu':
                available['jax_gpu'] = True
                available['jax_device'] = str(d)
                break
    except ImportError:
        pass
    except Exception:
        pass
    
    # Check for Numba availability
    try:
        import numba
        available['numba'] = True
        available['numba_version'] = numba.__version__
    except ImportError:
        pass
    
    return available


def get_backend_names():
    """Get list of available backend names.
    
    Returns
    -------
    names : list of str
        List of backend names that can be used with get_backend().
    """
    names = ['numpy']  # Always available
    
    hw = detect_hardware()
    
    if hw.get('numba'):
        names.append('numba')
    
    try:
        import torch
        names.append('torch')
    except ImportError:
        pass
    
    try:
        import jax
        names.append('jax')
    except ImportError:
        pass
    
    return names


# =============================================================================
# Backend Selection
# =============================================================================

_BACKEND_CACHE = {}


def get_backend(prefer=None):
    """Get the best available compute backend.
    
    Parameters
    ----------
    prefer : str | None
        Preferred backend: 'numpy', 'numba', 'torch', 'jax', or None (auto).
        Can also be set via AUTOREJECT_BACKEND environment variable.
        If the preferred backend is not available, falls back to the next
        best option.
    
    Returns
    -------
    backend : Backend
        A backend instance with methods for array operations.
    
    Notes
    -----
    Backend selection priority (when prefer=None):
    1. If CUDA GPU available: JAX > PyTorch > Numba > NumPy
    2. If MPS (Apple Silicon) available: PyTorch > Numba > NumPy
    3. CPU only: Numba > NumPy
    
    Examples
    --------
    >>> backend = get_backend()
    >>> print(f"Using {backend.name} on {backend.device}")
    
    >>> # Force NumPy backend
    >>> backend = get_backend(prefer='numpy')
    """
    # Check environment variable
    prefer = prefer or os.environ.get('AUTOREJECT_BACKEND', None)
    
    # Normalize preference
    if prefer is not None:
        prefer = prefer.lower().strip()
    
    # Check cache
    cache_key = prefer or 'auto'
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]
    
    # Try to load preferred backend
    if prefer is not None:
        backend = _try_load_backend(prefer)
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend
        warnings.warn(
            f"Preferred backend '{prefer}' not available, "
            "falling back to auto-detection.",
            RuntimeWarning
        )
    
    # Auto-detect best backend
    hw = detect_hardware()
    
    # Priority 1: GPU acceleration
    if hw.get('cuda') or hw.get('jax_gpu'):
        backend = _try_load_backend('jax')
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend
        backend = _try_load_backend('torch')
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend
    
    # Priority 2: Apple Silicon MPS
    if hw.get('mps'):
        backend = _try_load_backend('torch')
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend
    
    # Priority 3: CPU parallelization with Numba
    if hw.get('numba'):
        backend = _try_load_backend('numba')
        if backend is not None:
            _BACKEND_CACHE[cache_key] = backend
            return backend
    
    # Fallback: NumPy (always available)
    backend = NumpyBackend()
    _BACKEND_CACHE[cache_key] = backend
    return backend


def _try_load_backend(name):
    """Try to load a specific backend.
    
    Parameters
    ----------
    name : str
        Backend name: 'numpy', 'numba', 'torch', or 'jax'.
    
    Returns
    -------
    backend : Backend | None
        Backend instance if successful, None otherwise.
    """
    try:
        if name == 'numpy':
            return NumpyBackend()
        elif name == 'numba':
            return NumbaBackend()
        elif name == 'torch':
            return TorchBackend()
        elif name == 'jax':
            return JaxBackend()
        else:
            warnings.warn(f"Unknown backend: {name}", RuntimeWarning)
            return None
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"Failed to initialize {name} backend: {e}", RuntimeWarning)
        return None


def clear_backend_cache():
    """Clear the backend cache.
    
    Useful for testing or when hardware configuration changes.
    """
    _BACKEND_CACHE.clear()
    detect_hardware.cache_clear()


# =============================================================================
# Base Backend Class
# =============================================================================

class BaseBackend:
    """Abstract base class for compute backends.
    
    All backends must implement these methods with identical signatures
    to ensure interchangeability.
    
    Attributes
    ----------
    name : str
        Backend name ('numpy', 'numba', 'torch', 'jax').
    device : str
        Device description ('cpu', 'cuda:0', 'mps', etc.).
    """
    
    name = 'base'
    device = 'cpu'
    
    def ptp(self, data, axis=-1):
        """Compute peak-to-peak (max - min) along an axis.
        
        Parameters
        ----------
        data : array-like
            Input array.
        axis : int
            Axis along which to compute ptp. Default: -1.
        
        Returns
        -------
        result : ndarray
            Peak-to-peak values.
        """
        raise NotImplementedError
    
    def median(self, data, axis=None):
        """Compute median along an axis.
        
        Parameters
        ----------
        data : array-like
            Input array.
        axis : int | None
            Axis along which to compute median. Default: None (all elements).
        
        Returns
        -------
        result : ndarray | scalar
            Median values.
        """
        raise NotImplementedError
    
    def correlation(self, x, y):
        """Compute correlation between two arrays.
        
        Parameters
        ----------
        x : array-like, shape (n_times, n_channels)
            First array.
        y : array-like, shape (n_times, n_channels)
            Second array.
        
        Returns
        -------
        corr : ndarray, shape (n_channels,)
            Correlation coefficients.
        """
        raise NotImplementedError
    
    def to_numpy(self, arr):
        """Convert array to NumPy ndarray.
        
        Parameters
        ----------
        arr : array-like
            Input array (may be on GPU or in different format).
        
        Returns
        -------
        result : ndarray
            NumPy array on CPU.
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"


# =============================================================================
# NumPy Backend (Baseline)
# =============================================================================

class NumpyBackend(BaseBackend):
    """NumPy backend (baseline, always available).
    
    This is the reference implementation. All operations use standard
    NumPy functions without parallelization.
    """
    
    name = 'numpy'
    device = 'cpu'
    
    def ptp(self, data, axis=-1):
        """Compute peak-to-peak using np.ptp."""
        return np.ptp(data, axis=axis)
    
    def median(self, data, axis=None):
        """Compute median using np.median."""
        return np.median(data, axis=axis)
    
    def correlation(self, x, y):
        """Compute correlation between arrays.
        
        Uses the formula: corr = sum(x*y) / (||x|| * ||y||)
        """
        num = np.sum(x * y, axis=0)
        denom = np.sqrt(np.sum(x ** 2, axis=0)) * np.sqrt(np.sum(y ** 2, axis=0))
        return num / denom
    
    def to_numpy(self, arr):
        """Return array as-is (already NumPy)."""
        return np.asarray(arr)


# =============================================================================
# Numba Backend (CPU Parallel)
# =============================================================================

class NumbaBackend(BaseBackend):
    """Numba backend with CPU parallelization.
    
    Uses Numba's JIT compilation and parallel loops for acceleration.
    Falls back to NumPy if Numba is not installed.
    
    Notes
    -----
    The first call to each method may be slower due to JIT compilation.
    Subsequent calls will be much faster.
    """
    
    name = 'numba'
    device = 'cpu (parallel)'
    
    def __init__(self):
        """Initialize Numba backend."""
        import numba
        from numba import jit, prange
        
        self._numba = numba
        self._jit = jit
        self._prange = prange
        
        # Pre-compile parallel functions
        self._ptp_3d = self._create_ptp_3d()
        self._ptp_2d = self._create_ptp_2d()
        self._correlation_impl = self._create_correlation()
    
    def _create_ptp_3d(self):
        """Create JIT-compiled ptp for 3D arrays."""
        from numba import jit, prange
        
        @jit(nopython=True, parallel=True, cache=True)
        def _ptp_3d_impl(data):
            n_epochs, n_channels, n_times = data.shape
            result = np.empty((n_epochs, n_channels))
            for i in prange(n_epochs):
                for j in range(n_channels):
                    min_val = data[i, j, 0]
                    max_val = data[i, j, 0]
                    for k in range(1, n_times):
                        val = data[i, j, k]
                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val
                    result[i, j] = max_val - min_val
            return result
        
        return _ptp_3d_impl
    
    def _create_ptp_2d(self):
        """Create JIT-compiled ptp for 2D arrays."""
        from numba import jit, prange
        
        @jit(nopython=True, parallel=True, cache=True)
        def _ptp_2d_impl(data):
            n_rows, n_cols = data.shape
            result = np.empty(n_rows)
            for i in prange(n_rows):
                min_val = data[i, 0]
                max_val = data[i, 0]
                for j in range(1, n_cols):
                    val = data[i, j]
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val
                result[i] = max_val - min_val
            return result
        
        return _ptp_2d_impl
    
    def _create_correlation(self):
        """Create JIT-compiled correlation function."""
        from numba import jit, prange
        
        @jit(nopython=True, parallel=True, cache=True)
        def _corr_impl(x, y):
            n_times, n_channels = x.shape
            result = np.empty(n_channels)
            for ch in prange(n_channels):
                sum_xy = 0.0
                sum_x2 = 0.0
                sum_y2 = 0.0
                for t in range(n_times):
                    xi = x[t, ch]
                    yi = y[t, ch]
                    sum_xy += xi * yi
                    sum_x2 += xi * xi
                    sum_y2 += yi * yi
                denom = np.sqrt(sum_x2) * np.sqrt(sum_y2)
                if denom > 0:
                    result[ch] = sum_xy / denom
                else:
                    result[ch] = 0.0
            return result
        
        return _corr_impl
    
    def ptp(self, data, axis=-1):
        """Compute peak-to-peak with Numba parallelization."""
        data = np.asarray(data)
        
        # Use optimized implementations for common cases
        if data.ndim == 3 and axis == -1:
            return self._ptp_3d(data)
        elif data.ndim == 2 and axis == -1:
            return self._ptp_2d(data)
        else:
            # Fall back to NumPy for other cases
            return np.ptp(data, axis=axis)
    
    def median(self, data, axis=None):
        """Compute median (falls back to NumPy)."""
        # Numba doesn't have a good parallel median implementation
        return np.median(data, axis=axis)
    
    def correlation(self, x, y):
        """Compute correlation with Numba parallelization."""
        x = np.asarray(x)
        y = np.asarray(y)
        
        if x.ndim == 2 and y.ndim == 2:
            return self._correlation_impl(x, y)
        else:
            # Fall back to NumPy
            num = np.sum(x * y, axis=0)
            denom = np.sqrt(np.sum(x ** 2, axis=0)) * np.sqrt(np.sum(y ** 2, axis=0))
            return num / denom
    
    def to_numpy(self, arr):
        """Return array as-is (already NumPy)."""
        return np.asarray(arr)


# =============================================================================
# PyTorch Backend (CUDA/MPS)
# =============================================================================

class TorchBackend(BaseBackend):
    """PyTorch backend with CUDA/MPS GPU support.
    
    Automatically selects the best available device:
    - CUDA (NVIDIA GPUs)
    - MPS (Apple Silicon)
    - CPU (fallback)
    
    Notes
    -----
    Data is automatically transferred to/from GPU as needed.
    For best performance, keep data on GPU between operations.
    
    MPS (Apple Silicon) only supports float32, so data is automatically
    converted when using MPS device.
    """
    
    name = 'torch'
    
    def __init__(self):
        """Initialize PyTorch backend."""
        import torch
        self._torch = torch
        
        # Select device
        if torch.cuda.is_available():
            self.device = 'cuda'
            self._device = torch.device('cuda')
            self._dtype = torch.float64  # CUDA supports float64
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
            self._device = torch.device('mps')
            self._dtype = torch.float32  # MPS only supports float32
        else:
            self.device = 'cpu'
            self._device = torch.device('cpu')
            self._dtype = torch.float64  # CPU supports float64
    
    def _to_tensor(self, data):
        """Convert data to PyTorch tensor on device."""
        if isinstance(data, self._torch.Tensor):
            return data.to(device=self._device, dtype=self._dtype)
        arr = np.asarray(data)
        # Convert to appropriate dtype for device
        if self._dtype == self._torch.float32:
            arr = arr.astype(np.float32)
        return self._torch.from_numpy(arr).to(self._device)
    
    def ptp(self, data, axis=-1):
        """Compute peak-to-peak using PyTorch."""
        t = self._to_tensor(data)
        result = t.max(dim=axis).values - t.min(dim=axis).values
        return result.cpu().numpy()
    
    def median(self, data, axis=None):
        """Compute median using PyTorch."""
        t = self._to_tensor(data)
        if axis is None:
            result = t.median()
        else:
            result = t.median(dim=axis).values
        return result.cpu().numpy()
    
    def correlation(self, x, y):
        """Compute correlation using PyTorch."""
        tx = self._to_tensor(x)
        ty = self._to_tensor(y)
        
        num = (tx * ty).sum(dim=0)
        denom = tx.pow(2).sum(dim=0).sqrt() * ty.pow(2).sum(dim=0).sqrt()
        result = num / denom
        
        return result.cpu().numpy()
    
    def to_numpy(self, arr):
        """Convert tensor to NumPy array."""
        if isinstance(arr, self._torch.Tensor):
            return arr.cpu().numpy()
        return np.asarray(arr)


# =============================================================================
# JAX Backend (CUDA/TPU)
# =============================================================================

class JaxBackend(BaseBackend):
    """JAX backend with GPU/TPU support.
    
    Uses JAX's XLA compilation for high-performance operations.
    Automatically detects and uses available accelerators.
    
    Notes
    -----
    JAX operations are JIT-compiled on first call for each input shape.
    Subsequent calls with the same shape will be much faster.
    """
    
    name = 'jax'
    
    def __init__(self):
        """Initialize JAX backend."""
        import jax
        import jax.numpy as jnp
        
        self._jax = jax
        self._jnp = jnp
        
        # Determine device
        devices = jax.devices()
        if devices and devices[0].platform == 'gpu':
            self.device = f'gpu:{devices[0].id}'
        elif devices and devices[0].platform == 'tpu':
            self.device = f'tpu:{devices[0].id}'
        else:
            self.device = 'cpu'
        
        # Create JIT-compiled functions
        self._ptp_jit = jax.jit(self._ptp_impl)
        self._median_jit = jax.jit(self._median_impl, static_argnums=(1,))
        self._correlation_jit = jax.jit(self._correlation_impl)
    
    def _ptp_impl(self, data):
        """Peak-to-peak implementation for JIT."""
        return self._jnp.max(data, axis=-1) - self._jnp.min(data, axis=-1)
    
    def _median_impl(self, data, axis):
        """Median implementation for JIT."""
        return self._jnp.median(data, axis=axis)
    
    def _correlation_impl(self, x, y):
        """Correlation implementation for JIT."""
        num = self._jnp.sum(x * y, axis=0)
        denom = self._jnp.sqrt(self._jnp.sum(x ** 2, axis=0)) * \
                self._jnp.sqrt(self._jnp.sum(y ** 2, axis=0))
        return num / denom
    
    def ptp(self, data, axis=-1):
        """Compute peak-to-peak using JAX."""
        data = self._jnp.asarray(data)
        if axis == -1:
            result = self._ptp_jit(data)
        else:
            result = self._jnp.max(data, axis=axis) - self._jnp.min(data, axis=axis)
        return np.asarray(result)
    
    def median(self, data, axis=None):
        """Compute median using JAX."""
        data = self._jnp.asarray(data)
        result = self._median_jit(data, axis)
        return np.asarray(result)
    
    def correlation(self, x, y):
        """Compute correlation using JAX."""
        x = self._jnp.asarray(x)
        y = self._jnp.asarray(y)
        result = self._correlation_jit(x, y)
        return np.asarray(result)
    
    def to_numpy(self, arr):
        """Convert JAX array to NumPy."""
        return np.asarray(arr)
