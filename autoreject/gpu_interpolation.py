"""GPU-accelerated spherical spline interpolation for EEG channels.

This module provides PyTorch implementations of MNE's spherical spline
interpolation functions, enabling the entire interpolation pipeline to
run on GPU without CPU<->GPU transfers.

The implementation is based on:
    Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
    Spherical splines for scalp potential and current density mapping.
    Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
"""

import numpy as np
from .backends import get_backend, is_device_array, DeviceArray

__all__ = [
    'gpu_make_interpolation_matrix',
    'gpu_do_interp_dots',
    'gpu_interpolate_bads_eeg',
    'gpu_interpolate_bad_epochs',
    'legval_torch',
]


def legval_torch(x, c):
    """Evaluate Legendre polynomial using Clenshaw's algorithm.
    
    PyTorch implementation of numpy.polynomial.legendre.legval.
    Follows the exact same algorithm as NumPy.
    
    Parameters
    ----------
    x : torch.Tensor
        Evaluation points.
    c : list or torch.Tensor
        Legendre coefficients ordered from low to high degree.
        
    Returns
    -------
    torch.Tensor
        Polynomial evaluation at x.
    """
    import torch
    
    # Ensure x is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    device = x.device
    dtype = x.dtype
    
    # Convert coefficients to list for indexing
    if isinstance(c, torch.Tensor):
        c_list = c.tolist()
    else:
        c_list = list(c)
    
    n = len(c_list)
    
    if n == 1:
        return torch.full_like(x, c_list[0])
    elif n == 2:
        return c_list[0] + c_list[1] * x
    else:
        # Clenshaw recurrence - match NumPy exactly
        # nd starts as len(c) and decrements BEFORE each use
        nd = n
        c0 = torch.full_like(x, c_list[-2])
        c1 = torch.full_like(x, c_list[-1])
        
        for i in range(3, n + 1):
            tmp = c0.clone()
            nd = nd - 1  # Decrement BEFORE use (this is key!)
            # Clenshaw recurrence for Legendre
            c0 = c_list[-i] - c1 * ((nd - 1) / nd)
            c1 = tmp + c1 * x * ((2 * nd - 1) / nd)
        
        return c0 + c1 * x


def _calc_g_torch(cosang, stiffness=4, n_legendre_terms=50):
    """Calculate spherical spline G function on GPU.
    
    PyTorch implementation of MNE's _calc_g.
    
    Parameters
    ----------
    cosang : torch.Tensor, shape (n, m)
        Cosine of angles between pairs of points on a spherical surface.
    stiffness : float
        Stiffness of the spline.
    n_legendre_terms : int
        Number of Legendre terms to evaluate.
        
    Returns
    -------
    torch.Tensor, shape (n, m)
        The G matrix.
    """
    import torch
    
    device = cosang.device
    dtype = cosang.dtype
    
    # Compute Legendre coefficients
    # factors[n] = (2n+1) / (n^stiffness * (n+1)^stiffness * 4*pi)
    factors = [0.0]  # c[0] = 0
    for n in range(1, n_legendre_terms + 1):
        factor = (2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness * 4 * np.pi)
        factors.append(factor)
    
    return legval_torch(cosang, factors)


def _normalize_vectors_torch(pos):
    """Normalize position vectors to unit sphere.
    
    Parameters
    ----------
    pos : torch.Tensor, shape (n, 3)
        Position vectors.
        
    Returns
    -------
    torch.Tensor, shape (n, 3)
        Normalized position vectors.
    """
    import torch
    norms = torch.norm(pos, dim=1, keepdim=True)
    return pos / norms


def gpu_make_interpolation_matrix(pos_from, pos_to, alpha=1e-5, device=None):
    """Compute interpolation matrix based on spherical splines on GPU.
    
    PyTorch implementation of MNE's _make_interpolation_matrix.
    Uses float64 internally for numerical stability (like MNE), then
    converts to float32 for GPU compute efficiency.
    
    Parameters
    ----------
    pos_from : np.ndarray or torch.Tensor, shape (n_good, 3)
        The positions to interpolate from (good sensors).
    pos_to : np.ndarray or torch.Tensor, shape (n_bad, 3)
        The positions to interpolate to (bad sensors).
    alpha : float
        Regularization parameter. Defaults to 1e-5.
    device : str or torch.device, optional
        Device to run on. If None, uses 'mps' if available, else 'cpu'.
        
    Returns
    -------
    DeviceArray
        The interpolation matrix that maps good signals to bad signal locations.
        Shape: (n_bad, n_good)
    """
    import torch
    
    backend = get_backend()
    if backend.name != 'torch':
        raise RuntimeError("gpu_make_interpolation_matrix requires torch backend")
    
    if device is None:
        device = backend.device
    
    # Use CPU and float64 for matrix inversion (numerical stability)
    # This matches MNE's behavior exactly
    compute_device = 'cpu'  # pinv on MPS falls back to CPU anyway
    compute_dtype = torch.float64
    
    # Convert to torch tensors (float64 for precision)
    if isinstance(pos_from, np.ndarray):
        pos_from = torch.tensor(pos_from, dtype=compute_dtype, device=compute_device)
    else:
        pos_from = pos_from.clone().to(device=compute_device, dtype=compute_dtype)
    
    if isinstance(pos_to, np.ndarray):
        pos_to = torch.tensor(pos_to, dtype=compute_dtype, device=compute_device)
    else:
        pos_to = pos_to.clone().to(device=compute_device, dtype=compute_dtype)
    
    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]
    
    # Normalize sensor positions to unit sphere
    pos_from = _normalize_vectors_torch(pos_from)
    pos_to = _normalize_vectors_torch(pos_to)
    
    # Cosine angles between source positions (dot product of unit vectors)
    cosang_from = pos_from @ pos_from.T  # (n_from, n_from)
    cosang_to_from = pos_to @ pos_from.T  # (n_to, n_from)
    
    # Compute G matrices
    G_from = _calc_g_torch(cosang_from)
    G_to_from = _calc_g_torch(cosang_to_from)
    
    # Add regularization
    if alpha is not None:
        G_from = G_from + alpha * torch.eye(n_from, device=compute_device, dtype=compute_dtype)
    
    # Build the C matrix and compute pseudo-inverse
    # C = [[G_from, ones], [ones.T, 0]]
    ones_col = torch.ones((n_from, 1), device=compute_device, dtype=compute_dtype)
    ones_row = torch.ones((1, n_from), device=compute_device, dtype=compute_dtype)
    zero = torch.zeros((1, 1), device=compute_device, dtype=compute_dtype)
    
    C = torch.cat([
        torch.cat([G_from, ones_col], dim=1),
        torch.cat([ones_row, zero], dim=1)
    ], dim=0)
    
    # Pseudo-inverse (computed on CPU with float64 for numerical stability)
    C_inv = torch.linalg.pinv(C)
    
    # Compute interpolation matrix
    # interpolation = [G_to_from, ones] @ C_inv[:, :-1]
    ones_to = torch.ones((n_to, 1), device=compute_device, dtype=compute_dtype)
    interpolation = torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]
    
    assert interpolation.shape == (n_to, n_from)
    
    # Convert to float32 and move to target device for efficient GPU compute
    interpolation = interpolation.to(device=device, dtype=torch.float32)
    
    return DeviceArray(interpolation, backend='torch', device=str(device))


def gpu_do_interp_dots(data, interpolation, goods_idx, bads_idx, keep_on_device=True):
    """Apply interpolation matrix to data on GPU.
    
    Uses batch matrix multiplication (bmm) for optimal GPU performance.
    
    Parameters
    ----------
    data : np.ndarray or DeviceArray, shape (..., n_channels, n_times)
        The data to interpolate.
    interpolation : DeviceArray or torch.Tensor, shape (n_bad, n_good)
        The interpolation matrix.
    goods_idx : np.ndarray of bool, shape (n_channels,)
        Boolean mask for good channels.
    bads_idx : np.ndarray of bool, shape (n_channels,)
        Boolean mask for bad channels.
    keep_on_device : bool
        If True, returns DeviceArray. If False, returns numpy array.
        
    Returns
    -------
    DeviceArray or np.ndarray
        Interpolated data with bad channels replaced.
    """
    import torch
    
    backend = get_backend()
    if backend.name != 'torch':
        raise RuntimeError("gpu_do_interp_dots requires torch backend")
    
    device = backend.device
    
    # Get interpolation matrix tensor
    if is_device_array(interpolation):
        interp_tensor = interpolation.data
    else:
        interp_tensor = interpolation
    
    # Convert data to tensor if needed
    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    elif is_device_array(data):
        data_tensor = data.data.clone()
    else:
        data_tensor = data.clone()
    
    # Convert indices to tensors
    goods_idx_t = torch.tensor(goods_idx, dtype=torch.bool, device=device)
    bads_idx_t = torch.tensor(bads_idx, dtype=torch.bool, device=device)
    
    # Get good channel data
    good_data = data_tensor[..., goods_idx_t, :]
    
    # Apply interpolation using bmm (batch matrix multiplication) for speed
    if data_tensor.ndim == 2:
        # (n_channels, n_times) - single epoch/evoked
        interpolated = interp_tensor @ good_data
    elif data_tensor.ndim == 3:
        # (n_epochs, n_channels, n_times)
        # interp_tensor: (n_bad, n_good)
        # good_data: (n_epochs, n_good, n_times)
        n_epochs = good_data.shape[0]
        n_bad = interp_tensor.shape[0]
        
        # Expand interp to (n_epochs, n_bad, n_good) for bmm
        interp_expanded = interp_tensor.unsqueeze(0).expand(n_epochs, -1, -1)
        # bmm: (n_epochs, n_bad, n_good) @ (n_epochs, n_good, n_times) -> (n_epochs, n_bad, n_times)
        interpolated = torch.bmm(interp_expanded, good_data)
    else:
        raise ValueError(f"Unsupported data dimensions: {data_tensor.ndim}")
    
    # Replace bad channel data
    data_tensor[..., bads_idx_t, :] = interpolated
    
    if keep_on_device:
        return DeviceArray(data_tensor, backend='torch', device=str(device))
    else:
        return data_tensor.cpu().numpy()


def gpu_interpolate_bads_eeg(inst, picks=None, keep_on_device=True):
    """Interpolate bad EEG channels using GPU.
    
    GPU-accelerated version of MNE's _interpolate_bads_eeg.
    
    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    picks : array-like of int or None
        Channels to consider for interpolation.
    keep_on_device : bool
        If True, returns DeviceArray. If False, modifies inst in place.
        
    Returns
    -------
    DeviceArray or None
        If keep_on_device=True, returns interpolated data as DeviceArray.
        If False, modifies inst in place and returns None.
    """
    import torch
    from mne.bem import _fit_sphere
    from mne import pick_types, pick_channels
    from .utils import _handle_picks
    
    backend = get_backend()
    if backend.name != 'torch':
        raise RuntimeError("gpu_interpolate_bads_eeg requires torch backend")
    
    device = backend.device
    
    if picks is None:
        picks = pick_types(inst.info, meg=False, eeg=True, exclude=[])
    else:
        picks = _handle_picks(inst.info, picks)
    
    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)
    bads_idx[picks] = [inst.ch_names[ch] in inst.info['bads'] for ch in picks]
    
    if len(picks) == 0 or bads_idx.sum() == 0:
        if keep_on_device:
            return DeviceArray(
                torch.tensor(inst._data, dtype=torch.float32, device=device),
                backend='torch',
                device=str(device)
            )
        return None
    
    goods_idx[picks] = True
    goods_idx[bads_idx] = False
    
    pos = inst._get_channel_positions(picks)
    
    # Make sure only good EEG are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]
    pos_good = pos[goods_idx_pos]
    pos_bad = pos[bads_idx_pos]
    
    # Compute interpolation matrix on GPU
    interpolation = gpu_make_interpolation_matrix(pos_good, pos_bad, device=device)
    
    # Apply interpolation
    result = gpu_do_interp_dots(
        inst._data, 
        interpolation, 
        goods_idx, 
        bads_idx, 
        keep_on_device=keep_on_device
    )
    
    if keep_on_device:
        return result
    else:
        inst._data = result if isinstance(result, np.ndarray) else result.cpu().numpy()
        return None


def gpu_clean_by_interp(inst, picks=None, device=None, verbose=True):
    """Clean epochs/evoked by LOOCV interpolation on GPU.
    
    GPU-accelerated version of clean_by_interp that keeps data on GPU
    throughout the entire interpolation process.
    
    Parameters
    ----------
    inst : mne.Evoked or mne.Epochs
        The evoked or epochs object.
    picks : array-like or None
        Channels to include for interpolation.
    device : str or torch.device, optional
        Device to run on.
    verbose : bool
        Whether to show progress.
        
    Returns
    -------
    DeviceArray
        Interpolated data staying on GPU.
    """
    import torch
    import mne
    from mne import pick_channels
    from .utils import _handle_picks, _pbar, _get_epochs_type
    
    backend = get_backend()
    if backend.name != 'torch':
        raise RuntimeError("gpu_clean_by_interp requires torch backend")
    
    if device is None:
        device = backend.device
    
    picks = _handle_picks(info=inst.info, picks=picks)
    BaseEpochs = _get_epochs_type()
    ch_names = [inst.info['ch_names'][p] for p in picks]
    
    # Transfer data to GPU once
    data_gpu = torch.tensor(inst._data, dtype=torch.float32, device=device)
    result_gpu = data_gpu.clone()
    
    # Pre-compute positions and normalize once
    pos_all = inst._get_channel_positions(picks)
    pos_all_t = torch.tensor(pos_all, dtype=torch.float32, device=device)
    pos_all_t = _normalize_vectors_torch(pos_all_t)
    
    # Pre-compute G matrix for all positions
    cosang_all = pos_all_t @ pos_all_t.T
    G_all = _calc_g_torch(cosang_all)
    
    mesg = 'Creating augmented epochs (GPU)'
    for ch_idx, (pick, ch) in enumerate(_pbar(list(zip(picks, ch_names)),
                                        desc=mesg, verbose=verbose)):
        # Find the index of this channel in picks
        pick_in_picks = np.where(picks == pick)[0][0]
        
        # Create goods_idx (all True except current channel)
        goods_mask = torch.ones(len(picks), dtype=torch.bool, device=device)
        goods_mask[pick_in_picks] = False
        
        # Get positions for good and bad channels
        pos_good_idx = goods_mask.cpu().numpy()
        pos_bad_idx = ~goods_mask.cpu().numpy()
        
        # Extract submatrices from pre-computed G
        G_from = G_all[pos_good_idx][:, pos_good_idx]
        G_to_from = G_all[pos_bad_idx][:, pos_good_idx]
        
        # Add regularization
        alpha = 1e-5
        n_from = G_from.shape[0]
        G_from_reg = G_from + alpha * torch.eye(n_from, device=device, dtype=torch.float32)
        
        # Build C matrix and compute pseudo-inverse
        ones_col = torch.ones((n_from, 1), device=device, dtype=torch.float32)
        ones_row = torch.ones((1, n_from), device=device, dtype=torch.float32)
        zero = torch.zeros((1, 1), device=device, dtype=torch.float32)
        
        C = torch.cat([
            torch.cat([G_from_reg, ones_col], dim=1),
            torch.cat([ones_row, zero], dim=1)
        ], dim=0)
        
        C_inv = torch.linalg.pinv(C)
        
        # Compute interpolation (1, n_good)
        ones_to = torch.ones((1, 1), device=device, dtype=torch.float32)
        interpolation = torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]
        
        # Get picks for good channels in the original data
        good_picks = picks[pos_good_idx]
        
        # Apply interpolation
        if isinstance(inst, mne.Evoked):
            # (1, n_good) @ (n_good, n_times) -> (1, n_times)
            good_data = data_gpu[good_picks, :]
            interpolated = interpolation @ good_data
            result_gpu[pick, :] = interpolated.squeeze(0)
        elif isinstance(inst, BaseEpochs):
            # For epochs: (n_epochs, n_good, n_times)
            good_data = data_gpu[:, good_picks, :]
            # (1, n_good) @ (n_epochs, n_good, n_times) -> (n_epochs, 1, n_times)
            interpolated = torch.einsum('bg,egt->ebt', interpolation, good_data)
            result_gpu[:, pick, :] = interpolated.squeeze(1)
    
    return DeviceArray(result_gpu, backend='torch', device=str(device))


def benchmark_interpolation_gpu(n_epochs=100, n_channels=64, n_times=1000, n_iters=3):
    """Benchmark GPU vs CPU interpolation.
    
    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    n_times : int
        Number of time points.
    n_iters : int
        Number of iterations for timing.
        
    Returns
    -------
    dict
        Benchmark results.
    """
    import time
    import torch
    from mne.channels.interpolation import _make_interpolation_matrix as cpu_make_interp
    
    backend = get_backend()
    if backend.name != 'torch':
        print("Benchmark requires torch backend")
        return None
    
    device = backend.device
    print(f"Benchmarking on device: {device}")
    
    # Create random sensor positions on unit sphere
    np.random.seed(42)
    theta = np.random.uniform(0, np.pi, n_channels)
    phi = np.random.uniform(0, 2 * np.pi, n_channels)
    pos = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Simulate 5 bad channels
    n_bad = 5
    pos_good = pos[n_bad:]
    pos_bad = pos[:n_bad]
    
    # Create fake data
    data = np.random.randn(n_epochs, n_channels, n_times).astype(np.float32)
    
    # Benchmark CPU
    times_cpu = []
    for _ in range(n_iters):
        start = time.perf_counter()
        interp_cpu = cpu_make_interp(pos_good, pos_bad)
        # Simulate applying to epochs
        for e in range(n_epochs):
            _ = interp_cpu @ data[e, n_bad:, :]
        times_cpu.append(time.perf_counter() - start)
    
    cpu_time = np.median(times_cpu) * 1000
    
    # Benchmark GPU
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times_gpu = []
    for _ in range(n_iters):
        start = time.perf_counter()
        
        interp_gpu = gpu_make_interpolation_matrix(pos_good, pos_bad, device=device)
        
        # Transfer data once
        data_gpu = torch.tensor(data, dtype=torch.float32, device=device)
        
        # Apply interpolation - batched
        interp_tensor = interp_gpu.data
        good_data = data_gpu[:, n_bad:, :]
        interpolated = torch.einsum('bg,egt->ebt', interp_tensor, good_data)
        
        # Force synchronization
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times_gpu.append(time.perf_counter() - start)
    
    gpu_time = np.median(times_gpu) * 1000
    
    print(f"\nInterpolation Benchmark ({n_epochs} epochs, {n_channels} channels, {n_times} times):")
    print(f"  CPU: {cpu_time:.1f} ms")
    print(f"  GPU: {gpu_time:.1f} ms")
    print(f"  Speedup: {cpu_time / gpu_time:.1f}x")
    
    return {
        'cpu_ms': cpu_time,
        'gpu_ms': gpu_time,
        'speedup': cpu_time / gpu_time
    }


def gpu_interpolate_bad_epochs(data, interp_channels, picks, pos, device=None):
    """GPU-accelerated interpolation for epochs with per-epoch bad channels.
    
    This is a GPU version of _interpolate_bad_epochs that keeps all data
    on GPU throughout the interpolation process.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_epochs, n_channels_total, n_times)
        The epoch data to interpolate.
    interp_channels : list of list of int
        For each epoch, list of channel INDICES (within picks) to interpolate.
    picks : np.ndarray
        Channel indices that were picked.
    pos : np.ndarray, shape (n_picks, 3)
        3D positions of picked channels, normalized to unit vectors.
    device : str or torch.device, optional
        Device to run on.
        
    Returns
    -------
    torch.Tensor
        Interpolated data on GPU, shape (n_epochs, n_channels_total, n_times).
    """
    import torch
    
    backend = get_backend()
    if backend.name != 'torch':
        raise RuntimeError("gpu_interpolate_bad_epochs requires torch backend")
    
    if device is None:
        device = backend.device
    
    n_epochs, n_channels_total, n_times = data.shape
    n_picks = len(picks)
    picks = np.asarray(picks)  # Ensure picks is numpy array
    
    # Transfer data to GPU once
    if isinstance(data, torch.Tensor):
        data_gpu = data.clone()
    else:
        data_gpu = torch.tensor(data, dtype=torch.float32, device=device)
    
    # Pre-compute positions tensor
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    
    # Pre-compute full G matrix for all positions (n_picks x n_picks)
    cosang_all = pos_t @ pos_t.T
    G_all = _calc_g_torch(cosang_all)
    
    # Cache interpolation matrices to avoid recomputing for same bad channel patterns
    interp_cache = {}
    
    for epoch_idx, bad_ch_indices in enumerate(interp_channels):
        if len(bad_ch_indices) == 0:
            continue
        
        # Create cache key from bad channel pattern
        cache_key = tuple(sorted(bad_ch_indices))
        
        if cache_key not in interp_cache:
            # Create masks for good/bad channels
            goods_mask = np.ones(n_picks, dtype=bool)
            for bad_idx in bad_ch_indices:
                goods_mask[bad_idx] = False
            bads_mask = ~goods_mask
            
            # Get indices of good and bad channels within picks
            good_idx_in_picks = np.where(goods_mask)[0]
            bad_idx_in_picks = np.where(bads_mask)[0]
            
            # Create torch index tensors for GPU operations on G matrix
            good_idx_t = torch.tensor(good_idx_in_picks, device=device, dtype=torch.long)
            bad_idx_t = torch.tensor(bad_idx_in_picks, device=device, dtype=torch.long)
            
            # Extract submatrices from pre-computed G using advanced indexing
            G_from = G_all[good_idx_t][:, good_idx_t]
            G_to_from = G_all[bad_idx_t][:, good_idx_t]
            
            # Add regularization
            n_from = len(good_idx_in_picks)
            G_from_reg = G_from + 1e-5 * torch.eye(n_from, device=device, dtype=torch.float32)
            
            # Build C matrix and compute pseudo-inverse
            ones_col = torch.ones((n_from, 1), device=device, dtype=torch.float32)
            ones_row = torch.ones((1, n_from), device=device, dtype=torch.float32)
            zero = torch.zeros((1, 1), device=device, dtype=torch.float32)
            
            C = torch.cat([
                torch.cat([G_from_reg, ones_col], dim=1),
                torch.cat([ones_row, zero], dim=1)
            ], dim=0)
            
            C_inv = torch.linalg.pinv(C)
            
            # Compute interpolation matrix (n_bad, n_good)
            n_bad = len(bad_idx_in_picks)
            ones_to = torch.ones((n_bad, 1), device=device, dtype=torch.float32)
            interpolation = torch.cat([G_to_from, ones_to], dim=1) @ C_inv[:, :-1]
            
            # Store original picks for good and bad channels (in full data indexing)
            good_picks = picks[goods_mask]
            bad_picks = picks[bads_mask]
            
            interp_cache[cache_key] = (interpolation, good_picks, bad_picks)
        
        interpolation, good_picks, bad_picks = interp_cache[cache_key]
        
        # Apply interpolation for this epoch
        # (n_bad, n_good) @ (n_good, n_times) -> (n_bad, n_times)
        good_data = data_gpu[epoch_idx, good_picks, :]
        interpolated = interpolation @ good_data
        data_gpu[epoch_idx, bad_picks, :] = interpolated
    
    return data_gpu


if __name__ == '__main__':
    # Run benchmark when executed directly
    import os
    os.environ['AUTOREJECT_BACKEND'] = 'torch'
    
    # Reload backend
    from . import backends
    backends._backend = None
    
    benchmark_interpolation_gpu()
