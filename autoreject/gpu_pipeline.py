"""
GPU Pipeline for Autoreject - Complete End-to-End Implementation.

This module provides GPU-accelerated versions of the entire autoreject pipeline:
1. _compute_thresh_gpu - Per-channel threshold computation (replaces sklearn cross_val_score)
2. _compute_thresholds_gpu - Parallel threshold computation for all channels
3. _run_local_reject_cv_gpu - Full cross-validation loop on GPU

Key insight: Instead of doing N sequential cross_val_score calls,
we batch ALL threshold evaluations into single GPU operations.
"""

import numpy as np
import warnings

__all__ = [
    'GPUThresholdOptimizer',
    'compute_thresholds_gpu',
    'is_gpu_available',
    'run_local_reject_cv_gpu',
]


def _get_torch():
    """Import torch lazily."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def _get_device():
    """Get the best available torch device."""
    torch = _get_torch()
    if torch is None:
        return None
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def is_gpu_available():
    """Check if GPU acceleration is available."""
    torch = _get_torch()
    if torch is None:
        return False
    device = _get_device()
    return device in ('mps', 'cuda')


def _torch_median(tensor, dim):
    """Compute median matching numpy.median behavior.
    
    torch.median returns the lower of two middle values for even-length arrays,
    while numpy.median returns their average. This function matches numpy behavior.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which to compute median.
    
    Returns
    -------
    torch.Tensor
        Median values along the specified dimension.
    """
    torch = _get_torch()
    n = tensor.shape[dim]
    sorted_t, _ = torch.sort(tensor, dim=dim)
    
    if n % 2 == 1:
        # Odd: return middle element
        idx = n // 2
        return torch.select(sorted_t, dim, idx)
    else:
        # Even: return average of two middle elements
        idx1, idx2 = n // 2 - 1, n // 2
        v1 = torch.select(sorted_t, dim, idx1)
        v2 = torch.select(sorted_t, dim, idx2)
        return (v1 + v2) / 2


class GPUThresholdOptimizer:
    """
    GPU-accelerated threshold optimizer for autoreject.
    
    This class replaces sklearn's cross_val_score with batched GPU operations.
    Instead of evaluating thresholds one at a time, we evaluate ALL thresholds
    simultaneously using tensor broadcasting.
    
    Parameters
    ----------
    device : str or None
        Device to use ('mps', 'cuda', 'cpu'). Auto-detected if None.
    """
    
    def __init__(self, device=None):
        """Initialize the optimizer."""
        self.torch = _get_torch()
        if self.torch is None:
            raise ImportError("PyTorch is required for GPUThresholdOptimizer")
        
        self.device = device or _get_device()
        self._cache = {}
    
    def _to_tensor(self, data, dtype=None, cache_key=None):
        """Convert numpy array to GPU tensor.
        
        Parameters
        ----------
        data : np.ndarray
            Data to convert
        dtype : torch.dtype, optional
            Tensor dtype
        cache_key : str, optional
            If provided, cache the tensor with this key. Only use for
            static data that won't change (e.g., epochs data that's
            transferred once at the start).
        """
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]
        
        if dtype is None:
            dtype = self.torch.float32
        tensor = self.torch.tensor(data, dtype=dtype, device=self.device)
        
        if cache_key is not None:
            self._cache[cache_key] = tensor
        return tensor
    
    def _sync(self):
        """Synchronize GPU operations (for accurate timing)."""
        if self.device == 'mps':
            self.torch.mps.synchronize()
        elif self.device == 'cuda':
            self.torch.cuda.synchronize()
    
    def clear_cache(self):
        """Clear the tensor cache."""
        self._cache.clear()
    
    def compute_ptp_1d(self, data):
        """
        Compute peak-to-peak for single-channel data.
        
        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_times)
            Single channel epoch data.
        
        Returns
        -------
        ptp : torch.Tensor, shape (n_epochs,)
            Peak-to-peak values on GPU.
        """
        data_gpu = self._to_tensor(data)
        return data_gpu.max(dim=-1).values - data_gpu.min(dim=-1).values
    
    def compute_ptp_2d(self, data):
        """
        Compute peak-to-peak for multi-channel data.
        
        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            Multi-channel epoch data.
        
        Returns
        -------
        ptp : torch.Tensor, shape (n_epochs, n_channels)
            Peak-to-peak values on GPU.
        """
        data_gpu = self._to_tensor(data)
        return data_gpu.max(dim=-1).values - data_gpu.min(dim=-1).values
    
    def batched_channel_cv_loss(self, data_1d, thresholds, cv_splits, y=None):
        """
        Compute cross-validated loss for ALL thresholds at once for a single channel.
        
        This replaces the inner loop of _compute_thresh where cross_val_score
        is called for each threshold sequentially.
        
        Fully vectorized - no Python loops over thresholds.
        
        Parameters
        ----------
        data_1d : torch.Tensor, shape (n_epochs, n_times)
            Raw data for one channel (not just ptp values).
        thresholds : ndarray, shape (n_thresh,)
            All threshold values to evaluate.
        cv_splits : list of (train_idx, test_idx) tuples
            Cross-validation split indices.
        y : ndarray or None
            Labels for stratified splitting (augmented data).
        
        Returns
        -------
        losses : torch.Tensor, shape (n_thresh,)
            Mean CV loss for each threshold (lower = better).
        """
        n_thresh = len(thresholds)
        n_folds = len(cv_splits)
        thresh_gpu = self._to_tensor(thresholds)
        
        # Compute ptp for determining "good" epochs
        ptp_1d = data_1d.max(dim=-1).values - data_1d.min(dim=-1).values  # (n_epochs,)
        
        # Accumulate fold losses
        fold_losses = self.torch.zeros((n_folds, n_thresh), device=self.device)
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            # Get train/test indices
            train_idx_t = self.torch.tensor(train_idx, device=self.device, dtype=self.torch.long)
            test_idx_t = self.torch.tensor(test_idx, device=self.device, dtype=self.torch.long)
            
            # Get train/test data and ptp
            data_train = data_1d[train_idx_t]  # (n_train, n_times)
            data_test = data_1d[test_idx_t]    # (n_test, n_times)
            ptp_train = ptp_1d[train_idx_t]    # (n_train,)
            
            # Vectorized computation for ALL thresholds at once
            # ptp_train: (n_train,) -> (n_train, 1)
            ptp_train_exp = ptp_train.unsqueeze(-1)
            thresh_exp = thresh_gpu.unsqueeze(0)
            
            # good_train: (n_train, n_thresh) - True where epoch passes threshold
            good_train = ptp_train_exp <= thresh_exp
            
            # Counts
            n_good_train = good_train.sum(dim=0)  # (n_thresh,)
            
            # For each threshold, compute mean of good training epochs
            # data_train: (n_train, n_times) -> expand for broadcast
            # mean_ = sum(data * good_mask) / count  for each threshold
            # Expand dimensions: data_train: (n_train, n_times, 1)
            #                    good_train: (n_train, 1, n_thresh)
            data_train_exp = data_train.unsqueeze(-1)  # (n_train, n_times, 1)
            good_train_exp = good_train.unsqueeze(1)   # (n_train, 1, n_thresh)
            
            # Masked sum across epochs
            masked_sum = (data_train_exp * good_train_exp).sum(dim=0)  # (n_times, n_thresh)
            mean_train = masked_sum / n_good_train.clamp(min=1).unsqueeze(0)  # (n_times, n_thresh)
            
            # Compute score: -sqrt(mean((median(X_test) - mean_)^2))
            # median(X_test): median across epochs, shape (n_times,)
            # Use _torch_median for numpy-compatible behavior with even-length arrays
            median_test = _torch_median(data_test, dim=0)  # (n_times,)
            
            # Expand for all thresholds: (n_times, 1)
            median_test_exp = median_test.unsqueeze(-1)
            
            # RMSE for each threshold
            sq_diff = (median_test_exp - mean_train) ** 2  # (n_times, n_thresh)
            rmse = sq_diff.mean(dim=0).sqrt()  # (n_thresh,)
            
            # Score is -rmse, loss is positive, so loss = rmse
            # But sklearn cross_val_score returns score, and we negate in bayes_opt
            # so loss = -score = rmse (we want to minimize)
            
            # Handle cases with no good epochs
            no_good = (n_good_train == 0)
            rmse[no_good] = float('inf')
            
            fold_losses[fold_idx] = rmse
        
        # Mean across folds (this is what cross_val_score returns, negated)
        return fold_losses.mean(dim=0)
    
    def compute_thresh_gpu(self, data_1d, method='bayesian_optimization', 
                           cv_splits=None, n_cv=10, y=None, 
                           random_state=None, n_iter=20):
        """
        Compute optimal threshold for one channel using GPU.
        
        This replaces _compute_thresh() completely.
        
        Parameters
        ----------
        data_1d : ndarray, shape (n_epochs, n_times)
            Data for one channel.
        method : str
            'bayesian_optimization' or 'random_search'
        cv_splits : list or None
            Pre-computed CV splits. If None, creates KFold splits.
        n_cv : int
            Number of CV folds if cv_splits is None.
        y : ndarray or None
            Labels for stratified splits.
        random_state : int or None
            Random seed.
        n_iter : int
            Number of iterations for random_search.
        
        Returns
        -------
        best_thresh : float
            Optimal threshold value.
        """
        # Transfer data to GPU
        data_gpu = self._to_tensor(data_1d)
        
        # Compute PTP on GPU for thresholds
        ptp = data_gpu.max(dim=-1).values - data_gpu.min(dim=-1).values
        
        # Get all possible thresholds (sorted unique PTP values)
        ptp_np = ptp.cpu().numpy()
        all_threshes = np.sort(ptp_np)
        
        # Create CV splits if not provided
        if cv_splits is None:
            n_epochs = len(data_1d)
            rng = np.random.RandomState(random_state)
            indices = rng.permutation(n_epochs)
            fold_sizes = np.full(n_cv, n_epochs // n_cv)
            fold_sizes[:n_epochs % n_cv] += 1
            
            cv_splits = []
            current = 0
            for fs in fold_sizes:
                test_idx = indices[current:current + fs]
                train_idx = np.concatenate([indices[:current], indices[current + fs:]])
                cv_splits.append((train_idx, test_idx))
                current += fs
        
        if method == 'random_search':
            # Sample n_iter thresholds uniformly
            rng = np.random.RandomState(random_state)
            sample_idx = rng.choice(len(all_threshes), size=min(n_iter, len(all_threshes)), 
                                    replace=False)
            candidate_threshes = all_threshes[sample_idx]
            
            # Evaluate all candidates at once (pass full data, not just ptp)
            losses = self.batched_channel_cv_loss(data_gpu, candidate_threshes, cv_splits, y)
            best_idx = losses.argmin().item()
            best_thresh = candidate_threshes[best_idx]
            
        elif method == 'bayesian_optimization':
            # Use the EXACT same bayes_opt algorithm as CPU, but with GPU-accelerated
            # loss function evaluation
            from .bayesopt import bayes_opt, expected_improvement
            
            # Pre-compute ALL losses at once on GPU (this is the key optimization)
            # Instead of calling the loss function one-by-one during bayes_opt,
            # we evaluate all thresholds upfront and cache the results
            all_losses = self.batched_channel_cv_loss(data_gpu, all_threshes, cv_splits, y)
            all_losses_np = all_losses.cpu().numpy()
            
            # Create a cache for the loss function
            loss_cache = {thresh: loss for thresh, loss in zip(all_threshes, all_losses_np)}
            
            # Define the loss function that uses the cache
            def cached_loss_func(thresh):
                # Find the closest threshold in all_threshes (same as CPU does)
                idx = np.where(thresh - all_threshes >= 0)[0][-1]
                thresh = all_threshes[idx]
                return loss_cache[thresh]
            
            # Initial points: same as CPU
            n_epochs_thresh = len(all_threshes)
            idx = np.concatenate((
                np.linspace(0, n_epochs_thresh, 40, endpoint=False, dtype=int),
                [n_epochs_thresh - 1]
            ))
            idx = np.unique(idx)
            initial_x = all_threshes[idx]
            
            # Run the exact same bayes_opt as CPU
            best_thresh, _ = bayes_opt(cached_loss_func, initial_x,
                                       all_threshes,
                                       expected_improvement,
                                       max_iter=10, debug=False,
                                       random_state=random_state)
        
        return best_thresh


def compute_thresholds_gpu(epochs, method='bayesian_optimization',
                           random_state=None, picks=None, augment=True,
                           verbose=True, n_jobs=1, device=None, dots=None):
    """
    Compute channel-wise thresholds using GPU acceleration.
    
    This is a drop-in replacement for _compute_thresholds() that uses
    GPU-batched operations instead of sklearn cross_val_score.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    method : str
        'bayesian_optimization' or 'random_search'
    random_state : int or None
        Random seed.
    picks : array-like or None
        Channel indices to process.
    augment : bool
        Whether to augment data with interpolated epochs.
    verbose : bool
        Verbosity.
    n_jobs : int
        Not used (GPU is inherently parallel).
    device : str or None
        GPU device to use.
    dots : tuple or None
        Precomputed dots for interpolation (passed through to _clean_by_interp).
    
    Returns
    -------
    threshes : dict
        Channel name -> threshold mapping.
    """
    from .autoreject import _handle_picks, _check_data, _GDKW
    from .autoreject import _clean_by_interp
    from sklearn.model_selection import StratifiedShuffleSplit
    
    picks = _handle_picks(info=epochs.info, picks=picks)
    _check_data(epochs, picks, verbose=verbose, check_loc=augment,
                ch_constraint='data_channels')
    
    n_epochs = len(epochs)
    data = epochs.get_data(**_GDKW)
    y = np.ones((n_epochs,))
    
    if augment:
        epochs_interp = _clean_by_interp(epochs, picks=picks, dots=dots, verbose=verbose)
        data = np.concatenate((data, epochs_interp.get_data(**_GDKW)), axis=0)
        y = np.r_[np.zeros((n_epochs,)), np.ones((n_epochs,))]
    
    # Create CV splits once
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)
    cv_splits = list(cv.split(data, y))
    
    # Initialize GPU optimizer
    optimizer = GPUThresholdOptimizer(device=device)
    
    # Transfer all data to GPU once
    data_gpu = optimizer._to_tensor(data)
    
    # Compute thresholds for each channel
    ch_names = epochs.ch_names
    threshes = {}
    
    if verbose:
        from .autoreject import _pbar
        picks_iter = _pbar(picks, desc='Computing thresholds ...', 
                          position=0, verbose=verbose)
    else:
        picks_iter = picks
    
    for pick in picks_iter:
        # Extract single channel data (still on GPU if using tensor slicing)
        data_1d = data[:, pick, :]
        
        thresh = optimizer.compute_thresh_gpu(
            data_1d, method=method, cv_splits=cv_splits, 
            y=y, random_state=random_state
        )
        threshes[ch_names[pick]] = thresh
    
    optimizer.clear_cache()
    
    return threshes


def run_local_reject_cv_gpu(epochs, thresh_func, picks_, n_interpolate, cv,
                            consensus, dots=None, verbose=True, n_jobs=1,
                            device=None):
    """
    GPU-accelerated version of _run_local_reject_cv.
    
    This replaces the CPU-bound cross-validation loop with GPU operations.
    OPTIMIZED: Uses GPU interpolation and keeps data on GPU throughout.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    thresh_func : callable
        Function to compute thresholds (will be called once).
    picks_ : array-like
        Channel indices.
    n_interpolate : array-like
        Values of n_interpolate to try.
    cv : sklearn CV splitter
        Cross-validation object.
    consensus : array-like
        Values of consensus to try.
    dots : tuple or None
        Precomputed interpolation dots.
    verbose : bool
        Verbosity.
    n_jobs : int
        Parallel jobs for interpolation (unused with GPU).
    device : str or None
        GPU device.
    
    Returns
    -------
    local_reject : _AutoReject
        Fitted local reject object.
    loss : ndarray
        Loss array.
    """
    from .autoreject import (
        _AutoReject, _interpolate_bad_epochs, _get_interp_chs,
        _slicemean, _pbar, _GDKW
    )
    
    n_folds = cv.get_n_splits()
    loss = np.zeros((len(consensus), len(n_interpolate), n_folds))
    
    # Fit thresholds on entire data (this uses GPU via thresh_func)
    local_reject = _AutoReject(thresh_func=thresh_func,
                               verbose=verbose, picks=picks_,
                               dots=dots)
    local_reject.fit(epochs)
    
    assert len(local_reject.consensus_) == 1
    ch_type = next(iter(local_reject.consensus_))
    
    labels, bad_sensor_counts = local_reject._vote_bad_epochs(epochs, picks=picks_)
    
    # Initialize GPU 
    torch = _get_torch()
    if torch is None or device == 'cpu':
        use_gpu = False
    else:
        use_gpu = True
        optimizer = GPUThresholdOptimizer(device=device)
        
        # OPTIMIZATION 1: Transfer original data to GPU ONCE
        X_full = epochs.get_data(**_GDKW)
        X_gpu = optimizer._to_tensor(X_full)
        
        # Pre-compute positions for GPU interpolation
        pos = epochs._get_channel_positions(picks_)
        # Normalize positions to unit vectors
        norms = np.linalg.norm(pos, axis=1, keepdims=True)
        pos_normalized = pos / norms
    
    # Pre-compute CV splits once
    cv_splits = list(cv.split(np.zeros(len(epochs))))
    
    desc = 'n_interp'
    for jdx, n_interp in enumerate(_pbar(n_interpolate, desc=desc,
                                         position=1, verbose=verbose)):
        local_reject.n_interpolate_[ch_type] = n_interp
        labels = local_reject._get_epochs_interpolation(
            epochs, labels=labels, picks=picks_, n_interpolate=n_interp
        )
        
        interp_channels = _get_interp_chs(labels, epochs.ch_names, picks_)
        
        if use_gpu:
            # OPTIMIZATION 2: Use GPU interpolation
            # Convert channel names to indices within picks_
            ch_name_to_pick_idx = {epochs.ch_names[p]: i for i, p in enumerate(picks_)}
            interp_ch_indices = [
                [ch_name_to_pick_idx[ch] for ch in epoch_chs if ch in ch_name_to_pick_idx]
                for epoch_chs in interp_channels
            ]
            
            # GPU interpolation - returns tensor on GPU
            from .gpu_interpolation import gpu_interpolate_bad_epochs
            X_interp_gpu = gpu_interpolate_bad_epochs(
                X_full, interp_ch_indices, picks_, pos_normalized, 
                device=optimizer.device
            )
            # Extract only picked channels
            picks_t = optimizer.torch.tensor(picks_, device=optimizer.device)
            X_interp_picks_gpu = X_interp_gpu[:, picks_t, :]
            X_picks_gpu = X_gpu[:, picks_t, :]
        else:
            # CPU fallback
            epochs_interp = epochs.copy()
            _interpolate_bad_epochs(
                epochs_interp, interp_channels=interp_channels,
                picks=picks_, dots=dots, verbose=verbose, n_jobs=n_jobs
            )
            X = epochs.get_data(picks_, **_GDKW)
            X_interp = epochs_interp.get_data(picks_, **_GDKW)
        
        for fold, (train, test) in enumerate(_pbar(cv_splits, desc='Fold',
                                                   position=3, verbose=verbose)):
            if use_gpu:
                # OPTIMIZATION 3: Batch all consensus values, single sync per fold
                train_t = optimizer.torch.tensor(train, device=optimizer.device)
                test_t = optimizer.torch.tensor(test, device=optimizer.device)
                
                # Pre-compute test median once per fold (shared across consensus)
                X_test = X_picks_gpu[test_t]
                median_X = _torch_median(X_test, dim=0)
                
                # Allocate tensor for all scores in this fold
                n_consensus = len(consensus)
                scores_gpu = optimizer.torch.zeros(n_consensus, device=optimizer.device)
                
                for idx, this_consensus in enumerate(consensus):
                    n_channels = len(picks_)
                    if this_consensus * n_channels <= n_interp:
                        scores_gpu[idx] = float('-inf')
                        continue
                    
                    local_reject.consensus_[ch_type] = this_consensus
                    bad_epochs = local_reject._get_bad_epochs(
                        bad_sensor_counts[train], picks=picks_, ch_type=ch_type
                    )
                    
                    good_epochs_idx = np.nonzero(np.invert(bad_epochs))[0]
                    
                    if len(good_epochs_idx) == 0:
                        scores_gpu[idx] = float('-inf')
                        continue
                    
                    good_idx_t = optimizer.torch.tensor(good_epochs_idx, device=optimizer.device)
                    
                    # Index into interpolated data
                    X_train_interp = X_interp_picks_gpu[train_t]
                    X_good = X_train_interp[good_idx_t]
                    mean_gpu = X_good.mean(dim=0)  # (n_channels, n_times)
                    
                    # score = -sqrt(mean((median_X - mean_)^2))
                    sq_diff = (median_X - mean_gpu) ** 2
                    scores_gpu[idx] = -sq_diff.mean().sqrt()
                
                # SINGLE sync per fold instead of per consensus
                scores_np = scores_gpu.cpu().numpy()
                for idx in range(n_consensus):
                    if scores_np[idx] == float('-inf'):
                        loss[idx, jdx, fold] = np.inf
                    else:
                        loss[idx, jdx, fold] = -scores_np[idx]
            else:
                # CPU fallback
                for idx, this_consensus in enumerate(consensus):
                    n_channels = len(picks_)
                    if this_consensus * n_channels <= n_interp:
                        loss[idx, jdx, fold] = np.inf
                        continue
                    
                    local_reject.consensus_[ch_type] = this_consensus
                    bad_epochs = local_reject._get_bad_epochs(
                        bad_sensor_counts[train], picks=picks_, ch_type=ch_type
                    )
                    
                    good_epochs_idx = np.nonzero(np.invert(bad_epochs))[0]
                    
                    local_reject.mean_ = _slicemean(
                        X_interp[train][good_epochs_idx], axis=0
                    )
                    score = local_reject.score(X[test])
                    loss[idx, jdx, fold] = -score
    
    if use_gpu:
        optimizer.clear_cache()
    
    return local_reject, loss


# Utility function to check if GPU should be used
def should_use_gpu(n_epochs, n_channels, device=None):
    """
    Determine if GPU acceleration would be beneficial.
    
    GPU is beneficial for larger datasets where the parallelism
    outweighs the transfer overhead.
    
    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    device : str or None
        Requested device.
    
    Returns
    -------
    use_gpu : bool
        Whether to use GPU.
    device : str
        Device to use.
    """
    import os
    
    # OPTIMIZATION 4: Respect AUTOREJECT_BACKEND environment variable
    backend_env = os.environ.get('AUTOREJECT_BACKEND', '').lower()
    if backend_env == 'numpy' or backend_env == 'numba':
        return False, 'cpu'
    
    if device == 'cpu':
        return False, 'cpu'
    
    if not is_gpu_available():
        return False, 'cpu'
    
    # Heuristic: GPU beneficial for larger datasets
    # Based on benchmarks, GPU overhead is ~200ms
    # Each CV iteration saves ~1ms, so need >200 iterations to benefit
    # With 10 folds × 20 thresholds × n_channels, we have 200×n_channels iterations
    # So GPU is beneficial when n_channels >= 1 (i.e., always for realistic data)
    
    # But for very small datasets, the transfer overhead dominates
    if n_epochs < 50:
        return False, 'cpu'
    
    return True, _get_device()
