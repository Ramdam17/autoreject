"""GPU-only threshold selection via exact argmin.

Replaces the Bayesian optimization step with a direct argmin over
pre-computed CV losses. Since the GPU batch pipeline already computes
ALL losses for ALL channels × ALL thresholds, the optimal threshold
is simply the one with the lowest loss — no surrogate model needed.

This eliminates:
- The per-channel Python loop (128 iterations)
- sklearn GaussianProcessRegressor fitting (128 × 10 GP fits)
- Two CPU↔GPU synchronization round-trips
- All stochasticity in threshold selection

The only remaining source of variance across seeds is the
cross-validation split randomness, not the optimization method.

References
----------
Jas et al. (2017). Autoreject. NeuroImage, 159, 417-429.
  Section "Candidate thresholds using Bayesian optimization":
  "This motivated us to use Bayesian optimization to estimate the
   optimal thresholds." — motivation was computational efficiency,
  not methodological. With GPU batch evaluation, this is moot.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>


def compute_all_thresholds_gpu_argmin(optimizer, data_all_channels, cv_splits):
    """Compute optimal thresholds for all channels — fully on GPU.

    Replaces compute_all_thresholds_gpu() by eliminating the Bayesian
    optimization step. Everything stays on GPU until the final result.

    Parameters
    ----------
    optimizer : GPUThresholdOptimizer
        Initialized optimizer with device set.
    data_all_channels : torch.Tensor, shape (n_epochs, n_channels, n_times)
        Full data for all channels on GPU.
    cv_splits : list of (train_idx, test_idx) tuples
        Pre-computed CV splits.

    Returns
    -------
    best_thresholds : np.ndarray, shape (n_channels,)
        Optimal threshold for each channel.
    """
    torch = optimizer.torch
    n_epochs, n_channels, n_times = data_all_channels.shape

    # Step 1: PTP — stays on GPU
    ptp_all = (
        data_all_channels.max(dim=-1).values
        - data_all_channels.min(dim=-1).values
    )  # (n_epochs, n_channels) on GPU

    # Step 2: Sort PTP to get candidate thresholds — on GPU
    # torch.sort along epoch dimension for each channel
    ptp_transposed = ptp_all.T  # (n_channels, n_epochs)
    threshes_all, _ = torch.sort(ptp_transposed, dim=1)
    # threshes_all: (n_channels, n_epochs) on GPU, sorted

    # Step 3: Compute CV losses for ALL channels × ALL thresholds — on GPU
    all_losses = optimizer.batched_all_channels_cv_loss_parallel(
        data_all_channels, ptp_all, threshes_all, cv_splits
    )  # (n_channels, n_thresh) on GPU

    # Step 4: Exact argmin — on GPU, one operation
    best_idx = torch.argmin(all_losses, dim=1)  # (n_channels,) on GPU

    # Gather the optimal threshold for each channel
    best_thresholds_gpu = threshes_all[
        torch.arange(n_channels, device=optimizer.device), best_idx
    ]  # (n_channels,) on GPU

    # Single transfer to CPU at the end
    return best_thresholds_gpu.cpu().numpy()
