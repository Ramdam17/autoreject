"""Batched consensus scoring for the CV loop.

Instead of computing scores one consensus value at a time (each requiring
a separate GPU indexing + mean + RMSE), this module computes ALL consensus
scores in a single GPU operation per fold using a weight matrix approach.

The key insight: for each consensus value c, the "good epochs" are a subset
of training epochs. We build a weight matrix W of shape (n_consensus, n_train)
where W[c, e] = 1/n_good_c if epoch e is good for consensus c, else 0.
Then: ``mean_c = W[c, :] @ X_train`` (for all c at once via matmul).

This reduces ~11 separate (index + mean) GPU operations to 1 matmul.

References
----------
.. [1] Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A.
       (2017). Autoreject: Automated artifact rejection for MEG and EEG data.
       NeuroImage, 159, 417-429. doi:10.1016/j.neuroimage.2017.06.030
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

from __future__ import annotations

from typing import Any

import numpy as np


def build_consensus_weights(bad_sensor_counts_train: np.ndarray,
                            consensus_values: np.ndarray | list[float],
                            n_channels: int,
                            picks: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build weight matrix for batched consensus scoring.

    Parameters
    ----------
    bad_sensor_counts_train : np.ndarray, shape (n_train,)
        Number of bad sensors per training epoch.
    consensus_values : array-like of float
        Consensus values to evaluate (e.g., [0.0, 0.1, ..., 1.0]).
    n_channels : int
        Number of channels.
    picks : array-like
        Channel indices (used for n_channels computation).

    Returns
    -------
    weights : np.ndarray, shape (n_consensus, n_train)
        Weight matrix. weights[c, e] = 1/n_good_c if epoch e is good
        for consensus value c, else 0. Rows with no good epochs are all-zero.
    valid_mask : np.ndarray, shape (n_consensus,)
        Boolean mask indicating which consensus values have valid scores
        (at least one good epoch).
    """
    n_train = len(bad_sensor_counts_train)
    n_consensus = len(consensus_values)

    weights = np.zeros((n_consensus, n_train), dtype=np.float32)
    valid_mask = np.zeros(n_consensus, dtype=bool)

    # Pre-sort once (shared across consensus values)
    sorted_idx = np.argsort(bad_sensor_counts_train)[::-1]
    sorted_counts = bad_sensor_counts_train[sorted_idx]

    for c_idx, this_consensus in enumerate(consensus_values):
        n_consensus_ch = this_consensus * n_channels

        # Determine bad epochs (same logic as _get_bad_epochs)
        bad_epochs = np.zeros(n_train, dtype=bool)
        if len(sorted_counts) > 0 and np.max(sorted_counts) >= n_consensus_ch:
            n_drop = np.sum(sorted_counts >= n_consensus_ch)
            bad_epochs[sorted_idx[:n_drop]] = True

        good_mask = ~bad_epochs
        n_good = good_mask.sum()

        if n_good > 0:
            weights[c_idx, good_mask] = 1.0 / n_good
            valid_mask[c_idx] = True

    return weights, valid_mask


def batched_consensus_score(X_train_interp_gpu: Any, median_gpu: Any,
                            weights_gpu: Any, valid_mask: np.ndarray,
                            torch_module: Any) -> np.ndarray:
    """Compute RMSE scores for all consensus values in one GPU operation.

    Parameters
    ----------
    X_train_interp_gpu : torch.Tensor, shape (n_train, n_channels, n_times)
        Interpolated training data on GPU.
    median_gpu : torch.Tensor, shape (n_channels, n_times)
        Test median on GPU.
    weights_gpu : torch.Tensor, shape (n_consensus, n_train)
        Weight matrix on GPU (from build_consensus_weights).
    valid_mask : np.ndarray, shape (n_consensus,)
        Which consensus values have valid scores.
    torch_module : module
        The torch module (for tensor ops).

    Returns
    -------
    scores : np.ndarray, shape (n_consensus,)
        RMSE scores. Invalid consensus values get -inf.
    """
    n_consensus = weights_gpu.shape[0]

    # Weighted mean for ALL consensus values at once:
    # mean_all[c, ch, t] = sum_e(weights[c, e] * X[e, ch, t])
    # einsum: (n_consensus, n_train) @ (n_train, n_ch, n_times)
    #       → (n_consensus, n_ch, n_times)
    mean_all = torch_module.einsum('ce,eij->cij', weights_gpu, X_train_interp_gpu)

    # RMSE for all consensus values:
    # sq_diff[c, ch, t] = (median[ch, t] - mean_all[c, ch, t])^2
    sq_diff = (median_gpu.unsqueeze(0) - mean_all) ** 2

    # rmse[c] = sqrt(mean over ch,t of sq_diff[c])
    rmse = sq_diff.mean(dim=(1, 2)).sqrt()  # (n_consensus,)

    # Transfer back and apply validity mask
    scores = torch_module.full((n_consensus,), float('-inf'),
                               device=rmse.device)
    valid_t = torch_module.tensor(valid_mask, device=rmse.device)
    scores[valid_t] = -rmse[valid_t]

    return scores.cpu().numpy()


def fast_median(tensor: Any, dim: int, torch_module: Any) -> Any:
    """Compute median using torch.topk instead of full sort.

    For small sizes along the median dimension (typical: 40-80 test epochs),
    topk finding the k-th element is faster than full sort because it uses
    a partial heap selection algorithm.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which to compute median.
    torch_module : module
        The torch module.

    Returns
    -------
    torch.Tensor
        Median values.
    """
    n = tensor.shape[dim]
    mid = n // 2

    if n % 2 == 1:
        # Odd: need (mid+1)-th largest = mid-th smallest
        # topk returns the k largest values
        vals, _ = torch_module.topk(tensor, mid + 1, dim=dim, largest=True,
                                    sorted=False)
        return vals.min(dim=dim).values
    else:
        # Even: need mid-th and (mid+1)-th largest, average them
        vals, _ = torch_module.topk(tensor, mid + 1, dim=dim, largest=True,
                                    sorted=False)
        # The two middle values are the min and second-min of top-(mid+1)
        top_min = vals.min(dim=dim).values

        vals2, _ = torch_module.topk(tensor, mid, dim=dim, largest=True,
                                     sorted=False)
        top2_min = vals2.min(dim=dim).values

        return (top_min + top2_min) / 2
