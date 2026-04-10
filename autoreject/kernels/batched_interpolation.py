"""Batched per-epoch interpolation by bad-channel pattern grouping.

Instead of a Python loop doing one matmul per epoch (400 iterations),
this groups epochs sharing the same bad-channel pattern and applies
a single torch.bmm() per group.

In typical EEG data, there are far fewer unique bad-channel patterns
than epochs (e.g., 20-50 patterns for 400 epochs), so this reduces
400 matmul launches to 20-50 bmm launches.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

from collections import defaultdict

import numpy as np


def batched_interpolate_epochs(data_gpu, interp_channels, interp_cache,
                               torch_module):
    """Apply interpolation using batched matmul grouped by pattern.

    Parameters
    ----------
    data_gpu : torch.Tensor, shape (n_epochs, n_picks, n_times)
        Data on GPU. Modified in-place.
    interp_channels : list of list of int
        Per-epoch list of bad channel indices (within picks).
    interp_cache : dict
        Mapping from bad-channel pattern (tuple) to
        (interpolation_matrix, good_idx, bad_idx).
    torch_module : module
        The torch module.

    Returns
    -------
    data_gpu : torch.Tensor
        Same tensor, modified in-place with interpolated channels.
    """
    device = data_gpu.device

    # Group epochs by bad-channel pattern
    pattern_groups = defaultdict(list)
    for epoch_idx, bad_ch_indices in enumerate(interp_channels):
        if len(bad_ch_indices) == 0:
            continue
        pattern_key = tuple(sorted(bad_ch_indices))
        pattern_groups[pattern_key].append(epoch_idx)

    # Process each group with batched matmul
    for pattern_key, epoch_indices in pattern_groups.items():
        if pattern_key not in interp_cache:
            continue

        interpolation, good_idx, bad_idx = interp_cache[pattern_key]
        n_bad = len(bad_idx)
        n_good = len(good_idx)
        n_epochs_group = len(epoch_indices)

        # Gather good channel data for all epochs in this group
        epoch_idx_t = torch_module.tensor(epoch_indices, device=device)

        # data_gpu[epoch_indices, good_idx, :] → (n_group, n_good, n_times)
        good_data = data_gpu[epoch_idx_t][:, good_idx, :]

        # Batched interpolation:
        # interp: (n_bad, n_good) → expand to (n_group, n_bad, n_good)
        # good_data: (n_group, n_good, n_times)
        # result: (n_group, n_bad, n_times)
        interp_expanded = interpolation.unsqueeze(0).expand(
            n_epochs_group, -1, -1
        )
        interpolated = torch_module.bmm(interp_expanded, good_data)

        # Scatter back
        # Use advanced indexing for batch assignment
        for i, epoch_idx in enumerate(epoch_indices):
            data_gpu[epoch_idx, bad_idx, :] = interpolated[i]

    return data_gpu
