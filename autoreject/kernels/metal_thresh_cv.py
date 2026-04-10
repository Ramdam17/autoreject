"""Metal kernel for fused threshold cross-validation scoring.

This kernel fuses the following PyTorch operations into a single GPU pass:
1. PTP threshold comparison: ptp[e, ch] <= threshold → boolean mask
2. Masked mean: sum of data for good epochs / n_good
3. RMSE: sqrt(mean((median - mean)^2))

The key advantage: eliminates O(n_train × n_ch × n_thresh) boolean tensor
and O(n_ch × n_times × n_thresh) masked_sum intermediate.

Kernel design
-------------
- Grid: (n_channels, n_thresh, 1) threadgroups
- Each threadgroup: 256 threads cooperating on the time dimension
- Phase 1: Determine which epochs pass threshold (shared memory bitmask)
- Phase 2: Each thread handles ceil(n_times/256) timepoints,
           accumulating masked mean and MSE contribution
- Phase 3: Tree reduction of per-thread MSE → final RMSE

Memory per threadgroup: ~4 KB shared memory
  - good_mask: n_train bools (≤ 1024)
  - mse_reduction: 256 floats (1024 bytes)
  - n_good: 1 uint (4 bytes)

References
----------
Jas et al. (2017). Autoreject. NeuroImage, 159, 417-429.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

from functools import lru_cache

import numpy as np

from . import METAL_AVAILABLE
from ._metal_dispatch import (
    compile_metal_function,
    dispatch_threadgroups,
    make_buffer_from_numpy,
    make_const_buffer,
    make_output_buffer,
    read_buffer_float32,
)

# Maximum number of train epochs supported by the kernel.
# Shared memory: MAX_TRAIN bools = MAX_TRAIN bytes.
MAX_TRAIN = 1024

# Threadgroup size — must match the shader constant.
THREADGROUP_SIZE = 256


# =========================================================================
# Metal Shader Source
# =========================================================================

_THRESH_CV_SHADER = """
#include <metal_stdlib>
using namespace metal;

// Must match Python-side constants
constant uint TG_SIZE = 256;
constant uint MAX_TRAIN = 1024;

kernel void thresh_cv_fused(
    device const float* data         [[buffer(0)]],   // (n_train, n_ch, n_times)
    device const float* ptp          [[buffer(1)]],   // (n_train, n_ch)
    device const float* threshes     [[buffer(2)]],   // (n_ch, n_thresh)
    device const float* median       [[buffer(3)]],   // (n_ch, n_times)
    device float* rmse_out           [[buffer(4)]],   // (n_ch, n_thresh)
    device const float* min_ptp      [[buffer(5)]],   // (n_ch,) - min PTP per channel
    constant uint& n_train           [[buffer(6)]],
    constant uint& n_ch              [[buffer(7)]],
    constant uint& n_times           [[buffer(8)]],
    constant uint& n_thresh          [[buffer(9)]],
    uint3 group_id    [[threadgroup_position_in_grid]],
    uint3 tid3        [[thread_position_in_threadgroup]],
    uint3 tg_size3    [[threads_per_threadgroup]])
{
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint ch = group_id.x;
    uint th = group_id.y;

    if (ch >= n_ch || th >= n_thresh) return;

    float threshold = threshes[ch * n_thresh + th];

    // ================================================================
    // Phase 1: Determine which epochs are "good" (PTP <= threshold)
    // Store as FLOAT mask in shared memory for branchless accumulation.
    // Using float instead of bool eliminates branch divergence in Phase 2.
    // ================================================================
    threadgroup float good_mask[MAX_TRAIN];  // 0.0 or 1.0
    threadgroup uint n_good_val;

    if (tid == 0) n_good_val = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread checks a subset of epochs
    uint local_count = 0;
    for (uint e = tid; e < n_train; e += tg_size) {
        float is_good = (ptp[e * n_ch + ch] <= threshold) ? 1.0f : 0.0f;
        good_mask[e] = is_good;
        if (is_good > 0.5f) local_count++;
    }

    // Reduce count across threads using shared memory
    threadgroup uint count_buf[TG_SIZE];
    count_buf[tid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for count
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            count_buf[tid] += count_buf[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) n_good_val = count_buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_good = n_good_val;

    // ================================================================
    // Fallback: if no epochs pass threshold, use epochs with min PTP
    // This matches the CPU behavior in batched_all_channels_cv_loss_parallel
    // ================================================================
    if (n_good == 0) {
        float min_ptp_ch = min_ptp[ch];

        if (tid == 0) n_good_val = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        local_count = 0;
        for (uint e = tid; e < n_train; e += tg_size) {
            float is_good = (ptp[e * n_ch + ch] <= min_ptp_ch) ? 1.0f : 0.0f;
            good_mask[e] = is_good;
            if (is_good > 0.5f) local_count++;
        }

        count_buf[tid] = local_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                count_buf[tid] += count_buf[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) n_good_val = count_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        n_good = n_good_val;
    }

    // If still no good epochs (edge case), output infinity
    if (n_good == 0) {
        if (tid == 0) {
            rmse_out[ch * n_thresh + th] = INFINITY;
        }
        return;
    }

    float inv_n_good = 1.0f / float(n_good);

    // ================================================================
    // Phase 2: Each thread computes MSE for its assigned timepoints.
    // Thread tid handles timepoints: tid, tid+tg_size, tid+2*tg_size, ...
    //
    // BRANCHLESS: multiply by good_mask[e] (0.0/1.0) instead of branching.
    // This eliminates warp/SIMD divergence in the inner loop — all threads
    // execute the same instruction sequence regardless of mask values.
    // Data layout: (n_train, n_ch, n_times) — original format, no copy.
    // ================================================================
    float local_mse = 0.0f;

    for (uint t = tid; t < n_times; t += tg_size) {
        float masked_sum = 0.0f;
        float med = median[ch * n_times + t];

        // Branchless accumulation — every epoch contributes 0 or data value
        for (uint e = 0; e < n_train; e++) {
            masked_sum += data[(e * n_ch + ch) * n_times + t] * good_mask[e];
        }

        float mean_val = masked_sum * inv_n_good;
        float diff = med - mean_val;
        local_mse += diff * diff;
    }

    // ================================================================
    // Phase 3: Tree reduction of MSE across threads
    // ================================================================
    threadgroup float mse_buf[TG_SIZE];
    mse_buf[tid] = local_mse;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mse_buf[tid] += mse_buf[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the final RMSE
    if (tid == 0) {
        float mse = mse_buf[0] / float(n_times);
        rmse_out[ch * n_thresh + th] = sqrt(mse);
    }
}
"""


# =========================================================================
# Compilation (cached)
# =========================================================================


@lru_cache(maxsize=1)
def _compile_thresh_cv():
    """Compile the fused threshold-CV Metal kernel. Cached."""
    return compile_metal_function(_THRESH_CV_SHADER, "thresh_cv_fused")


# =========================================================================
# Python dispatch
# =========================================================================


def metal_batched_cv_loss(data_train, ptp_train, threshes_all, median_test):
    """Compute CV loss for all channels and thresholds using Metal kernel.

    Drop-in replacement for
    ``GPUThresholdOptimizer.batched_all_channels_cv_loss_parallel()``
    (one fold at a time).

    Parameters
    ----------
    data_train : np.ndarray, shape (n_train, n_channels, n_times)
        Training data for this fold. float32.
    ptp_train : np.ndarray, shape (n_train, n_channels)
        Peak-to-peak values for training epochs. float32.
    threshes_all : np.ndarray, shape (n_channels, n_thresh)
        Threshold values for all channels. float32.
    median_test : np.ndarray, shape (n_channels, n_times)
        Median of test data for this fold. float32.

    Returns
    -------
    rmse : np.ndarray, shape (n_channels, n_thresh)
        RMSE loss for each channel and threshold. float32.
    """
    if not METAL_AVAILABLE:
        raise RuntimeError("Metal not available — install PyObjC")

    device, pipeline = _compile_thresh_cv()

    n_train, n_channels, n_times = data_train.shape
    n_thresh = threshes_all.shape[1]

    if n_train > MAX_TRAIN:
        raise ValueError(
            f"n_train={n_train} exceeds MAX_TRAIN={MAX_TRAIN}. "
            "Increase MAX_TRAIN in the shader source."
        )

    # Ensure float32 and contiguous
    data_train = np.ascontiguousarray(data_train, dtype=np.float32)
    ptp_train = np.ascontiguousarray(ptp_train, dtype=np.float32)
    threshes_all = np.ascontiguousarray(threshes_all, dtype=np.float32)
    median_test = np.ascontiguousarray(median_test, dtype=np.float32)

    # Compute min PTP per channel for fallback logic
    min_ptp = ptp_train.min(axis=0).astype(np.float32)  # (n_channels,)

    # Create Metal buffers
    buf_data = make_buffer_from_numpy(device, data_train)
    buf_ptp = make_buffer_from_numpy(device, ptp_train)
    buf_thresh = make_buffer_from_numpy(device, threshes_all)
    buf_median = make_buffer_from_numpy(device, median_test)
    buf_min_ptp = make_buffer_from_numpy(device, min_ptp)

    out_nbytes = n_channels * n_thresh * 4  # float32
    buf_out = make_output_buffer(device, out_nbytes)

    # Build buffer list with indices
    buffers = [
        (buf_data, 0),
        (buf_ptp, 1),
        (buf_thresh, 2),
        (buf_median, 3),
        (buf_out, 4),
        (buf_min_ptp, 5),
        (make_const_buffer(device, n_train), 6),
        (make_const_buffer(device, n_channels), 7),
        (make_const_buffer(device, n_times), 8),
        (make_const_buffer(device, n_thresh), 9),
    ]

    # Dispatch: one threadgroup per (channel, threshold) pair
    grid_size = (n_channels, n_thresh, 1)

    dispatch_threadgroups(
        device, pipeline, buffers, grid_size,
        threadgroup_size=THREADGROUP_SIZE,
    )

    # Read result
    return read_buffer_float32(buf_out, (n_channels, n_thresh))
