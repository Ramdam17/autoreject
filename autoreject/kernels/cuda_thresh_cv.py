"""CUDA kernel for fused threshold cross-validation scoring.

Same algorithm as the Metal version but using float64 for exact precision
on NVIDIA GPUs (A100: 9.7 TFLOPS fp64, RTX 4090: ~1.3 TFLOPS fp64).

Uses CuPy RawKernel for inline CUDA source.

References
----------
Jas et al. (2017). Autoreject. NeuroImage, 159, 417-429.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import numpy as np

from . import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


MAX_TRAIN = 1024
BLOCK_SIZE = 256


# =========================================================================
# CUDA Kernel Source
# =========================================================================

_THRESH_CV_CUDA = r"""
extern "C" __global__ void thresh_cv_fused(
    const double* __restrict__ data,       // (n_train, n_ch, n_times)
    const double* __restrict__ ptp,        // (n_train, n_ch)
    const double* __restrict__ threshes,   // (n_ch, n_thresh)
    const double* __restrict__ median,     // (n_ch, n_times)
    double* __restrict__ rmse_out,         // (n_ch, n_thresh)
    const double* __restrict__ min_ptp,    // (n_ch,)
    int n_train,
    int n_ch,
    int n_times,
    int n_thresh)
{
    // Block = one (channel, threshold) pair
    // blockIdx.x = ch * n_thresh + th (flattened)
    int flat_idx = blockIdx.x;
    if (flat_idx >= n_ch * n_thresh) return;

    int ch = flat_idx / n_thresh;
    int th = flat_idx % n_thresh;
    int tid = threadIdx.x;
    int tg_size = blockDim.x;

    double threshold = threshes[ch * n_thresh + th];

    // ================================================================
    // Phase 1: Determine good epochs
    // ================================================================
    __shared__ double good_mask[1024];  // MAX_TRAIN, 0.0 or 1.0 for branchless
    __shared__ unsigned int n_good_shared;

    if (tid == 0) n_good_shared = 0;
    __syncthreads();

    // Each thread checks a subset of epochs (float mask for branchless)
    unsigned int local_count = 0;
    for (int e = tid; e < n_train; e += tg_size) {
        double is_good = (ptp[e * n_ch + ch] <= threshold) ? 1.0 : 0.0;
        good_mask[e] = is_good;
        if (is_good > 0.5) local_count++;
    }

    // Shared memory reduction for count
    __shared__ unsigned int count_buf[256];
    count_buf[tid] = local_count;
    __syncthreads();

    for (int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            count_buf[tid] += count_buf[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) n_good_shared = count_buf[0];
    __syncthreads();

    unsigned int n_good = n_good_shared;

    // ================================================================
    // Fallback: use min-PTP epochs if threshold too low
    // ================================================================
    if (n_good == 0) {
        double min_ptp_ch = min_ptp[ch];

        if (tid == 0) n_good_shared = 0;
        __syncthreads();

        local_count = 0;
        for (int e = tid; e < n_train; e += tg_size) {
            double is_good = (ptp[e * n_ch + ch] <= min_ptp_ch) ? 1.0 : 0.0;
            good_mask[e] = is_good;
            if (is_good > 0.5) local_count++;
        }

        count_buf[tid] = local_count;
        __syncthreads();

        for (int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                count_buf[tid] += count_buf[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) n_good_shared = count_buf[0];
        __syncthreads();

        n_good = n_good_shared;
    }

    if (n_good == 0) {
        if (tid == 0) rmse_out[ch * n_thresh + th] = INFINITY;
        return;
    }

    double inv_n_good = 1.0 / (double)n_good;

    // ================================================================
    // Phase 2: Compute MSE across timepoints
    // BRANCHLESS: multiply by good_mask (0.0/1.0) instead of branching
    // ================================================================
    double local_mse = 0.0;

    for (int t = tid; t < n_times; t += tg_size) {
        double masked_sum = 0.0;
        double med = median[ch * n_times + t];

        for (int e = 0; e < n_train; e++) {
            masked_sum += data[(e * n_ch + ch) * n_times + t] * good_mask[e];
        }

        double mean_val = masked_sum * inv_n_good;
        double diff = med - mean_val;
        local_mse += diff * diff;
    }

    // ================================================================
    // Phase 3: Tree reduction of MSE
    // ================================================================
    __shared__ double mse_buf[256];
    mse_buf[tid] = local_mse;
    __syncthreads();

    for (int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mse_buf[tid] += mse_buf[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double mse = mse_buf[0] / (double)n_times;
        rmse_out[ch * n_thresh + th] = sqrt(mse);
    }
}
"""


# =========================================================================
# Compilation (cached)
# =========================================================================

_kernel = None


def _get_kernel():
    """Get or compile the CUDA kernel (singleton)."""
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_THRESH_CV_CUDA, "thresh_cv_fused")
    return _kernel


# =========================================================================
# Python dispatch
# =========================================================================


def cuda_batched_cv_loss(data_train, ptp_train, threshes_all, median_test):
    """Compute CV loss for all channels and thresholds using CUDA kernel.

    Drop-in replacement for
    ``GPUThresholdOptimizer.batched_all_channels_cv_loss_parallel()``
    (one fold at a time).

    Parameters
    ----------
    data_train : np.ndarray, shape (n_train, n_channels, n_times)
        Training data for this fold. float64.
    ptp_train : np.ndarray, shape (n_train, n_channels)
        Peak-to-peak values for training epochs. float64.
    threshes_all : np.ndarray, shape (n_channels, n_thresh)
        Threshold values for all channels. float64.
    median_test : np.ndarray, shape (n_channels, n_times)
        Median of test data for this fold. float64.

    Returns
    -------
    rmse : np.ndarray, shape (n_channels, n_thresh)
        RMSE loss for each channel and threshold. float64.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available — install cupy-cuda*")

    kernel = _get_kernel()

    n_train, n_channels, n_times = data_train.shape
    n_thresh = threshes_all.shape[1]

    if n_train > MAX_TRAIN:
        raise ValueError(
            f"n_train={n_train} exceeds MAX_TRAIN={MAX_TRAIN}."
        )

    # Transfer to GPU as float64
    d_data = cp.asarray(np.ascontiguousarray(data_train), dtype=cp.float64)
    d_ptp = cp.asarray(np.ascontiguousarray(ptp_train), dtype=cp.float64)
    d_thresh = cp.asarray(np.ascontiguousarray(threshes_all), dtype=cp.float64)
    d_median = cp.asarray(np.ascontiguousarray(median_test), dtype=cp.float64)

    # Min PTP per channel for fallback
    d_min_ptp = d_ptp.min(axis=0)  # (n_channels,)

    d_out = cp.zeros((n_channels, n_thresh), dtype=cp.float64)

    # Dispatch: one block per (channel, threshold) pair
    total_blocks = n_channels * n_thresh
    grid_size = (total_blocks,)
    block_size = (BLOCK_SIZE,)

    kernel(
        grid_size, block_size,
        (d_data, d_ptp, d_thresh, d_median, d_out, d_min_ptp,
         n_train, n_channels, n_times, n_thresh),
    )

    return cp.asnumpy(d_out)
