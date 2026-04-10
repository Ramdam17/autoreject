"""Tests for custom Metal/CUDA kernels.

Validates that kernel outputs match the PyTorch reference implementation
(GPUThresholdOptimizer.batched_all_channels_cv_loss_parallel) to within
acceptable tolerance.

MPS float32: rtol=1e-4, atol=1e-5 (float32 precision)
CUDA float64: rtol=1e-10, atol=1e-12 (float64 precision)
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import numpy as np
import pytest

from autoreject.kernels import METAL_AVAILABLE, CUPY_AVAILABLE


# =========================================================================
# NumPy reference implementation (CPU, float64 gold standard)
# =========================================================================


def numpy_batched_cv_loss(data_train, ptp_train, threshes_all, median_test):
    """Pure NumPy reference for threshold CV loss.

    This is the gold standard: no GPU, no PyTorch, float64.
    Matches the logic of batched_all_channels_cv_loss_parallel exactly.

    Parameters
    ----------
    data_train : np.ndarray, shape (n_train, n_channels, n_times)
    ptp_train : np.ndarray, shape (n_train, n_channels)
    threshes_all : np.ndarray, shape (n_channels, n_thresh)
    median_test : np.ndarray, shape (n_channels, n_times)

    Returns
    -------
    rmse : np.ndarray, shape (n_channels, n_thresh)
    """
    n_train, n_channels, n_times = data_train.shape
    n_thresh = threshes_all.shape[1]

    rmse = np.zeros((n_channels, n_thresh), dtype=np.float64)

    for ch in range(n_channels):
        for th_idx in range(n_thresh):
            threshold = threshes_all[ch, th_idx]

            # Determine good epochs
            good_mask = ptp_train[:, ch] <= threshold
            n_good = good_mask.sum()

            # Fallback: use min-PTP epochs if none pass
            if n_good == 0:
                min_ptp = ptp_train[:, ch].min()
                good_mask = ptp_train[:, ch] <= min_ptp
                n_good = good_mask.sum()

            if n_good == 0:
                rmse[ch, th_idx] = np.inf
                continue

            # Masked mean
            mean_train = data_train[good_mask, ch, :].mean(axis=0)

            # RMSE
            sq_diff = (median_test[ch, :] - mean_train) ** 2
            rmse[ch, th_idx] = np.sqrt(sq_diff.mean())

    return rmse


# =========================================================================
# Test data generation
# =========================================================================


def make_test_data(n_train, n_channels, n_times, n_thresh=None,
                   random_state=42):
    """Generate synthetic data for kernel testing.

    Returns data with realistic PTP distribution where some epochs
    have higher amplitude (simulating artifacts).
    """
    rng = np.random.RandomState(random_state)

    # Realistic EEG-like data (~10 µV scale)
    data_train = rng.randn(n_train, n_channels, n_times).astype(np.float64)
    data_train *= 1e-5

    # Add channel-specific variance
    channel_scales = rng.uniform(0.5, 2.0, size=n_channels)
    data_train *= channel_scales[np.newaxis, :, np.newaxis]

    # Add a few artifact epochs
    n_bad = max(1, n_train // 10)
    bad_idx = rng.choice(n_train, n_bad, replace=False)
    data_train[bad_idx] *= rng.uniform(2.0, 5.0, size=(n_bad, 1, 1))

    # PTP
    ptp_train = data_train.max(axis=-1) - data_train.min(axis=-1)

    # Thresholds: sorted PTPs per channel
    if n_thresh is None:
        n_thresh = n_train
    threshes_all = np.zeros((n_channels, n_thresh), dtype=np.float64)
    for ch in range(n_channels):
        sorted_ptp = np.sort(ptp_train[:, ch])
        if n_thresh == n_train:
            threshes_all[ch] = sorted_ptp
        else:
            idx = np.linspace(0, n_train - 1, n_thresh, dtype=int)
            threshes_all[ch] = sorted_ptp[idx]

    # Median test (use random subset as "test" median)
    test_idx = rng.choice(n_train, max(1, n_train // 5), replace=False)
    median_test = np.median(data_train[test_idx], axis=0)

    return data_train, ptp_train, threshes_all, median_test


# =========================================================================
# Metal kernel tests
# =========================================================================


@pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
class TestMetalThreshCV:
    """Test Metal fused threshold-CV kernel against NumPy reference."""

    def test_small(self):
        """Small data: 20 train × 4 channels × 100 times."""
        from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

        data, ptp, thresh, median = make_test_data(20, 4, 100)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = metal_batched_cv_loss(
            data.astype(np.float32), ptp.astype(np.float32),
            thresh.astype(np.float32), median.astype(np.float32),
        )

        np.testing.assert_allclose(result, ref.astype(np.float32),
                                   rtol=1e-4, atol=1e-5)

    def test_medium(self):
        """Medium data: 160 train × 32 channels × 500 times."""
        from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

        data, ptp, thresh, median = make_test_data(160, 32, 500)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = metal_batched_cv_loss(
            data.astype(np.float32), ptp.astype(np.float32),
            thresh.astype(np.float32), median.astype(np.float32),
        )

        np.testing.assert_allclose(result, ref.astype(np.float32),
                                   rtol=1e-4, atol=1e-5)

    def test_realistic(self):
        """Realistic data: 320 train × 64 channels × 1000 times."""
        from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

        data, ptp, thresh, median = make_test_data(320, 64, 1000)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = metal_batched_cv_loss(
            data.astype(np.float32), ptp.astype(np.float32),
            thresh.astype(np.float32), median.astype(np.float32),
        )

        np.testing.assert_allclose(result, ref.astype(np.float32),
                                   rtol=1e-3, atol=1e-4)

    def test_edge_all_good(self):
        """Edge case: threshold so high all epochs pass."""
        from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

        data, ptp, thresh, median = make_test_data(20, 4, 100)
        # Set all thresholds very high
        thresh[:] = 1e10

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = metal_batched_cv_loss(
            data.astype(np.float32), ptp.astype(np.float32),
            thresh.astype(np.float32), median.astype(np.float32),
        )

        np.testing.assert_allclose(result, ref.astype(np.float32),
                                   rtol=1e-4, atol=1e-5)

    def test_edge_none_good_fallback(self):
        """Edge case: threshold so low that fallback to min-PTP kicks in."""
        from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

        data, ptp, thresh, median = make_test_data(20, 4, 100)
        # Set first threshold per channel below minimum PTP
        # This forces the fallback path
        for ch in range(4):
            thresh[ch, 0] = ptp[:, ch].min() * 0.5

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = metal_batched_cv_loss(
            data.astype(np.float32), ptp.astype(np.float32),
            thresh.astype(np.float32), median.astype(np.float32),
        )

        np.testing.assert_allclose(result, ref.astype(np.float32),
                                   rtol=1e-4, atol=1e-5)

    def test_single_channel(self):
        """Edge case: only 1 channel."""
        from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

        data, ptp, thresh, median = make_test_data(20, 1, 100)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = metal_batched_cv_loss(
            data.astype(np.float32), ptp.astype(np.float32),
            thresh.astype(np.float32), median.astype(np.float32),
        )

        np.testing.assert_allclose(result, ref.astype(np.float32),
                                   rtol=1e-4, atol=1e-5)

    def test_single_threshold(self):
        """Edge case: only 1 threshold per channel."""
        from autoreject.kernels.metal_thresh_cv import metal_batched_cv_loss

        data, ptp, thresh, median = make_test_data(20, 4, 100, n_thresh=1)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = metal_batched_cv_loss(
            data.astype(np.float32), ptp.astype(np.float32),
            thresh.astype(np.float32), median.astype(np.float32),
        )

        np.testing.assert_allclose(result, ref.astype(np.float32),
                                   rtol=1e-4, atol=1e-5)


# =========================================================================
# CUDA kernel tests
# =========================================================================


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestCUDAThreshCV:
    """Test CUDA fused threshold-CV kernel against NumPy reference."""

    def test_small(self):
        """Small data: 20 train × 4 channels × 100 times."""
        from autoreject.kernels.cuda_thresh_cv import cuda_batched_cv_loss

        data, ptp, thresh, median = make_test_data(20, 4, 100)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = cuda_batched_cv_loss(data, ptp, thresh, median)

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)

    def test_medium(self):
        """Medium data: 160 train × 32 channels × 500 times."""
        from autoreject.kernels.cuda_thresh_cv import cuda_batched_cv_loss

        data, ptp, thresh, median = make_test_data(160, 32, 500)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = cuda_batched_cv_loss(data, ptp, thresh, median)

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)

    def test_realistic(self):
        """Realistic data: 320 train × 64 channels × 1000 times."""
        from autoreject.kernels.cuda_thresh_cv import cuda_batched_cv_loss

        data, ptp, thresh, median = make_test_data(320, 64, 1000)

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = cuda_batched_cv_loss(data, ptp, thresh, median)

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)

    def test_edge_fallback(self):
        """Edge case: fallback to min-PTP."""
        from autoreject.kernels.cuda_thresh_cv import cuda_batched_cv_loss

        data, ptp, thresh, median = make_test_data(20, 4, 100)
        for ch in range(4):
            thresh[ch, 0] = ptp[:, ch].min() * 0.5

        ref = numpy_batched_cv_loss(data, ptp, thresh, median)
        result = cuda_batched_cv_loss(data, ptp, thresh, median)

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-12)


# =========================================================================
# NumPy reference self-test
# =========================================================================


class TestNumpyReference:
    """Sanity checks for the NumPy reference implementation."""

    def test_all_same_threshold(self):
        """If all thresholds equal, all should give same RMSE."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 2, 50)
        ptp = data.max(axis=-1) - data.min(axis=-1)
        median = np.median(data[:5], axis=0)

        # All thresholds set to max PTP (all epochs pass)
        max_ptp = ptp.max()
        thresh = np.full((2, 10), max_ptp)

        result = numpy_batched_cv_loss(data, ptp, thresh, median)

        # All thresholds should give identical RMSE
        for ch in range(2):
            np.testing.assert_allclose(
                result[ch, :], result[ch, 0],
                rtol=1e-14,
            )

    def test_shape(self):
        """Output shape should be (n_channels, n_thresh)."""
        data, ptp, thresh, median = make_test_data(20, 4, 100)
        result = numpy_batched_cv_loss(data, ptp, thresh, median)
        assert result.shape == (4, 20)

    def test_non_negative(self):
        """RMSE should be non-negative."""
        data, ptp, thresh, median = make_test_data(20, 4, 100)
        result = numpy_batched_cv_loss(data, ptp, thresh, median)
        assert np.all(result >= 0)
