"""Tests for GPU pipeline module."""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import os

import numpy as np
import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from autoreject.backends import use_backend  # noqa: E402
from autoreject.gpu_pipeline import (  # noqa: E402
    GPUThresholdOptimizer,
    _torch_median,
    _get_device,
    is_gpu_available,
    should_use_gpu,
)


@pytest.fixture(autouse=True)
def setup_torch_backend():
    """Force torch backend for all tests in this module."""
    with use_backend('torch'):
        yield


# ─── _torch_median ────────────────────────────────────────────────────


class TestTorchMedian:
    """Tests for _torch_median (matches numpy.median behavior)."""

    def test_odd_length(self):
        """Odd-length: torch.median and numpy.median agree."""
        arr = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        t = torch.tensor(arr)
        result = _torch_median(t, dim=0).item()
        assert result == pytest.approx(np.median(arr))

    def test_even_length(self):
        """Even-length: _torch_median averages the two middle values."""
        arr = np.array([3.0, 1.0, 4.0, 2.0])
        t = torch.tensor(arr)
        result = _torch_median(t, dim=0).item()
        expected = np.median(arr)  # 2.5
        assert result == pytest.approx(expected)

    def test_2d_along_rows(self):
        """2D median along axis=0 matches numpy."""
        arr = np.random.RandomState(42).randn(5, 3).astype(np.float32)
        t = torch.tensor(arr)
        result = _torch_median(t, dim=0).numpy()
        np.testing.assert_allclose(result, np.median(arr, axis=0),
                                   rtol=1e-5)

    def test_2d_along_cols(self):
        """2D median along axis=1 matches numpy."""
        arr = np.random.RandomState(42).randn(4, 6).astype(np.float32)
        t = torch.tensor(arr)
        result = _torch_median(t, dim=1).numpy()
        np.testing.assert_allclose(result, np.median(arr, axis=1),
                                   rtol=1e-5)

    def test_single_element(self):
        """Single-element array returns that element."""
        t = torch.tensor([42.0])
        result = _torch_median(t, dim=0).item()
        assert result == pytest.approx(42.0)


# ─── GPUThresholdOptimizer ────────────────────────────────────────────


class TestGPUThresholdOptimizer:
    """Tests for GPUThresholdOptimizer."""

    def test_init_auto_device(self):
        """Auto-detected device is a valid string."""
        opt = GPUThresholdOptimizer()
        assert opt.device in ('mps', 'cuda', 'cpu')

    def test_init_explicit_cpu(self):
        """Explicit device='cpu' is respected."""
        opt = GPUThresholdOptimizer(device='cpu')
        assert opt.device == 'cpu'

    def test_compute_ptp_1d_matches_numpy(self):
        """compute_ptp_1d matches np.ptp on 2D data."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 100).astype(np.float32)
        opt = GPUThresholdOptimizer(device='cpu')
        result = opt.compute_ptp_1d(data).cpu().numpy()
        expected = np.ptp(data, axis=1)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_compute_ptp_2d_matches_numpy(self):
        """compute_ptp_2d matches np.ptp on 3D data."""
        rng = np.random.RandomState(42)
        data = rng.randn(20, 5, 100).astype(np.float32)
        opt = GPUThresholdOptimizer(device='cpu')
        result = opt.compute_ptp_2d(data).cpu().numpy()
        expected = np.ptp(data, axis=-1)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_tensor_caching(self):
        """Cached tensors are reused on repeated calls."""
        opt = GPUThresholdOptimizer(device='cpu')
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t1 = opt._to_tensor(data, cache_key='test_data')
        t2 = opt._to_tensor(data, cache_key='test_data')
        # Same object — not just equal values
        assert t1 is t2

    def test_clear_cache(self):
        """clear_cache empties the cache."""
        opt = GPUThresholdOptimizer(device='cpu')
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        opt._to_tensor(data, cache_key='test')
        assert len(opt._cache) == 1
        opt.clear_cache()
        assert len(opt._cache) == 0

    def test_batched_channel_cv_loss_shape(self):
        """batched_channel_cv_loss returns array with correct shape."""
        from sklearn.model_selection import KFold

        rng = np.random.RandomState(42)
        n_epochs, n_times = 60, 100
        data_np = rng.randn(n_epochs, n_times).astype(np.float32)
        # batched_channel_cv_loss expects a tensor on device
        data_1d = torch.tensor(data_np, device='cpu')
        # Thresholds must span the ptp range of the data
        ptps = np.ptp(data_np, axis=1)
        thresholds = np.linspace(ptps.min(), ptps.max(), 10)

        cv = KFold(n_splits=5)
        splits = list(cv.split(data_np))

        opt = GPUThresholdOptimizer(device='cpu')
        loss = opt.batched_channel_cv_loss(data_1d, thresholds, splits)

        # Shape: (n_thresholds,) — mean across folds
        assert loss.shape == (len(thresholds),)
        # Most thresholds should produce finite loss
        # (the lowest threshold may reject all epochs -> inf)
        assert torch.any(torch.isfinite(loss))

    def test_batched_cv_loss_matches_lower_threshold(self):
        """Lower thresholds reject more epochs, giving higher loss."""
        from sklearn.model_selection import KFold

        rng = np.random.RandomState(42)
        data_np = rng.randn(60, 100).astype(np.float32)
        data_1d = torch.tensor(data_np, device='cpu')
        # Very low threshold vs very high threshold
        thresholds = np.array([0.01, 100.0])

        cv = KFold(n_splits=3)
        splits = list(cv.split(data_np))

        opt = GPUThresholdOptimizer(device='cpu')
        loss = opt.batched_channel_cv_loss(data_1d, thresholds, splits)

        # Low threshold -> more rejection -> higher loss
        assert loss[0].item() >= loss[1].item()


# ─── should_use_gpu ───────────────────────────────────────────────────


class TestShouldUseGPU:
    """Tests for should_use_gpu heuristic."""

    def test_cpu_device_returns_false(self):
        """Explicit device='cpu' always returns False."""
        use, dev = should_use_gpu(1000, 64, device='cpu')
        assert use is False
        assert dev == 'cpu'

    def test_small_dataset_returns_false(self):
        """Datasets with <50 epochs use CPU."""
        use, dev = should_use_gpu(30, 64)
        assert use is False
        assert dev == 'cpu'

    def test_env_variable_forces_cpu(self):
        """AUTOREJECT_BACKEND=numpy forces CPU."""
        old = os.environ.get('AUTOREJECT_BACKEND')
        try:
            os.environ['AUTOREJECT_BACKEND'] = 'numpy'
            use, dev = should_use_gpu(1000, 64)
            assert use is False
            assert dev == 'cpu'
        finally:
            if old is None:
                os.environ.pop('AUTOREJECT_BACKEND', None)
            else:
                os.environ['AUTOREJECT_BACKEND'] = old

    def test_large_dataset_uses_gpu_if_available(self):
        """Large datasets (>50 epochs) use GPU when available."""
        use, dev = should_use_gpu(200, 64)
        if is_gpu_available():
            assert use is True
            assert dev in ('mps', 'cuda')
        else:
            assert use is False
            assert dev == 'cpu'


# ─── is_gpu_available ─────────────────────────────────────────────────


class TestIsGpuAvailable:
    """Tests for is_gpu_available."""

    def test_returns_bool(self):
        """Return type is bool."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_consistent_with_device(self):
        """If GPU available, _get_device is not 'cpu'."""
        if is_gpu_available():
            assert _get_device() in ('mps', 'cuda')
        else:
            assert _get_device() in ('cpu', None)


# ─── Integration tests ────────────────────────────────────────────────


@pytest.mark.slow
class TestComputeThresholdsGPUIntegration:
    """Integration tests comparing GPU vs CPU thresholds."""

    def test_thresholds_match_cpu(self):
        """GPU thresholds should be close to CPU thresholds."""
        import mne
        from autoreject.gpu_pipeline import compute_thresholds_gpu
        from autoreject.autoreject import _compute_thresholds

        # Create synthetic epochs with valid channel positions
        rng = np.random.RandomState(42)
        n_channels, n_times = 10, 100
        n_epochs = 60
        sfreq = 100.0
        # Use standard montage channel names for valid positions
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_names = montage.ch_names[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                               ch_types='eeg')
        info.set_montage(montage)
        data = rng.randn(n_epochs, n_channels, n_times) * 1e-5
        epochs = mne.EpochsArray(data, info)

        picks = np.arange(n_channels)

        # CPU thresholds
        cpu_threshes = _compute_thresholds(
            epochs, picks=picks, method='bayesian_optimization',
            random_state=42)

        # GPU thresholds
        gpu_threshes = compute_thresholds_gpu(
            epochs, picks=picks, method='bayesian_optimization',
            random_state=42, device='cpu')

        # Same channel names
        assert set(cpu_threshes.keys()) == set(gpu_threshes.keys())

        # Thresholds should be in the same ballpark
        # Note: argmin vs BO may give different exact values,
        # but both should be valid thresholds for each channel
        for ch in cpu_threshes:
            cpu_t = cpu_threshes[ch]
            gpu_t = gpu_threshes[ch]
            # Both should be positive
            assert cpu_t > 0
            assert gpu_t > 0
            # Both should be in a reasonable range relative to each other
            ratio = max(cpu_t, gpu_t) / min(cpu_t, gpu_t)
            assert ratio < 5.0, (
                f'Threshold ratio for {ch} is {ratio:.1f}x '
                f'(cpu={cpu_t:.6f}, gpu={gpu_t:.6f})')

    def test_returns_dict_with_channel_names(self):
        """compute_thresholds_gpu returns dict with correct channel names."""
        import mne
        from autoreject.gpu_pipeline import compute_thresholds_gpu

        rng = np.random.RandomState(42)
        n_channels, n_times = 5, 100
        n_epochs = 60
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_names = montage.ch_names[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=100.0,
                               ch_types='eeg')
        info.set_montage(montage)
        data = rng.randn(n_epochs, n_channels, n_times) * 1e-5
        epochs = mne.EpochsArray(data, info)
        picks = np.arange(n_channels)

        threshes = compute_thresholds_gpu(
            epochs, picks=picks, device='cpu')

        assert isinstance(threshes, dict)
        assert set(threshes.keys()) == set(ch_names)
        for v in threshes.values():
            assert isinstance(v, float)
            assert v > 0
