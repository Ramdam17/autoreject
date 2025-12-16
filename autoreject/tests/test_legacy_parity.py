"""Test parity between legacy CPU implementation and current implementation.

This module ensures that when using AUTOREJECT_BACKEND='numpy', the current
implementation produces IDENTICAL results to the legacy/official code.

This is critical for:
1. Validating that GPU optimizations don't break CPU fallback
2. Ensuring PR to official repo maintains backward compatibility
3. Detecting any numerical divergences early

Usage:
    AUTOREJECT_BACKEND=numpy pytest autoreject/tests/test_legacy_parity.py -v
"""

import os
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

import mne
from mne.datasets import testing

# Get paths
REPO_ROOT = Path(__file__).parent.parent.parent

# Test data
data_path = testing.data_path(download=False)
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def force_numpy_backend():
    """Force NumPy backend for duration of test."""
    original_value = os.environ.get('AUTOREJECT_BACKEND', None)
    os.environ['AUTOREJECT_BACKEND'] = 'numpy'
    
    # Clear backend cache to force re-detection
    from autoreject.backends import _BACKEND_CACHE
    _BACKEND_CACHE.clear()
    
    yield
    
    # Restore original value
    if original_value is None:
        if 'AUTOREJECT_BACKEND' in os.environ:
            del os.environ['AUTOREJECT_BACKEND']
    else:
        os.environ['AUTOREJECT_BACKEND'] = original_value
    
    # Clear cache again
    _BACKEND_CACHE.clear()


@pytest.fixture
def synthetic_epochs():
    """Create synthetic epochs for fast testing with proper montage."""
    np.random.seed(42)
    
    n_epochs = 30
    n_times = 100
    sfreq = 250.0
    
    # Use standard 10-20 montage channel names
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:21]  # Use first 21 channels
    n_channels = len(ch_names)
    
    # Create synthetic data with some bad epochs
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
    
    # Add some artifacts to specific epochs
    bad_epoch_indices = [5, 12, 23]
    for idx in bad_epoch_indices:
        # Add large amplitude artifact to random channel
        data[idx, idx % n_channels, :] *= 10
    
    # Create MNE info and epochs
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )
    info.set_montage(montage)
    
    epochs = mne.EpochsArray(data, info)
    
    return epochs


@pytest.fixture
def real_epochs():
    """Create real epochs from MNE testing data."""
    if not data_path.exists():
        pytest.skip("MNE testing data not available")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
        raw.del_proj()
        raw.info['bads'] = []
        
        events = mne.find_events(raw, verbose=False)
        
        # Pick only EEG channels for simpler testing
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                               eog=False, exclude=[])
        
        epochs = mne.Epochs(
            raw, events, event_id=None, tmin=-0.2, tmax=0.5,
            picks=picks, baseline=(None, 0), decim=10,
            reject=None, preload=True, verbose=False
        )[:15]  # Limit epochs for speed
    
    return epochs


# =============================================================================
# Test: AutoReject.fit() - Legacy vs Current PARITY
# =============================================================================

@pytest.mark.slow
@pytest.mark.legacy_parity
def test_autoreject_fit_matches_legacy(force_numpy_backend, synthetic_epochs):
    """Test that current AutoReject.fit() produces IDENTICAL results to legacy.
    
    This is the main end-to-end parity test.
    """
    # Import both implementations
    from autoreject import AutoReject as CurrentAutoReject
    from legacy.autoreject_original import AutoReject as LegacyAutoReject
    
    epochs = synthetic_epochs
    random_state = 42
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Current implementation
        ar_current = CurrentAutoReject(
            n_interpolate=[1, 2],
            consensus=[0.5, 1.0],
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        ar_current.fit(epochs.copy())
        
        # Legacy implementation
        ar_legacy = LegacyAutoReject(
            n_interpolate=[1, 2],
            consensus=[0.5, 1.0],
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        ar_legacy.fit(epochs.copy())
    
    # Compare thresholds - must be IDENTICAL
    assert set(ar_current.threshes_.keys()) == set(ar_legacy.threshes_.keys()), \
        "Different channel sets in thresholds"
    
    for ch_name in ar_current.threshes_:
        assert_allclose(
            ar_current.threshes_[ch_name],
            ar_legacy.threshes_[ch_name],
            rtol=1e-14,
            err_msg=f"Threshold mismatch for channel {ch_name}: "
                    f"current={ar_current.threshes_[ch_name]}, "
                    f"legacy={ar_legacy.threshes_[ch_name]}"
        )
    
    # Compare n_interpolate_ and consensus_
    assert ar_current.n_interpolate_ == ar_legacy.n_interpolate_, \
        f"n_interpolate_ mismatch: current={ar_current.n_interpolate_}, legacy={ar_legacy.n_interpolate_}"
    assert ar_current.consensus_ == ar_legacy.consensus_, \
        f"consensus_ mismatch: current={ar_current.consensus_}, legacy={ar_legacy.consensus_}"


@pytest.mark.slow
@pytest.mark.legacy_parity
def test_autoreject_transform_matches_legacy(force_numpy_backend, synthetic_epochs):
    """Test that current AutoReject.fit_transform() produces IDENTICAL results to legacy."""
    from autoreject import AutoReject as CurrentAutoReject
    from legacy.autoreject_original import AutoReject as LegacyAutoReject
    
    epochs = synthetic_epochs
    random_state = 42
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Current implementation
        ar_current = CurrentAutoReject(
            n_interpolate=[1, 2],
            consensus=[0.5, 1.0],
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        epochs_clean_current, log_current = ar_current.fit_transform(
            epochs.copy(), return_log=True
        )
        
        # Legacy implementation
        ar_legacy = LegacyAutoReject(
            n_interpolate=[1, 2],
            consensus=[0.5, 1.0],
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        epochs_clean_legacy, log_legacy = ar_legacy.fit_transform(
            epochs.copy(), return_log=True
        )
    
    # Compare cleaned data - must be IDENTICAL
    assert_allclose(
        epochs_clean_current.get_data(),
        epochs_clean_legacy.get_data(),
        rtol=1e-14,
        err_msg="Cleaned epochs data mismatch between current and legacy"
    )
    
    # Compare reject logs - must be IDENTICAL
    assert_array_equal(
        log_current.bad_epochs,
        log_legacy.bad_epochs,
        err_msg="bad_epochs mismatch between current and legacy"
    )
    assert_array_equal(
        log_current.labels,
        log_legacy.labels,
        err_msg="labels mismatch between current and legacy"
    )


# =============================================================================
# Test: compute_thresholds() - Legacy vs Current PARITY
# =============================================================================

@pytest.mark.slow
@pytest.mark.legacy_parity
def test_compute_thresholds_matches_legacy(force_numpy_backend, synthetic_epochs):
    """Test that current compute_thresholds() produces IDENTICAL results to legacy."""
    from autoreject import compute_thresholds as current_compute_thresholds
    from legacy.autoreject_original import compute_thresholds as legacy_compute_thresholds
    
    epochs = synthetic_epochs
    random_state = 42
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Current implementation
        thresholds_current = current_compute_thresholds(
            epochs.copy(),
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        
        # Legacy implementation
        thresholds_legacy = legacy_compute_thresholds(
            epochs.copy(),
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
    
    # Compare all thresholds - must be IDENTICAL
    assert set(thresholds_current.keys()) == set(thresholds_legacy.keys()), \
        "Different channel sets in thresholds"
    
    for ch_name in thresholds_current:
        assert_allclose(
            thresholds_current[ch_name],
            thresholds_legacy[ch_name],
            rtol=1e-14,
            err_msg=f"Threshold mismatch for channel {ch_name}: "
                    f"current={thresholds_current[ch_name]}, "
                    f"legacy={thresholds_legacy[ch_name]}"
        )


# =============================================================================
# Test: RANSAC - Legacy vs Current PARITY
# =============================================================================

@pytest.mark.slow
@pytest.mark.legacy_parity
def test_ransac_fit_matches_legacy(force_numpy_backend, synthetic_epochs):
    """Test that current Ransac.fit() produces IDENTICAL results to legacy."""
    from autoreject.ransac import Ransac as CurrentRansac
    from legacy.ransac_original import Ransac as LegacyRansac
    
    epochs = synthetic_epochs
    random_state = 42
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Current implementation
        ransac_current = CurrentRansac(
            n_resample=20,
            min_channels=0.25,
            min_corr=0.75,
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        ransac_current.fit(epochs.copy())
        
        # Legacy implementation
        ransac_legacy = LegacyRansac(
            n_resample=20,
            min_channels=0.25,
            min_corr=0.75,
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        ransac_legacy.fit(epochs.copy())
    
    # Compare bad_chs_ - must be IDENTICAL
    assert set(ransac_current.bad_chs_) == set(ransac_legacy.bad_chs_), \
        f"bad_chs_ mismatch: current={ransac_current.bad_chs_}, legacy={ransac_legacy.bad_chs_}"


# NOTE: test_ransac_correlations_match_legacy was removed because:
# 1. The internal _fit_ransac() method exists only in legacy, not in current
# 2. The current implementation abstracts this through the backend system
# 3. test_ransac_fit_matches_legacy already validates the final result is identical
# 4. Testing internal methods is less important than testing public API parity


# =============================================================================
# Test: Real data (requires MNE testing data)
# =============================================================================

@pytest.mark.slow
@pytest.mark.legacy_parity
def test_autoreject_real_data_matches_legacy(force_numpy_backend, real_epochs):
    """Test AutoReject parity with real EEG data."""
    from autoreject import AutoReject as CurrentAutoReject
    from legacy.autoreject_original import AutoReject as LegacyAutoReject
    
    epochs = real_epochs
    random_state = 42
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Current implementation
        ar_current = CurrentAutoReject(
            n_interpolate=[1, 2],
            consensus=[0.5, 1.0],
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        epochs_clean_current, log_current = ar_current.fit_transform(
            epochs.copy(), return_log=True
        )
        
        # Legacy implementation
        ar_legacy = LegacyAutoReject(
            n_interpolate=[1, 2],
            consensus=[0.5, 1.0],
            random_state=random_state,
            n_jobs=1,
            verbose=False
        )
        epochs_clean_legacy, log_legacy = ar_legacy.fit_transform(
            epochs.copy(), return_log=True
        )
    
    # Compare thresholds
    for ch_name in ar_current.threshes_:
        assert_allclose(
            ar_current.threshes_[ch_name],
            ar_legacy.threshes_[ch_name],
            rtol=1e-14,
            err_msg=f"Real data: threshold mismatch for {ch_name}"
        )
    
    # Compare cleaned data
    assert_allclose(
        epochs_clean_current.get_data(),
        epochs_clean_legacy.get_data(),
        rtol=1e-14,
        err_msg="Real data: cleaned epochs mismatch"
    )
    
    # Compare reject logs
    assert_array_equal(
        log_current.bad_epochs,
        log_legacy.bad_epochs,
        err_msg="Real data: bad_epochs mismatch"
    )


# =============================================================================
# Test: Backend infrastructure
# =============================================================================

def test_force_numpy_backend_fixture(force_numpy_backend):
    """Verify that the fixture correctly forces NumPy backend."""
    from autoreject.backends import get_backend
    
    backend = get_backend()
    
    assert backend.name == 'numpy', \
        f"Expected 'numpy' backend but got '{backend.name}'"
    assert os.environ.get('AUTOREJECT_BACKEND') == 'numpy', \
        "AUTOREJECT_BACKEND environment variable not set correctly"


def test_backend_ptp_matches_numpy_ptp(force_numpy_backend):
    """Test that backend.ptp() produces same results as np.ptp()."""
    from autoreject.backends import get_backend
    
    np.random.seed(42)
    data = np.random.randn(50, 32, 100)
    
    backend = get_backend()
    
    ptp_backend = backend.ptp(data, axis=-1)
    ptp_numpy = np.ptp(data, axis=-1)
    
    assert_allclose(
        ptp_backend,
        ptp_numpy,
        rtol=1e-14,
        err_msg="backend.ptp() doesn't match np.ptp()"
    )


def test_backend_median_matches_numpy_median(force_numpy_backend):
    """Test that backend.median() produces same results as np.median()."""
    from autoreject.backends import get_backend
    
    np.random.seed(42)
    data = np.random.randn(50, 32, 100)
    
    backend = get_backend()
    
    for axis in [0, 1, 2, -1]:
        median_backend = backend.median(data, axis=axis)
        median_numpy = np.median(data, axis=axis)
        
        assert_allclose(
            median_backend,
            median_numpy,
            rtol=1e-14,
            err_msg=f"backend.median() doesn't match np.median() for axis={axis}"
        )


def test_backend_correlation_matches_legacy_formula(force_numpy_backend):
    """Test that backend.correlation() matches the legacy correlation formula."""
    from autoreject.backends import get_backend
    
    np.random.seed(42)
    
    x = np.random.randn(100, 32)
    y = np.random.randn(100, 32)
    
    backend = get_backend()
    corr_backend = backend.correlation(x, y)
    
    # Legacy formula from ransac_original.py lines 127-134
    num = np.sum(x * y, axis=0)
    denom = np.sqrt(np.sum(x ** 2, axis=0)) * np.sqrt(np.sum(y ** 2, axis=0))
    corr_legacy = num / denom
    
    assert_allclose(
        corr_backend,
        corr_legacy,
        rtol=1e-14,
        err_msg="backend.correlation() doesn't match legacy formula"
    )


# =============================================================================
# Test: Numba backend parity (if available)
# =============================================================================

def test_numba_ptp_matches_numpy():
    """Test that NumbaBackend.ptp() matches NumpyBackend.ptp() exactly."""
    from autoreject.backends import NumpyBackend
    try:
        from autoreject.backends import NumbaBackend
        numba_backend = NumbaBackend()
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Numba not available")
    
    np.random.seed(42)
    data_3d = np.random.randn(50, 32, 100)
    
    numpy_backend = NumpyBackend()
    
    ptp_numpy = numpy_backend.ptp(data_3d, axis=-1)
    ptp_numba = numba_backend.ptp(data_3d, axis=-1)
    
    assert_allclose(
        ptp_numba,
        ptp_numpy,
        rtol=1e-14,
        err_msg="ptp mismatch between NumPy and Numba backends"
    )


def test_numba_correlation_matches_numpy():
    """Test that NumbaBackend.correlation() matches NumpyBackend.correlation()."""
    from autoreject.backends import NumpyBackend
    try:
        from autoreject.backends import NumbaBackend
        numba_backend = NumbaBackend()
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Numba not available")
    
    np.random.seed(42)
    x = np.random.randn(100, 32)
    y = np.random.randn(100, 32)
    
    numpy_backend = NumpyBackend()
    
    corr_numpy = numpy_backend.correlation(x, y)
    corr_numba = numba_backend.correlation(x, y)
    
    assert_allclose(
        corr_numba,
        corr_numpy,
        rtol=1e-12,  # Slightly looser due to parallel reduction order
        err_msg="correlation mismatch between NumPy and Numba backends"
    )


# =============================================================================
# Test: PyTorch backend parity (if available)
# =============================================================================

def test_torch_median_matches_numpy():
    """Test that PyTorch backend median matches NumPy for even/odd arrays."""
    try:
        from autoreject.backends import TorchBackend, NumpyBackend
        torch_backend = TorchBackend()
    except (ImportError, ModuleNotFoundError, RuntimeError):
        pytest.skip("PyTorch not available")
    
    numpy_backend = NumpyBackend()
    np.random.seed(42)
    
    # Test with odd number of elements
    data_odd = np.random.randn(51, 32, 100)
    median_np_odd = numpy_backend.median(data_odd, axis=0)
    median_torch_odd = torch_backend.median(data_odd, axis=0)
    
    assert_allclose(
        median_torch_odd,
        median_np_odd,
        rtol=1e-6,  # Looser tolerance for float32 on some devices
        err_msg="PyTorch median doesn't match NumPy for odd-length arrays"
    )
    
    # Test with even number of elements
    data_even = np.random.randn(50, 32, 100)
    median_np_even = numpy_backend.median(data_even, axis=0)
    median_torch_even = torch_backend.median(data_even, axis=0)
    
    assert_allclose(
        median_torch_even,
        median_np_even,
        rtol=1e-6,
        err_msg="PyTorch median doesn't match NumPy for even-length arrays"
    )
