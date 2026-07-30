"""Retrocompatibility tests for autoreject.

These tests ensure that performance optimizations do not change
the numerical results of the core algorithms.

The tests compare current outputs against stored reference data
generated from the original (unoptimized) implementation.

Note: These tests force the NumPy backend to ensure exact numerical
reproducibility with the reference data.
"""

# Authors: autoreject contributors

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

import mne

from autoreject import Ransac
from autoreject.autoreject import (
    _AutoReject, _compute_thresholds,
    _interpolate_bad_epochs, _get_interp_chs, _run_local_reject_cv
)
from autoreject.utils import _handle_picks, _GDKW
from autoreject.backends import clear_backend_cache

# Path to reference data
REFERENCES_DIR = Path(__file__).parent / "references"


# =============================================================================
# Module-level setup: Force NumPy backend for ALL tests in this module
# =============================================================================

# Store original value ONCE at module import time
_ORIGINAL_BACKEND_ENV = os.environ.get('AUTOREJECT_BACKEND')

# Force NumPy backend at module import time (BEFORE any fixtures run)
os.environ['AUTOREJECT_BACKEND'] = 'numpy'
clear_backend_cache()


@pytest.fixture(scope='session', autouse=True)
def restore_backend_after_retrocompat():
    """Restore original backend setting after all retrocompat tests."""
    yield
    # Restore original value after all tests
    if _ORIGINAL_BACKEND_ENV is None:
        os.environ.pop('AUTOREJECT_BACKEND', None)
    else:
        os.environ['AUTOREJECT_BACKEND'] = _ORIGINAL_BACKEND_ENV
    clear_backend_cache()


@pytest.fixture(autouse=True)
def ensure_numpy_backend_per_test():
    """Ensure NumPy backend is active for each test.
    
    This provides a second layer of protection to ensure exact numerical
    reproducibility with reference data generated using NumPy operations.
    """
    # Clear cache and re-apply at start of each test
    clear_backend_cache()
    os.environ['AUTOREJECT_BACKEND'] = 'numpy'
    
    yield
    
    # Keep numpy backend for subsequent tests in this module


def _load_reference(name):
    """Load a reference data file.
    
    Parameters
    ----------
    name : str
        Name of the reference file (without .npz extension).
    
    Returns
    -------
    data : dict
        Dictionary with reference data.
    """
    path = REFERENCES_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(
            f"Reference file {path} not found. "
            "Run 'python tools/generate_references.py' first."
        )
    return dict(np.load(path, allow_pickle=True))


def _create_test_epochs(n_epochs=30, n_channels=32, n_times=256,
                        sfreq=256, seed=42):
    """Create deterministic test epochs matching the reference data.
    
    Must produce identical data to tools/generate_references.py
    Uses isolated RandomState to avoid interference from other tests.
    """
    # Use isolated RandomState to ensure reproducibility regardless of
    # what other tests have done to the global numpy RNG
    rng = np.random.RandomState(seed)
    
    # Create channel info - use 1-based naming to avoid EEG000
    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create custom montage with spherical positions
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    phi = np.linspace(np.pi/4, 3*np.pi/4, n_channels)
    radius = 0.09
    
    pos = np.column_stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi)
    ])
    
    ch_pos = {ch: p for ch, p in zip(ch_names, pos)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage)
    
    # Generate synthetic data using isolated RNG
    data = rng.randn(n_epochs, n_channels, n_times) * 20e-6
    common_signal = rng.randn(n_epochs, 1, n_times) * 5e-6
    data += common_signal
    
    # Add bad epochs
    for idx in [5, 12, 22]:
        data[idx] *= 3.0
    
    # Add bad channels in specific epochs
    data[3, 10, :] += rng.randn(n_times) * 100e-6
    data[8, 5, :] += rng.randn(n_times) * 80e-6
    data[15, 20, :] += rng.randn(n_times) * 90e-6
    data[18, 15, :] += rng.randn(n_times) * 120e-6
    
    events = np.column_stack([
        np.arange(0, n_epochs * n_times, n_times),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])
    
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)
    return epochs


@pytest.fixture(scope='module')
def test_epochs():
    """Create test epochs for retrocompatibility tests."""
    return _create_test_epochs()


@pytest.fixture(scope='module')
def test_picks(test_epochs):
    """Get picks for test epochs."""
    return _handle_picks(test_epochs.info, picks=None)


# =============================================================================
# Vote Bad Epochs Tests
# =============================================================================

@pytest.mark.retrocompat
class TestVoteBadEpochsRetrocompat:
    """Retrocompatibility tests for _AutoReject._vote_bad_epochs."""
    
    @pytest.fixture(scope='class')
    def reference(self):
        """Load reference data."""
        return _load_reference("vote_bad_epochs_v1")
    
    @pytest.fixture(scope='class')
    def computed_result(self, test_epochs, test_picks, reference):
        """Compute result with current implementation."""
        # Reconstruct thresholds from reference
        threshes = dict(zip(
            reference['threshes_keys'],
            reference['threshes_values']
        ))
        
        ar = _AutoReject(
            n_interpolate=4, consensus=0.5,
            picks=test_picks, verbose=False
        )
        ar.threshes_ = threshes
        
        labels, bad_sensor_counts = ar._vote_bad_epochs(
            test_epochs, picks=test_picks
        )
        return labels, bad_sensor_counts
    
    def test_labels_shape(self, computed_result, reference):
        """Test that labels shape matches reference."""
        labels, _ = computed_result
        ref_labels = reference['labels']
        assert labels.shape == ref_labels.shape, (
            f"Labels shape mismatch: {labels.shape} vs {ref_labels.shape}"
        )
    
    def test_labels_values(self, computed_result, reference):
        """Test that labels values match reference exactly."""
        labels, _ = computed_result
        ref_labels = reference['labels']
        # Use assert_array_equal for exact match (including NaN positions)
        assert_array_equal(
            np.isnan(labels), np.isnan(ref_labels),
            err_msg="NaN positions in labels do not match reference"
        )
        # Compare non-NaN values
        mask = ~np.isnan(labels)
        assert_array_equal(
            labels[mask], ref_labels[mask],
            err_msg="Labels values do not match reference"
        )
    
    def test_bad_sensor_counts(self, computed_result, reference):
        """Test that bad sensor counts match reference."""
        _, bad_sensor_counts = computed_result
        ref_counts = reference['bad_sensor_counts']
        assert_array_equal(
            bad_sensor_counts, ref_counts,
            err_msg="Bad sensor counts do not match reference"
        )


# =============================================================================
# Compute Thresholds Tests
# =============================================================================

@pytest.mark.retrocompat
class TestComputeThresholdsRetrocompat:
    """Retrocompatibility tests for _compute_thresholds."""
    
    @pytest.fixture(scope='class')
    def reference(self):
        """Load reference data."""
        return _load_reference("compute_thresholds_v1")
    
    def test_bayesian_optimization_thresholds(self, test_epochs, test_picks,
                                               reference):
        """Test Bayesian optimization thresholds match reference."""
        np.random.seed(42)
        
        threshes = _compute_thresholds(
            test_epochs, method='bayesian_optimization',
            random_state=42, picks=test_picks,
            augment=False, verbose=False, n_jobs=1
        )
        
        ref_threshes = dict(zip(
            reference['bayes_keys'],
            reference['bayes_values']
        ))
        
        # Check all channels present
        assert set(threshes.keys()) == set(ref_threshes.keys()), (
            "Channel sets do not match"
        )
        
        # Check values match
        for ch in threshes:
            assert_allclose(
                threshes[ch], ref_threshes[ch], rtol=1e-5,
                err_msg=f"Threshold for {ch} does not match reference"
            )
    
    def test_random_search_thresholds(self, test_epochs, test_picks, reference):
        """Test random search thresholds match reference."""
        np.random.seed(42)
        
        threshes = _compute_thresholds(
            test_epochs, method='random_search',
            random_state=42, picks=test_picks,
            augment=False, verbose=False, n_jobs=1
        )
        
        ref_threshes = dict(zip(
            reference['random_keys'],
            reference['random_values']
        ))
        
        # Check all channels present
        assert set(threshes.keys()) == set(ref_threshes.keys())
        
        # Check values match
        for ch in threshes:
            assert_allclose(
                threshes[ch], ref_threshes[ch], rtol=1e-5,
                err_msg=f"Threshold for {ch} does not match reference"
            )


# =============================================================================
# RANSAC Tests
# =============================================================================

@pytest.mark.retrocompat
class TestRansacRetrocompat:
    """Retrocompatibility tests for Ransac."""
    
    @pytest.fixture(scope='class')
    def reference(self):
        """Load reference data."""
        return _load_reference("ransac_v1")
    
    @pytest.fixture(scope='class')
    def computed_ransac(self, test_epochs, test_picks):
        """Compute RANSAC with current implementation."""
        ransac = Ransac(
            n_resample=25,
            min_channels=0.25,
            min_corr=0.75,
            unbroken_time=0.4,
            n_jobs=1,
            random_state=42,
            picks=test_picks,
            verbose=False
        )
        ransac.fit(test_epochs)
        return ransac
    
    def test_correlations_shape(self, computed_ransac, reference):
        """Test correlations shape matches reference."""
        assert computed_ransac.corr_.shape == reference['corr_'].shape, (
            f"Correlations shape mismatch: "
            f"{computed_ransac.corr_.shape} vs {reference['corr_'].shape}"
        )
    
    def test_correlations_values(self, computed_ransac, reference):
        """Test correlations values match reference."""
        assert_allclose(
            computed_ransac.corr_, reference['corr_'], rtol=1e-5,
            err_msg="RANSAC correlations do not match reference"
        )
    
    def test_bad_log(self, computed_ransac, reference):
        """Test bad_log matches reference."""
        assert_array_equal(
            computed_ransac.bad_log, reference['bad_log'],
            err_msg="RANSAC bad_log does not match reference"
        )
    
    def test_bad_channels(self, computed_ransac, reference):
        """Test detected bad channels match reference."""
        ref_bad_chs = list(reference['bad_chs_'])
        assert computed_ransac.bad_chs_ == ref_bad_chs, (
            f"Bad channels mismatch: "
            f"{computed_ransac.bad_chs_} vs {ref_bad_chs}"
        )


# =============================================================================
# Interpolate Epochs Tests
# =============================================================================

@pytest.mark.retrocompat
class TestInterpolateEpochsRetrocompat:
    """Retrocompatibility tests for _interpolate_bad_epochs."""
    
    @pytest.fixture(scope='class')
    def reference(self):
        """Load reference data."""
        return _load_reference("interpolate_epochs_v1")
    
    @pytest.fixture(scope='class')
    def vote_reference(self):
        """Load vote_bad_epochs reference for thresholds."""
        return _load_reference("vote_bad_epochs_v1")
    
    def test_interpolation_labels(self, test_epochs, test_picks,
                                   vote_reference, reference):
        """Test interpolation labels match reference."""
        # Reconstruct thresholds
        threshes = dict(zip(
            vote_reference['threshes_keys'],
            vote_reference['threshes_values']
        ))
        
        ar = _AutoReject(
            n_interpolate=4, consensus=0.5,
            picks=test_picks, verbose=False
        )
        ar.threshes_ = threshes
        ar.consensus_ = {'eeg': 0.5}
        ar.n_interpolate_ = {'eeg': 4}
        
        labels, _ = ar._vote_bad_epochs(test_epochs, picks=test_picks)
        labels_interp = ar._get_epochs_interpolation(
            test_epochs, labels=labels, picks=test_picks, n_interpolate=4
        )
        
        ref_labels = reference['labels_interp']
        
        # Compare NaN positions
        assert_array_equal(
            np.isnan(labels_interp), np.isnan(ref_labels),
            err_msg="NaN positions in interpolation labels do not match"
        )
        
        # Compare non-NaN values
        mask = ~np.isnan(labels_interp)
        assert_array_equal(
            labels_interp[mask], ref_labels[mask],
            err_msg="Interpolation labels do not match reference"
        )
    
    def test_interpolated_data_sample(self, test_epochs, test_picks,
                                       vote_reference, reference):
        """Test interpolated data sample matches reference."""
        # Reconstruct thresholds
        threshes = dict(zip(
            vote_reference['threshes_keys'],
            vote_reference['threshes_values']
        ))
        
        ar = _AutoReject(
            n_interpolate=4, consensus=0.5,
            picks=test_picks, verbose=False
        )
        ar.threshes_ = threshes
        ar.consensus_ = {'eeg': 0.5}
        ar.n_interpolate_ = {'eeg': 4}
        
        labels, _ = ar._vote_bad_epochs(test_epochs, picks=test_picks)
        labels_interp = ar._get_epochs_interpolation(
            test_epochs, labels=labels, picks=test_picks, n_interpolate=4
        )
        
        interp_channels = _get_interp_chs(
            labels_interp, test_epochs.ch_names, test_picks
        )
        
        epochs_copy = test_epochs.copy()
        _interpolate_bad_epochs(
            epochs_copy, interp_channels=interp_channels,
            picks=test_picks, dots=None, verbose=False
        )
        
        # Compare sample of interpolated data
        data_sample = epochs_copy.get_data(**_GDKW)[:5, :10, :]
        ref_sample = reference['data_sample']
        
        assert_allclose(
            data_sample, ref_sample, rtol=1e-5,
            err_msg="Interpolated data sample does not match reference"
        )


# =============================================================================
# Local Reject CV Tests
# =============================================================================

@pytest.mark.retrocompat
class TestLocalRejectCVRetrocompat:
    """Retrocompatibility tests for _run_local_reject_cv."""
    
    @pytest.fixture(scope='class')
    def reference(self):
        """Load reference data."""
        return _load_reference("local_reject_cv_v1")
    
    def test_cv_loss_values(self, test_epochs, test_picks, reference):
        """Test CV loss values match reference."""
        from functools import partial
        from sklearn.model_selection import KFold
        
        np.random.seed(42)
        
        thresh_func = partial(
            _compute_thresholds, n_jobs=1,
            method='bayesian_optimization',
            random_state=42, dots=None
        )
        
        n_interpolate = np.array([1, 4])
        consensus = np.array([0.2, 0.5, 0.8])
        cv = KFold(n_splits=3, shuffle=False)
        
        local_reject, loss = _run_local_reject_cv(
            test_epochs, thresh_func, test_picks, n_interpolate, cv,
            consensus, dots=None, verbose=False
        )
        
        ref_loss = reference['loss']
        
        assert loss.shape == ref_loss.shape, (
            f"Loss shape mismatch: {loss.shape} vs {ref_loss.shape}"
        )
        
        # Note: Some loss values may be inf, handle them separately
        finite_mask = np.isfinite(loss) & np.isfinite(ref_loss)
        inf_mask = np.isinf(loss) & np.isinf(ref_loss)
        
        # All inf positions should match
        assert_array_equal(
            np.isinf(loss), np.isinf(ref_loss),
            err_msg="Inf positions in loss do not match reference"
        )
        
        # Finite values should match closely
        if np.any(finite_mask):
            assert_allclose(
                loss[finite_mask], ref_loss[finite_mask], rtol=1e-5,
                err_msg="CV loss values do not match reference"
            )
    
    def test_thresholds_from_cv(self, test_epochs, test_picks, reference):
        """Test thresholds computed during CV match reference."""
        from functools import partial
        from sklearn.model_selection import KFold
        
        np.random.seed(42)
        
        thresh_func = partial(
            _compute_thresholds, n_jobs=1,
            method='bayesian_optimization',
            random_state=42, dots=None
        )
        
        n_interpolate = np.array([1, 4])
        consensus = np.array([0.2, 0.5, 0.8])
        cv = KFold(n_splits=3, shuffle=False)
        
        local_reject, _ = _run_local_reject_cv(
            test_epochs, thresh_func, test_picks, n_interpolate, cv,
            consensus, dots=None, verbose=False
        )
        
        ref_threshes = dict(zip(
            reference['threshes_keys'],
            reference['threshes_values']
        ))
        
        for ch in local_reject.threshes_:
            assert_allclose(
                local_reject.threshes_[ch], ref_threshes[ch], rtol=1e-5,
                err_msg=f"Threshold for {ch} from CV does not match reference"
            )
