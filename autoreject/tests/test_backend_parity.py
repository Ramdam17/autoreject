"""Backend parity tests -- lock the numerical guarantees, per device.

Complements ``test_retrocompat.py``, which forces the NumPy backend
module-wide and therefore never confronts the torch paths with the frozen
legacy reference. This module does confront them, and it chooses its
tolerance per device instead of applying one uniform value everywhere.

Why per-device and why so tight
-------------------------------
AutoReject's threshold search is a *discrete* selection over a grid of
candidate peak-to-peak values, and the stages that consume its output
(ICA, ICLabel's exclusion cut, the consensus/n_interpolate grid) are
discrete too. A rounding of ~1e-7 relative does not produce a slightly
different threshold -- it can move the selected grid point, and from
there a different set of retained epochs. A uniform ``rtol=1e-5`` is two
orders of magnitude larger than that effect, so it cannot see it.

On any device that supports float64 the GPU threshold search is expected
to reproduce ``legacy/`` *bit for bit*; the tolerance there is 0. On MPS,
float32 is a hardware limit and 1e-5 is the honest bound.

See Also
--------
test_retrocompat : NumPy path versus the frozen legacy implementation.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

torch = pytest.importorskip("torch")

from autoreject import AutoReject  # noqa: E402
from autoreject.autoreject import _compute_thresholds  # noqa: E402
from autoreject.backends import (  # noqa: E402
    use_backend, get_backend_names, clear_backend_cache,
)
from autoreject.gpu_pipeline import (  # noqa: E402
    GPUThresholdOptimizer, compute_thresholds_gpu,
)

# Reuse the reference fixtures rather than duplicating the generator: the
# epochs must stay byte-identical to what generated the .npz files.
sys.path.insert(0, str(Path(__file__).parent))
from test_retrocompat import _create_test_epochs, _load_reference  # noqa: E402

# Relative tolerance against the frozen legacy reference, per device.
#
# cpu/cuda: float64 is available, so bit-exactness is achievable *and*
#   required -- a non-zero deviation there means a dtype is being narrowed
#   somewhere, which is exactly the regression these tests exist to catch.
# mps: float32 only. 1e-5 per the multi-backend tolerance table.
#
# NOTE: the cuda entry is an inference, not a local measurement -- it was
# validated on cpu float64 and cuda takes the same float64 branch of
# GPUThresholdOptimizer._resolve_dtype. A failure on CUDA hardware is a
# real finding, not a flaky test.
DEVICE_RTOL = {"cpu": 0.0, "cuda": 0.0, "mps": 1e-5}

FLOAT64_DEVICES = ("cpu", "cuda")


def _skip_if_unavailable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS not available")
    if "torch" not in get_backend_names():
        pytest.skip("PyTorch backend not available")


@pytest.fixture
def epochs():
    """Epochs matching the reference generation exactly."""
    return _create_test_epochs()


@pytest.fixture
def picks(epochs):
    from autoreject.utils import _handle_picks
    return _handle_picks(epochs.info, picks=None)


@pytest.fixture
def ref_threshes():
    """Frozen bayesian-optimization thresholds from legacy/."""
    ref = _load_reference("compute_thresholds_v1")
    return dict(zip(ref["bayes_keys"], ref["bayes_values"]))


# =============================================================================
# Regression locks on the dtype policy itself
# =============================================================================


@pytest.mark.backends
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_to_tensor_follows_device_dtype_policy(device):
    """_to_tensor must not narrow to float32 on a float64-capable device.

    Regression lock. ``_to_tensor`` used to default to ``torch.float32``
    unconditionally, which silently overrode the float64 that
    ``backends.TorchBackend`` selects for CPU and CUDA -- so the threshold
    search had never actually run in float64 on CUDA.
    """
    _skip_if_unavailable(device)
    opt = GPUThresholdOptimizer(device=device)
    expected = torch.float32 if device == "mps" else torch.float64

    assert opt.dtype == expected
    tensor = opt._to_tensor(np.zeros((4, 8), dtype=np.float64))
    assert tensor.dtype == expected, (
        f"_to_tensor narrowed {expected} to {tensor.dtype} on {device}"
    )


@pytest.mark.backends
@pytest.mark.parametrize("device", FLOAT64_DEVICES)
def test_cv_loss_is_not_narrowed(device):
    """The CV losses bayes_opt compares must keep the device dtype.

    Regression lock. The loss accumulator was allocated without a dtype
    (defaulting to float32) and the boolean mask was cast with ``.float()``,
    either of which narrows a float64 computation at the point where the
    values are compared.
    """
    _skip_if_unavailable(device)
    opt = GPUThresholdOptimizer(device=device)
    rng = np.random.RandomState(42)

    n_epochs, n_channels, n_times = 12, 3, 32
    data = opt._to_tensor(rng.randn(n_epochs, n_channels, n_times) * 30e-6)
    ptp = data.max(dim=-1).values - data.min(dim=-1).values
    threshes = torch.stack([
        torch.sort(ptp[:, c]).values for c in range(n_channels)
    ])
    cv_splits = [(np.arange(0, 8), np.arange(8, 12)),
                 (np.arange(4, 12), np.arange(0, 4))]

    losses = opt.batched_all_channels_cv_loss_parallel(
        data, ptp, threshes, cv_splits
    )
    assert losses.dtype == torch.float64, (
        f"CV losses narrowed to {losses.dtype} on {device}"
    )


# =============================================================================
# Parity against the frozen legacy reference
# =============================================================================


@pytest.mark.backends
@pytest.mark.parametrize("device", FLOAT64_DEVICES)
def test_gpu_thresholds_bit_exact_on_float64_device(
    device, epochs, picks, ref_threshes
):
    """On a float64 device the GPU search must reproduce legacy exactly.

    ``augment=False`` matches the reference generation, so this isolates
    the threshold search with no interpolation confound.
    """
    _skip_if_unavailable(device)
    clear_backend_cache()
    try:
        with use_backend("torch"):
            np.random.seed(42)
            threshes = compute_thresholds_gpu(
                epochs, method="bayesian_optimization", random_state=42,
                picks=picks, augment=False, verbose=False, device=device,
            )
    finally:
        clear_backend_cache()

    assert set(threshes) == set(ref_threshes)
    for ch in sorted(threshes):
        assert_allclose(
            threshes[ch], ref_threshes[ch],
            rtol=DEVICE_RTOL[device], atol=0.0,
            err_msg=(
                f"{ch}: GPU threshold on {device} is not bit-exact with "
                f"legacy. A dtype is being narrowed somewhere."
            ),
        )


@pytest.mark.backends
def test_gpu_thresholds_within_tolerance_on_default_device(
    epochs, picks, ref_threshes
):
    """The auto-selected device must stay within its documented bound."""
    if "torch" not in get_backend_names():
        pytest.skip("PyTorch backend not available")
    clear_backend_cache()
    try:
        with use_backend("torch"):
            opt = GPUThresholdOptimizer()
            device = str(opt.device)
            np.random.seed(42)
            threshes = compute_thresholds_gpu(
                epochs, method="bayesian_optimization", random_state=42,
                picks=picks, augment=False, verbose=False, device=device,
            )
    finally:
        clear_backend_cache()

    rtol = DEVICE_RTOL.get(device.split(":")[0], 1e-5)
    for ch in sorted(threshes):
        assert_allclose(threshes[ch], ref_threshes[ch], rtol=rtol, atol=0.0,
                        err_msg=f"{ch} exceeds the {device} tolerance {rtol}")


# =============================================================================
# Discrete decisions -- sets, never counts
# =============================================================================


@pytest.mark.backends
@pytest.mark.slow
def test_discrete_decisions_identical_on_float64_device(epochs):
    """Rejected-epoch *sets* and the AR grid must match exactly.

    Equal epoch *counts* are not evidence of equivalence: two runs can
    retain the same number of epochs while retaining different ones, and
    per-task differences of opposite sign cancel in any total. This
    asserts set identity and exact grid equality instead.
    """
    _skip_if_unavailable("cpu")
    kwargs = dict(n_interpolate=[1, 2], consensus=np.linspace(0, 1.0, 3),
                  cv=3, random_state=42, verbose=False)

    clear_backend_cache()
    try:
        with use_backend("numpy"):
            ar_np = AutoReject(**kwargs)
            ar_np.fit(epochs)
            log_np = ar_np.get_reject_log(epochs)
    finally:
        clear_backend_cache()

    clear_backend_cache()
    try:
        with use_backend("torch"):
            ar_t = AutoReject(**kwargs)
            ar_t.fit(epochs)
            log_t = ar_t.get_reject_log(epochs)
    finally:
        clear_backend_cache()

    # Discrete hyperparameters: exact equality, no tolerance applies.
    assert ar_np.consensus_ == ar_t.consensus_, (
        f"consensus differs: {ar_np.consensus_} vs {ar_t.consensus_}"
    )
    assert ar_np.n_interpolate_ == ar_t.n_interpolate_, (
        f"n_interpolate differs: "
        f"{ar_np.n_interpolate_} vs {ar_t.n_interpolate_}"
    )

    # Which epochs, not how many.
    bad_np = set(np.flatnonzero(log_np.bad_epochs).tolist())
    bad_t = set(np.flatnonzero(log_t.bad_epochs).tolist())
    assert bad_np == bad_t, (
        f"rejected-epoch sets differ: only numpy {sorted(bad_np - bad_t)}, "
        f"only torch {sorted(bad_t - bad_np)} "
        f"(counts {len(bad_np)} vs {len(bad_t)} -- equal counts would not "
        f"have caught this)"
    )

    # Which channels were interpolated in which epoch.
    assert_array_equal(
        log_np.labels, log_t.labels,
        err_msg="per-epoch per-channel labels differ between backends",
    )


@pytest.mark.backends
def test_cpu_path_still_matches_legacy_under_torch_backend(
    epochs, picks, ref_threshes
):
    """The non-GPU code path under the torch backend keeps its bound.

    ``_compute_thresholds`` with the torch backend active goes through
    ``backends.TorchBackend``, not through ``compute_thresholds_gpu``.
    That is a separate path and it needs its own guard.
    """
    if "torch" not in get_backend_names():
        pytest.skip("PyTorch backend not available")
    clear_backend_cache()
    try:
        with use_backend("torch"):
            from autoreject.backends import get_backend
            device = str(get_backend().device).split(":")[0]
            np.random.seed(42)
            threshes = _compute_thresholds(
                epochs, method="bayesian_optimization", random_state=42,
                picks=picks, augment=False, verbose=False, n_jobs=1,
            )
    finally:
        clear_backend_cache()

    rtol = DEVICE_RTOL.get(device, 1e-5)
    for ch in sorted(threshes):
        assert_allclose(threshes[ch], ref_threshes[ch], rtol=rtol, atol=0.0,
                        err_msg=f"{ch} exceeds the {device} tolerance {rtol}")
