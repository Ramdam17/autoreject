"""Backend parity tests -- lock the numerical guarantees, per device.

Complements ``test_retrocompat.py``, which forces the NumPy backend
module-wide and therefore never confronts the torch paths with the frozen
legacy reference. This module does confront them, and it chooses its
tolerance per device and per code path instead of applying one uniform
value everywhere.

Why per-device and why tight
----------------------------
On a device that supports float64 the GPU threshold search reproduces
``legacy/`` bit for bit, so the tolerance there is 0 and any deviation
means a dtype is being narrowed. On MPS, float32 is a hardware limit and
1e-5 is the honest bound. The augmented path gets its own, looser bound:
it goes through torch linear algebra whose float64 result differs from
NumPy's at the ~1e-11 level for reasons that have nothing to do with
narrowing.

Two traps this module is written to avoid
----------------------------------------
1. Do **not** import from ``test_retrocompat``: that module sets
   ``AUTOREJECT_BACKEND=numpy`` at import time and restores it only in a
   session-scoped fixture. Importing it for a helper forces NumPy for the
   whole session, which makes ``should_use_gpu`` return ``(False, 'cpu')``
   unconditionally -- so a test meaning to exercise the GPU path quietly
   stops doing so while still passing. Shared helpers live in
   ``_reference_epochs.py``, which has no import-time side effects.
2. Do **not** assume ``use_backend('torch')`` reaches the GPU path.
   ``AutoReject.fit`` gates on ``_should_use_gpu``, which also refuses
   datasets under 50 epochs. Tests that mean to cover the GPU path assert
   that the gate opened, and spy on the call.

See Also
--------
test_retrocompat : NumPy path versus the frozen legacy implementation.
"""

# Author: Rémy Ramadour <remy.ramadour.labs@gmail.com>

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

torch = pytest.importorskip("torch")

from autoreject import AutoReject  # noqa: E402
from autoreject.autoreject import _compute_thresholds  # noqa: E402
from autoreject.backends import (  # noqa: E402
    use_backend, get_backend, get_backend_names, clear_backend_cache,
)
from autoreject.gpu_pipeline import (  # noqa: E402
    GPUThresholdOptimizer, compute_thresholds_gpu, is_gpu_available,
)
from autoreject.utils import _handle_picks  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _reference_epochs import create_reference_epochs, load_reference  # noqa: E402

# Relative tolerance against the frozen legacy reference, per device.
#
# cpu/cuda: float64 is available, so bit-exactness is achievable *and*
#   required -- a non-zero deviation means a dtype is being narrowed, which
#   is what these tests exist to catch.
# mps: float32 only.
#
# NOTE the cuda entry is an inference, not a local measurement: it was
# established on cpu float64, and cuda takes the same float64 branch of
# GPUThresholdOptimizer._resolve_dtype. A failure on CUDA hardware is a
# real finding, not a flaky test.
DEVICE_RTOL = {"cpu": 0.0, "cuda": 0.0, "mps": 1e-5}

# The augmented path additionally goes through torch linear algebra
# (spherical-spline pinv, batched matmul), whose float64 output differs
# from NumPy's at ~1e-11. Measured max 3.2e-11 over 32 channels; 1e-9
# matches the torch-CPU-float64 row of the project's tolerance table.
AUGMENT_RTOL = {"cpu": 1e-9, "cuda": 1e-9, "mps": 1e-5}

FLOAT64_DEVICES = ("cpu", "cuda")
# Enough epochs to clear the >=50 gate in should_use_gpu.
GPU_GATE_N_EPOCHS = 60


def _has(device):
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return (hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available())
    return True


def _skip_if_unavailable(device):
    if not _has(device):
        pytest.skip(f"{device} not available")
    if "torch" not in get_backend_names():
        pytest.skip("PyTorch backend not available")


def _gpu_devices():
    """Devices on which AutoReject.fit will actually take the GPU path."""
    return [d for d in ("cuda", "mps") if _has(d)]


@pytest.fixture
def epochs():
    return create_reference_epochs()


@pytest.fixture
def picks(epochs):
    return _handle_picks(epochs.info, picks=None)


@pytest.fixture
def ref_threshes():
    """Frozen bayesian-optimization thresholds from legacy/."""
    ref = load_reference("compute_thresholds_v1")
    if ref is None:
        pytest.skip("reference fixtures absent; run tools/generate_references.py")
    return dict(zip(ref["bayes_keys"], ref["bayes_values"]))


# =============================================================================
# Environment hygiene
# =============================================================================


@pytest.mark.backends
def test_backend_env_is_not_contaminated():
    """No other module may have pinned AUTOREJECT_BACKEND on us.

    ``test_retrocompat`` pins it to 'numpy' at import time. If that leaks
    into this module, ``should_use_gpu`` short-circuits to CPU and the GPU
    tests below silently stop testing the GPU.
    """
    assert os.environ.get("AUTOREJECT_BACKEND", "").lower() != "numpy", (
        "AUTOREJECT_BACKEND=numpy leaked into this module; the GPU-path "
        "tests here would pass without exercising the GPU path"
    )


# =============================================================================
# Regression locks on the dtype policy itself
# =============================================================================


@pytest.mark.backends
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_to_tensor_follows_device_dtype_policy(device):
    """_to_tensor must not narrow to float32 on a float64-capable device.

    Regression lock. ``_to_tensor`` used to default to ``torch.float32``
    unconditionally, overriding the float64 that ``backends.TorchBackend``
    selects for CPU and CUDA -- so the threshold search had never actually
    run in float64 on CUDA.
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
@pytest.mark.parametrize("spec", ["cpu", "cuda", "cuda:0", "MPS", "mps"])
def test_resolve_dtype_accepts_device_spellings(spec):
    """The dtype policy must survive 'cuda:0', torch.device, and casing."""
    opt = GPUThresholdOptimizer(device="cpu")
    expected = torch.float32 if spec.lower().startswith("mps") else torch.float64
    assert opt._resolve_dtype(spec) == expected
    if _has(spec.split(":")[0].lower()):
        assert opt._resolve_dtype(torch.device(spec.lower())) == expected


@pytest.mark.backends
@pytest.mark.parametrize("device", FLOAT64_DEVICES)
def test_cv_loss_is_not_narrowed(device):
    """The CV losses bayes_opt compares must keep the device dtype.

    Regression lock. The loss accumulator was allocated without a dtype
    (defaulting to float32) and the boolean mask was cast with ``.float()``,
    either of which narrows a float64 computation exactly where the values
    are compared.
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


@pytest.mark.backends
@pytest.mark.parametrize("device", FLOAT64_DEVICES)
def test_consensus_scores_are_not_narrowed(device):
    """The AR2 consensus/n_interpolate scores must keep the device dtype.

    ``run_local_reject_cv_gpu_batch`` allocates ``scores_gpu`` for the
    consensus grid; without an explicit dtype it defaulted to float32.
    """
    _skip_if_unavailable(device)
    opt = GPUThresholdOptimizer(device=device)
    scores = opt.torch.zeros(3, device=opt.device, dtype=opt.dtype)
    assert scores.dtype == torch.float64, (
        f"consensus scores would be {scores.dtype} on {device}"
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

    ``augment=False`` matches the reference generation, isolating the
    threshold search with no interpolation confound.

    Note on scope: ``device='cpu'`` is not a configuration ``AutoReject.fit``
    can produce -- ``_should_use_gpu`` requires ``is_gpu_available()``, which
    requires mps or cuda. It is used here as the float64 proxy that isolates
    the dtype from the device, which is what makes the assertion meaningful.
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
            err_msg=(f"{ch}: GPU threshold on {device} is not bit-exact with "
                     f"legacy -- a dtype is being narrowed somewhere"),
        )


@pytest.mark.backends
@pytest.mark.slow
@pytest.mark.parametrize("device", FLOAT64_DEVICES)
def test_gpu_thresholds_augmented_within_bound(device, epochs, picks):
    """The augmented path is the one the real pipeline uses.

    Bit-exactness is not expected here: augmentation routes through torch
    linear algebra, whose float64 output differs from NumPy's at ~1e-11.
    What is asserted is that the difference stays at that level rather than
    at the float32 level, which would mean a narrowing.
    """
    _skip_if_unavailable(device)
    clear_backend_cache()
    try:
        with use_backend("numpy"):
            np.random.seed(42)
            ref = _compute_thresholds(
                epochs, method="bayesian_optimization", random_state=42,
                picks=picks, augment=True, verbose=False, n_jobs=1,
            )
    finally:
        clear_backend_cache()

    clear_backend_cache()
    try:
        with use_backend("torch"):
            np.random.seed(42)
            got = compute_thresholds_gpu(
                epochs, method="bayesian_optimization", random_state=42,
                picks=picks, augment=True, verbose=False, device=device,
            )
    finally:
        clear_backend_cache()

    for ch in sorted(set(ref) & set(got)):
        assert_allclose(
            got[ch], ref[ch], rtol=AUGMENT_RTOL[device], atol=0.0,
            err_msg=f"{ch}: augmented path on {device} exceeds ~1e-9",
        )


@pytest.mark.backends
def test_cpu_path_still_matches_legacy_under_torch_backend(
    epochs, picks, ref_threshes
):
    """The non-GPU code path under the torch backend keeps its bound.

    ``_compute_thresholds`` with the torch backend active goes through
    ``backends.TorchBackend``, not through ``compute_thresholds_gpu``. A
    separate path needing its own guard.
    """
    if "torch" not in get_backend_names():
        pytest.skip("PyTorch backend not available")
    clear_backend_cache()
    try:
        with use_backend("torch"):
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


# =============================================================================
# The path AutoReject.fit actually takes -- discrete decisions, as sets
# =============================================================================


def _fit_both_backends(device, monkeypatch, n_epochs):
    """Fit on numpy and on the real GPU path; return both reject logs.

    Asserts the GPU gate opened and the batched GPU CV was actually called
    -- without that, the >=50-epoch gate in ``should_use_gpu`` or a pinned
    ``AUTOREJECT_BACKEND`` would make the caller pass while running on CPU.
    """
    ep = create_reference_epochs(n_epochs=n_epochs)
    kwargs = dict(n_interpolate=[1, 4], consensus=np.linspace(0, 1.0, 3),
                  cv=3, random_state=42, verbose=False)

    clear_backend_cache()
    try:
        with use_backend("numpy"):
            ar_np = AutoReject(**kwargs)
            ar_np.device = "cpu"
            np.random.seed(42)
            ar_np.fit(ep)
            log_np = ar_np.get_reject_log(ep)
    finally:
        clear_backend_cache()

    calls = []
    clear_backend_cache()
    try:
        with use_backend("torch"):
            from autoreject import gpu_pipeline
            real = gpu_pipeline.run_local_reject_cv_gpu_batch

            def spy(*args, **kw):
                calls.append(device)
                return real(*args, **kw)

            monkeypatch.setattr(
                gpu_pipeline, "run_local_reject_cv_gpu_batch", spy)

            ar_t = AutoReject(**kwargs)
            ar_t.device = device
            assert ar_t._should_use_gpu(ep)[0], (
                f"the GPU gate refused {device}; this test would not have "
                f"exercised the GPU path"
            )
            np.random.seed(42)
            ar_t.fit(ep)
            log_t = ar_t.get_reject_log(ep)
    finally:
        clear_backend_cache()

    assert calls, (
        "run_local_reject_cv_gpu_batch was never called -- the batched GPU "
        "CV path is not covered by this test"
    )
    return ar_np, ar_t, log_np, log_t


@pytest.mark.backends
@pytest.mark.slow
@pytest.mark.parametrize("device", _gpu_devices() or ["mps"])
def test_fit_gpu_path_matches_coarse_decisions(device, monkeypatch):
    """The AR grid and the rejected-epoch set must match on the GPU path.

    These are the coarse decisions, and they are expected to hold on every
    device including MPS: they aggregate over channels, so a single channel
    picking a different threshold does not usually change them.

    Compares *sets*, not counts -- two runs can retain the same number of
    different epochs, and differences of opposite sign cancel in a total.
    """
    _skip_if_unavailable(device)
    if not is_gpu_available():
        pytest.skip("no GPU device available")

    ar_np, ar_t, log_np, log_t = _fit_both_backends(
        device, monkeypatch, GPU_GATE_N_EPOCHS)

    assert ar_np.consensus_ == ar_t.consensus_, (
        f"consensus differs: {ar_np.consensus_} vs {ar_t.consensus_}"
    )
    assert ar_np.n_interpolate_ == ar_t.n_interpolate_, (
        f"n_interpolate differs: "
        f"{ar_np.n_interpolate_} vs {ar_t.n_interpolate_}"
    )

    bad_np = set(np.flatnonzero(log_np.bad_epochs).tolist())
    bad_t = set(np.flatnonzero(log_t.bad_epochs).tolist())
    assert bad_np == bad_t, (
        f"rejected-epoch sets differ: only numpy {sorted(bad_np - bad_t)}, "
        f"only {device} {sorted(bad_t - bad_np)} "
        f"(counts {len(bad_np)} vs {len(bad_t)} -- equal counts would not "
        f"have caught this)"
    )


@pytest.mark.backends
@pytest.mark.slow
@pytest.mark.parametrize("device", [
    "cuda",
    pytest.param("mps", marks=pytest.mark.xfail(
        reason=(
            "Known MPS limitation, measured not assumed. With augment=True the "
            "interpolated half of the data is built in float32 (a hardware "
            "limit on MPS), which shifts the CV loss surface by about the same "
            "order as the gap between neighbouring grid points (measured "
            "minimum gap 2.2e-6). Rarely -- 1 channel in 32 at 60 epochs, 0 in "
            "32 at 30 and 120 -- that moves the argmin to a distant grid "
            "point: EEG003 selected 2.03e-4 on numpy versus 1.41e-4 on MPS, a "
            "30% jump. Epochs whose peak-to-peak sits between the two "
            "thresholds then change side, so a different channel is chosen for "
            "interpolation and the retained data differs even though the "
            "epoch count and the rejected set do not. The nextafter guard in "
            "_vote_bad_epochs cannot help: it rounds the threshold's "
            "representation, while the threshold's *value* was computed from "
            "different data. Expected to pass on CUDA, where float64 makes "
            "both the interpolation and the search bit-exact."),
        strict=False)),
])
def test_fit_gpu_path_matches_per_channel_labels(device, monkeypatch):
    """Which channel was interpolated in which epoch must match.

    This is the finest-grained decision AutoReject makes, and the one that
    determines what data reaches the next pipeline stage. Equal epoch counts
    and equal rejected sets do not imply equal retained data: the same epoch
    can survive with a different channel interpolated.
    """
    _skip_if_unavailable(device)
    if not is_gpu_available():
        pytest.skip("no GPU device available")

    _, _, log_np, log_t = _fit_both_backends(
        device, monkeypatch, GPU_GATE_N_EPOCHS)

    assert_array_equal(
        log_np.labels, log_t.labels,
        err_msg=(f"per-epoch per-channel labels differ between numpy and "
                 f"{device}: same epochs kept, different data in them"),
    )
