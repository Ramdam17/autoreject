# Potential Bug: Missing Sphere Centering in EEG Interpolation

**Date discovered**: December 1, 2025  
**Branch**: `fix/gpu-pipeline-divergence`  
**Status**: Documented for discussion with maintainer

---

## Summary

During GPU implementation work, we noticed that `autoreject.utils._interpolate_bads_eeg()` differs from MNE's `mne.channels.interpolation._interpolate_bads_eeg()` in how it handles electrode positions before computing the interpolation matrix.

## The Difference

### MNE's implementation (mne/channels/interpolation.py)
```python
def _interpolate_bads_eeg(inst, origin, exclude=None, ecog=False, verbose=None):
    # ...
    pos_good = pos[goods_idx_pos] - origin  # ← Centers positions around origin
    pos_bad = pos[bads_idx_pos] - origin    # ← Centers positions around origin
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)
```

### autoreject's implementation (autoreject/utils.py)
```python
def _interpolate_bads_eeg(inst, picks=None):
    # ...
    pos_good = pos[goods_idx_pos]  # ← Does NOT center positions
    pos_bad = pos[bads_idx_pos]    # ← Does NOT center positions
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)
```

## Impact

The spherical spline interpolation uses cosine of angles between electrodes. When positions are not centered around the fitted sphere's origin, the computed angles may be slightly different.

### Test results:
| Version | Max difference vs MNE |
|---------|----------------------|
| autoreject (current) | 5.78e-06 |
| autoreject + centering | 1.57e-06 |

The difference is small but measurable.

## Why We Didn't Fix It

**Conformity with existing behavior is the priority.** Users of autoreject expect consistent results. Changing the interpolation behavior would:
1. Break reproducibility with previous versions
2. Potentially change research results that depend on autoreject

## Recommendation

If this is considered a bug worth fixing:
1. Add centering to match MNE behavior
2. Document as a breaking change in release notes
3. Consider a deprecation warning or version flag

## Files Affected

If a fix is desired:
- `autoreject/utils.py` - `_interpolate_bads_eeg()` function (~line 330)

## Test to Validate

```python
import mne
from autoreject.utils import _interpolate_bads_eeg

# Compare autoreject vs MNE interpolation
evoked_ar = evoked.copy()
evoked_mne = evoked.copy()

_interpolate_bads_eeg(evoked_ar, picks=None)
evoked_mne.interpolate_bads(reset_bads=True)

# Should be very close (currently ~5.78e-06 max diff)
assert np.allclose(evoked_ar.data, evoked_mne.data, atol=1e-5)
```
