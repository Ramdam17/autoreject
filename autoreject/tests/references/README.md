# Reference Data for Retrocompatibility Tests

This directory contains reference outputs from the original (unoptimized) 
implementation of autoreject. These files are used to validate that 
performance optimizations do not change the numerical results.

## Files

| File | Function Tested | Contents |
|------|-----------------|----------|
| `vote_bad_epochs_v1.npz` | `_AutoReject._vote_bad_epochs` | labels, bad_sensor_counts |
| `compute_thresholds_v1.npz` | `_compute_thresholds` | thresholds dict |
| `ransac_v1.npz` | `Ransac.fit` | correlations, bad_chs_ |
| `interpolate_epochs_v1.npz` | `_interpolate_bad_epochs` | sample interpolated data |

## Generation

Reference files are generated using:

```bash
python tools/generate_references.py
```

## Versioning

- `v1`: Original implementation (baseline before optimization)
- Future versions may be added if intentional algorithmic changes are made

## Determinism

All reference data is generated with:
- `numpy.random.seed(42)`
- Fixed test data parameters (see `generate_references.py`)

## Important

**Do not modify these files unless making intentional algorithmic changes.**
If optimization changes numerical results (even slightly), the retrocompatibility 
tests will fail, indicating a potential regression.
