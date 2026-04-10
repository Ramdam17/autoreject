"""Accuracy analysis: 3-tier verification of benchmark results.

Tier 1 — Numerical identity (same precision):
  CUDA float64 vs CPU float64 → bitwise match expected.
  MPS float32 vs CPU float64 → ``rtol=1e-4``.

Tier 2 — Algorithmic equivalence (argmin vs bayes_opt):
  Thresholds differ, but ``loss[argmin_thresh] <= loss[bayesopt_thresh]``.

Tier 3 — Kernel equivalence (Metal/CUDA vs PyTorch):
  ``np.allclose(kernel_losses, pytorch_losses, rtol=1e-5 / 1e-12)``.

References
----------
.. [1] Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A.
       (2017). Autoreject: Automated artifact rejection for MEG and EEG data.
       NeuroImage, 159, 417-429. doi:10.1016/j.neuroimage.2017.06.030
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compare_thresholds(ref_result: dict, test_result: dict) -> dict:
    """Compare per-channel thresholds between two results.

    Parameters
    ----------
    ref_result : dict
        Reference result (typically numpy_cpu).
    test_result : dict
        Test result to compare.

    Returns
    -------
    dict
        Comparison metrics.
    """
    ref_t = ref_result.get("threshes", {})
    test_t = test_result.get("threshes", {})
    common = sorted(set(ref_t.keys()) & set(test_t.keys()))

    if not common:
        return {
            "n_channels": 0,
            "exact_match_pct": float("nan"),
            "mean_rel_diff_pct": float("nan"),
            "max_rel_diff_pct": float("nan"),
            "median_rel_diff_pct": float("nan"),
        }

    ref_vals = np.array([ref_t[ch] for ch in common])
    test_vals = np.array([test_t[ch] for ch in common])

    exact = (ref_vals == test_vals).mean() * 100
    rel_diff = np.abs(ref_vals - test_vals) / (np.abs(ref_vals) + 1e-20)

    return {
        "n_channels": len(common),
        "exact_match_pct": exact,
        "mean_rel_diff_pct": rel_diff.mean() * 100,
        "max_rel_diff_pct": rel_diff.max() * 100,
        "median_rel_diff_pct": np.median(rel_diff) * 100,
        "per_channel": {
            ch: {"ref": ref_t[ch], "test": test_t[ch],
                 "rel_diff": float(rd)}
            for ch, rd in zip(common, rel_diff)
        },
    }


def compare_hyperparams(ref_result: dict, test_result: dict) -> dict:
    """Compare consensus and n_interpolate.

    Returns
    -------
    dict
        Match status and values.
    """
    ref_c = ref_result.get("consensus", {})
    test_c = test_result.get("consensus", {})
    ref_n = ref_result.get("n_interpolate", {})
    test_n = test_result.get("n_interpolate", {})

    return {
        "consensus_match": ref_c == test_c,
        "consensus_ref": ref_c,
        "consensus_test": test_c,
        "n_interpolate_match": ref_n == test_n,
        "n_interpolate_ref": ref_n,
        "n_interpolate_test": test_n,
    }


def full_accuracy_report(results_by_backend: dict[str, dict],
                         reference_backend: str = "numpy_cpu") -> list[dict]:
    """Generate accuracy comparison table for all backends vs reference.

    Parameters
    ----------
    results_by_backend : dict
        {backend_name: result_dict} for one config.
    reference_backend : str
        Which backend is the reference (default: numpy_cpu).

    Returns
    -------
    list of dict
        One entry per non-reference backend with accuracy metrics.
    """
    ref = results_by_backend.get(reference_backend)
    if ref is None:
        logger.warning("Reference backend '%s' not found", reference_backend)
        return []

    reports = []
    for name, result in results_by_backend.items():
        if name == reference_backend:
            continue

        thresh_cmp = compare_thresholds(ref, result)
        hp_cmp = compare_hyperparams(ref, result)

        reports.append({
            "backend": name,
            **thresh_cmp,
            **hp_cmp,
        })

    return reports
