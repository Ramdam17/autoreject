"""Cross-seed variance analysis.

Compares threshold stability across multiple random seeds for each backend.
Key question: does argmin have lower/equal/higher variance than bayes_opt?

Metrics:
- Per-channel coefficient of variation (CV = std/mean)
- Distribution comparison: bayes_opt vs argmin
- Determinism test: same seed N times → identical results for argmin
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def compute_variance_analysis(results_list):
    """Compute cross-seed variance from multiple results for one config.

    Parameters
    ----------
    results_list : list of dict
        Results from the same config across different seeds.
        Each dict has: backend_name, seed, threshes.

    Returns
    -------
    analysis : dict
        {backend_name: {
            'n_seeds': int,
            'per_channel_cv': {ch: float},
            'mean_cv': float,
            'std_cv': float,
            'median_cv': float,
        }}
    """
    # Group by backend
    by_backend = defaultdict(list)
    for r in results_list:
        backend = r.get("backend_name", "unknown")
        by_backend[backend].append(r)

    analysis = {}

    for backend, results in by_backend.items():
        if len(results) < 2:
            logger.info("  %s: only %d seed(s), skipping variance",
                        backend, len(results))
            continue

        # Build threshold matrix: (n_seeds, n_channels)
        ch_names = sorted(results[0].get("threshes", {}).keys())
        if not ch_names:
            continue

        matrix = np.array([
            [r["threshes"].get(ch, np.nan) for ch in ch_names]
            for r in results
        ])

        # Per-channel CV
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        cvs = stds / (np.abs(means) + 1e-20)

        per_channel_cv = {ch: float(cv) for ch, cv in zip(ch_names, cvs)}

        analysis[backend] = {
            "n_seeds": len(results),
            "per_channel_cv": per_channel_cv,
            "mean_cv": float(cvs.mean()),
            "std_cv": float(cvs.std()),
            "median_cv": float(np.median(cvs)),
            "max_cv": float(cvs.max()),
            "threshold_matrix": matrix,  # for figures
            "ch_names": ch_names,
        }

    return analysis


def compare_variance_envelopes(analysis, ref_backend="torch_gpu",
                               test_backend="torch_gpu_argmin"):
    """Compare variance envelopes between two backends.

    Parameters
    ----------
    analysis : dict
        Output of compute_variance_analysis.
    ref_backend : str
        Reference backend (typically bayes_opt).
    test_backend : str
        Test backend (typically argmin).

    Returns
    -------
    comparison : dict
        Comparison metrics.
    """
    ref = analysis.get(ref_backend)
    test = analysis.get(test_backend)

    if ref is None or test is None:
        return {"available": False}

    ref_cvs = np.array(list(ref["per_channel_cv"].values()))
    test_cvs = np.array(list(test["per_channel_cv"].values()))

    # Paired comparison (same channels)
    common = sorted(set(ref["per_channel_cv"].keys()) &
                    set(test["per_channel_cv"].keys()))
    if not common:
        return {"available": False}

    ref_paired = np.array([ref["per_channel_cv"][ch] for ch in common])
    test_paired = np.array([test["per_channel_cv"][ch] for ch in common])

    # Which has lower variance per channel?
    test_lower = (test_paired < ref_paired).mean() * 100
    test_higher = (test_paired > ref_paired).mean() * 100
    equal = 100 - test_lower - test_higher

    # Effect size: mean difference in CV
    cv_diff = test_paired - ref_paired
    mean_diff = cv_diff.mean()

    # Paired t-test (if scipy available)
    p_value = float("nan")
    try:
        from scipy import stats
        _, p_value = stats.ttest_rel(test_paired, ref_paired)
    except ImportError:
        pass

    return {
        "available": True,
        "ref_backend": ref_backend,
        "test_backend": test_backend,
        "n_channels": len(common),
        "ref_mean_cv": float(ref_paired.mean()),
        "test_mean_cv": float(test_paired.mean()),
        "test_lower_pct": test_lower,
        "test_higher_pct": test_higher,
        "equal_pct": equal,
        "mean_cv_diff": float(mean_diff),
        "p_value": float(p_value),
    }


def check_determinism(results_list, backend_name="torch_gpu_argmin"):
    """Check if a backend produces identical results for the same seed.

    Parameters
    ----------
    results_list : list of dict
        Results that should include multiple runs with the same seed.

    Returns
    -------
    is_deterministic : bool
    details : str
    """
    # Group by (backend, seed)
    groups = defaultdict(list)
    for r in results_list:
        if r.get("backend_name") == backend_name:
            groups[r.get("seed")].append(r)

    for seed, runs in groups.items():
        if len(runs) < 2:
            continue

        # Compare thresholds across runs with same seed
        for i in range(1, len(runs)):
            t0 = runs[0].get("threshes", {})
            ti = runs[i].get("threshes", {})
            for ch in t0:
                if ch in ti and t0[ch] != ti[ch]:
                    return False, (
                        f"Non-deterministic: seed={seed}, ch={ch}, "
                        f"run0={t0[ch]}, run{i}={ti[ch]}"
                    )

    return True, "All same-seed runs produced identical thresholds"
