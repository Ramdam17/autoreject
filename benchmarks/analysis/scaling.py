"""Scaling law extraction from benchmark results.

Fits speedup vs data size (channels, epochs) to identify:

- How GPU advantage scales with problem size
- Crossover point: where GPU breaks even with CPU
- Which phase dominates at each scale
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_scaling_data(results_list: list[dict],
                         group_by: str = "n_channels",
                         ref_backend: str = "numpy_cpu") -> dict:
    """Extract scaling curves from benchmark results.

    Parameters
    ----------
    results_list : list of dict
        Results from scaling suite configs.
    group_by : str
        'n_channels' or 'n_epochs'.
    ref_backend : str
        Baseline backend for speedup computation.

    Returns
    -------
    scaling : dict
        {backend_name: {
            'x': array of group_by values,
            'times_ms': array of wall times,
            'speedups': array of speedups vs ref,
        }}
    """
    from collections import defaultdict

    # Group results by (config, backend)
    by_config = defaultdict(dict)
    for r in results_list:
        cfg = r.get("config_name", "?")
        backend = r.get("backend_name", "?")
        by_config[cfg][backend] = r

    # Extract x-values and times per backend
    backends_data = defaultdict(lambda: {"x": [], "times_ms": [],
                                         "speedups": []})

    for cfg_name, backends in sorted(by_config.items()):
        ref = backends.get(ref_backend)
        if ref is None:
            continue

        x_val = ref.get(group_by, 0)
        ref_time = ref.get("elapsed_ms", 0)

        if x_val == 0 or ref_time == 0:
            continue

        for backend_name, r in backends.items():
            test_time = r.get("elapsed_ms", 0)
            if test_time == 0 or r.get("error"):
                continue

            speedup = ref_time / test_time if test_time > 0 else 0

            backends_data[backend_name]["x"].append(x_val)
            backends_data[backend_name]["times_ms"].append(test_time)
            backends_data[backend_name]["speedups"].append(speedup)

    # Convert to arrays
    scaling = {}
    for name, data in backends_data.items():
        if len(data["x"]) < 2:
            continue
        scaling[name] = {
            "x": np.array(data["x"]),
            "times_ms": np.array(data["times_ms"]),
            "speedups": np.array(data["speedups"]),
        }

    return scaling


def fit_scaling_law(x: np.ndarray, y: np.ndarray,
                    model: str = "power") -> dict:
    """Fit a scaling law to (x, y) data.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., n_channels).
    y : array-like
        Dependent variable (e.g., wall time in ms).
    model : str
        'power' for y = a * x^b, 'linear' for y = a * x + b.

    Returns
    -------
    params : dict
        Fitted parameters and R-squared.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return {"model": model, "r_squared": float("nan")}

    if model == "power":
        # log(y) = log(a) + b * log(x)
        mask = (x > 0) & (y > 0)
        if mask.sum() < 2:
            return {"model": model, "r_squared": float("nan")}

        log_x = np.log(x[mask])
        log_y = np.log(y[mask])
        coeffs = np.polyfit(log_x, log_y, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])

        # R-squared
        y_pred = a * x[mask] ** b
        ss_res = np.sum((y[mask] - y_pred) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-20)

        return {
            "model": "power",
            "a": float(a),
            "b": float(b),
            "equation": f"y = {a:.2e} * x^{b:.2f}",
            "r_squared": float(r_sq),
        }

    elif model == "linear":
        coeffs = np.polyfit(x, y, 1)
        a, b = coeffs

        y_pred = a * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-20)

        return {
            "model": "linear",
            "slope": float(a),
            "intercept": float(b),
            "equation": f"y = {a:.2e} * x + {b:.2e}",
            "r_squared": float(r_sq),
        }

    return {"model": model, "r_squared": float("nan")}


def find_crossover_point(scaling_data: dict,
                         cpu_backend: str = "numpy_cpu",
                         gpu_backend: str = "torch_gpu") -> float | None:
    """Find where GPU breaks even with CPU.

    Returns
    -------
    crossover : float or None
        The x-value where GPU becomes faster, or None if always faster/slower.
    """
    cpu = scaling_data.get(cpu_backend)
    gpu = scaling_data.get(gpu_backend)

    if cpu is None or gpu is None:
        return None

    # Find where GPU time crosses below CPU time
    # Interpolate between data points
    cpu_x = cpu["x"]
    gpu_x = gpu["x"]

    # Match x-values
    common_x = sorted(set(cpu_x) & set(gpu_x))
    if len(common_x) < 2:
        return None

    cpu_times = np.array([
        cpu["times_ms"][np.where(cpu_x == x)[0][0]] for x in common_x
    ])
    gpu_times = np.array([
        gpu["times_ms"][np.where(gpu_x == x)[0][0]] for x in common_x
    ])

    # Find sign change in (cpu_time - gpu_time)
    diff = cpu_times - gpu_times
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # Linear interpolation
            x0, x1 = common_x[i], common_x[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            crossover = x0 + (x1 - x0) * (-d0) / (d1 - d0)
            return float(crossover)

    # No crossover: GPU always faster or always slower
    if diff[0] > 0:
        return 0.0  # GPU already faster at smallest size
    return None  # GPU never faster
