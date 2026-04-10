"""Figure generation for benchmark reports.

Produces matplotlib figures matching the Legacy benchmark style
(speedup heatmaps, scaling curves, variance distributions, etc.).
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Color palette
BACKEND_COLORS = {
    "numpy_cpu": "#4A90D9",
    "torch_gpu": "#F5A623",
    "torch_gpu_argmin": "#7ED321",
    "metal_kernel": "#BD10E0",
    "cuda_kernel": "#D0021B",
}

BACKEND_LABELS = {
    "numpy_cpu": "CPU (NumPy)",
    "torch_gpu": "GPU (PyTorch)",
    "torch_gpu_argmin": "GPU (argmin)",
    "metal_kernel": "Metal kernel",
    "cuda_kernel": "CUDA kernel",
}


def save_figure(fig, output_dir, name, fmt="png", dpi=150):
    """Save a figure to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info("  Saved %s", path)
    return str(path)


def generate_all_figures(grouped_results, output_dir, scaling_data=None,
                         variance_data=None):
    """Generate all benchmark figures.

    Parameters
    ----------
    grouped_results : dict
        {config_name: {backend_name: result_dict}}
    output_dir : str or Path
    scaling_data : dict or None
        Output of scaling.extract_scaling_data.
    variance_data : dict or None
        Output of variance.compute_variance_analysis.

    Returns
    -------
    figure_paths : dict
        {figure_name: file_path}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}

    # 1. Timing comparison
    fig = plot_timing_comparison(grouped_results)
    if fig:
        paths["timing_comparison"] = save_figure(fig, output_dir,
                                                  "timing_comparison")
        plt.close(fig)

    # 2. Speedup vs channels (if scaling data available)
    if scaling_data:
        fig = plot_speedup_vs_x(scaling_data, xlabel="Channels",
                                title="Speedup vs Channel Count")
        if fig:
            paths["speedup_vs_channels"] = save_figure(
                fig, output_dir, "speedup_vs_channels")
            plt.close(fig)

    # 3. Memory usage
    fig = plot_memory_usage(grouped_results)
    if fig:
        paths["memory_usage"] = save_figure(fig, output_dir, "memory_usage")
        plt.close(fig)

    # 4. Accuracy scatter
    fig = plot_accuracy_scatter(grouped_results)
    if fig:
        paths["accuracy_scatter"] = save_figure(fig, output_dir,
                                                 "accuracy_scatter")
        plt.close(fig)

    # 5. Variance envelope
    if variance_data:
        fig = plot_variance_envelope(variance_data)
        if fig:
            paths["variance_envelope"] = save_figure(
                fig, output_dir, "variance_envelope")
            plt.close(fig)

    # 6. Validation summary
    fig = plot_validation_summary(grouped_results)
    if fig:
        paths["validation_summary"] = save_figure(fig, output_dir,
                                                    "validation_summary")
        plt.close(fig)

    return paths


def plot_timing_comparison(grouped_results):
    """Grouped bar chart: wall time per config × backend."""
    import matplotlib.pyplot as plt

    configs = sorted(grouped_results.keys())
    if not configs:
        return None

    # Collect backends present in any config
    all_backends = []
    for cfg in configs:
        for b in grouped_results[cfg]:
            if b not in all_backends:
                all_backends.append(b)

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.5), 5))

    x = np.arange(len(configs))
    width = 0.8 / max(len(all_backends), 1)

    for i, backend in enumerate(all_backends):
        times = []
        for cfg in configs:
            r = grouped_results[cfg].get(backend, {})
            t = r.get("elapsed_ms", 0)
            if r.get("error"):
                t = 0
            times.append(t / 1000)  # Convert to seconds

        offset = (i - len(all_backends) / 2 + 0.5) * width
        color = BACKEND_COLORS.get(backend, "#999999")
        label = BACKEND_LABELS.get(backend, backend)
        bars = ax.bar(x + offset, times, width, label=label, color=color)

        # Add time labels on bars
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{t:.1f}s", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("AutoReject.fit() Timing Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


def plot_speedup_vs_x(scaling_data, xlabel="Channels",
                      title="Speedup vs Data Size"):
    """Line chart: speedup vs x for each backend."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for backend, data in sorted(scaling_data.items()):
        if backend == "numpy_cpu":
            continue  # Baseline is always 1x

        color = BACKEND_COLORS.get(backend, "#999999")
        label = BACKEND_LABELS.get(backend, backend)

        sort_idx = np.argsort(data["x"])
        ax.plot(data["x"][sort_idx], data["speedups"][sort_idx],
                "o-", color=color, label=label, markersize=6)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="CPU baseline")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speedup vs CPU")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig


def plot_memory_usage(grouped_results):
    """Bar chart: peak GPU memory per config × backend."""
    import matplotlib.pyplot as plt

    configs = sorted(grouped_results.keys())
    gpu_backends = []
    for cfg in configs:
        for b in grouped_results[cfg]:
            if b != "numpy_cpu" and b not in gpu_backends:
                gpu_backends.append(b)

    if not gpu_backends:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.5), 5))

    x = np.arange(len(configs))
    width = 0.8 / max(len(gpu_backends), 1)

    for i, backend in enumerate(gpu_backends):
        mems = []
        for cfg in configs:
            r = grouped_results[cfg].get(backend, {})
            mem = r.get("peak_gpu_mb", 0)
            if np.isnan(mem):
                mem = 0
            mems.append(mem)

        offset = (i - len(gpu_backends) / 2 + 0.5) * width
        color = BACKEND_COLORS.get(backend, "#999999")
        label = BACKEND_LABELS.get(backend, backend)
        ax.bar(x + offset, mems, width, label=label, color=color)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("GPU Memory Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


def plot_accuracy_scatter(grouped_results, ref_backend="numpy_cpu"):
    """Scatter plot: CPU threshold vs GPU threshold per channel."""
    import matplotlib.pyplot as plt

    # Collect all (ref, test) threshold pairs across configs
    pairs_by_backend = defaultdict(lambda: {"ref": [], "test": []})

    for cfg_name, backends in grouped_results.items():
        ref = backends.get(ref_backend, {})
        ref_t = ref.get("threshes", {})
        if not ref_t:
            continue

        for backend_name, r in backends.items():
            if backend_name == ref_backend:
                continue
            test_t = r.get("threshes", {})
            for ch in ref_t:
                if ch in test_t:
                    pairs_by_backend[backend_name]["ref"].append(ref_t[ch])
                    pairs_by_backend[backend_name]["test"].append(test_t[ch])

    if not pairs_by_backend:
        return None

    n_backends = len(pairs_by_backend)
    fig, axes = plt.subplots(1, n_backends,
                              figsize=(5 * n_backends, 5), squeeze=False)

    for i, (backend, pairs) in enumerate(sorted(pairs_by_backend.items())):
        ax = axes[0, i]
        ref_vals = np.array(pairs["ref"])
        test_vals = np.array(pairs["test"])

        color = BACKEND_COLORS.get(backend, "#999999")
        ax.scatter(ref_vals, test_vals, alpha=0.4, s=10, color=color)

        # Diagonal
        lims = [min(ref_vals.min(), test_vals.min()),
                max(ref_vals.max(), test_vals.max())]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=0.5)

        ax.set_xlabel(f"CPU threshold ({ref_backend})")
        ax.set_ylabel(f"GPU threshold ({backend})")
        ax.set_title(BACKEND_LABELS.get(backend, backend), fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle("Threshold Accuracy: CPU vs GPU", fontsize=12)
    fig.tight_layout()

    return fig


def plot_variance_envelope(variance_data):
    """Box/violin plot: per-channel CV for each backend."""
    import matplotlib.pyplot as plt

    backends = sorted(variance_data.keys())
    if len(backends) < 2:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(backends) * 2), 5))

    data = []
    labels = []
    colors = []

    for backend in backends:
        info = variance_data[backend]
        cvs = list(info["per_channel_cv"].values())
        data.append(cvs)
        labels.append(BACKEND_LABELS.get(backend, backend))
        colors.append(BACKEND_COLORS.get(backend, "#999999"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Coefficient of Variation (per channel)")
    ax.set_title("Cross-Seed Threshold Variance by Backend")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    return fig


def plot_validation_summary(grouped_results, ref_backend="numpy_cpu"):
    """Summary figure: horizontal bar chart of parameter differences."""
    import matplotlib.pyplot as plt
    from .accuracy import compare_thresholds, compare_hyperparams

    configs = sorted(grouped_results.keys())
    gpu_backends = []
    for cfg in configs:
        for b in grouped_results[cfg]:
            if b != ref_backend and b not in gpu_backends:
                gpu_backends.append(b)

    if not gpu_backends or not configs:
        return None

    # For simplicity, use the first GPU backend
    backend = gpu_backends[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(configs) * 0.5)))

    # Left: threshold mean relative difference
    diffs = []
    c_matches = []
    n_matches = []
    cfg_labels = []

    for cfg in configs:
        ref = grouped_results[cfg].get(ref_backend)
        test = grouped_results[cfg].get(backend)
        if ref is None or test is None:
            continue

        t_cmp = compare_thresholds(ref, test)
        h_cmp = compare_hyperparams(ref, test)

        diffs.append(t_cmp["mean_rel_diff_pct"])
        c_matches.append(h_cmp["consensus_match"])
        n_matches.append(h_cmp["n_interpolate_match"])
        cfg_labels.append(cfg)

    if not cfg_labels:
        plt.close(fig)
        return None

    y = np.arange(len(cfg_labels))
    color = BACKEND_COLORS.get(backend, "#F5A623")

    ax1.barh(y, diffs, color=color, alpha=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(cfg_labels, fontsize=8)
    ax1.set_xlabel("Mean Relative Threshold Difference (%)")
    ax1.set_title(f"GPU vs CPU Threshold Differences\n({backend})")
    ax1.grid(axis="x", alpha=0.3)

    # Right: consensus + n_interpolate match table
    cell_text = []
    for i in range(len(cfg_labels)):
        c_str = "match" if c_matches[i] else "DIFF"
        n_str = "match" if n_matches[i] else "DIFF"
        cell_text.append([cfg_labels[i], f"{diffs[i]:.2f}%", c_str, n_str])

    ax2.axis("off")
    table = ax2.table(
        cellText=cell_text,
        colLabels=["Config", "Thresh diff", "Consensus", "n_interpolate"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Color cells
    for i in range(len(cfg_labels)):
        for j in range(4):
            cell = table[i + 1, j]
            if j >= 2:  # consensus and n_interpolate columns
                val = cell_text[i][j]
                if val == "match":
                    cell.set_facecolor("#D5F5D5")
                else:
                    cell.set_facecolor("#F5D5D5")

    ax2.set_title("Accuracy Summary", fontsize=10)
    fig.tight_layout()

    return fig
