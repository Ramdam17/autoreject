"""Generate Markdown benchmark report from results.

Assembles timing tables, accuracy comparisons, and optimization summary
into a structured Markdown document.

References
----------
.. [1] Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A.
       (2017). Autoreject: Automated artifact rejection for MEG and EEG data.
       NeuroImage, 159, 417-429. doi:10.1016/j.neuroimage.2017.06.030
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .accuracy import full_accuracy_report
from .figures import generate_all_figures

logger = logging.getLogger(__name__)


def load_results(results_dir: str | Path) -> list[dict]:
    """Load all JSON result files from a directory.

    Returns
    -------
    results : list of dict
    """
    results_dir = Path(results_dir)
    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["_filename"] = f.name
                results.append(data)
        except Exception as e:
            logger.warning("Could not load %s: %s", f, e)
    return results


def group_by_config(results: list[dict]) -> dict[str, dict]:
    """Group results by config_name → {backend_name: result}.

    Returns
    -------
    dict of dict
        {config_name: {backend_name: result_dict}}
    """
    grouped = {}
    for r in results:
        cfg = r.get("config_name", "unknown")
        backend = r.get("backend_name", "unknown")
        if cfg not in grouped:
            grouped[cfg] = {}
        grouped[cfg][backend] = r
    return grouped


def generate_report(results_dir: str | Path,
                    machine_info: dict | None = None,
                    output_path: str | Path | None = None) -> str:
    """Generate a complete Markdown benchmark report.

    Parameters
    ----------
    results_dir : str or Path
    machine_info : dict or None
    output_path : str or Path or None

    Returns
    -------
    report : str
        Markdown content.
    """
    results = load_results(results_dir)
    if not results:
        logger.error("No results found in %s", results_dir)
        return ""

    grouped = group_by_config(results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []

    # Header
    lines.append("# AutoReject GPU Benchmark Report")
    lines.append("")
    lines.append(f"**Date:** {timestamp}")

    if machine_info:
        lines.append(f"**Machine:** {machine_info.get('hostname', '?')}")
        lines.append(f"**CPU:** {machine_info.get('cpu', '?')}")
        lines.append(f"**GPU:** {machine_info.get('gpu', '?')} "
                      f"({machine_info.get('gpu_type', '?')})")
        lines.append(f"**Python:** {machine_info.get('python_version', '?')}")
        lines.append(f"**PyTorch:** {machine_info.get('torch_version', '?')}")
        lines.append(f"**MNE:** {machine_info.get('mne_version', '?')}")
        lines.append(f"**Metal:** {machine_info.get('metal_available', '?')}")
        lines.append(f"**CuPy:** {machine_info.get('cupy_available', '?')}")
        if "git_hash" in machine_info:
            lines.append(f"**Git:** {machine_info['git_hash']}")
    lines.append("")

    # Per-config sections
    for config_name, backends in sorted(grouped.items()):
        lines.extend(_config_section(config_name, backends))

    # Generate figures
    if output_path:
        figures_dir = Path(output_path).parent.parent / "figures"
    else:
        figures_dir = Path(results_dir).parent / "figures"

    try:
        figure_paths = generate_all_figures(grouped, figures_dir)
        if figure_paths:
            lines.append("## Figures")
            lines.append("")
            for name, path in sorted(figure_paths.items()):
                rel_path = Path(path).name
                lines.append(f"![{name}](../figures/{rel_path})")
                lines.append("")
    except Exception as e:
        logger.warning("Figure generation failed: %s", e)

    # Optimization summary
    lines.extend(_optimization_summary())

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info("Report saved to %s", output_path)

    return report


def _config_section(config_name: str, backends: dict) -> list[str]:
    """Generate report section for one config."""
    lines = []
    lines.append(f"## {config_name}")
    lines.append("")

    # Get metadata from any result
    sample = next(iter(backends.values()))
    meta = sample.get("metadata", {})
    lines.append(
        f"Data: {sample.get('n_epochs', '?')} epochs "
        f"x {sample.get('n_channels', '?')} channels "
        f"x {sample.get('n_times', '?')} times"
    )
    if meta.get("description"):
        lines.append(f"  *{meta['description']}*")
    lines.append("")

    # Performance table
    lines.append("### Performance")
    lines.append("")
    lines.append("| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |")
    lines.append("|---------|---------------|--------|---------------|")

    cpu_ms = backends.get("numpy_cpu", {}).get("elapsed_ms", float("nan"))

    for name in ("numpy_cpu", "torch_gpu", "torch_gpu_argmin",
                 "metal_kernel", "cuda_kernel"):
        if name not in backends:
            continue
        r = backends[name]
        ms = r.get("elapsed_ms", float("nan"))
        if r.get("error"):
            lines.append(f"| {name} | ERROR | - | - |")
            continue

        vs_cpu = f"{cpu_ms / ms:.1f}x" if cpu_ms > 0 and ms > 0 else "-"
        mem = r.get("peak_gpu_mb", float("nan"))
        mem_str = f"{mem:.1f}" if not np.isnan(mem) else "-"
        lines.append(f"| {name} | {ms:.1f} | {vs_cpu} | {mem_str} |")

    lines.append("")

    # Accuracy table
    accuracy = full_accuracy_report(backends)
    if accuracy:
        lines.append("### Accuracy (vs CPU reference)")
        lines.append("")
        lines.append("| Backend | Thresh match | Mean diff "
                      "| Consensus | n_interpolate |")
        lines.append("|---------|-------------|---------|"
                      "-----------|---------------|")

        for acc in accuracy:
            c_match = "match" if acc["consensus_match"] else "DIFF"
            n_match = "match" if acc["n_interpolate_match"] else "DIFF"
            lines.append(
                f"| {acc['backend']} "
                f"| {acc['exact_match_pct']:.1f}% "
                f"| {acc['mean_rel_diff_pct']:.2f}% "
                f"| {c_match} "
                f"| {n_match} |"
            )
        lines.append("")

    return lines


def _optimization_summary() -> list[str]:
    """Static optimization summary section."""
    lines = []
    lines.append("## Optimizations Explored")
    lines.append("")
    lines.append("| Optimization | Hotspot | Isolated speedup | Status |")
    lines.append("|-------------|---------|-----------------|--------|")
    lines.append(
        "| Metal fused threshold-CV kernel | batched_cv_loss (22.6%) "
        "| 4-14x vs PyTorch (small-medium) | Implemented |"
    )
    lines.append(
        "| Batched consensus scoring (einsum) | cv_scoring_loop (27.6%) "
        "| 3.7-6.4x | Implemented |"
    )
    lines.append(
        "| GPU argmin (replace bayes_opt) | bayesian_opt (11.5%) "
        "| Exact solution, deterministic | Implemented |"
    )
    lines.append(
        "| Median topk | cv_median (16.4%) "
        "| 0.5x (slower) | Abandoned |"
    )
    lines.append(
        "| Batched interpolation | per_epoch_interp (16.6%) "
        "| 1.1-1.3x (marginal) | Marginal |"
    )
    lines.append("")
    lines.append("## Key Finding")
    lines.append("")
    lines.append(
        "Bayesian optimization is redundant in the GPU pipeline: all CV losses "
        "are pre-computed in batch, so the GP surrogate models a fully known "
        "function. `argmin` gives the exact minimum (deterministic) instead of "
        "a stochastic approximation, while eliminating CPU<->GPU round-trips."
    )
    lines.append("")
    lines.append(
        "*Ref: Jas et al. (2017) 'Candidate thresholds using Bayesian "
        "optimization' — motivation was computational efficiency, not "
        "methodological.*"
    )
    lines.append("")
    return lines
