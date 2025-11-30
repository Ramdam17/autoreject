#!/usr/bin/env python
"""
Generate benchmark report with figures.

Usage:
    python generate_report.py           # Generate report from all results
    python generate_report.py --format pdf  # Also export figures as PDF
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    """Load all JSON results from results directory."""
    results_dir = Path(results_dir)
    results = []
    
    for f in results_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["_filename"] = f.name
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    return results


def extract_metrics(results):
    """Extract key metrics from results for plotting."""
    metrics = []
    
    for r in results:
        try:
            config = r.get("config", {})
            data_info = r.get("data_info", {})
            cpu_time = r.get("cpu", {}).get("time_seconds", 0)
            gpu_time = r.get("gpu", {}).get("time_seconds", 0)
            
            if cpu_time > 0 and gpu_time > 0:
                metrics.append({
                    "name": config.get("name", r.get("name", r["_filename"])),
                    "n_channels": data_info.get("n_channels", config.get("channels", 0)),
                    "n_epochs": data_info.get("n_epochs", 0),
                    "sfreq": config.get("sfreq", 0),
                    "n_cv": config.get("cv_folds", 0),
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": r.get("comparison", {}).get("speedup", cpu_time / gpu_time),
                    "cpu_memory_mb": r.get("cpu", {}).get("memory_peak_mb", 0),
                    "gpu_memory_mb": r.get("gpu", {}).get("memory_peak_mb", 0),
                    "outputs_match": r.get("comparison", {}).get("results_match", False),
                    "timestamp": r.get("timestamp", ""),
                })
        except Exception as e:
            print(f"Warning: Could not extract metrics from {r.get('_filename')}: {e}")
    
    return metrics


def plot_speedup_vs_channels(metrics, figures_dir, fmt="png"):
    """Plot speedup vs number of channels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by channel count
    channel_groups = {}
    for m in metrics:
        ch = m["n_channels"]
        if ch not in channel_groups:
            channel_groups[ch] = []
        channel_groups[ch].append(m["speedup"])
    
    channels = sorted(channel_groups.keys())
    means = [np.mean(channel_groups[ch]) for ch in channels]
    stds = [np.std(channel_groups[ch]) for ch in channels]
    
    ax.bar(range(len(channels)), means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels([str(ch) for ch in channels])
    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("Speedup (GPU vs CPU)")
    ax.set_title("GPU Speedup vs Channel Count")
    ax.axhline(y=1, color='r', linestyle='--', label='CPU baseline')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.3, f"{mean:.1f}x", ha='center', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(figures_dir / f"speedup_vs_channels.{fmt}", dpi=150)
    plt.close()
    
    return figures_dir / f"speedup_vs_channels.{fmt}"


def plot_speedup_vs_epochs(metrics, figures_dir, fmt="png"):
    """Plot speedup vs number of epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by epoch count
    epoch_groups = {}
    for m in metrics:
        ep = m["n_epochs"]
        if ep not in epoch_groups:
            epoch_groups[ep] = []
        epoch_groups[ep].append(m["speedup"])
    
    epochs = sorted(epoch_groups.keys())
    means = [np.mean(epoch_groups[ep]) for ep in epochs]
    stds = [np.std(epoch_groups[ep]) for ep in epochs]
    
    ax.bar(range(len(epochs)), means, yerr=stds, capsize=5, color='seagreen', alpha=0.8)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(ep) for ep in epochs])
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Speedup (GPU vs CPU)")
    ax.set_title("GPU Speedup vs Epoch Count")
    ax.axhline(y=1, color='r', linestyle='--', label='CPU baseline')
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.3, f"{mean:.1f}x", ha='center', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(figures_dir / f"speedup_vs_epochs.{fmt}", dpi=150)
    plt.close()
    
    return figures_dir / f"speedup_vs_epochs.{fmt}"


def plot_timing_comparison(metrics, figures_dir, fmt="png"):
    """Plot CPU vs GPU timing comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by CPU time
    sorted_metrics = sorted(metrics, key=lambda x: x["cpu_time"])
    
    names = [m["name"] for m in sorted_metrics]
    cpu_times = [m["cpu_time"] for m in sorted_metrics]
    gpu_times = [m["gpu_time"] for m in sorted_metrics]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, cpu_times, width, label='CPU', color='indianred', alpha=0.8)
    ax.bar(x + width/2, gpu_times, width, label='GPU', color='steelblue', alpha=0.8)
    
    ax.set_ylabel("Time (seconds)")
    ax.set_title("CPU vs GPU Execution Time")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(figures_dir / f"timing_comparison.{fmt}", dpi=150)
    plt.close()
    
    return figures_dir / f"timing_comparison.{fmt}"


def plot_memory_usage(metrics, figures_dir, fmt="png"):
    """Plot memory usage comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out zero memory values
    valid = [m for m in metrics if m["cpu_memory_mb"] > 0 and m["gpu_memory_mb"] > 0]
    
    if not valid:
        # Create placeholder
        ax.text(0.5, 0.5, "No memory data available", ha='center', va='center', transform=ax.transAxes)
        fig.savefig(figures_dir / f"memory_usage.{fmt}", dpi=150)
        plt.close()
        return figures_dir / f"memory_usage.{fmt}"
    
    names = [m["name"] for m in valid]
    cpu_mem = [m["cpu_memory_mb"] for m in valid]
    gpu_mem = [m["gpu_memory_mb"] for m in valid]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, cpu_mem, width, label='CPU', color='indianred', alpha=0.8)
    ax.bar(x + width/2, gpu_mem, width, label='GPU', color='steelblue', alpha=0.8)
    
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Usage: CPU vs GPU")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(figures_dir / f"memory_usage.{fmt}", dpi=150)
    plt.close()
    
    return figures_dir / f"memory_usage.{fmt}"


def plot_speedup_heatmap(metrics, figures_dir, fmt="png"):
    """Plot heatmap of speedup by channels and epochs."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pivot table
    channels = sorted(set(m["n_channels"] for m in metrics))
    epochs = sorted(set(m["n_epochs"] for m in metrics))
    
    if len(channels) < 2 or len(epochs) < 2:
        ax.text(0.5, 0.5, "Not enough data for heatmap", ha='center', va='center', transform=ax.transAxes)
        fig.savefig(figures_dir / f"speedup_heatmap.{fmt}", dpi=150)
        plt.close()
        return figures_dir / f"speedup_heatmap.{fmt}"
    
    # Build matrix
    speedup_matrix = np.zeros((len(channels), len(epochs)))
    count_matrix = np.zeros((len(channels), len(epochs)))
    
    for m in metrics:
        i = channels.index(m["n_channels"])
        j = epochs.index(m["n_epochs"])
        speedup_matrix[i, j] += m["speedup"]
        count_matrix[i, j] += 1
    
    # Average where multiple values
    with np.errstate(divide='ignore', invalid='ignore'):
        speedup_matrix = np.where(count_matrix > 0, speedup_matrix / count_matrix, np.nan)
    
    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=1)
    
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(e) for e in epochs])
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([str(c) for c in channels])
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Number of Channels")
    ax.set_title("GPU Speedup Heatmap (Channels × Epochs)")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup (x)")
    
    # Add text annotations
    for i in range(len(channels)):
        for j in range(len(epochs)):
            if not np.isnan(speedup_matrix[i, j]):
                ax.text(j, i, f"{speedup_matrix[i, j]:.1f}x", ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(figures_dir / f"speedup_heatmap.{fmt}", dpi=150)
    plt.close()
    
    return figures_dir / f"speedup_heatmap.{fmt}"


def generate_markdown_report(metrics, figures, output_path):
    """Generate markdown summary report."""
    report = []
    report.append("# AutoReject GPU Benchmark Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary statistics
    if metrics:
        speedups = [m["speedup"] for m in metrics]
        report.append("## Summary Statistics\n")
        report.append(f"- **Number of benchmarks:** {len(metrics)}")
        report.append(f"- **Average speedup:** {np.mean(speedups):.2f}x")
        report.append(f"- **Max speedup:** {np.max(speedups):.2f}x")
        report.append(f"- **Min speedup:** {np.min(speedups):.2f}x")
        report.append(f"- **Median speedup:** {np.median(speedups):.2f}x")
        
        # Verification
        verified = sum(1 for m in metrics if m["outputs_match"])
        report.append(f"- **Verified outputs match:** {verified}/{len(metrics)}")
        report.append("")
    
    # Figures
    report.append("## Figures\n")
    for fig_path in figures:
        report.append(f"![{fig_path.stem}]({fig_path.name})")
        report.append("")
    
    # Detailed results table
    report.append("## Detailed Results\n")
    report.append("| Config | Channels | Epochs | CPU (s) | GPU (s) | Speedup | Verified |")
    report.append("|--------|----------|--------|---------|---------|---------|----------|")
    
    for m in sorted(metrics, key=lambda x: x["speedup"], reverse=True):
        verified = "✅" if m["outputs_match"] else "❌"
        report.append(
            f"| {m['name']} | {m['n_channels']} | {m['n_epochs']} | "
            f"{m['cpu_time']:.2f} | {m['gpu_time']:.2f} | {m['speedup']:.2f}x | {verified} |"
        )
    
    report.append("")
    
    # Best configurations
    if metrics:
        report.append("## Best Speedups\n")
        top_5 = sorted(metrics, key=lambda x: x["speedup"], reverse=True)[:5]
        for i, m in enumerate(top_5, 1):
            report.append(f"{i}. **{m['name']}**: {m['speedup']:.2f}x speedup "
                         f"({m['n_channels']}ch × {m['n_epochs']}ep)")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Figure format")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Loaded {len(results)} result files")
    
    metrics = extract_metrics(results)
    print(f"Extracted metrics from {len(metrics)} benchmarks")
    
    if not metrics:
        print("No valid metrics to plot")
        return
    
    print("\nGenerating figures...")
    figures = []
    
    fig = plot_speedup_vs_channels(metrics, figures_dir, args.format)
    print(f"  ✅ {fig.name}")
    figures.append(fig)
    
    fig = plot_speedup_vs_epochs(metrics, figures_dir, args.format)
    print(f"  ✅ {fig.name}")
    figures.append(fig)
    
    fig = plot_timing_comparison(metrics, figures_dir, args.format)
    print(f"  ✅ {fig.name}")
    figures.append(fig)
    
    fig = plot_memory_usage(metrics, figures_dir, args.format)
    print(f"  ✅ {fig.name}")
    figures.append(fig)
    
    fig = plot_speedup_heatmap(metrics, figures_dir, args.format)
    print(f"  ✅ {fig.name}")
    figures.append(fig)
    
    print("\nGenerating report...")
    report_path = figures_dir / "benchmark_report.md"
    generate_markdown_report(metrics, figures, report_path)
    print(f"  ✅ {report_path.name}")
    
    print("\n" + "=" * 50)
    print("REPORT GENERATION COMPLETE")
    print("=" * 50)
    print(f"Figures saved to: {figures_dir}")
    print(f"Report saved to: {report_path}")
    
    # Print summary
    if metrics:
        speedups = [m["speedup"] for m in metrics]
        print(f"\n📊 Average speedup: {np.mean(speedups):.2f}x")
        print(f"🚀 Max speedup: {np.max(speedups):.2f}x")


if __name__ == "__main__":
    main()
