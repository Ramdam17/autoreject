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
            
            # Extract times for all implementations
            cpu_legacy_time = r.get("cpu_legacy", {}).get("time_seconds", 0)
            cpu_time = r.get("cpu", {}).get("time_seconds", 0)
            gpu_time = r.get("gpu", {}).get("time_seconds", 0)
            comparison = r.get("comparison", {})
            
            if cpu_time > 0 and gpu_time > 0:
                # Determine match status
                exact_match = comparison.get("exact_match", False)
                neighbor_match = comparison.get("neighbor_match", False)
                # Legacy fallback
                legacy_match = comparison.get("results_match", False)
                
                if exact_match:
                    match_status = "exact"
                elif neighbor_match:
                    match_status = "neighbor"
                elif legacy_match:
                    match_status = "exact"  # Legacy format
                else:
                    match_status = "mismatch"
                
                # Check if CPU Legacy matches CPU Current
                legacy_vs_current_match = comparison.get("legacy_vs_current_match", True)
                
                metrics.append({
                    "name": config.get("name", r.get("name", r["_filename"])),
                    "n_channels": data_info.get("n_channels", config.get("channels", 0)),
                    "n_epochs": data_info.get("n_epochs", 0),
                    "sfreq": config.get("sfreq", 0),
                    "n_cv": config.get("cv_folds", 0),
                    "cpu_legacy_time": cpu_legacy_time,
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": comparison.get("speedup", cpu_time / gpu_time),
                    "cpu_legacy_memory_mb": r.get("cpu_legacy", {}).get("memory_peak_mb", 0),
                    "cpu_memory_mb": r.get("cpu", {}).get("memory_peak_mb", 0),
                    "gpu_memory_mb": r.get("gpu", {}).get("memory_peak_mb", 0),
                    "match_status": match_status,
                    "legacy_vs_current_match": legacy_vs_current_match,
                    "outputs_match": match_status in ("exact", "neighbor"),  # For compatibility
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
    """Plot CPU Legacy vs CPU Current vs GPU timing comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by CPU time
    sorted_metrics = sorted(metrics, key=lambda x: x["cpu_time"])
    
    names = [m["name"] for m in sorted_metrics]
    cpu_legacy_times = [m.get("cpu_legacy_time", 0) for m in sorted_metrics]
    cpu_times = [m["cpu_time"] for m in sorted_metrics]
    gpu_times = [m["gpu_time"] for m in sorted_metrics]
    
    x = np.arange(len(names))
    width = 0.25
    
    # Only show CPU Legacy if we have data
    has_legacy = any(t > 0 for t in cpu_legacy_times)
    
    if has_legacy:
        ax.bar(x - width, cpu_legacy_times, width, label='CPU Legacy', color='darkred', alpha=0.8)
        ax.bar(x, cpu_times, width, label='CPU Current', color='indianred', alpha=0.8)
        ax.bar(x + width, gpu_times, width, label='GPU', color='steelblue', alpha=0.8)
    else:
        ax.bar(x - width/2, cpu_times, width, label='CPU', color='indianred', alpha=0.8)
        ax.bar(x + width/2, gpu_times, width, label='GPU', color='steelblue', alpha=0.8)
    
    ax.set_ylabel("Time (seconds)")
    ax.set_title("CPU Legacy vs CPU Current vs GPU Execution Time")
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


def plot_validation_summary(results, figures_dir, fmt="png"):
    """Plot validation summary showing CPU Legacy, CPU Current, and GPU parameters."""
    # Extract configs with comparison data
    valid_results = []
    for r in results:
        comp = r.get("comparison", {})
        if comp:
            valid_results.append({
                "name": r.get("config", {}).get("name", r.get("name", "?")),
                "speedup": comp.get("speedup", 0),
                "exact_match": comp.get("exact_match", False),
                "neighbor_match": comp.get("neighbor_match", False),
                "legacy_vs_current_match": comp.get("legacy_vs_current_match", True),
                "legacy_vs_gpu_match": comp.get("legacy_vs_gpu_match", False),
                "consensus_diff": comp.get("consensus_diff_steps", 0),
                "n_interp_diff": comp.get("n_interpolate_diff_steps", 0),
                # CPU Legacy
                "legacy_consensus": r.get("cpu_legacy", {}).get("consensus", {}).get("eeg", "?"),
                "legacy_n_interp": r.get("cpu_legacy", {}).get("n_interpolate", {}).get("eeg", "?"),
                # CPU Current
                "cpu_consensus": r.get("cpu", {}).get("consensus", {}).get("eeg", "?"),
                "cpu_n_interp": r.get("cpu", {}).get("n_interpolate", {}).get("eeg", "?"),
                # GPU
                "gpu_consensus": r.get("gpu", {}).get("consensus", {}).get("eeg", "?"),
                "gpu_n_interp": r.get("gpu", {}).get("n_interpolate", {}).get("eeg", "?"),
                "n_channels": r.get("data_info", {}).get("n_channels", 0),
                "n_epochs": r.get("data_info", {}).get("n_epochs", 0),
            })
    
    if not valid_results:
        return None
    
    # Sort by name
    valid_results = sorted(valid_results, key=lambda x: x["name"])
    n_configs = len(valid_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n_configs * 0.5)))
    
    # Left plot: Parameter differences
    ax1 = axes[0]
    y_pos = np.arange(n_configs)
    
    names = [r["name"] for r in valid_results]
    cons_diff = [r["consensus_diff"] for r in valid_results]
    interp_diff = [r["n_interp_diff"] for r in valid_results]
    
    # Total difference (sum of both)
    total_diff = [c + i for c, i in zip(cons_diff, interp_diff)]
    
    # Color by severity
    colors = []
    for r in valid_results:
        if r["exact_match"]:
            colors.append("green")
        elif r["neighbor_match"]:
            colors.append("orange")
        else:
            colors.append("red")
    
    # Horizontal bar chart
    bars = ax1.barh(y_pos, total_diff, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, r) in enumerate(zip(bars, valid_results)):
        width = bar.get_width()
        label = f"cons={r['consensus_diff']}, interp={r['n_interp_diff']}"
        if width > 0:
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    label, va='center', fontsize=8)
        else:
            ax1.text(0.1, bar.get_y() + bar.get_height()/2,
                    "exact", va='center', fontsize=8, color='green')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Total Parameter Difference (steps)")
    ax1.set_title("GPU vs CPU Current Parameter Differences\n(0 = exact match, 1 = neighbor, >1 = mismatch)")
    ax1.axvline(x=1, color='orange', linestyle='--', alpha=0.5, label='Neighbor threshold')
    ax1.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='Mismatch threshold')
    ax1.set_xlim(left=-0.2)
    ax1.legend(loc='lower right', fontsize=8)
    
    # Right plot: Detailed comparison table with CPU Legacy
    ax2 = axes[1]
    ax2.axis('off')
    
    # Build table data
    table_data = []
    for r in valid_results:
        # GPU vs CPU Current match
        if r["exact_match"]:
            gpu_status = "OK"
        elif r["neighbor_match"]:
            gpu_status = "~1"
        else:
            gpu_status = "FAIL"
        
        # CPU Legacy vs Current match
        legacy_status = "OK" if r["legacy_vs_current_match"] else "DIFF"
        
        # Format parameters
        def fmt_param(cons, n_int):
            if isinstance(cons, float):
                return f"{cons:.2f}/{n_int}"
            return f"{cons}/{n_int}"
        
        legacy_params = fmt_param(r["legacy_consensus"], r["legacy_n_interp"])
        cpu_params = fmt_param(r["cpu_consensus"], r["cpu_n_interp"])
        gpu_params = fmt_param(r["gpu_consensus"], r["gpu_n_interp"])
        
        table_data.append([
            r["name"][:18],
            f"{r['n_channels']}x{r['n_epochs']}",
            legacy_params,
            cpu_params,
            gpu_params,
            f"{r['speedup']:.1f}x",
            legacy_status,
            gpu_status
        ])
    
    col_labels = ["Config", "Size", "CPU Legacy", "CPU Current", "GPU", "Speedup", "Leg=Cur", "GPU=Cur"]
    
    table = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.5)
    
    # Color header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color rows by status
    for i, r in enumerate(valid_results, 1):
        # Base color by GPU match
        if r["exact_match"]:
            color = '#C6EFCE'  # Light green
        elif r["neighbor_match"]:
            color = '#FFEB9C'  # Light orange
        else:
            color = '#FFC7CE'  # Light red
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)
        
        # Color Legacy=Current column
        if r["legacy_vs_current_match"]:
            table[(i, 6)].set_facecolor('#C6EFCE')
        else:
            table[(i, 6)].set_facecolor('#FFC7CE')
    
    ax2.set_title("Detailed Comparison: CPU Legacy vs CPU Current vs GPU\n(consensus/n_interpolate)")
    
    plt.tight_layout()
    fig.savefig(figures_dir / f"validation_summary.{fmt}", dpi=150, bbox_inches='tight')
    plt.close()
    
    return figures_dir / f"validation_summary.{fmt}"


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
        exact = sum(1 for m in metrics if m.get("match_status") == "exact")
        neighbor = sum(1 for m in metrics if m.get("match_status") == "neighbor")
        mismatch = sum(1 for m in metrics if m.get("match_status") == "mismatch")
        legacy_match = sum(1 for m in metrics if m.get("legacy_vs_current_match", True))
        
        report.append("")
        report.append("### GPU vs CPU Current Conformity")
        report.append(f"- **Exact match:** {exact}/{len(metrics)}")
        report.append(f"- **Neighbor match (±1 step):** {neighbor}/{len(metrics)}")
        if mismatch > 0:
            report.append(f"- **⚠️ Mismatch:** {mismatch}/{len(metrics)}")
        
        report.append("")
        report.append("### CPU Legacy vs CPU Current Conformity")
        report.append(f"- **Legacy = Current:** {legacy_match}/{len(metrics)} ✅")
        report.append("")
    
    # Figures
    report.append("## Figures\n")
    for fig_path in figures:
        report.append(f"![{fig_path.stem}]({fig_path.name})")
        report.append("")
    
    # Detailed results table with all 3 implementations
    report.append("## Detailed Results\n")
    report.append("| Config | Ch×Ep | CPU Legacy (s) | CPU Current (s) | GPU (s) | Speedup | GPU Match |")
    report.append("|--------|-------|----------------|-----------------|---------|---------|-----------|")
    
    for m in sorted(metrics, key=lambda x: x["speedup"], reverse=True):
        match_icons = {"exact": "✅", "neighbor": "≈", "mismatch": "❌"}
        match_icon = match_icons.get(m.get("match_status", "mismatch"), "?")
        
        cpu_legacy_time = m.get("cpu_legacy_time", 0)
        legacy_str = f"{cpu_legacy_time:.1f}" if cpu_legacy_time > 0 else "-"
        
        report.append(
            f"| {m['name']} | {m['n_channels']}×{m['n_epochs']} | "
            f"{legacy_str} | {m['cpu_time']:.1f} | {m['gpu_time']:.2f} | "
            f"{m['speedup']:.1f}x | {match_icon} |"
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
    
    # Validation summary (needs raw results, not just metrics)
    fig = plot_validation_summary(results, figures_dir, args.format)
    if fig:
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
