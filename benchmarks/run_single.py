#!/usr/bin/env python
"""
Run a single benchmark configuration.

Usage:
    python run_single.py --config small_fast
    python run_single.py --config standard_64ch --overwrite
    python run_single.py --channels 128 --sfreq 500 --duration 10 --cv 10
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def sanitize_for_json(obj):
    """Recursively replace inf/nan with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def is_neighbor_match(cpu_val, gpu_val, grid):
    """Check if GPU value is within ±1 step of CPU value in the grid."""
    if cpu_val == gpu_val:
        return True, 0
    
    try:
        cpu_idx = list(grid).index(cpu_val)
        gpu_idx = list(grid).index(gpu_val)
        diff = abs(cpu_idx - gpu_idx)
        return diff <= 1, diff
    except ValueError:
        return False, -1


def generate_comparison_figure(results, n_interpolate_grid, consensus_grid, output_path):
    """Generate a figure comparing CPU and GPU loss grids."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    cpu_loss = results.get("cpu", {}).get("loss_grid")
    gpu_loss = results.get("gpu", {}).get("loss_grid")
    
    if cpu_loss is None or gpu_loss is None:
        # No loss grids available, show parameters only
        ax = axes[1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        cpu_cons = results.get("cpu", {}).get("consensus", {}).get("eeg", "?")
        gpu_cons = results.get("gpu", {}).get("consensus", {}).get("eeg", "?")
        cpu_nint = results.get("cpu", {}).get("n_interpolate", {}).get("eeg", "?")
        gpu_nint = results.get("gpu", {}).get("n_interpolate", {}).get("eeg", "?")
        
        text = f"CPU: consensus={cpu_cons}, n_interpolate={cpu_nint}\n"
        text += f"GPU: consensus={gpu_cons}, n_interpolate={gpu_nint}\n\n"
        
        match_info = results.get("comparison", {})
        if match_info.get("exact_match"):
            text += "✅ Exact match"
        elif match_info.get("neighbor_match"):
            text += f"≈ Neighbor match (within ±1 step)"
        else:
            text += "❌ Results differ"
        
        ax.text(5, 5, text, ha='center', va='center', fontsize=14, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.axis('off')
        ax.set_title("Parameter Comparison")
        
        # Hide other axes
        axes[0].axis('off')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    cpu_loss = np.array(cpu_loss)
    gpu_loss = np.array(gpu_loss)
    
    # Plot 1: CPU loss grid
    im1 = axes[0].imshow(cpu_loss, aspect='auto', cmap='viridis')
    axes[0].set_title("CPU Loss Grid (float64)")
    axes[0].set_xlabel("n_interpolate")
    axes[0].set_ylabel("consensus")
    axes[0].set_xticks(range(len(n_interpolate_grid)))
    axes[0].set_xticklabels(n_interpolate_grid)
    axes[0].set_yticks(range(len(consensus_grid)))
    axes[0].set_yticklabels([f"{c:.2f}" for c in consensus_grid])
    plt.colorbar(im1, ax=axes[0], label='Loss')
    
    # Mark CPU minimum
    cpu_min = np.unravel_index(np.nanargmin(cpu_loss), cpu_loss.shape)
    axes[0].scatter(cpu_min[1], cpu_min[0], c='red', s=200, marker='*', 
                    edgecolors='white', linewidths=2, zorder=5)
    
    # Plot 2: GPU loss grid
    im2 = axes[1].imshow(gpu_loss, aspect='auto', cmap='viridis')
    axes[1].set_title("GPU Loss Grid (float32)")
    axes[1].set_xlabel("n_interpolate")
    axes[1].set_ylabel("consensus")
    axes[1].set_xticks(range(len(n_interpolate_grid)))
    axes[1].set_xticklabels(n_interpolate_grid)
    axes[1].set_yticks(range(len(consensus_grid)))
    axes[1].set_yticklabels([f"{c:.2f}" for c in consensus_grid])
    plt.colorbar(im2, ax=axes[1], label='Loss')
    
    # Mark GPU minimum
    gpu_min = np.unravel_index(np.nanargmin(gpu_loss), gpu_loss.shape)
    axes[1].scatter(gpu_min[1], gpu_min[0], c='red', s=200, marker='*',
                    edgecolors='white', linewidths=2, zorder=5)
    
    # Plot 3: Difference (CPU - GPU)
    diff = cpu_loss - gpu_loss
    max_diff = np.nanmax(np.abs(diff))
    im3 = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r', 
                         vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f"Difference (CPU - GPU)\nmax|diff|={max_diff:.2e}")
    axes[2].set_xlabel("n_interpolate")
    axes[2].set_ylabel("consensus")
    axes[2].set_xticks(range(len(n_interpolate_grid)))
    axes[2].set_xticklabels(n_interpolate_grid)
    axes[2].set_yticks(range(len(consensus_grid)))
    axes[2].set_yticklabels([f"{c:.2f}" for c in consensus_grid])
    plt.colorbar(im3, ax=axes[2], label='Δ Loss')
    
    # Mark both minima on diff plot
    axes[2].scatter(cpu_min[1], cpu_min[0], c='blue', s=150, marker='o',
                    edgecolors='white', linewidths=2, zorder=5, label='CPU min')
    axes[2].scatter(gpu_min[1], gpu_min[0], c='red', s=150, marker='s',
                    edgecolors='white', linewidths=2, zorder=5, label='GPU min')
    axes[2].legend(loc='upper right')
    
    # Title with match status
    match_info = results.get("comparison", {})
    if match_info.get("exact_match"):
        status = "✅ Exact match"
    elif match_info.get("neighbor_match"):
        status = "≈ Neighbor match (within ±1 step)"
    else:
        status = "❌ Results differ"
    
    speedup = match_info.get("speedup", 0)
    fig.suptitle(f"{results.get('name', 'Benchmark')} - Speedup: {speedup:.1f}x - {status}",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def get_machine_info():
    """Collect machine information."""
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count(),
    }
    
    # Try to get more CPU info on macOS
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info["cpu"] = result.stdout.strip()
        except Exception:
            pass
    
    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["ram_gb"] = "Unknown (install psutil)"
    
    # GPU info
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
            info["gpu_type"] = "CUDA"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["gpu"] = "Apple Silicon (MPS)"
            info["gpu_type"] = "MPS"
            # Try to get chip info
            if platform.system() == "Darwin":
                try:
                    import subprocess
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True
                    )
                    if "Apple" in result.stdout:
                        info["gpu"] = f"Apple Silicon MPS ({result.stdout.strip()})"
                except Exception:
                    pass
        else:
            info["gpu"] = "None (CPU only)"
            info["gpu_type"] = "CPU"
    except ImportError:
        info["torch_version"] = "Not installed"
        info["gpu"] = "Unknown"
        info["gpu_type"] = "Unknown"
    
    # MNE version
    try:
        import mne
        info["mne_version"] = mne.__version__
    except ImportError:
        info["mne_version"] = "Not installed"
    
    # AutoReject version
    try:
        import autoreject
        info["autoreject_version"] = autoreject.__version__
    except (ImportError, AttributeError):
        info["autoreject_version"] = "Unknown"
    
    return info


def create_synthetic_data(n_channels, sfreq, epoch_duration, recording_duration,
                          artifact_pct, random_state=42):
    """Create synthetic EEG data with artifacts."""
    import mne
    
    np.random.seed(random_state)
    
    n_times = int(epoch_duration * sfreq)
    n_epochs = int(recording_duration * 60 / epoch_duration)
    
    # Choose appropriate montage based on channel count
    if n_channels <= 32:
        montage_name = 'standard_1020'
    elif n_channels <= 64:
        montage_name = 'standard_1005'
    elif n_channels <= 128:
        montage_name = 'GSN-HydroCel-128'
    else:
        montage_name = 'GSN-HydroCel-256'
    
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        ch_names = montage.ch_names[:n_channels]
    except Exception:
        # Fallback: generate channel names
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        montage = None
    
    # Create info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    if montage is not None:
        try:
            info.set_montage(montage)
        except Exception:
            pass  # Some montages may not work with all channel counts
    
    # Generate base signal (pink noise-like)
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-5
    
    # Add some structure (slow oscillations)
    t = np.arange(n_times) / sfreq
    for epoch_idx in range(n_epochs):
        # Add alpha-like oscillation (8-12 Hz)
        freq = np.random.uniform(8, 12)
        phase = np.random.uniform(0, 2 * np.pi)
        alpha = np.sin(2 * np.pi * freq * t + phase) * 2e-5
        data[epoch_idx] += alpha
    
    # Add artifacts
    n_bad_epochs = int(n_epochs * artifact_pct)
    artifact_epochs = np.random.choice(n_epochs, n_bad_epochs, replace=False)
    
    for epoch_idx in artifact_epochs:
        # Random number of bad channels per epoch (1-20% of channels)
        n_bad_chs = np.random.randint(1, max(2, int(n_channels * 0.2)))
        bad_chs = np.random.choice(n_channels, n_bad_chs, replace=False)
        
        # Add large amplitude artifacts
        artifact_amplitude = np.random.uniform(3, 10) * 1e-4
        data[epoch_idx, bad_chs, :] += (
            np.random.randn(n_bad_chs, n_times) * artifact_amplitude
        )
    
    # Create epochs object
    epochs = mne.EpochsArray(data, info, verbose=False)
    
    return epochs, {
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_artifact_epochs": n_bad_epochs,
        "data_size_mb": data.nbytes / 1e6,
    }


def run_benchmark(config, logger):
    """Run a single benchmark (CPU Legacy, CPU Current, and GPU)."""
    import mne
    mne.set_log_level('ERROR')
    
    from autoreject import AutoReject
    from autoreject.backends import clear_backend_cache
    
    # Try to import GPU cache
    try:
        from autoreject.gpu_interpolation import _LOOCV_INTERP_CACHE
        has_loocv_cache = True
    except ImportError:
        has_loocv_cache = False
        _LOOCV_INTERP_CACHE = {}
    
    # Import legacy module for CPU Legacy benchmark
    try:
        from legacy import utils_original
        has_legacy = True
    except ImportError:
        has_legacy = False
        logger.warning("Legacy module not found, skipping CPU Legacy benchmark")
    
    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "machine": get_machine_info(),
    }
    
    # Resolve n_interpolate and consensus from config names
    n_interpolate_map = {
        "light": [1, 4],
        "medium": [1, 4, 8],
        "aggressive": [1, 2, 4, 8, 12, 16],
    }
    consensus_map = {
        "light": [0.1, 0.3, 0.5],
        "standard": np.linspace(0.1, 0.5, 5).tolist(),
    }
    
    n_interpolate = n_interpolate_map.get(
        config.get("n_interpolate", "medium"), 
        config.get("n_interpolate", [1, 4, 8])
    )
    consensus = consensus_map.get(
        config.get("consensus", "standard"),
        config.get("consensus", np.linspace(0.1, 0.5, 5).tolist())
    )
    
    logger.info(f"Creating synthetic data...")
    epochs, data_info = create_synthetic_data(
        n_channels=config["channels"],
        sfreq=config["sfreq"],
        epoch_duration=config["epoch_duration"],
        recording_duration=config["recording_duration"],
        artifact_pct=config["artifact_pct"],
        random_state=config.get("random_state", 42),
    )
    results["data_info"] = data_info
    logger.info(f"  Created {data_info['n_epochs']} epochs × {data_info['n_channels']} channels")
    logger.info(f"  Data size: {data_info['data_size_mb']:.1f} MB")
    
    # ========== CPU Legacy Benchmark ==========
    if has_legacy:
        logger.info("=" * 60)
        logger.info("Running CPU Legacy benchmark (original utils.py)...")
        
        os.environ['AUTOREJECT_BACKEND'] = 'numpy'
        clear_backend_cache()
        if has_loocv_cache:
            _LOOCV_INTERP_CACHE.clear()
        
        # Monkey-patch to use legacy interpolation
        import autoreject.utils as utils_module
        original_interpolate = utils_module._interpolate_bads_eeg
        utils_module._interpolate_bads_eeg = utils_original._interpolate_bads_eeg
        
        ar_legacy = AutoReject(
            n_interpolate=n_interpolate,
            consensus=consensus,
            cv=config["cv_folds"],
            random_state=config.get("random_state", 42),
            n_jobs=1,
            verbose=True,
            device='cpu'
        )
        
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            ar_legacy.fit(epochs)
            legacy_time = time.perf_counter() - start_time
            legacy_success = True
            legacy_error = None
        except Exception as e:
            legacy_time = time.perf_counter() - start_time
            legacy_success = False
            legacy_error = str(e)
            logger.error(f"CPU Legacy benchmark failed: {e}")
        
        legacy_memory_peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        
        # Restore original function
        utils_module._interpolate_bads_eeg = original_interpolate
        
        results["cpu_legacy"] = {
            "time_seconds": legacy_time,
            "memory_peak_mb": legacy_memory_peak,
            "success": legacy_success,
            "error": legacy_error,
        }
        
        if legacy_success:
            results["cpu_legacy"]["consensus"] = {
                k: float(v) for k, v in ar_legacy.consensus_.items()
            }
            results["cpu_legacy"]["n_interpolate"] = {
                k: int(v) for k, v in ar_legacy.n_interpolate_.items()
            }
            logger.info(f"  CPU Legacy time: {legacy_time:.2f}s")
            logger.info(f"  CPU Legacy memory peak: {legacy_memory_peak:.1f} MB")
            logger.info(f"  Results: consensus={ar_legacy.consensus_}, n_interpolate={ar_legacy.n_interpolate_}")
    else:
        results["cpu_legacy"] = {"success": False, "error": "Legacy module not available"}
    
    # ========== CPU Current Benchmark ==========
    logger.info("=" * 60)
    logger.info("Running CPU Current benchmark (with sphere centering fix)...")
    
    os.environ['AUTOREJECT_BACKEND'] = 'numpy'
    clear_backend_cache()
    if has_loocv_cache:
        _LOOCV_INTERP_CACHE.clear()
    
    ar_cpu = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        cv=config["cv_folds"],
        random_state=config.get("random_state", 42),
        n_jobs=1,
        verbose=True,
        device='cpu'
    )
    
    # Memory tracking
    tracemalloc.start()
    
    start_time = time.perf_counter()
    try:
        ar_cpu.fit(epochs)
        cpu_time = time.perf_counter() - start_time
        cpu_success = True
        cpu_error = None
    except Exception as e:
        cpu_time = time.perf_counter() - start_time
        cpu_success = False
        cpu_error = str(e)
        logger.error(f"CPU Current benchmark failed: {e}")
    
    cpu_memory_peak = tracemalloc.get_traced_memory()[1] / 1e6  # MB
    tracemalloc.stop()
    
    results["cpu"] = {
        "time_seconds": cpu_time,
        "memory_peak_mb": cpu_memory_peak,
        "success": cpu_success,
        "error": cpu_error,
    }
    
    if cpu_success:
        results["cpu"]["consensus"] = {
            k: float(v) for k, v in ar_cpu.consensus_.items()
        }
        results["cpu"]["n_interpolate"] = {
            k: int(v) for k, v in ar_cpu.n_interpolate_.items()
        }
        logger.info(f"  CPU Current time: {cpu_time:.2f}s")
        logger.info(f"  CPU Current memory peak: {cpu_memory_peak:.1f} MB")
        logger.info(f"  Results: consensus={ar_cpu.consensus_}, n_interpolate={ar_cpu.n_interpolate_}")
    
    # ========== GPU Benchmark ==========
    logger.info("=" * 60)
    logger.info("Running GPU benchmark...")
    
    os.environ['AUTOREJECT_BACKEND'] = 'torch'
    clear_backend_cache()
    if has_loocv_cache:
        _LOOCV_INTERP_CACHE.clear()
    
    ar_gpu = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        cv=config["cv_folds"],
        random_state=config.get("random_state", 42),
        n_jobs=1,
        verbose=True,
        device='gpu'
    )
    
    # Memory tracking
    tracemalloc.start()
    
    start_time = time.perf_counter()
    try:
        ar_gpu.fit(epochs)
        gpu_time = time.perf_counter() - start_time
        gpu_success = True
        gpu_error = None
    except Exception as e:
        gpu_time = time.perf_counter() - start_time
        gpu_success = False
        gpu_error = str(e)
        logger.error(f"GPU benchmark failed: {e}")
        logger.error(traceback.format_exc())
    
    gpu_memory_peak = tracemalloc.get_traced_memory()[1] / 1e6  # MB
    tracemalloc.stop()
    
    results["gpu"] = {
        "time_seconds": gpu_time,
        "memory_peak_mb": gpu_memory_peak,
        "success": gpu_success,
        "error": gpu_error,
    }
    
    if gpu_success:
        results["gpu"]["consensus"] = {
            k: float(v) for k, v in ar_gpu.consensus_.items()
        }
        results["gpu"]["n_interpolate"] = {
            k: int(v) for k, v in ar_gpu.n_interpolate_.items()
        }
        logger.info(f"  GPU time: {gpu_time:.2f}s")
        logger.info(f"  GPU memory peak: {gpu_memory_peak:.1f} MB")
        logger.info(f"  Results: consensus={ar_gpu.consensus_}, n_interpolate={ar_gpu.n_interpolate_}")
    
    # ========== Comparison ==========
    if cpu_success and gpu_success:
        speedup = cpu_time / gpu_time
        
        # Exact match check (GPU vs CPU Current)
        exact_match = (
            ar_cpu.consensus_ == ar_gpu.consensus_ and
            ar_cpu.n_interpolate_ == ar_gpu.n_interpolate_
        )
        
        # Neighbor match check (within ±1 step in the grid)
        consensus_neighbor, consensus_diff = is_neighbor_match(
            ar_cpu.consensus_.get('eeg'), 
            ar_gpu.consensus_.get('eeg'),
            consensus
        )
        n_interp_neighbor, n_interp_diff = is_neighbor_match(
            ar_cpu.n_interpolate_.get('eeg'),
            ar_gpu.n_interpolate_.get('eeg'),
            n_interpolate
        )
        neighbor_match = consensus_neighbor and n_interp_neighbor
        
        # Store loss grids if available
        if hasattr(ar_cpu, 'loss_') and ar_cpu.loss_ is not None:
            # loss_ shape: (n_consensus, n_interpolate) after mean over folds
            cpu_loss = ar_cpu.loss_.get('eeg')
            if cpu_loss is not None:
                # Average over folds if needed
                if cpu_loss.ndim == 3:
                    cpu_loss = np.mean(cpu_loss, axis=2)
                results["cpu"]["loss_grid"] = cpu_loss.tolist()
        
        if hasattr(ar_gpu, 'loss_') and ar_gpu.loss_ is not None:
            gpu_loss = ar_gpu.loss_.get('eeg')
            if gpu_loss is not None:
                if gpu_loss.ndim == 3:
                    gpu_loss = np.mean(gpu_loss, axis=2)
                results["gpu"]["loss_grid"] = gpu_loss.tolist()
        
        results["comparison"] = {
            "speedup": speedup,
            "exact_match": exact_match,
            "neighbor_match": neighbor_match,
            "consensus_diff_steps": consensus_diff,
            "n_interpolate_diff_steps": n_interp_diff,
            # Legacy field for compatibility
            "results_match": exact_match or neighbor_match,
        }
        
        # Compare with CPU Legacy if available
        if results.get("cpu_legacy", {}).get("success"):
            legacy_cons = results["cpu_legacy"]["consensus"].get("eeg")
            legacy_nint = results["cpu_legacy"]["n_interpolate"].get("eeg")
            current_cons = ar_cpu.consensus_.get("eeg")
            current_nint = ar_cpu.n_interpolate_.get("eeg")
            gpu_cons = ar_gpu.consensus_.get("eeg")
            gpu_nint = ar_gpu.n_interpolate_.get("eeg")
            
            results["comparison"]["legacy_vs_current_match"] = (
                legacy_cons == current_cons and legacy_nint == current_nint
            )
            results["comparison"]["legacy_vs_gpu_match"] = (
                legacy_cons == gpu_cons and legacy_nint == gpu_nint
            )
        
        # Store grids for reference
        results["grids"] = {
            "n_interpolate": n_interpolate,
            "consensus": consensus,
        }
        
        logger.info("=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info(f"SPEEDUP (GPU vs CPU Current): {speedup:.2f}x")
        
        if exact_match:
            logger.info("✅ GPU vs CPU Current: EXACT MATCH")
        elif neighbor_match:
            logger.info(f"≈ GPU vs CPU Current: Within tolerance (±1 step)")
            logger.info(f"   Consensus: CPU={ar_cpu.consensus_} vs GPU={ar_gpu.consensus_} (diff={consensus_diff} steps)")
            logger.info(f"   N_interpolate: CPU={ar_cpu.n_interpolate_} vs GPU={ar_gpu.n_interpolate_} (diff={n_interp_diff} steps)")
        else:
            logger.warning("❌ GPU vs CPU Current: Results differ significantly!")
            logger.warning(f"   Consensus diff: {consensus_diff} steps")
            logger.warning(f"   N_interpolate diff: {n_interp_diff} steps")
        
        # Log legacy comparison
        if results.get("cpu_legacy", {}).get("success"):
            if results["comparison"]["legacy_vs_current_match"]:
                logger.info("ℹ️  CPU Legacy vs CPU Current: SAME (no bug impact on this data)")
            else:
                logger.info("📝 CPU Legacy vs CPU Current: DIFFERENT (bug fix changed results)")
                logger.info(f"   Legacy: consensus={legacy_cons}, n_interpolate={legacy_nint}")
                logger.info(f"   Current: consensus={current_cons}, n_interpolate={current_nint}")
    
    return results, n_interpolate, consensus


def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark")
    
    # Config-based
    parser.add_argument("--config", type=str, help="Config name from config.yaml")
    
    # Manual parameters
    parser.add_argument("--channels", type=int, help="Number of channels")
    parser.add_argument("--sfreq", type=float, help="Sampling frequency (Hz)")
    parser.add_argument("--epoch-duration", type=float, default=2.0, help="Epoch duration (s)")
    parser.add_argument("--duration", type=float, help="Recording duration (minutes)")
    parser.add_argument("--cv", type=int, default=10, help="CV folds")
    parser.add_argument("--n-interpolate", type=str, default="medium", 
                        help="n_interpolate config: light, medium, aggressive")
    parser.add_argument("--consensus", type=str, default="standard",
                        help="consensus config: light, standard")
    parser.add_argument("--artifact-pct", type=float, default=0.3,
                        help="Artifact percentage (0-1)")
    
    # Execution options
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Determine output directory
    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # Load config
    if args.config:
        config_path = script_dir / "config.yaml"
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        
        # Find the named config
        config = None
        for cfg in full_config.get("subset_configs", []):
            if cfg["name"] == args.config:
                config = cfg
                break
        
        if config is None:
            print(f"Error: Config '{args.config}' not found in config.yaml")
            print("Available configs:", [c["name"] for c in full_config.get("subset_configs", [])])
            sys.exit(1)
        
        config_name = args.config
    else:
        # Build config from CLI args
        if not args.channels or not args.sfreq or not args.duration:
            print("Error: Must specify --config or (--channels, --sfreq, --duration)")
            sys.exit(1)
        
        config = {
            "channels": args.channels,
            "sfreq": args.sfreq,
            "epoch_duration": args.epoch_duration,
            "recording_duration": args.duration,
            "cv_folds": args.cv,
            "n_interpolate": args.n_interpolate,
            "consensus": args.consensus,
            "artifact_pct": args.artifact_pct,
        }
        config_name = f"bench_{args.channels}ch_{int(args.sfreq)}hz_{int(args.duration)}min"
    
    # Check if results already exist
    result_file = results_dir / f"{config_name}.json"
    log_file = logs_dir / f"{config_name}.log"
    
    if result_file.exists() and not args.overwrite:
        print(f"Results already exist: {result_file}")
        print("Use --overwrite to rerun")
        sys.exit(0)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"BENCHMARK: {config_name}")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Run benchmark
    try:
        results, n_interpolate_grid, consensus_grid = run_benchmark(config, logger)
        results["name"] = config_name
        
        # Save results (sanitize inf/nan for valid JSON)
        with open(result_file, 'w') as f:
            json.dump(sanitize_for_json(results), f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {result_file}")
        logger.info(f"Log saved to: {log_file}")
        
        # Generate comparison figure
        figure_path = figures_dir / f"{config_name}.png"
        try:
            fig_result = generate_comparison_figure(
                results, n_interpolate_grid, consensus_grid, figure_path
            )
            if fig_result:
                logger.info(f"Figure saved to: {figure_path}")
        except Exception as e:
            logger.warning(f"Could not generate figure: {e}")
        
        # Print summary
        if results.get("comparison"):
            comp = results["comparison"]
            print(f"\n{'='*60}")
            print(f"✅ BENCHMARK COMPLETE: {config_name}")
            print(f"   Speedup: {comp['speedup']:.2f}x")
            if comp.get('exact_match'):
                print(f"   Match: ✅ Exact")
            elif comp.get('neighbor_match'):
                print(f"   Match: ≈ Within tolerance (±1 step)")
            else:
                print(f"   Match: ❌ Results differ")
            print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
