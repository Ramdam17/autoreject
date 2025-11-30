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
    """Run a single benchmark (CPU and GPU)."""
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
    
    # ========== CPU Benchmark ==========
    logger.info("=" * 60)
    logger.info("Running CPU benchmark...")
    
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
        verbose=False,
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
        logger.error(f"CPU benchmark failed: {e}")
    
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
        logger.info(f"  CPU time: {cpu_time:.2f}s")
        logger.info(f"  CPU memory peak: {cpu_memory_peak:.1f} MB")
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
        verbose=False,
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
        results_match = (
            ar_cpu.consensus_ == ar_gpu.consensus_ and
            ar_cpu.n_interpolate_ == ar_gpu.n_interpolate_
        )
        
        results["comparison"] = {
            "speedup": speedup,
            "results_match": results_match,
        }
        
        logger.info("=" * 60)
        logger.info(f"SPEEDUP: {speedup:.2f}x")
        logger.info(f"Results match: {results_match}")
        
        if not results_match:
            logger.warning("⚠️  CPU and GPU results differ!")
    
    return results


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
    
    # Run benchmark
    try:
        results = run_benchmark(config, logger)
        results["name"] = config_name
        
        # Save results
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {result_file}")
        logger.info(f"Log saved to: {log_file}")
        
        # Print summary
        if results.get("comparison"):
            print(f"\n{'='*60}")
            print(f"✅ BENCHMARK COMPLETE: {config_name}")
            print(f"   Speedup: {results['comparison']['speedup']:.2f}x")
            print(f"   Results match: {results['comparison']['results_match']}")
            print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
