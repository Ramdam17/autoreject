"""Machine info collection and GPU memory tracking.

Collects hardware details for reproducible benchmark reports.
"""

import os
import platform
import subprocess


def get_machine_info():
    """Collect machine information for benchmark reports.

    Returns
    -------
    info : dict
        Machine details: hostname, CPU, RAM, GPU, library versions.
    """
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count(),
    }

    # macOS: get detailed CPU name
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True,
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
        info["ram_gb"] = "Unknown"

    # PyTorch + GPU
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_type"] = "CUDA"
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_type"] = "MPS"
            info["gpu"] = _get_apple_chip_name()
        else:
            info["gpu_type"] = "none"
            info["gpu"] = "none"
    except ImportError:
        info["torch_version"] = "not installed"
        info["gpu_type"] = "none"

    # MNE version
    try:
        import mne
        info["mne_version"] = mne.__version__
    except ImportError:
        pass

    # Metal availability
    try:
        import Metal  # noqa: F401
        info["metal_available"] = True
    except ImportError:
        info["metal_available"] = False

    # CuPy availability
    try:
        import cupy  # noqa: F401
        info["cupy_available"] = True
    except ImportError:
        info["cupy_available"] = False

    # Git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            info["git_hash"] = result.stdout.strip()
    except Exception:
        pass

    return info


def _get_apple_chip_name():
    """Get Apple Silicon chip name on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "Apple Silicon"


def detect_device():
    """Auto-detect best available GPU device.

    Returns
    -------
    device : str
        'cuda', 'mps', or 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def reset_gpu_memory(device):
    """Reset GPU memory tracking counters."""
    try:
        import torch
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def get_peak_gpu_mb(device):
    """Get peak GPU memory usage in MB.

    Returns
    -------
    float
        Peak memory in MB, or NaN if unavailable.
    """
    try:
        import torch
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return float("nan")


def sync_device(device):
    """Synchronize GPU device."""
    try:
        import torch
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass
