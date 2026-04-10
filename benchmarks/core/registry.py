"""Backend registry: discover, validate, and filter available backends.

Probes the current machine to determine which backends can run,
based on config.yaml backend definitions and installed dependencies.
"""

import logging
from dataclasses import dataclass, field

from .hardware import detect_device

logger = logging.getLogger(__name__)


@dataclass
class BackendSpec:
    """Specification for a benchmark backend."""

    name: str
    label: str
    env_backend: str  # 'numpy' or 'torch'
    device: str  # 'cpu', 'mps', 'cuda', or 'auto'
    precision: str  # 'float32', 'float64', or 'auto'
    requires: list = field(default_factory=list)
    use_argmin: bool = False
    use_kernel: bool = False
    available: bool = True
    skip_reason: str = ""

    @property
    def resolved_device(self):
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            return detect_device()
        return self.device

    @property
    def resolved_precision(self):
        """Resolve 'auto' to actual precision."""
        if self.precision == "auto":
            return "float32" if self.resolved_device == "mps" else "float64"
        return self.precision


def probe_backends(backend_configs):
    """Probe which backends are available on this machine.

    Parameters
    ----------
    backend_configs : dict
        The 'backends' section from config.yaml.

    Returns
    -------
    backends : list of BackendSpec
        All backends with availability status.
    """
    backends = []

    for name, cfg in backend_configs.items():
        spec = BackendSpec(
            name=name,
            label=cfg.get("label", name),
            env_backend=cfg.get("env_backend", "numpy"),
            device=cfg.get("device", "cpu"),
            precision=cfg.get("precision", "float64"),
            requires=cfg.get("requires", []),
            use_argmin=cfg.get("use_argmin", False),
            use_kernel=cfg.get("use_kernel", False),
        )

        # Check dependencies
        for dep in spec.requires:
            if not _check_dependency(dep):
                spec.available = False
                spec.skip_reason = f"Missing dependency: {dep}"
                break

        # Check device availability
        if spec.available and spec.device not in ("cpu", "auto"):
            actual_device = detect_device()
            if spec.device == "mps" and actual_device != "mps":
                spec.available = False
                spec.skip_reason = "MPS not available"
            elif spec.device == "cuda" and actual_device != "cuda":
                spec.available = False
                spec.skip_reason = "CUDA not available"

        status = "available" if spec.available else f"skipped ({spec.skip_reason})"
        logger.info("  Backend %-20s: %s", name, status)

        backends.append(spec)

    return backends


def filter_backends(backends, requested):
    """Filter backends by a requested list.

    Parameters
    ----------
    backends : list of BackendSpec
    requested : list of str or 'all'
        Backend names to include, or 'all' for all available.

    Returns
    -------
    list of BackendSpec
        Filtered, available backends.
    """
    if requested == "all" or requested is None:
        return [b for b in backends if b.available]

    result = []
    for name in requested:
        match = [b for b in backends if b.name == name]
        if not match:
            logger.warning("Unknown backend '%s', skipping", name)
            continue
        b = match[0]
        if not b.available:
            logger.warning("Backend '%s' not available: %s", name,
                           b.skip_reason)
            continue
        result.append(b)
    return result


def _check_dependency(dep):
    """Check if a Python dependency is importable."""
    try:
        __import__(dep)
        return True
    except ImportError:
        return False
