"""Multi-config benchmark orchestrator.

Iterates over benchmark suites, loads data once per config,
runs all backends, and saves results incrementally.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .datasets import load_dataset
from .registry import filter_backends
from .runner import run_single

logger = logging.getLogger(__name__)


def run_suite(suite_configs, backends, global_config, output_dir):
    """Run a complete benchmark suite.

    Parameters
    ----------
    suite_configs : list of dict
        Configs from one suite in config.yaml.
    backends : list of BackendSpec
        Available backends (already probed).
    global_config : dict
        Global settings (warmup_runs, timing_runs, etc.).
    output_dir : Path
        Directory to save JSON results.

    Returns
    -------
    results : list of BenchmarkResult
        All results from this suite.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup = global_config.get("warmup_runs", 1)
    timing = global_config.get("timing_runs", 3)
    base_seed = global_config.get("random_state", 42)

    all_results = []
    total_configs = len(suite_configs)

    for cfg_idx, config in enumerate(suite_configs, 1):
        config_name = config["name"]

        # Filter backends for this config
        requested_backends = config.get("backends", "all")
        active_backends = filter_backends(backends, requested_backends)

        if not active_backends:
            logger.warning("[%d/%d] %s: no backends available, skipping",
                           cfg_idx, total_configs, config_name)
            continue

        # Handle variance configs (multiple seeds)
        n_seeds = config.get("n_seeds", 1)
        seed_start = config.get("seed_start", base_seed)
        seeds = list(range(seed_start, seed_start + n_seeds))

        # Load data once per config (same data for all backends/seeds)
        logger.info("[%d/%d] Loading data for '%s'...",
                    cfg_idx, total_configs, config_name)
        try:
            epochs, metadata = load_dataset(config)
        except Exception as e:
            logger.error("  Failed to load data: %s", e)
            continue

        logger.info("  Data: %d epochs × %d ch × %d times",
                     metadata["n_epochs"], metadata["n_channels"],
                     metadata.get("n_times", "?"))

        for seed in seeds:
            seed_label = f" (seed={seed})" if n_seeds > 1 else ""

            for backend in active_backends:
                label = f"{config_name}/{backend.name}{seed_label}"

                # Check if result already exists
                result_file = _result_path(output_dir, config_name,
                                           backend.name, seed)
                if result_file.exists():
                    logger.info("  %s: result exists, skipping", label)
                    # Load existing result
                    with open(result_file) as f:
                        existing = json.load(f)
                    all_results.append(existing)
                    continue

                logger.info("  Running %s...", label)

                result = run_single(
                    config, backend, epochs, seed=seed,
                    warmup_runs=warmup, timing_runs=timing,
                )

                # Log summary
                if result.error:
                    logger.error("    FAILED: %s", result.error)
                else:
                    logger.info("    %.1f ms", result.elapsed_ms)

                # Save incrementally
                result_dict = result.to_dict()
                result_dict["metadata"] = metadata
                result_dict["timestamp"] = datetime.now().isoformat()

                with open(result_file, "w") as f:
                    json.dump(result_dict, f, indent=2)

                all_results.append(result_dict)

    return all_results


def _result_path(output_dir, config_name, backend_name, seed):
    """Generate the result file path."""
    return Path(output_dir) / f"{config_name}_{backend_name}_s{seed}.json"
