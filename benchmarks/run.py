"""CLI entry point for running benchmarks.

Usage
-----
    python -m benchmarks.run --suite quick
    python -m benchmarks.run --suite real_data scaling
    python -m benchmarks.run --suite variance --seeds 15
    python -m benchmarks.run --config smoke_32ch
    python -m benchmarks.run --suite all --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .core.hardware import get_machine_info, detect_device
from .core.registry import probe_backends
from .core.orchestrator import run_suite

logger = logging.getLogger(__name__)


def load_config(config_path=None):
    """Load benchmark configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_suites(config, suite_names):
    """Resolve suite names to list of config dicts."""
    suites = config.get("suites", {})
    all_configs = []

    for name in suite_names:
        if name == "all":
            for suite_configs in suites.values():
                all_configs.extend(suite_configs)
        elif name in suites:
            all_configs.extend(suites[name])
        else:
            logger.warning("Unknown suite '%s', skipping", name)

    return all_configs


def main():
    parser = argparse.ArgumentParser(
        description="AutoReject GPU Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suites:
  quick      Smoke test (< 2 min)
  real_data  3 real EEG datasets (MNE sample, ds002778, ds000117)
  scaling    Synthetic scaling analysis (channels, epochs, artifacts)
  variance   Cross-seed reproducibility test (15 seeds)
  all        Everything
        """,
    )
    parser.add_argument(
        "--suite", nargs="+", default=["quick"],
        help="Suite(s) to run (default: quick)",
    )
    parser.add_argument(
        "--config-file", type=str, default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--backends", nargs="+", default=None,
        help="Filter to specific backends",
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Override n_seeds for variance suite",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config(args.config_file)
    global_config = config.get("global", {})

    # Machine info
    machine = get_machine_info()
    device = detect_device()
    logger.info("Machine: %s", machine.get("hostname", "unknown"))
    logger.info("Device: %s (%s)", device, machine.get("gpu", "none"))

    # Probe backends
    logger.info("Probing backends...")
    all_backends = probe_backends(config.get("backends", {}))

    if args.backends:
        from .core.registry import filter_backends
        active_backends = filter_backends(all_backends, args.backends)
    else:
        active_backends = [b for b in all_backends if b.available]

    logger.info("Available backends: %s",
                [b.name for b in active_backends])

    # Resolve suites
    suite_configs = resolve_suites(config, args.suite)

    if not suite_configs:
        logger.error("No configs resolved from suites: %s", args.suite)
        return 1

    # Apply seed override
    if args.seeds is not None:
        for cfg in suite_configs:
            if "n_seeds" in cfg:
                cfg["n_seeds"] = args.seeds

    # Output directory
    output_dir = args.output_dir or str(
        Path(__file__).parent / global_config.get("output_dir", "results")
    )

    # Dry run
    if args.dry_run:
        print()
        print("DRY RUN — would execute:")
        print(f"  Suites: {args.suite}")
        print(f"  Configs: {len(suite_configs)}")
        print(f"  Backends: {[b.name for b in active_backends]}")
        print(f"  Output: {output_dir}")
        print()
        for cfg in suite_configs:
            n_seeds = cfg.get("n_seeds", 1)
            req_backends = cfg.get("backends", "all")
            print(f"  {cfg['name']:<30} seeds={n_seeds:<3} "
                  f"backends={req_backends}")
        return 0

    # Run
    logger.info("Running %d configs across %d backends...",
                len(suite_configs), len(active_backends))

    results = run_suite(
        suite_configs, all_backends, global_config, output_dir,
    )

    # Summary
    n_ok = sum(1 for r in results if not r.get("error"))
    n_fail = sum(1 for r in results if r.get("error"))
    logger.info("Done: %d succeeded, %d failed", n_ok, n_fail)

    return 0


if __name__ == "__main__":
    sys.exit(main())
