"""CLI entry point for generating benchmark reports.

Usage
-----
    python -m benchmarks.report
    python -m benchmarks.report --results-dir results/
    python -m benchmarks.report --output reports/my_report.md
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .core.hardware import get_machine_info
from .analysis.report_generator import generate_report

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from results",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory with JSON results (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output markdown path",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = args.results_dir or str(
        Path(__file__).parent / "results"
    )

    machine = get_machine_info()
    device = machine.get("gpu_type", "cpu").lower()

    output_path = args.output or str(
        Path(__file__).parent / "reports"
        / f"benchmark_{device}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    )

    report = generate_report(results_dir, machine_info=machine,
                             output_path=output_path)

    if report:
        print(report)
    else:
        logger.error("No report generated")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
