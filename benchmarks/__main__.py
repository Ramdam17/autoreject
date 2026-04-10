"""Allow `python -m benchmarks` to run the benchmark CLI."""

from .run import main

raise SystemExit(main())
