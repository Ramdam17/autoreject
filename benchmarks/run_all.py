#!/usr/bin/env python
"""
Orchestrator: Run all benchmark configurations.

Usage:
    python run_all.py                    # Run all configs
    python run_all.py --dry-run          # Show what would be run
    python run_all.py --parallel 2       # Run 2 configs in parallel
    python run_all.py --filter "scaling" # Only run configs matching pattern
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml


def load_configs(config_path):
    """Load configurations from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("subset_configs", [])


def get_existing_results(results_dir):
    """Get list of already completed benchmarks."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return set()
    return {f.stem for f in results_dir.glob("*.json")}


def run_single_config(config_name, script_path, overwrite=False, stream_output=True):
    """Run a single benchmark configuration."""
    cmd = [sys.executable, str(script_path), "--config", config_name]
    if overwrite:
        cmd.append("--overwrite")
    
    try:
        if stream_output:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            output_lines = []
            for line in process.stdout:
                print(line, end='', flush=True)
                output_lines.append(line)
            
            process.wait()
            
            return {
                "name": config_name,
                "success": process.returncode == 0,
                "stdout": ''.join(output_lines),
                "returncode": process.returncode,
            }
        else:
            # Capture output (for parallel execution)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 2,  # 2 hour timeout
            )
            return {
                "name": config_name,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
    except subprocess.TimeoutExpired:
        return {
            "name": config_name,
            "success": False,
            "error": "Timeout (2 hours)",
        }
    except Exception as e:
        return {
            "name": config_name,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel runs")
    parser.add_argument("--filter", type=str, help="Only run configs matching pattern")
    parser.add_argument("--config-file", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--log-file", type=str, default="full_initial_run.log", help="Main run log filename")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config_file
    results_dir = script_dir / "results"
    run_single_script = script_dir / "run_single.py"
    
    # Load configs
    configs = load_configs(config_path)
    
    # Filter if requested
    if args.filter:
        configs = [c for c in configs if args.filter.lower() in c["name"].lower()]
    
    if not configs:
        print("No configurations to run.")
        return
    
    # Get existing results
    existing = get_existing_results(results_dir)
    
    # Determine what to run
    to_run = []
    skipped = []
    
    for config in configs:
        name = config["name"]
        if name in existing and not args.overwrite:
            skipped.append(name)
        else:
            to_run.append(name)
    
    # Print summary
    print("=" * 60)
    print("BENCHMARK ORCHESTRATOR")
    print("=" * 60)
    print(f"Total configurations: {len(configs)}")
    print(f"Already completed: {len(skipped)}")
    print(f"To run: {len(to_run)}")
    
    if skipped:
        print(f"\nSkipping (already done): {', '.join(skipped)}")
    
    if to_run:
        print(f"\nWill run: {', '.join(to_run)}")
    
    if args.dry_run:
        print("\n[DRY RUN - No benchmarks executed]")
        return
    
    if not to_run:
        print("\nNothing to run. Use --overwrite to rerun existing benchmarks.")
        return
    
    print(f"\nStarting {len(to_run)} benchmarks...")
    print("=" * 60)
    
    # Track results
    run_results = []
    start_time = datetime.now()
    
    if args.parallel > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single_config, name, run_single_script, args.overwrite): name
                for name in to_run
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    run_results.append(result)
                    status = "✅" if result["success"] else "❌"
                    print(f"{status} {name}")
                except Exception as e:
                    run_results.append({"name": name, "success": False, "error": str(e)})
                    print(f"❌ {name}: {e}")
    else:
        # Sequential execution
        for i, name in enumerate(to_run, 1):
            print(f"\n[{i}/{len(to_run)}] Running {name}...")
            result = run_single_config(name, run_single_script, args.overwrite)
            run_results.append(result)
            
            if result["success"]:
                print(f"✅ {name} completed")
            else:
                print(f"❌ {name} failed")
                if "stderr" in result:
                    print(result["stderr"][-500:])  # Last 500 chars
    
    # Summary
    elapsed = datetime.now() - start_time
    successful = sum(1 for r in run_results if r["success"])
    failed = len(run_results) - successful
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total run: {len(run_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed}")
    
    if failed > 0:
        print("\nFailed benchmarks:")
        for r in run_results:
            if not r["success"]:
                error = r.get("error", r.get("stderr", "Unknown error"))
                print(f"  - {r['name']}: {error[:100]}")
    
    # Save run log
    log_file = script_dir / args.log_file
    with open(log_file, 'w') as f:
        json.dump({
            "start_time": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "configs_run": len(run_results),
            "successful": successful,
            "failed": failed,
            "results": run_results,
        }, f, indent=2, default=str)
    
    print(f"\nOrchestrator log saved to: {log_file}")
    
    if successful > 0:
        print(f"\n💡 Run 'python generate_report.py' to generate figures and report")


if __name__ == "__main__":
    main()
