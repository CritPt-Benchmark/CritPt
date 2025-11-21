#!/usr/bin/env python3
"""
Batch evaluation script for all generated results.

This script finds all submission files in the results directory,
submits them as ONE SINGLE batch to the evaluation server,
and saves the aggregate result.

The server handles batching and concurrency internally.

Usage:
    python evaluate_all_results.py --api-key YOUR_API_KEY
    python evaluate_all_results.py --server-url http://localhost:8000  # for local server
    python evaluate_all_results.py --api-key YOUR_API_KEY --results-dir results/generations
    python evaluate_all_results.py --api-key YOUR_API_KEY --output-dir results/evaluations
"""

import argparse
import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from critpt.submission import Submission
from critpt.evaluation.eval_client import AsyncEvaluationClient


def find_all_submission_files(base_dir: Path) -> list[Path]:
    """
    Find all submission JSON files recursively.

    Returns:
        List of paths to submission JSON files
    """
    submission_files = []

    for json_file in base_dir.rglob("*.json"):
        # Skip metadata and evaluation files
        if json_file.name in ["batch_metadata.json", "evaluation_summary.json", "aggregate_report.json"]:
            continue
        if "_eval.json" in json_file.name:
            continue
        if json_file.name in ["main.json"] + [f"sub_{i}.json" for i in range(10)]:
            continue

        # Check if this looks like a submission file
        if "_main.json" in json_file.name or "_sub_" in json_file.name:
            submission_files.append(json_file)

    return sorted(submission_files)


async def main_async(args):
    """Async main function."""
    # Convert to Path objects
    results_base = Path(args.results_dir)
    output_base = Path(args.output_dir)

    if not results_base.exists():
        print(f"Error: Results directory does not exist: {results_base}")
        sys.exit(1)

    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Results directory: {results_base.absolute()}")
    print(f"Output directory:  {output_base.absolute()}")
    print(f"Server URL:        {args.server_url}")
    print(f"Skip if exists:    {args.skip_if_exists}")
    print("=" * 70)
    print()

    # Check if aggregate report already exists and skip if requested
    aggregate_report_file = output_base / "aggregate_report.json"
    if args.skip_if_exists and aggregate_report_file.exists():
        print(f"Aggregate report already exists: {aggregate_report_file}")
        print("Skipping evaluation (--skip-if-exists enabled)")
        sys.exit(0)

    # Find all submission files
    print("Searching for submission files...")
    submission_files = find_all_submission_files(results_base)

    if not submission_files:
        print("No submission files found!")
        print(f"Searched in: {results_base.absolute()}")
        sys.exit(1)

    print(f"Found {len(submission_files)} submission file(s)")
    print()

    if args.dry_run:
        print("DRY RUN - Would evaluate these files:")
        for i, file_path in enumerate(submission_files[:10], 1):
            rel_path = file_path.relative_to(results_base)
            print(f"  [{i}] {rel_path}")
        if len(submission_files) > 10:
            print(f"  ... and {len(submission_files) - 10} more")
        print()
        print("DRY RUN - No evaluation performed")
        sys.exit(0)

    # Load all submissions and track original file paths
    print("Loading all submissions...")
    submissions = []
    submission_file_paths = []  # Track original paths to preserve directory structure
    failed_to_load = []

    for file_path in submission_files:
        try:
            submission = Submission.from_json(file_path)
            submissions.append(submission)
            submission_file_paths.append(file_path)  # Keep track of original path
        except Exception as e:
            rel_path = file_path.relative_to(results_base)
            failed_to_load.append({"file": str(rel_path), "error": str(e)})
            print(f"Warning: Failed to load {rel_path}: {e}")

    print(f"Successfully loaded {len(submissions)} submission(s)")
    if failed_to_load:
        print(f"Failed to load {len(failed_to_load)} file(s)")
    print()

    if len(submissions) == 0:
        print("Error: No valid submissions loaded")
        sys.exit(1)

    print("=" * 70)
    print("SUBMITTING TO SERVER")
    print("=" * 70)
    print()

    # Connect to evaluation server and submit ALL as ONE batch
    async with AsyncEvaluationClient(args.server_url, api_key=args.api_key) as client:
        # Check server health (optional - public APIs may not have this endpoint)
        # try:
        #     health = await client.check_health()
        #     print(f"Server status: {health.get('status', 'unknown')}")
        #     if not health.get('initialized'):
        #         print("Warning: Server is not initialized")
        # except Exception as e:
        #     # Health check may not be available on public APIs - continue anyway
        #     print(f"Note: Health check not available ({e})")
        #     print("Continuing with evaluation...")

        print(f"\nSubmitting ALL {len(submissions)} submission(s) as ONE SINGLE BATCH...")
        print("(Server will handle batching and concurrency internally)")
        print()

        progress_bar = None
        progress_log_step = max(1, len(submissions) // 10)
        last_logged = 0

        if tqdm is not None:
            progress_bar = tqdm(total=len(submissions), desc="Evaluating", unit="submission", leave=True)

        def progress_callback(completed: int, total: int) -> None:
            nonlocal progress_log_step, last_logged
            if total > 0:
                progress_log_step = max(1, total // 10)
            if progress_bar is not None:
                if total:
                    progress_bar.total = total
                progress_bar.n = completed
                progress_bar.refresh()
            else:
                if total and (completed == total or completed - last_logged >= progress_log_step):
                    pct = (completed / total * 100) if total else 0.0
                    print(f"  Progress: {completed}/{total} ({pct:.1f}%)")
                    last_logged = completed

        try:
            # Submit ALL as ONE batch
            batch_metadata = {
                "total_files": len(submission_files),
                "loaded_submissions": len(submissions),
                "failed_to_load": len(failed_to_load),
                "timestamp": datetime.now().isoformat()
            }

            batch_payload = await client.evaluate_batch(
                submissions=submissions,
                batch_metadata=batch_metadata,
                stream_progress=True,
                progress_callback=progress_callback
            )

            if progress_bar is not None:
                progress_bar.n = progress_bar.total
                progress_bar.refresh()
                progress_bar.close()

            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE")
            print("=" * 70)

        except Exception as e:
            if progress_bar is not None:
                progress_bar.close()
            print(f"\nError: Batch evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    metrics: Dict[str, Any] = {}

    if isinstance(batch_payload, dict):
        metrics = batch_payload

    # Compose human readable summary (adds total counts on top of raw metrics)
    summary = {
        "total_submissions": len(submissions),
        "accuracy": metrics.get("accuracy", 0.0),
        "timeout_rate": metrics.get("timeout_rate", 0.0),
        "server_timeout_count": metrics.get("server_timeout_count", 0)
    }

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total submissions: {summary['total_submissions']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Server timeout rate: {summary['timeout_rate']:.2%}")
    print(f"Server timeouts: {summary['server_timeout_count']}")

    print("=" * 70)

    # Save aggregate report
    output_base.mkdir(parents=True, exist_ok=True)

    aggregate_report = {
        "timestamp": datetime.now().isoformat(),
        "total_files_found": len(submission_files),
        "total_submissions_loaded": len(submissions),
        "failed_to_load": failed_to_load,
        "summary": summary,
        "metrics": metrics,
    }


    with open(aggregate_report_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate_report, f, indent=2, ensure_ascii=False)

    print()
    print(f"Aggregate report saved to: {aggregate_report_file}")
    print()
    print("=" * 70)
    print("✓ Batch evaluation complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate all generated results as ONE single batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using public API (default)
  python evaluate_all_results.py --api-key YOUR_API_KEY

  # Evaluate using local server
  python evaluate_all_results.py --server-url http://localhost:8000

  # Specify custom results directory
  python evaluate_all_results.py --api-key YOUR_API_KEY --results-dir results/generations

  # Save to custom output directory
  python evaluate_all_results.py --api-key YOUR_API_KEY --output-dir results/evaluations_2024

  # Skip if aggregate report already exists
  python evaluate_all_results.py --api-key YOUR_API_KEY --skip-if-exists

Note: The server handles batching and concurrency control internally.
Use server startup parameters to control batch size and concurrency.
        """
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/generations",
        help="Base directory containing generated results (default: results/generations)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluations",
        help="Directory to save aggregate report (default: results/evaluations)"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="https://artificialanalysis.ai/api/v2/critpt/evaluate",
        help="URL of the evaluation server (default: https://artificialanalysis.ai/api/v2/critpt/evaluate)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (required for public server)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be evaluated, don't actually evaluate"
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip evaluation if aggregate report file already exists (default: False)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming progress updates (use if public API doesn't support streaming)"
    )

    args = parser.parse_args()

    try:
        # Run async main
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
