"""Entry point for running the Financial Fine-Tuning Laboratory benchmark.

Usage::

    # Full benchmark (all 4 approaches × 3 runs)
    python3 -m evaluation.run_benchmark

    # Quick dry run — first 5 questions only
    python3 -m evaluation.run_benchmark --dry-run

    # Single approach without hallucination judge
    python3 -m evaluation.run_benchmark --approach zero_shot --no-judge

    # Specific approaches with custom run count
    python3 -m evaluation.run_benchmark --approaches zero_shot,rag --runs 2
"""

import argparse
import logging
import sys

from approaches import (
    ChainOfThoughtApproach,
    FewShotApproach,
    RAGApproach,
    ZeroShotApproach,
)
from approaches.base import BaseApproach
from evaluation.benchmark_runner import BenchmarkRunner

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# Mapping from CLI-friendly names to approach classes
_APPROACH_MAP: dict[str, type[BaseApproach]] = {
    "zero_shot": ZeroShotApproach,
    "few_shot": FewShotApproach,
    "cot": ChainOfThoughtApproach,
    "chain_of_thought": ChainOfThoughtApproach,
    "rag": RAGApproach,
}

_DEFAULT_APPROACHES = "zero_shot,few_shot,cot,rag"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Financial Fine-Tuning Laboratory benchmark.",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default=_DEFAULT_APPROACHES,
        help=(
            "Comma-separated list of approach names to run. "
            f"Valid: {', '.join(_APPROACH_MAP.keys())}. "
            f"Default: {_DEFAULT_APPROACHES}"
        ),
    )
    parser.add_argument(
        "--approach",
        type=str,
        default=None,
        help="Run a single approach by name (shortcut for --approaches).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per approach. Default: 3",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip hallucination judge (faster, saves API calls).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the first 5 questions per approach to verify the pipeline.",
    )
    return parser


def _instantiate_approaches(names: list[str]) -> list[BaseApproach]:
    """Create approach instances from CLI names."""
    instances: list[BaseApproach] = []
    for name in names:
        name = name.strip()
        if name not in _APPROACH_MAP:
            logger.error(
                "Unknown approach '%s'. Valid: %s",
                name,
                ", ".join(_APPROACH_MAP.keys()),
            )
            sys.exit(1)
        instances.append(_APPROACH_MAP[name]())
    return instances


def _print_summary_table(summary: dict) -> None:
    """Print a clean summary table to the console."""
    approaches = summary.get("approaches", {})
    if not approaches:
        logger.warning("No approach results to display.")
        return

    header = (
        f"{'Approach':<20} {'Accuracy':>10} {'Halluc.%':>10} "
        f"{'Avg Lat.':>10} {'$/Correct':>12}"
    )
    sep = "-" * len(header)

    # Use print for the final user-facing table (not production logging)
    print(f"\n{sep}")
    print("BENCHMARK RESULTS")
    print(sep)
    print(header)
    print(sep)

    for name in summary.get("ranking", []):
        data = approaches[name]
        print(
            f"{name:<20} {data['overall_accuracy_pct']:>9.1f}% "
            f"{data['hallucination_rate_pct']:>9.1f}% "
            f"{data['avg_latency_ms']:>8.0f}ms "
            f"${data['cost_per_correct_usd']:>10.6f}"
        )

    print(sep)
    best = summary.get("best_approach", "")
    if best:
        print(f"Best approach: {best}")
    print(summary.get("statistical_notes", ""))
    print(sep)
    print()


def main() -> None:
    """Parse arguments and run the benchmark."""
    parser = _build_parser()
    args = parser.parse_args()

    # Determine which approaches to run
    if args.approach:
        approach_names = [args.approach]
    else:
        approach_names = [n.strip() for n in args.approaches.split(",")]

    logger.info("Approaches: %s", approach_names)
    logger.info("Runs per approach: %d", args.runs)
    logger.info("Judge enabled: %s", not args.no_judge)
    logger.info("Dry run: %s", args.dry_run)

    # Instantiate approaches
    instances = _instantiate_approaches(approach_names)

    # Setup RAG if included
    for inst in instances:
        if hasattr(inst, "setup"):
            logger.info("Setting up %s...", inst.approach_name)
            inst.setup()

    # Create runner
    runner = BenchmarkRunner(
        approaches=instances,
        n_runs=args.runs,
        run_judge=not args.no_judge,
    )

    # For dry-run, limit the eval set to first 5 questions
    if args.dry_run:
        logger.info("DRY RUN — limiting to first 5 questions")
        runner.eval_set.questions = runner.eval_set.questions[:5]
        runner.eval_set.total_questions = len(runner.eval_set.questions)

    # Run
    all_results = runner.run_all()

    # Generate and print summary
    summary = runner.generate_summary(all_results)
    _print_summary_table(summary)


if __name__ == "__main__":
    main()
