"""Evaluation package for Financial Fine-Tuning Laboratory.

Provides scoring metrics, LLM-as-judge evaluation, statistical
analysis, and the benchmark runner that orchestrates all approaches
across the 100-question eval set.
"""

from evaluation.benchmark_runner import (
    BenchmarkRunner,
    QuestionResult,
    RunResults,
)
from evaluation.llm_judge import (
    compute_hallucination_rate,
    judge_batch,
    judge_single,
)
from evaluation.metrics import (
    MetricResult,
    compute_cost_per_correct_answer,
    compute_factual_accuracy,
    compute_hallucination_score,
)
from evaluation.statistics import (
    compare_approaches,
    compute_approach_statistics,
    compute_bootstrap_ci,
    compute_latency_percentiles,
)

__all__ = [
    "BenchmarkRunner",
    "MetricResult",
    "QuestionResult",
    "RunResults",
    "compare_approaches",
    "compute_approach_statistics",
    "compute_bootstrap_ci",
    "compute_cost_per_correct_answer",
    "compute_factual_accuracy",
    "compute_hallucination_rate",
    "compute_hallucination_score",
    "compute_latency_percentiles",
    "judge_batch",
    "judge_single",
]
