"""Statistical analysis for Financial Fine-Tuning Laboratory.

A benchmark that reports only a point-estimate accuracy ("this approach
scored 78%") is incomplete.  With 100 questions and inherent
randomness in LLM output, the true accuracy could easily be anywhere
in a ±5% band.  Without confidence intervals, you cannot tell whether
one approach is genuinely better than another or whether the difference
is just noise.

This module provides:

* **Bootstrap confidence intervals** — resample the scores thousands of
  times to estimate the plausible range for the true mean.  Bootstrap
  is well-suited for our small sample (100 questions) where normality
  assumptions may not hold.
* **Latency percentiles** — P50 is what users *typically* experience;
  P95 is the worst-case that SREs care about.  Together they reveal
  whether an approach has a long tail.
* **Approach comparison** — ranks approaches by accuracy and checks
  whether their confidence intervals overlap (non-overlapping CIs
  indicate statistically meaningful differences).
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def compute_bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
) -> dict:
    """Compute a bootstrap confidence interval for the mean of *scores*.

    Args:
        scores: Sample of numeric scores (e.g. per-question accuracies).
        confidence: Confidence level for the interval (default 95%).
        n_resamples: Number of bootstrap resamples.

    Returns:
        Dict with *mean*, *std*, *ci_lower*, *ci_upper*, *ci_width*,
        *confidence_level*, *n_samples*, and *margin_of_error*.
    """
    if not scores:
        return {"error": "insufficient data", "mean": 0.0}

    if len(scores) < 2:
        return {"error": "insufficient data", "mean": float(scores[0])}

    arr = np.array(scores)
    result = stats.bootstrap(
        (arr,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        random_state=42,
    )

    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "ci_width": ci_high - ci_low,
        "confidence_level": confidence,
        "n_samples": len(scores),
        "margin_of_error": (ci_high - ci_low) / 2,
    }


def compute_latency_percentiles(latencies_ms: list[float]) -> dict:
    """Compute latency distribution statistics.

    Args:
        latencies_ms: Per-call latencies in milliseconds.

    Returns:
        Dict with P50, P75, P95, P99, mean, min, max, and std.
    """
    if not latencies_ms:
        return {
            "p50_ms": 0.0,
            "p75_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }

    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p75_ms": float(np.percentile(arr, 75)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def compute_approach_statistics(
    run_scores: list[list[float]],
    run_latencies: list[list[float]],
) -> dict:
    """Compute aggregate statistics across multiple benchmark runs.

    Args:
        run_scores: List of score lists, one inner list per run.
        run_latencies: List of latency lists, same structure.

    Returns:
        Dict with per-run accuracy, overall accuracy, bootstrap CI,
        latency percentiles, and run/question counts.
    """
    if not run_scores:
        return {
            "per_run_accuracy": [],
            "overall_accuracy": 0.0,
            "bootstrap_ci": {"error": "no data", "mean": 0.0},
            "latency": compute_latency_percentiles([]),
            "n_runs": 0,
            "n_questions_per_run": 0,
        }

    per_run_acc = [float(np.mean(run)) for run in run_scores]
    overall_acc = float(np.mean(per_run_acc))

    all_latencies = [lat for run in run_latencies for lat in run]

    return {
        "per_run_accuracy": [round(a, 4) for a in per_run_acc],
        "overall_accuracy": round(overall_acc, 4),
        "bootstrap_ci": compute_bootstrap_ci(per_run_acc),
        "latency": compute_latency_percentiles(all_latencies),
        "n_runs": len(run_scores),
        "n_questions_per_run": len(run_scores[0]) if run_scores else 0,
    }


def compare_approaches(approach_stats: dict[str, dict]) -> dict:
    """Rank approaches and check for statistically meaningful differences.

    Two approaches are considered *significantly different* when their
    95% bootstrap confidence intervals do not overlap.

    Args:
        approach_stats: Mapping of approach name to statistics dict
            (output of :func:`compute_approach_statistics`).

    Returns:
        Dict with *ranking*, *best_approach*, *accuracy_comparison*,
        and *latency_comparison*.
    """
    if not approach_stats:
        return {
            "ranking": [],
            "best_approach": "",
            "accuracy_comparison": {},
            "latency_comparison": {},
        }

    # Sort by overall accuracy descending
    ranking = sorted(
        approach_stats.keys(),
        key=lambda name: approach_stats[name].get("overall_accuracy", 0.0),
        reverse=True,
    )

    # Build per-approach CI info
    acc_comparison: dict[str, dict] = {}
    for name in ranking:
        st = approach_stats[name]
        ci = st.get("bootstrap_ci", {})
        ci_lower = ci.get("ci_lower", 0.0)
        ci_upper = ci.get("ci_upper", 0.0)

        acc_comparison[name] = {
            "accuracy": st.get("overall_accuracy", 0.0),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significantly_better_than": [],
        }

    # Determine significant differences (non-overlapping CIs)
    for name_a, info_a in acc_comparison.items():
        for name_b, info_b in acc_comparison.items():
            if name_a == name_b:
                continue
            # A is significantly better than B if A's lower bound
            # is above B's upper bound
            if info_a["ci_lower"] > info_b["ci_upper"]:
                info_a["significantly_better_than"].append(name_b)

    # Latency comparison
    lat_comparison: dict[str, dict] = {}
    for name in ranking:
        lat = approach_stats[name].get("latency", {})
        lat_comparison[name] = {
            "p50_ms": lat.get("p50_ms", 0.0),
            "p95_ms": lat.get("p95_ms", 0.0),
        }

    return {
        "ranking": ranking,
        "best_approach": ranking[0] if ranking else "",
        "accuracy_comparison": acc_comparison,
        "latency_comparison": lat_comparison,
    }
