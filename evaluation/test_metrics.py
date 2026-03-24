"""Smoke tests for the evaluation metrics and statistics layer.

Runs 7 tests covering exact match, fuzzy match, classification,
schema validation, bootstrap CI, latency percentiles, and one real
LLM-judge hallucination call.

Usage::

    python3 -m evaluation.test_metrics
"""

import logging
import math
import sys

from evaluation.metrics import (
    compute_factual_accuracy,
    compute_hallucination_score,
)
from evaluation.statistics import (
    compute_bootstrap_ci,
    compute_latency_percentiles,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

passed = 0
total = 7


def _ok(test_num: int, name: str) -> None:
    global passed
    passed += 1
    logger.info("TEST %d  PASS  %s", test_num, name)


def _fail(test_num: int, name: str, reason: str) -> None:
    logger.error("TEST %d  FAIL  %s — %s", test_num, name, reason)


def main() -> None:
    global passed

    # ── TEST 1: exact_match ───────────────────────────────────────────
    t1_name = "exact_match accuracy"
    try:
        r1a = compute_factual_accuracy("78,450", "78,450", "exact_match", "factual_extraction")
        r1b = compute_factual_accuracy("78450", "78,450", "exact_match", "factual_extraction")
        r1c = compute_factual_accuracy("78,000", "78,450", "exact_match", "factual_extraction")

        if r1a.score == 1.0 and r1b.score == 1.0 and r1c.score == 0.0:
            _ok(1, t1_name)
        else:
            _fail(1, t1_name,
                  f"scores: {r1a.score}, {r1b.score}, {r1c.score} "
                  f"(expected 1.0, 1.0, 0.0)")
    except Exception as exc:
        _fail(1, t1_name, str(exc))

    # ── TEST 2: fuzzy_match ───────────────────────────────────────────
    t2_name = "fuzzy_match accuracy"
    try:
        r2a = compute_factual_accuracy(
            "$57 billion", "$57.0 billion", "fuzzy_match", "factual_extraction")
        r2b = compute_factual_accuracy(
            "Revenue was fifty-seven billion dollars",
            "$57.0 billion", "fuzzy_match", "factual_extraction")

        ok_a = r2a.score >= 0.75
        ok_b = r2b.details.get("numeric_match") is False
        if ok_a and ok_b:
            _ok(2, t2_name)
        else:
            _fail(2, t2_name,
                  f"r2a.score={r2a.score} (>= 0.75? {ok_a}), "
                  f"r2b numeric_match={r2b.details.get('numeric_match')} "
                  f"(expected False — 'fifty-seven' has no digit '57')")
    except Exception as exc:
        _fail(2, t2_name, str(exc))

    # ── TEST 3: classification ────────────────────────────────────────
    t3_name = "classification accuracy"
    try:
        r3a = compute_factual_accuracy("positive", "positive", "classification", "sentiment")
        r3b = compute_factual_accuracy("Positive", "positive", "classification", "sentiment")
        r3c = compute_factual_accuracy("very positive", "positive", "classification", "sentiment")

        if r3a.score == 1.0 and r3b.score == 1.0 and r3c.score == 0.0:
            _ok(3, t3_name)
        else:
            _fail(3, t3_name,
                  f"scores: {r3a.score}, {r3b.score}, {r3c.score} "
                  f"(expected 1.0, 1.0, 0.0)")
    except Exception as exc:
        _fail(3, t3_name, str(exc))

    # ── TEST 4: schema_validation ─────────────────────────────────────
    t4_name = "schema_validation accuracy"
    try:
        r4a = compute_factual_accuracy(
            '{"revenue": "$57B", "net_income": "$18B"}',
            '{"revenue": "$57.0 billion", "net_income": "$18.1 billion"}',
            "schema_validation", "structured_output")
        r4b = compute_factual_accuracy(
            "not json at all",
            '{"revenue": "$57B"}',
            "schema_validation", "structured_output")
        r4c = compute_factual_accuracy(
            '{"revenue": "$57B"}',
            '{"revenue": "$57B", "net_income": "$18B"}',
            "schema_validation", "structured_output")

        ok_a = r4a.score == 1.0
        ok_b = r4b.score == 0.0
        ok_c = r4c.score == 0.5
        if ok_a and ok_b and ok_c:
            _ok(4, t4_name)
        else:
            _fail(4, t4_name,
                  f"scores: {r4a.score}, {r4b.score}, {r4c.score} "
                  f"(expected 1.0, 0.0, 0.5)")
    except Exception as exc:
        _fail(4, t4_name, str(exc))

    # ── TEST 5: bootstrap CI ──────────────────────────────────────────
    t5_name = "bootstrap confidence interval"
    try:
        scores = [0.8, 0.75, 0.82, 0.78, 0.80]
        ci = compute_bootstrap_ci(scores)
        mean = ci["mean"]
        ok_mean = math.isclose(mean, 0.79, abs_tol=0.01)
        ok_order = ci["ci_lower"] < mean < ci["ci_upper"]

        if ok_mean and ok_order:
            _ok(5, t5_name)
        else:
            _fail(5, t5_name,
                  f"mean={mean:.4f} (≈0.79?), "
                  f"ci=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    except Exception as exc:
        _fail(5, t5_name, str(exc))

    # ── TEST 6: latency percentiles ───────────────────────────────────
    t6_name = "latency percentiles"
    try:
        lats = [100.0, 200.0, 300.0, 400.0, 500.0,
                600.0, 700.0, 800.0, 900.0, 1000.0]
        lp = compute_latency_percentiles(lats)
        ok_p50 = math.isclose(lp["p50_ms"], 550.0, abs_tol=1.0)
        ok_p95 = math.isclose(lp["p95_ms"], 955.0, abs_tol=10.0)

        if ok_p50 and ok_p95:
            _ok(6, t6_name)
        else:
            _fail(6, t6_name,
                  f"p50={lp['p50_ms']:.1f} (≈550?), "
                  f"p95={lp['p95_ms']:.1f} (≈950?)")
    except Exception as exc:
        _fail(6, t6_name, str(exc))

    # ── TEST 7: real hallucination judge call ─────────────────────────
    t7_name = "hallucination judge (real API call)"
    try:
        result = compute_hallucination_score(
            question="What does EPS stand for?",
            predicted_answer="EPS stands for Earnings Per Share",
            source_text=(
                "EPS, or Earnings Per Share, measures the portion "
                "of profit allocated to each share."
            ),
        )
        if result.details.get("verdict") == "NO_HALLUCINATION":
            _ok(7, t7_name)
        else:
            _fail(7, t7_name,
                  f"verdict='{result.details.get('verdict')}' "
                  f"(expected NO_HALLUCINATION)")
    except Exception as exc:
        _fail(7, t7_name, str(exc))

    # ── summary ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 50)
    logger.info("%d/%d tests passed", passed, total)
    if passed == total:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error("%d test(s) FAILED", total - passed)
    logger.info("=" * 50)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
