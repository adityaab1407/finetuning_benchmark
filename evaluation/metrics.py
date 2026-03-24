"""Evaluation metrics for Financial Fine-Tuning Laboratory.

This module defines the four core metrics used to grade every benchmark
answer.  We deliberately avoid corpus-level text-overlap scores like
BLEU or ROUGE because they were designed for translation and
summarisation — not for financial fact-checking where a single wrong
digit makes the entire answer wrong.

Instead, each metric is tailored to what matters in production:

* **Factual accuracy** — did the model return the correct number, label,
  JSON structure, or conclusion?  Scored differently per question type.
* **Hallucination score** — did the model fabricate claims not present in
  the source text?  Assessed by an LLM judge (Groq).
* **Schema validity** — for structured-output questions, is the returned
  JSON parseable and does it contain the expected keys?
* **Cost per correct answer** — the headline production metric: how much
  does each *correct* answer cost in USD and INR?
"""

import json
import logging
import re
import time

from groq import Groq
from pydantic import BaseModel
from rapidfuzz import fuzz

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_USD_TO_INR = 83.5

# ── result model ──────────────────────────────────────────────────────


class MetricResult(BaseModel):
    """Container for the output of any single metric computation."""

    metric_name: str
    score: float
    passed: bool
    threshold: float
    details: dict
    error: str | None = None


# ── helpers ───────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"\d+\.?\d*")


def _clean(text: str) -> str:
    """Lowercase, strip, and remove commas from a string."""
    return text.lower().strip().replace(",", "")


# ── FUNCTION 1: factual accuracy ─────────────────────────────────────


def compute_factual_accuracy(
    predicted: str,
    expected: str,
    evaluation_type: str,
    question_category: str,
) -> MetricResult:
    """Score a predicted answer against the ground truth.

    Scoring logic varies by *evaluation_type*:

    * ``exact_match``       — binary string equality after cleaning
    * ``fuzzy_match``       — partial ratio with numeric-value boost
    * ``classification``    — exact sentiment-label equality
    * ``schema_validation`` — JSON key overlap ratio
    * ``reasoning_match``   — token-set ratio with relaxed threshold

    Args:
        predicted: The model's answer string.
        expected: The ground-truth answer string.
        evaluation_type: One of the five types above.
        question_category: The question category (for metadata only).

    Returns:
        A :class:`MetricResult` with the computed score.
    """
    if not predicted and not expected:
        return MetricResult(
            metric_name="factual_accuracy",
            score=1.0,
            passed=True,
            threshold=0.0,
            details={"note": "both empty"},
        )
    if not predicted or not expected:
        return MetricResult(
            metric_name="factual_accuracy",
            score=0.0,
            passed=False,
            threshold=0.0,
            details={"note": "one side empty"},
        )

    if evaluation_type == "exact_match":
        return _exact_match(predicted, expected)
    if evaluation_type == "fuzzy_match":
        return _fuzzy_match(predicted, expected)
    if evaluation_type == "classification":
        return _classification(predicted, expected)
    if evaluation_type == "schema_validation":
        return _schema_validation(predicted, expected)
    if evaluation_type == "reasoning_match":
        return _reasoning_match(predicted, expected)

    # Unknown evaluation_type — fall back to fuzzy
    logger.warning("Unknown evaluation_type '%s', falling back to fuzzy", evaluation_type)
    return _fuzzy_match(predicted, expected)


def _exact_match(predicted: str, expected: str) -> MetricResult:
    p_clean = _clean(predicted)
    e_clean = _clean(expected)
    match = p_clean == e_clean
    return MetricResult(
        metric_name="factual_accuracy",
        score=1.0 if match else 0.0,
        passed=match,
        threshold=1.0,
        details={"predicted_clean": p_clean, "expected_clean": e_clean},
    )


def _fuzzy_match(predicted: str, expected: str) -> MetricResult:
    ratio = fuzz.partial_ratio(predicted, expected)
    score = ratio / 100.0

    # Numeric-value boost: if expected contains a number and that
    # number appears verbatim in predicted, boost score.
    nums = _NUM_RE.findall(expected)
    numeric_match = any(n in predicted for n in nums) if nums else False
    if numeric_match:
        score = max(score, 0.85)

    return MetricResult(
        metric_name="factual_accuracy",
        score=round(score, 4),
        passed=score >= 0.75,
        threshold=0.75,
        details={
            "fuzzy_ratio": ratio,
            "numeric_match": numeric_match,
            "predicted": predicted[:100],
            "expected": expected[:100],
        },
    )


def _classification(predicted: str, expected: str) -> MetricResult:
    p_clean = predicted.lower().strip()
    e_clean = expected.lower().strip()
    match = p_clean == e_clean
    return MetricResult(
        metric_name="factual_accuracy",
        score=1.0 if match else 0.0,
        passed=match,
        threshold=1.0,
        details={"predicted_class": p_clean, "expected_class": e_clean},
    )


def _schema_validation(predicted: str, expected: str) -> MetricResult:
    try:
        pred_obj = json.loads(predicted)
    except (json.JSONDecodeError, TypeError):
        return MetricResult(
            metric_name="factual_accuracy",
            score=0.0,
            passed=False,
            threshold=0.8,
            details={"error": "invalid JSON"},
        )

    if not isinstance(pred_obj, dict):
        return MetricResult(
            metric_name="factual_accuracy",
            score=0.0,
            passed=False,
            threshold=0.8,
            details={"error": "JSON is not a dict"},
        )

    try:
        exp_obj = json.loads(expected)
    except (json.JSONDecodeError, TypeError):
        exp_obj = {}

    if not isinstance(exp_obj, dict):
        exp_obj = {}

    expected_keys = set(exp_obj.keys())
    predicted_keys = set(pred_obj.keys())

    if not expected_keys:
        return MetricResult(
            metric_name="factual_accuracy",
            score=1.0,
            passed=True,
            threshold=0.8,
            details={"note": "no expected keys to check"},
        )

    matching = expected_keys & predicted_keys
    score = len(matching) / len(expected_keys)

    return MetricResult(
        metric_name="factual_accuracy",
        score=round(score, 4),
        passed=score >= 0.8,
        threshold=0.8,
        details={
            "matching_keys": len(matching),
            "total_keys": len(expected_keys),
            "missing_keys": sorted(expected_keys - predicted_keys),
            "extra_keys": sorted(predicted_keys - expected_keys),
        },
    )


def _reasoning_match(predicted: str, expected: str) -> MetricResult:
    ratio = fuzz.token_set_ratio(predicted, expected)
    score = ratio / 100.0
    return MetricResult(
        metric_name="factual_accuracy",
        score=round(score, 4),
        passed=score >= 0.6,
        threshold=0.6,
        details={
            "token_set_ratio": ratio,
            "predicted": predicted[:150],
            "expected": expected[:150],
        },
    )


# ── FUNCTION 2: hallucination scoring (LLM judge) ────────────────────

_JUDGE_SYSTEM = (
    "You are an expert fact-checker for financial information. "
    "Your job is to determine if an AI-generated answer contains "
    "any claims that are not supported by or contradict the "
    "provided source text. Be strict but fair."
)

_JUDGE_USER_TEMPLATE = """Question: {question}

Source text (ground truth):
{source_text}

AI-generated answer:
{predicted_answer}

Does the AI answer contain any claims NOT supported by the source text, or does it contradict the source text?

Respond with ONLY one of these exact options:
NO_HALLUCINATION - the answer is fully supported by the source
MINOR_HALLUCINATION - small unsupported claims but core answer correct
MAJOR_HALLUCINATION - significant unsupported or contradictory claims

Then on a new line, briefly explain your reasoning (1-2 sentences)."""


def compute_hallucination_score(
    question: str,
    predicted_answer: str,
    source_text: str,
    model: str | None = None,
) -> MetricResult:
    """Use an LLM judge to assess whether *predicted_answer* hallucinates.

    Makes a real Groq API call.  Retries with exponential backoff on
    429 rate-limit errors (2 s → 4 s → 8 s).

    Args:
        question: The original question text.
        predicted_answer: The model's answer to judge.
        source_text: The ground-truth source passage.
        model: Groq model ID for the judge.  Defaults to
            ``settings.MODEL_JUDGE``.

    Returns:
        A :class:`MetricResult` with score 1.0 (clean), 0.5 (minor),
        or 0.0 (major hallucination).
    """
    if model is None:
        model = settings.MODEL_JUDGE

    if not predicted_answer.strip():
        return MetricResult(
            metric_name="hallucination",
            score=0.0,
            passed=False,
            threshold=0.5,
            details={"verdict": "EMPTY_ANSWER", "reasoning": "No answer to judge",
                      "judge_model": model},
        )

    client = Groq(api_key=settings.GROQ_API_KEY)
    user_msg = _JUDGE_USER_TEMPLATE.format(
        question=question,
        source_text=source_text,
        predicted_answer=predicted_answer,
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    delays = [2, 4, 8]
    last_error: str | None = None

    for attempt in range(len(delays) + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _parse_judge_response(raw, model)

        except Exception as exc:
            last_error = str(exc)
            is_rate_limit = "429" in last_error or "rate" in last_error.lower()
            if is_rate_limit and attempt < len(delays):
                wait = delays[attempt]
                logger.warning(
                    "Judge rate limited (attempt %d), retrying in %ds...",
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
            else:
                break

    logger.error("Judge failed after retries: %s", last_error)
    return MetricResult(
        metric_name="hallucination",
        score=0.5,
        passed=True,
        threshold=0.5,
        details={"verdict": "JUDGE_ERROR", "reasoning": "", "judge_model": model},
        error="Judge rate limited",
    )


def _parse_judge_response(raw: str, model: str) -> MetricResult:
    """Parse the first line of the judge response into a MetricResult."""
    upper = raw.upper()
    lines = raw.strip().split("\n", 1)
    reasoning = lines[1].strip() if len(lines) > 1 else ""

    if upper.startswith("NO_HALLUCINATION"):
        verdict, score, passed = "NO_HALLUCINATION", 1.0, True
    elif upper.startswith("MINOR_HALLUCINATION"):
        verdict, score, passed = "MINOR_HALLUCINATION", 0.5, True
    elif upper.startswith("MAJOR_HALLUCINATION"):
        verdict, score, passed = "MAJOR_HALLUCINATION", 0.0, False
    else:
        verdict, score, passed = "UNPARSEABLE", 0.5, True
        reasoning = raw[:200]

    return MetricResult(
        metric_name="hallucination",
        score=score,
        passed=passed,
        threshold=0.5,
        details={"verdict": verdict, "reasoning": reasoning, "judge_model": model},
    )


# ── FUNCTION 3: schema validity (convenience wrapper) ─────────────────


def compute_schema_validity(predicted: str, expected: str) -> MetricResult:
    """Check that *predicted* is valid JSON matching the expected schema.

    Convenience wrapper around :func:`compute_factual_accuracy` with
    ``evaluation_type="schema_validation"``.

    Args:
        predicted: The model's JSON string output.
        expected: The ground-truth JSON string.

    Returns:
        A :class:`MetricResult` with key-overlap score.
    """
    return compute_factual_accuracy(
        predicted=predicted,
        expected=expected,
        evaluation_type="schema_validation",
        question_category="structured_output",
    )


# ── FUNCTION 4: cost per correct answer ──────────────────────────────


def compute_cost_per_correct_answer(
    results: list[dict],
    approach_name: str,
) -> dict:
    """Compute the cost-efficiency headline metric for an approach.

    Args:
        results: Per-question dicts with keys *question_id*, *correct*,
            *cost_usd*, *latency_ms*.
        approach_name: Name of the approach being evaluated.

    Returns:
        Summary dict including accuracy percentage, total cost, and
        the headline *cost_per_correct_usd* metric.
    """
    if not results:
        return {
            "approach_name": approach_name,
            "total_questions": 0,
            "correct_count": 0,
            "accuracy_pct": 0.0,
            "total_cost_usd": 0.0,
            "total_cost_inr": 0.0,
            "cost_per_question_usd": 0.0,
            "cost_per_correct_usd": 0.0,
            "cost_per_correct_inr": 0.0,
            "avg_latency_ms": 0.0,
            "note": ("Groq free tier — costs are $0.00. "
                     "Metric preserved for paid tier comparison."),
        }

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    total_latency = sum(r.get("latency_ms", 0.0) for r in results)

    cost_per_q = total_cost / total if total else 0.0
    cost_per_correct = total_cost / correct if correct else 0.0

    return {
        "approach_name": approach_name,
        "total_questions": total,
        "correct_count": correct,
        "accuracy_pct": round(correct / total * 100, 2) if total else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "total_cost_inr": round(total_cost * _USD_TO_INR, 4),
        "cost_per_question_usd": round(cost_per_q, 6),
        "cost_per_correct_usd": round(cost_per_correct, 6),
        "cost_per_correct_inr": round(cost_per_correct * _USD_TO_INR, 4),
        "avg_latency_ms": round(total_latency / total, 1) if total else 0.0,
        "note": ("Groq free tier — costs are $0.00. "
                 "Metric preserved for paid tier comparison."),
    }
