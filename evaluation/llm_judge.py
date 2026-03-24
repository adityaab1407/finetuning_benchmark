"""LLM-as-judge evaluation for Financial Fine-Tuning Laboratory.

Traditional NLP metrics like BLEU and ROUGE measure surface-level text
overlap — they reward paraphrases that share words, not answers that
share *meaning*.  For financial Q&A, a correct answer can be phrased
many ways ("$94.9 billion", "approximately $95B", "94.9B USD") and a
BLEU-identical answer can still be factually wrong if a single digit
is off.

Instead, we use an LLM judge (Groq llama-3.1-8b) that reads the
source text, reads the model's answer, and classifies whether the
answer is supported by the source.  This catches hallucinated numbers,
fabricated context, and subtle factual errors that overlap metrics miss.

**Limitation:** the judge model can itself make mistakes.  We mitigate
this by using a smaller, faster model (lower cost per judgement) and
conservative verdicts (benefit of doubt on unparseable responses).
"""

import logging
import time

from evaluation.metrics import compute_hallucination_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def judge_single(
    question: str,
    predicted_answer: str,
    source_text: str,
    question_id: str = "",
) -> dict:
    """Judge a single predicted answer for hallucinations.

    Wraps :func:`evaluation.metrics.compute_hallucination_score` and
    returns a flat dict suitable for JSON serialization and
    downstream aggregation.

    Args:
        question: The original question text.
        predicted_answer: The model's answer to evaluate.
        source_text: The ground-truth source passage.
        question_id: Optional question identifier for logging.

    Returns:
        Dict with keys *question_id*, *verdict*, *hallucination_score*,
        *passed*, *reasoning*, and *judge_model*.
    """
    result = compute_hallucination_score(
        question=question,
        predicted_answer=predicted_answer,
        source_text=source_text,
    )

    return {
        "question_id": question_id,
        "verdict": result.details.get("verdict", "UNKNOWN"),
        "hallucination_score": result.score,
        "passed": result.passed,
        "reasoning": result.details.get("reasoning", ""),
        "judge_model": result.details.get("judge_model", ""),
    }


def judge_batch(
    items: list[dict],
    delay_seconds: float = 1.5,
) -> list[dict]:
    """Judge multiple answers sequentially with rate-limit delays.

    Args:
        items: List of dicts, each containing *question*,
            *predicted_answer*, *source_text*, and *question_id*.
        delay_seconds: Pause between Groq API calls.

    Returns:
        List of :func:`judge_single` result dicts.
    """
    results: list[dict] = []
    total = len(items)

    for i, item in enumerate(items, start=1):
        qid = item.get("question_id", f"item_{i}")
        logger.info("Judging %d/%d: %s", i, total, qid)

        result = judge_single(
            question=item["question"],
            predicted_answer=item["predicted_answer"],
            source_text=item["source_text"],
            question_id=qid,
        )
        results.append(result)

        if i < total:
            time.sleep(delay_seconds)

    return results


def compute_hallucination_rate(judge_results: list[dict]) -> dict:
    """Aggregate judge verdicts into summary statistics.

    Args:
        judge_results: Output of :func:`judge_batch`.

    Returns:
        Dict with total counts, hallucination rate percentage,
        and clean rate percentage.
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "no_hallucination_count": 0,
            "minor_hallucination_count": 0,
            "major_hallucination_count": 0,
            "hallucination_rate_pct": 0.0,
            "major_hallucination_rate_pct": 0.0,
            "clean_rate_pct": 0.0,
        }

    no_h = sum(1 for r in judge_results if r["verdict"] == "NO_HALLUCINATION")
    minor = sum(1 for r in judge_results if r["verdict"] == "MINOR_HALLUCINATION")
    major = sum(1 for r in judge_results if r["verdict"] == "MAJOR_HALLUCINATION")

    return {
        "total_judged": total,
        "no_hallucination_count": no_h,
        "minor_hallucination_count": minor,
        "major_hallucination_count": major,
        "hallucination_rate_pct": round((minor + major) / total * 100, 2),
        "major_hallucination_rate_pct": round(major / total * 100, 2),
        "clean_rate_pct": round(no_h / total * 100, 2),
    }
