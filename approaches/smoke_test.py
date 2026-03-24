"""Smoke test for the four available benchmark approaches.

Makes exactly 4 real Groq API calls to verify that zero-shot, few-shot,
chain-of-thought, and RAG approaches are wired up correctly and returning
non-empty answers.

Usage::

    python3 -m approaches.smoke_test
"""

import logging
import sys
import time

from approaches.cot import ChainOfThoughtApproach
from approaches.few_shot import FewShotApproach
from approaches.rag import RAGApproach
from approaches.zero_shot import ZeroShotApproach

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_TEST_QUESTION = "What does EPS stand for and what does it measure?"
_INTER_CALL_DELAY = 3.0


def main() -> None:
    """Run one question through each approach and print results."""
    # Instantiate all approaches
    rag = RAGApproach()
    rag.setup()

    approaches = [
        ZeroShotApproach(),
        FewShotApproach(),
        ChainOfThoughtApproach(),
        rag,
    ]

    results = []
    for i, approach in enumerate(approaches):
        if i > 0:
            time.sleep(_INTER_CALL_DELAY)
        result = approach.run(_TEST_QUESTION)
        results.append(result)

    # Print summary table
    logger.info("")
    logger.info(
        "%-18s | %-80s | %10s | %6s | %8s",
        "Approach", "Answer (first 80 chars)", "Latency ms", "Tokens", "Cost USD",
    )
    logger.info("-" * 135)

    all_passed = True
    for r in results:
        answer_preview = r.answer[:80] if r.answer else "(empty)"
        logger.info(
            "%-18s | %-80s | %10.1f | %6d | %8.4f",
            r.approach_name,
            answer_preview,
            r.latency_ms,
            r.tokens_total,
            r.cost_usd,
        )
        if not r.answer:
            all_passed = False
            logger.error("  ^ EMPTY answer from %s: %s", r.approach_name, r.error)

    logger.info("")
    if all_passed:
        logger.info("SMOKE TEST PASSED — all 4 approaches returned non-empty answers")
    else:
        logger.error("SMOKE TEST FAILED — one or more approaches returned empty answers")
        sys.exit(1)


if __name__ == "__main__":
    main()
