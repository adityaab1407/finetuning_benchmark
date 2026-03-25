"""Quality filtering for synthetic training data.

Not all teacher-generated pairs are good enough for training.  A naive
prompt (v1) may produce vague questions, hallucinated numbers, or
answers that don't match the source text.  Even the production prompt
(v3) can occasionally produce low-quality output.

This module scores generated pairs on four dimensions using a smaller,
faster LLM judge (llama-3.1-8b via Groq):

* **Relevance** — is the instruction about finance?
* **Accuracy** — is the response supported by the source text?
* **Specificity** — is the instruction precise, not generic?
* **Format** — does the response match the expected output type?

Pairs scoring below the quality threshold (default 0.65) are filtered
out before training.  The per-version quality scores populate the
Training Journey tab in the Streamlit dashboard, showing the impact of
prompt engineering across v1 → v2 → v3.

For efficiency, only a sample of pairs are scored by the LLM judge.
The remainder are filtered with a fast heuristic (minimum instruction
and response length).
"""

import logging
import random
import re
import time

from groq import Groq

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_SCORE_RE = re.compile(r"(RELEVANCE|ACCURACY|SPECIFICITY|FORMAT):\s*(\d)")


class QualityFilter:
    """Scores and filters synthetic instruction-response pairs.

    Args:
        judge_model: Groq model ID for the quality judge.  Defaults to
            ``settings.MODEL_SCORER``.
    """

    def __init__(self, judge_model: str | None = None) -> None:
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.judge_model = judge_model or settings.MODEL_SCORER
        self._log = logger

    # ── single-pair scoring ──────────────────────────────────────────

    def score_pair(
        self,
        instruction: str,
        response: str,
        source_text: str,
    ) -> dict:
        """Score a single pair on four quality dimensions.

        Returns:
            Dict with per-dimension scores, total, quality_score, and
            pass/fail flag.
        """
        system_msg = (
            "You are evaluating training data quality for a financial "
            "language model. Score strictly."
        )
        user_msg = (
            "Rate this instruction-response pair on 4 criteria.\n\n"
            f"Source text: {source_text[:1000]}\n"
            f"Instruction: {instruction}\n"
            f"Response: {response}\n\n"
            "Score each criterion 1-5:\n"
            "RELEVANCE: Is the instruction relevant to finance? "
            "(1=not financial, 5=clearly financial)\n"
            "ACCURACY: Is the response factually supported by the "
            "source text? (1=contradicts source, 5=fully supported)\n"
            "SPECIFICITY: Is the instruction specific not generic? "
            "(1=too vague, 5=very specific)\n"
            "FORMAT: Is the response in the correct format for its "
            "type (number/word/JSON/explanation)? "
            "(1=wrong format, 5=perfect format)\n\n"
            "Respond ONLY in this format:\n"
            "RELEVANCE: X\n"
            "ACCURACY: X\n"
            "SPECIFICITY: X\n"
            "FORMAT: X"
        )

        raw = self._call_judge(system_msg, user_msg)
        return self._parse_scores(raw)

    def _call_judge(self, system_msg: str, user_msg: str) -> str:
        """Call the judge model with exponential backoff on 429."""
        delays = [2, 4, 8]
        last_error = ""

        for attempt in range(len(delays) + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=128,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = str(exc)
                is_rate_limit = (
                    "429" in last_error or "rate" in last_error.lower()
                )
                if is_rate_limit and attempt < len(delays):
                    wait = delays[attempt]
                    self._log.warning(
                        "Scorer rate limited (attempt %d), waiting %ds...",
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    break

        self._log.error("Scorer call failed: %s", last_error)
        return ""

    @staticmethod
    def _parse_scores(raw: str) -> dict:
        """Extract four dimension scores from the judge's response."""
        scores: dict[str, int] = {
            "relevance": 3,
            "accuracy": 3,
            "specificity": 3,
            "format_score": 3,
        }

        key_map = {
            "RELEVANCE": "relevance",
            "ACCURACY": "accuracy",
            "SPECIFICITY": "specificity",
            "FORMAT": "format_score",
        }

        for match in _SCORE_RE.finditer(raw):
            dim = match.group(1)
            val = int(match.group(2))
            val = max(1, min(5, val))
            if dim in key_map:
                scores[key_map[dim]] = val

        total = sum(scores.values())
        quality = total / 20.0

        return {
            **scores,
            "total_score": total,
            "quality_score": round(quality, 4),
            "passed": quality >= settings.QUALITY_THRESHOLD,
        }

    # ── batch filtering ──────────────────────────────────────────────

    def filter_dataset(
        self,
        pairs: list[dict],
        source_material: dict,
        sample_size: int = 50,
    ) -> list[dict]:
        """Score a sample and fast-filter the full dataset.

        Args:
            pairs: Parsed pairs with ``instruction``, ``response``,
                ``source_chunk_id``, and optionally ``source_text``.
            source_material: Dict from
                :meth:`SyntheticDataGenerator.load_source_material`.
            sample_size: Number of pairs to score with the LLM judge.

        Returns:
            Filtered list of pairs with quality scores added.
        """
        if not pairs:
            return []

        # Build source lookup for scoring
        source_lookup: dict[str, str] = {}
        for c in source_material.get("edgar_chunks", []):
            source_lookup[c["chunk_id"]] = c["text"]
        for s in source_material.get("phrasebank", []):
            source_lookup[s["id"]] = s["sentence"]

        # Fast filter: minimum length requirements
        fast_passed: list[dict] = []
        for p in pairs:
            inst = p.get("instruction", "")
            resp = p.get("response", "")
            if len(inst) > 20 and len(resp) > 10:
                fast_passed.append(p)

        # Score a random sample with the LLM judge
        random.seed(42)
        sample = random.sample(
            fast_passed, min(sample_size, len(fast_passed))
        )
        sample_ids = {id(p) for p in sample}

        for i, pair in enumerate(sample):
            source_text = source_lookup.get(
                pair.get("source_chunk_id", ""), ""
            )
            scores = self.score_pair(
                instruction=pair["instruction"],
                response=pair["response"],
                source_text=source_text,
            )
            pair.update(scores)

            if (i + 1) % 10 == 0:
                self._log.info("Scored %d/%d sample pairs", i + 1, len(sample))

            if i < len(sample) - 1:
                time.sleep(settings.GROQ_CALL_DELAY_SECONDS)

        # For non-sampled pairs, assign default passing score
        for p in fast_passed:
            if id(p) not in sample_ids:
                p.setdefault("quality_score", 0.70)
                p.setdefault("passed", True)

        # Final filter
        kept = [p for p in fast_passed if p.get("passed", False)]

        total = len(pairs)
        self._log.info(
            "Quality filter: %d/%d pairs kept (%.1f%% pass rate)",
            len(kept),
            total,
            len(kept) / total * 100 if total else 0,
        )

        return kept

    # ── dataset statistics ───────────────────────────────────────────

    @staticmethod
    def compute_dataset_stats(pairs: list[dict]) -> dict:
        """Compute statistics for the Training Journey tab.

        Returns:
            Dict with counts, averages, category distribution,
            quality histogram, and pass rate.
        """
        if not pairs:
            return {
                "total_pairs": 0,
                "avg_instruction_length": 0.0,
                "avg_response_length": 0.0,
                "category_distribution": {},
                "quality_score_histogram": {},
                "pass_rate_pct": 0.0,
            }

        total = len(pairs)
        inst_lengths = [len(p.get("instruction", "")) for p in pairs]
        resp_lengths = [len(p.get("response", "")) for p in pairs]

        # Category detection via keywords
        categories: dict[str, int] = {
            "factual": 0,
            "sentiment": 0,
            "structured": 0,
            "reasoning": 0,
            "unknown": 0,
        }
        for p in pairs:
            inst_lower = p.get("instruction", "").lower()
            if "classify" in inst_lower or "sentiment" in inst_lower:
                categories["sentiment"] += 1
            elif "json" in inst_lower or "extract" in inst_lower:
                categories["structured"] += 1
            elif any(
                kw in inst_lower
                for kw in ("why", "indicate", "trend", "compare")
            ):
                categories["reasoning"] += 1
            else:
                categories["factual"] += 1

        # Quality histogram
        histogram: dict[str, int] = {
            "0.0-0.5": 0,
            "0.5-0.65": 0,
            "0.65-0.8": 0,
            "0.8-1.0": 0,
        }
        for p in pairs:
            qs = p.get("quality_score", 0.7)
            if qs < 0.5:
                histogram["0.0-0.5"] += 1
            elif qs < 0.65:
                histogram["0.5-0.65"] += 1
            elif qs < 0.8:
                histogram["0.65-0.8"] += 1
            else:
                histogram["0.8-1.0"] += 1

        passed = sum(1 for p in pairs if p.get("passed", False))

        return {
            "total_pairs": total,
            "avg_instruction_length": round(sum(inst_lengths) / total, 1),
            "avg_response_length": round(sum(resp_lengths) / total, 1),
            "category_distribution": categories,
            "quality_score_histogram": histogram,
            "pass_rate_pct": round(passed / total * 100, 1),
        }
