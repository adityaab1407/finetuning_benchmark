"""Dataset formatting for SFT and DPO training.

Fine-tuning a language model requires training data in specific formats
that the training loop understands.  This module converts filtered
synthetic pairs into two formats:

**SFT (Supervised Fine-Tuning)**
    The model learns to produce the correct response given an
    instruction.  Each example is a single instruction-response pair
    formatted with Mistral's special tokens::

        <s>[INST] {instruction} [/INST] {response}</s>

    The tokenizer uses ``[INST]`` / ``[/INST]`` delimiters to
    distinguish the user's question from the model's expected output.

**DPO (Direct Preference Optimization)**
    The model learns to *prefer* good responses over bad ones.  Each
    example has three fields: a prompt, a "chosen" response (the
    correct teacher answer), and a "rejected" response (a mismatched
    answer from a different question).  Shuffling responses across
    questions creates realistic rejections — a real answer to a
    different question looks like a plausible hallucination.

Both formats are saved as JSON lists for easy loading by the training
scripts and Hugging Face ``datasets`` library.
"""

import logging
import random

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


class DatasetFormatter:
    """Converts filtered synthetic pairs into SFT and DPO training formats."""

    # ── SFT formatting ───────────────────────────────────────────────

    @staticmethod
    def format_for_sft(pairs: list[dict]) -> list[dict]:
        """Convert pairs to Mistral instruction-tuning format.

        Each example includes the full ``text`` field with Mistral
        special tokens that the tokenizer will use during training.

        Args:
            pairs: Filtered pairs with ``instruction``, ``response``,
                and metadata fields.

        Returns:
            List of SFT-formatted dicts.
        """
        formatted: list[dict] = []

        for i, pair in enumerate(pairs):
            instruction = pair["instruction"]
            response = pair["response"]

            # Detect category from instruction keywords
            inst_lower = instruction.lower()
            if "classify" in inst_lower or "sentiment" in inst_lower:
                category = "sentiment"
            elif "json" in inst_lower or "extract" in inst_lower:
                category = "structured"
            elif any(
                kw in inst_lower
                for kw in ("why", "indicate", "trend", "compare")
            ):
                category = "reasoning"
            else:
                category = "factual"

            example = {
                "id": f"sft_{i + 1:04d}",
                "instruction": instruction,
                "input": "",
                "output": response,
                "source_ticker": pair.get("source_ticker", ""),
                "prompt_version": pair.get("prompt_version", ""),
                "category": category,
                "text": f"<s>[INST] {instruction} [/INST] {response}</s>",
            }
            formatted.append(example)

        logger.info("Formatted %d SFT examples", len(formatted))
        return formatted

    # ── DPO formatting ───────────────────────────────────────────────

    @staticmethod
    def format_for_dpo(
        pairs: list[dict],
        rejection_strategy: str = "shuffled",
    ) -> list[dict]:
        """Convert pairs to DPO preference format.

        The "rejected" response is drawn from a different question's
        answer (shuffled strategy), creating a realistic but incorrect
        response that the model should learn to avoid.

        Args:
            pairs: Filtered pairs with ``instruction``, ``response``,
                and ``quality_score``.
            rejection_strategy: Only ``"shuffled"`` is supported.

        Returns:
            List of DPO-formatted dicts.
        """
        # Only include pairs above quality threshold as "chosen"
        quality_pairs = [
            p for p in pairs if p.get("quality_score", 0.7) >= 0.65
        ]

        if len(quality_pairs) < 2:
            logger.warning("Not enough quality pairs for DPO (%d)", len(quality_pairs))
            return []

        # Build shuffled rejected responses
        random.seed(42)
        responses = [p["response"] for p in quality_pairs]
        rejected_responses = responses.copy()
        random.shuffle(rejected_responses)

        # Ensure no pair gets its own response as rejected
        for i in range(len(quality_pairs)):
            if rejected_responses[i] == quality_pairs[i]["response"]:
                # Swap with the next pair
                j = (i + 1) % len(quality_pairs)
                rejected_responses[i], rejected_responses[j] = (
                    rejected_responses[j],
                    rejected_responses[i],
                )

        formatted: list[dict] = []
        for i, pair in enumerate(quality_pairs):
            example = {
                "id": f"dpo_{i + 1:04d}",
                "prompt": pair["instruction"],
                "chosen": pair["response"],
                "rejected": rejected_responses[i],
                "chosen_score": pair.get("quality_score", 0.7),
                "rejected_score": max(
                    0.1, pair.get("quality_score", 0.7) - 0.3
                ),
            }
            formatted.append(example)

        logger.info("Formatted %d DPO pairs", len(formatted))
        return formatted
