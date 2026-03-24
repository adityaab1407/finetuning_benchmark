"""Zero-shot prompting approach for Financial Fine-Tuning Laboratory.

Zero-shot is the simplest baseline: the model receives only a system
instruction and the raw question — no examples, no reasoning scaffold,
no retrieved context.  It measures what the model already knows about
finance from pre-training alone.

Performance on zero-shot tells us the floor: any approach that cannot
beat zero-shot is adding complexity without value.
"""

from approaches.base import BaseApproach
from config import settings

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer financial "
    "questions accurately and concisely based on your knowledge. "
    "For factual questions, provide the specific number or figure. "
    "For sentiment questions, respond with exactly one word: "
    "positive, negative, or neutral. For questions asking for "
    "JSON output, return only valid JSON with no additional text. "
    "For reasoning questions, provide your conclusion followed "
    "by a brief explanation."
)


class ZeroShotApproach(BaseApproach):
    """Raw question in, answer out — no examples, no scaffolding."""

    def __init__(self) -> None:
        super().__init__(
            approach_name="zero_shot",
            model=settings.MODEL_ZERO_SHOT,
        )

    def _build_prompt(self, question: str) -> list[dict]:
        """Return system + user messages with no additional framing.

        Args:
            question: The evaluation question text.

        Returns:
            Two-element message list for the Groq chat API.
        """
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
