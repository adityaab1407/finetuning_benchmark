"""Few-shot prompting approach for Financial Fine-Tuning Laboratory.

Few-shot prompting prepends 3-5 worked examples before the real
question.  The examples teach the model the expected answer format for
each question category — exact figures for factual, single-word labels
for sentiment, valid JSON for structured output, and
conclusion-then-explanation for reasoning.

By comparing few-shot against zero-shot we isolate the value of
in-context examples: if few-shot wins, the model can learn answer
format from examples; if it doesn't, the model already knows the
format and the examples are redundant.
"""

from approaches.base import BaseApproach
from config import settings

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer financial "
    "questions accurately following the exact format shown in "
    "the examples. For factual questions, provide the specific "
    "figure. For sentiment, respond with exactly one word: "
    "positive, negative, or neutral. For JSON extraction, "
    "return only valid JSON. For reasoning, give conclusion "
    "then brief explanation."
)

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": (
            "What was NVIDIA's total revenue for Q3 FY2026?"
        ),
        "answer": "$57.0 billion",
    },
    {
        "question": (
            "Classify the sentiment of this financial statement "
            "as positive, negative, or neutral: 'The company reported "
            "record quarterly revenue, exceeding analyst expectations "
            "by 12%.'"
        ),
        "answer": "positive",
    },
    {
        "question": (
            "Read the following paragraph and extract these fields "
            "into a JSON object: revenue, net_income, eps_diluted. "
            "Use null for any field not mentioned.\n\n"
            "Paragraph: 'The company reported revenue of $25.3 billion "
            "and net income of $8.1 billion. Diluted EPS was $2.14.'"
        ),
        "answer": (
            '{"revenue": "$25.3 billion", '
            '"net_income": "$8.1 billion", '
            '"eps_diluted": "$2.14"}'
        ),
    },
    {
        "question": (
            "A company reported operating margin of 18% last quarter "
            "and 22% this quarter. Is profitability improving or "
            "deteriorating? Explain briefly."
        ),
        "answer": (
            "Improving. Operating margin increased by 4 percentage "
            "points from 18% to 22%, indicating the company is "
            "retaining more profit per dollar of revenue."
        ),
    },
]


class FewShotApproach(BaseApproach):
    """Prompt with worked examples to teach expected answer format."""

    def __init__(self) -> None:
        super().__init__(
            approach_name="few_shot",
            model=settings.MODEL_FEW_SHOT,
        )

    def _build_prompt(self, question: str) -> list[dict]:
        """Build a prompt that shows all four examples then the real question.

        Args:
            question: The evaluation question text.

        Returns:
            Two-element message list for the Groq chat API.
        """
        example_blocks: list[str] = []
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
            example_blocks.append(
                f"Example {i}:\n"
                f"Question: {ex['question']}\n"
                f"Answer: {ex['answer']}"
            )

        user_content = (
            "Here are examples of how to answer financial questions:\n\n"
            + "\n\n".join(example_blocks)
            + "\n\n"
            "Now answer this question:\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
