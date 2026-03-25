"""Chain-of-thought prompting approach for Financial Fine-Tuning Laboratory.

Chain-of-thought (CoT) forces the model to show its reasoning before
answering.  The system prompt defines a three-step scaffold — UNDERSTAND,
ANALYZE, ANSWER — and requires the model to end with a clearly delimited
``FINAL ANSWER:`` line.

CoT is expected to help most on reasoning and structured-output questions
where breaking the problem into steps reduces errors.  For simple
sentiment classification it may add unnecessary verbosity.  Comparing
CoT against zero-shot and few-shot reveals whether explicit reasoning
is worth the extra tokens.
"""

import time
from datetime import datetime, timezone

from approaches.base import ApproachResult, BaseApproach
from config import settings

_SYSTEM_PROMPT = (
    "You are a senior financial analyst. You reason carefully before answering.\n\n"
    "For every question, follow this exact three-step process:\n\n"
    "## STEP 1 — UNDERSTAND\n"
    "State the question type: factual lookup | trend analysis | sentiment | "
    "ratio/calculation | comparison. Identify the specific metric, company, "
    "and time period being asked about.\n\n"
    "## STEP 2 — ANALYZE\n"
    "Work through the evidence:\n"
    "- For factual: locate the exact figure and its reporting period.\n"
    "- For trend/analysis: list the data points in chronological order, "
    "compute direction (improving = rising metrics that are good, or falling "
    "costs/losses; deteriorating = the opposite). State whether the change is "
    "material (>1 pp is significant for margins).\n"
    "- For sentiment: quote the key phrases that signal tone, then weigh them.\n"
    "- For calculation: show each arithmetic step explicitly.\n\n"
    "## STEP 3 — ANSWER\n"
    "Give a concise, precise answer:\n"
    "- Factual: the exact figure with units.\n"
    "- Trend: one of 'improving' or 'deteriorating', followed by one sentence "
    "of justification.\n"
    "- Sentiment: exactly one word (positive / negative / neutral).\n"
    "- Calculation: the result with units.\n\n"
    "Always close your response with exactly this line (no extra text after it):\n"
    "FINAL ANSWER: <your concise answer>"
)


class ChainOfThoughtApproach(BaseApproach):
    """Structured reasoning scaffold with explicit FINAL ANSWER extraction."""

    def __init__(self) -> None:
        super().__init__(
            approach_name="chain_of_thought",
            model=settings.MODEL_COT,
        )

    def _build_prompt(self, question: str) -> list[dict]:
        """Build a prompt that triggers step-by-step reasoning.

        Args:
            question: The evaluation question text.

        Returns:
            Two-element message list for the Groq chat API.
        """
        user_content = (
            f"Question: {question}\n\n"
            "Work through STEP 1, STEP 2, and STEP 3, then end with FINAL ANSWER:"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def extract_final_answer(raw_response: str) -> str:
        """Pull the text after ``FINAL ANSWER:`` from a CoT response.

        If the delimiter is not found the full response is returned
        as-is (the model may have answered directly).

        Args:
            raw_response: The complete model output including reasoning.

        Returns:
            The extracted answer, stripped of whitespace.
        """
        marker = "FINAL ANSWER:"
        upper = raw_response.upper()
        idx = upper.rfind(marker)
        if idx != -1:
            return raw_response[idx + len(marker):].strip()
        return raw_response.strip()

    # ── override run() to post-process CoT reasoning ──────────────────

    def run(self, question: str) -> ApproachResult:
        """Execute CoT and extract the final answer from reasoning.

        The full chain-of-thought is preserved in
        ``metadata["full_reasoning"]``; the stored ``answer`` is the
        concise extracted value suitable for automated scoring.

        Args:
            question: The evaluation question text.

        Returns:
            An :class:`ApproachResult` with the extracted final answer.
        """
        start = time.perf_counter()
        base_meta: dict = {"approach_version": "1.0", "temperature": 0.1}

        try:
            messages = self._build_prompt(question)
            # Use shared retry helper; 1024 tokens to allow full reasoning chain.
            response = self._call_groq(messages, max_tokens=1024)

            raw_answer = response.choices[0].message.content or ""
            final_answer = self.extract_final_answer(raw_answer)
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            tokens_tot = usage.total_tokens if usage else 0
            latency = (time.perf_counter() - start) * 1000
            cost = self._calculate_cost(tokens_in, tokens_out)

            base_meta["full_reasoning"] = raw_answer

            self._log.info(
                "%s | model=%s | q='%.50s' | %.0fms | %d tok",
                self.approach_name,
                self.model,
                question,
                latency,
                tokens_tot,
            )

            return ApproachResult(
                approach_name=self.approach_name,
                question=question,
                answer=final_answer,
                latency_ms=round(latency, 1),
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                tokens_total=tokens_tot,
                cost_usd=cost,
                model_used=self.model,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                metadata=base_meta,
                error=None,
            )

        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            self._log.error(
                "%s | FAILED | q='%.50s' | %s",
                self.approach_name,
                question,
                exc,
            )
            return ApproachResult(
                approach_name=self.approach_name,
                question=question,
                answer="",
                latency_ms=round(latency, 1),
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                cost_usd=0.0,
                model_used=self.model,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                metadata=base_meta,
                error=str(exc),
            )
