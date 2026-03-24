"""Base approach interface for Financial Fine-Tuning Laboratory.

Every benchmark approach — zero-shot, few-shot, chain-of-thought, RAG,
SFT fine-tuned, and DPO-aligned — inherits from :class:`BaseApproach`.
This guarantees a uniform contract: the benchmark runner calls
``approach.run(question)`` and always receives an :class:`ApproachResult`,
regardless of which approach is executing.

The base class owns the Groq client, latency measurement, cost tracking,
and error handling.  Subclasses only override ``_build_prompt()`` to
define how the question is packaged into chat messages.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from groq import Groq
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


# ── result dataclass ──────────────────────────────────────────────────


class ApproachResult(BaseModel):
    """Standardised output produced by every approach's ``run()`` call."""

    approach_name: str
    question: str
    answer: str
    latency_ms: float
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: float
    model_used: str
    timestamp: str
    metadata: dict
    error: str | None = None

    @property
    def cost_inr(self) -> float:
        """Approximate cost in Indian Rupees (USD × 83.5)."""
        return self.cost_usd * 83.5


# ── abstract base approach ────────────────────────────────────────────


class BaseApproach(ABC):
    """Abstract base class for all benchmark approaches.

    Subclasses must implement :meth:`_build_prompt` which returns
    the list of chat-message dicts sent to the Groq API.  Everything
    else — API call, timing, error handling, result packaging — is
    handled here.
    """

    def __init__(self, approach_name: str, model: str) -> None:
        self.approach_name = approach_name
        self.model = model
        self._log = logger
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    # ── subclass hook ─────────────────────────────────────────────────

    @abstractmethod
    def _build_prompt(self, question: str) -> list[dict]:
        """Construct the chat messages sent to the Groq API.

        Args:
            question: The raw evaluation question text.

        Returns:
            A list of message dicts, e.g.
            ``[{"role": "system", "content": "..."}, ...]``
        """

    # ── public API ────────────────────────────────────────────────────

    def run(self, question: str) -> ApproachResult:
        """Execute this approach on a single question.

        Args:
            question: The evaluation question text.

        Returns:
            An :class:`ApproachResult` with answer, timing, and usage.
        """
        start = time.perf_counter()
        base_meta: dict = {"approach_version": "1.0", "temperature": 0.1}

        try:
            messages = self._build_prompt(question)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )

            answer = response.choices[0].message.content or ""
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            tokens_tot = usage.total_tokens if usage else 0
            latency = (time.perf_counter() - start) * 1000
            cost = self._calculate_cost(tokens_in, tokens_out)

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
                answer=answer.strip(),
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

    def batch_run(
        self,
        questions: list[str],
        delay_seconds: float | None = None,
    ) -> list[ApproachResult]:
        """Run multiple questions sequentially with rate-limit delays.

        Args:
            questions: List of question texts.
            delay_seconds: Pause between API calls.  Defaults to
                ``settings.GROQ_CALL_DELAY_SECONDS``.

        Returns:
            List of :class:`ApproachResult`, one per question.
        """
        delay = delay_seconds if delay_seconds is not None else settings.GROQ_CALL_DELAY_SECONDS
        results: list[ApproachResult] = []
        total = len(questions)

        for i, q in enumerate(questions, start=1):
            self._log.info("Processing question %d/%d", i, total)
            results.append(self.run(q))
            if i < total:
                time.sleep(delay)

        return results

    # ── cost calculation ──────────────────────────────────────────────

    @staticmethod
    def _calculate_cost(tokens_input: int, tokens_output: int) -> float:
        """Compute per-call cost in USD.

        Groq free tier — update with paid tier pricing if upgraded.
        """
        return 0.0
