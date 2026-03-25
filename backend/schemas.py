"""Pydantic request/response schemas for the Financial Fine-Tuning Laboratory API.

Every endpoint has typed request and response models.  FastAPI uses these for
automatic request validation, OpenAPI doc generation (/docs), and JSON
serialization.  Keeping schemas separate from route handlers keeps main.py
focused on orchestration logic.
"""

from pydantic import BaseModel, Field


# ── request models ───────────────────────────────────────────────────────


class QuestionRequest(BaseModel):
    """Request body for running approaches on a question."""

    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The financial question to answer.",
    )
    approaches: list[str] = Field(
        default=["zero_shot", "few_shot", "cot", "rag"],
        description="Approaches to run. Valid: zero_shot, few_shot, cot, rag, sft, dpo.",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Per-approach timeout in seconds.",
    )


# ── response models ─────────────────────────────────────────────────────


class ApproachResponse(BaseModel):
    """Result from a single approach run."""

    approach_name: str
    answer: str
    latency_ms: float
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: float
    cost_inr: float
    model_used: str
    timestamp: str
    error: str | None = None
    metadata: dict


class MultiApproachResponse(BaseModel):
    """Combined results from running multiple approaches on one question."""

    question: str
    results: list[ApproachResponse]
    total_latency_ms: float
    timestamp: str


class BenchmarkSummaryResponse(BaseModel):
    """Benchmark results summary for the dashboard."""

    available: bool
    summary: dict | None = None
    last_updated: str | None = None
    approaches_completed: list[str]


class HealthResponse(BaseModel):
    """System health check."""

    status: str
    timestamp: str
    approaches_loaded: list[str]
    chromadb_chunks: int
    eval_questions: int
    benchmark_available: bool


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    timestamp: str
