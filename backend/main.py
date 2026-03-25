"""FastAPI application for the Financial Fine-Tuning Laboratory.

This API serves all benchmark approaches via REST endpoints, exposes
pre-computed benchmark results for the Streamlit dashboard, and provides
a live demo where users can submit questions and see all approaches
respond side-by-side.

**Why async + ThreadPoolExecutor?**

The Groq API calls in each approach are synchronous (blocking I/O).
Running them directly in an ``async def`` handler would block the
entire event loop, preventing concurrent request handling.  Instead,
we dispatch each ``approach.run()`` call to a thread pool via
``loop.run_in_executor()``.  This lets FastAPI handle multiple
in-flight requests and, critically, lets the ``/api/run/all`` endpoint
fire all requested approaches in parallel via ``asyncio.gather()``.

**Note on parallel Groq calls:** running multiple approaches
concurrently means Groq sees simultaneous requests.  On the free tier
this may exhaust RPM limits faster than sequential calls.  The
approaches' internal backoff logic handles 429 errors, but callers
should expect occasional retries under heavy load.
"""

import asyncio
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from approaches import (
    ChainOfThoughtApproach,
    FewShotApproach,
    RAGApproach,
    ZeroShotApproach,
)
from approaches.base import ApproachResult, BaseApproach
from backend.schemas import (
    ApproachResponse,
    BenchmarkSummaryResponse,
    ErrorResponse,
    HealthResponse,
    MultiApproachResponse,
    QuestionRequest,
)
from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_USD_TO_INR = 83.5

# ── approach registry ────────────────────────────────────────────────────

_approaches: dict[str, BaseApproach] = {}
_executor = ThreadPoolExecutor(max_workers=4)


def get_approach(name: str) -> BaseApproach:
    """Lazily initialise and return the requested approach."""
    if name not in _approaches:
        if name == "zero_shot":
            _approaches[name] = ZeroShotApproach()
        elif name == "few_shot":
            _approaches[name] = FewShotApproach()
        elif name == "cot":
            _approaches[name] = ChainOfThoughtApproach()
        elif name == "rag":
            rag = RAGApproach()
            rag.setup()
            _approaches[name] = rag
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Approach '{name}' not found. "
                    "Available: zero_shot, few_shot, cot, rag"
                ),
            )
    return _approaches[name]


# ── lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Pre-load RAG approach on startup so the first request is fast."""
    try:
        rag = RAGApproach()
        rag.setup()
        _approaches["rag"] = rag
        logger.info("RAG approach pre-loaded on startup")
    except Exception as exc:
        logger.warning("RAG pre-load failed: %s", exc)
    yield
    _executor.shutdown(wait=False)


# ── application ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Fine-Tuning Laboratory API",
    description=(
        "Benchmarks LLM approaches on financial question answering. "
        "Serves zero-shot, few-shot, chain-of-thought, and RAG approaches "
        "via REST endpoints, exposes benchmark data, and provides a live "
        "demo for the Streamlit dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ──────────────────────────────────────────────────────────────


def _result_to_response(result: ApproachResult) -> ApproachResponse:
    """Convert an ApproachResult to an API response model."""
    return ApproachResponse(
        approach_name=result.approach_name,
        answer=result.answer,
        latency_ms=result.latency_ms,
        tokens_input=result.tokens_input,
        tokens_output=result.tokens_output,
        tokens_total=result.tokens_total,
        cost_usd=result.cost_usd,
        cost_inr=round(result.cost_usd * _USD_TO_INR, 6),
        model_used=result.model_used,
        timestamp=result.timestamp,
        error=result.error,
        metadata=result.metadata,
    )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ── ENDPOINT 1: health ──────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """System health check — never crashes."""
    approaches_loaded = list(_approaches.keys())

    # ChromaDB chunk count
    chromadb_chunks = 0
    try:
        rag = _approaches.get("rag")
        if rag is not None and hasattr(rag, "collection") and rag.collection is not None:
            chromadb_chunks = rag.collection.count()
    except Exception:
        pass

    # Eval question count
    eval_questions = 0
    try:
        if settings.EVAL_SET_PATH.exists():
            data = json.loads(settings.EVAL_SET_PATH.read_text(encoding="utf-8"))
            eval_questions = data.get("total_questions", len(data.get("questions", [])))
    except Exception:
        pass

    # Benchmark available
    benchmark_available = (settings.RESULTS_DIR / "benchmark_summary.json").exists()

    return HealthResponse(
        status="healthy",
        timestamp=_now_iso(),
        approaches_loaded=approaches_loaded,
        chromadb_chunks=chromadb_chunks,
        eval_questions=eval_questions,
        benchmark_available=benchmark_available,
    )


# ── ENDPOINT 2: run all approaches ──────────────────────────────────────
# NOTE: must be defined BEFORE /api/run/{approach_name} so FastAPI does not
# swallow "all" as an approach_name path parameter.


@app.post(
    "/api/run/all",
    response_model=MultiApproachResponse,
    responses={500: {"model": ErrorResponse}},
)
async def run_all(request: QuestionRequest) -> MultiApproachResponse:
    """Run all requested approaches concurrently on the given question.

    Approaches are dispatched in parallel via ``asyncio.gather()``.
    If one approach fails, its error is captured in the response —
    the other approaches still return successfully.

    **Warning:** parallel Groq calls may exhaust free-tier RPM limits
    faster than sequential execution.
    """
    start = time.perf_counter()
    logger.info(
        "POST /api/run/all — approaches=%s — q='%.60s'",
        request.approaches,
        request.question,
    )

    async def _run_one(name: str, stagger_secs: float = 0.0) -> ApproachResponse:
        if stagger_secs:
            await asyncio.sleep(stagger_secs)
        try:
            approach = get_approach(name)
            loop = asyncio.get_event_loop()
            result: ApproachResult = await asyncio.wait_for(
                loop.run_in_executor(_executor, approach.run, request.question),
                timeout=request.timeout_seconds,
            )
            return _result_to_response(result)
        except asyncio.TimeoutError:
            return ApproachResponse(
                approach_name=name,
                answer="",
                latency_ms=request.timeout_seconds * 1000,
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                cost_usd=0.0,
                cost_inr=0.0,
                model_used="",
                timestamp=_now_iso(),
                error=f"Timed out after {request.timeout_seconds}s",
                metadata={},
            )
        except Exception as exc:
            return ApproachResponse(
                approach_name=name,
                answer="",
                latency_ms=0.0,
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                cost_usd=0.0,
                cost_inr=0.0,
                model_used="",
                timestamp=_now_iso(),
                error=str(exc),
                metadata={},
            )

    # Stagger requests by 1.5 s each to avoid simultaneous free-tier 429s.
    tasks = [_run_one(name, stagger_secs=i * 1.5) for i, name in enumerate(request.approaches)]
    results = await asyncio.gather(*tasks)

    total_latency = (time.perf_counter() - start) * 1000
    logger.info("POST /api/run/all — %.0fms total", total_latency)

    return MultiApproachResponse(
        question=request.question,
        results=list(results),
        total_latency_ms=round(total_latency, 1),
        timestamp=_now_iso(),
    )


# ── ENDPOINT 3: run single approach ─────────────────────────────────────


@app.post(
    "/api/run/{approach_name}",
    response_model=ApproachResponse,
    responses={404: {"model": ErrorResponse}, 408: {"model": ErrorResponse}},
)
async def run_single(approach_name: str, request: QuestionRequest) -> ApproachResponse:
    """Run a single approach on the given question."""
    start = time.perf_counter()
    logger.info("POST /api/run/%s — q='%.60s'", approach_name, request.question)

    approach = get_approach(approach_name)

    try:
        loop = asyncio.get_event_loop()
        result: ApproachResult = await asyncio.wait_for(
            loop.run_in_executor(_executor, approach.run, request.question),
            timeout=request.timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Approach '{approach_name}' timed out after {request.timeout_seconds}s",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /api/run/%s — %.0fms — %s",
        approach_name,
        latency,
        "OK" if not result.error else f"ERROR: {result.error}",
    )

    return _result_to_response(result)


# ── ENDPOINT 4: benchmark summary ───────────────────────────────────────


@app.get("/api/results", response_model=BenchmarkSummaryResponse)
async def get_results() -> BenchmarkSummaryResponse:
    """Return the benchmark summary if available."""
    summary_path = settings.RESULTS_DIR / "benchmark_summary.json"

    if not summary_path.exists():
        return BenchmarkSummaryResponse(
            available=False,
            summary=None,
            last_updated=None,
            approaches_completed=[],
        )

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        approaches_completed = list(data.get("approaches", {}).keys())
        return BenchmarkSummaryResponse(
            available=True,
            summary=data,
            last_updated=data.get("generated_at"),
            approaches_completed=approaches_completed,
        )
    except Exception as exc:
        logger.error("Failed to load benchmark summary: %s", exc)
        return BenchmarkSummaryResponse(
            available=False,
            summary=None,
            last_updated=None,
            approaches_completed=[],
        )


# ── ENDPOINT 5: per-approach results ────────────────────────────────────


@app.get(
    "/api/results/{approach_name}",
    responses={404: {"model": ErrorResponse}},
)
async def get_approach_results(approach_name: str) -> dict:
    """Return all run results for a specific approach."""
    results_dir = settings.RESULTS_DIR
    run_files = sorted(results_dir.glob(f"{approach_name}_run*.json"))

    if not run_files:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for approach '{approach_name}'",
        )

    runs: list[dict] = []
    for path in run_files:
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    return {
        "approach_name": approach_name,
        "total_runs": len(runs),
        "runs": runs,
    }


# ── ENDPOINT 6: sample questions ────────────────────────────────────────


@app.get("/api/questions/sample")
async def get_sample_questions() -> list[dict]:
    """Return 5 sample eval questions — one per category plus one extra."""
    try:
        data = json.loads(settings.EVAL_SET_PATH.read_text(encoding="utf-8"))
        questions = data.get("questions", [])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load eval set: {exc}")

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        by_category.setdefault(cat, []).append(q)

    sample: list[dict] = []
    random.seed(42)

    # One per category
    for cat in ("factual_extraction", "sentiment", "structured_output", "reasoning"):
        pool = by_category.get(cat, [])
        if pool:
            picked = random.choice(pool)
            sample.append({
                "id": picked["id"],
                "category": picked["category"],
                "question": picked["question"],
                "difficulty": picked["difficulty"],
            })

    # One extra factual
    factual_pool = by_category.get("factual_extraction", [])
    extras = [q for q in factual_pool if q["id"] not in {s["id"] for s in sample}]
    if extras:
        picked = random.choice(extras)
        sample.append({
            "id": picked["id"],
            "category": picked["category"],
            "question": picked["question"],
            "difficulty": picked["difficulty"],
        })

    return sample


# ── ENDPOINT 7: system status ───────────────────────────────────────────


@app.get("/api/status")
async def get_status() -> dict:
    """Return current system status for the dashboard header."""
    results_dir = settings.RESULTS_DIR

    # Count completed benchmark runs
    run_files = list(results_dir.glob("*_run*.json"))
    complete_count = 0
    approaches_benchmarked: set[str] = set()
    for path in run_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") == "complete":
                complete_count += 1
                approaches_benchmarked.add(data.get("approach_name", ""))
        except Exception:
            pass

    # Fine-tuned models
    sft_adapter_path = Path("models/adapters/sft_r16_full")
    fine_tuned_ready: list[str] = []
    if sft_adapter_path.exists():
        fine_tuned_ready.append("sft_r16_full")

    dpo_adapter_path = Path("models/adapters/dpo_r16_full")
    if dpo_adapter_path.exists():
        fine_tuned_ready.append("dpo_r16_full")

    # Dataset on HuggingFace (proxy: local SFT dataset exists)
    dataset_exists = (settings.SYNTHETIC_DIR / "sft_dataset.json").exists()

    return {
        "benchmark_runs_complete": complete_count,
        "benchmark_runs_total": 3,
        "approaches_benchmarked": sorted(approaches_benchmarked),
        "fine_tuned_models_ready": fine_tuned_ready,
        "fine_tuned_models_training": [],
        "dataset_on_huggingface": dataset_exists,
        "kaggle_training_status": "not_started",
    }
