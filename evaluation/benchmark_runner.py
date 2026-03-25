"""Benchmark runner for Financial Fine-Tuning Laboratory.

Orchestrates running all benchmark approaches against the 100-question eval
set, scoring every answer, and producing per-approach result JSONs that the
Streamlit dashboard consumes.

Architecture
────────────
The runner processes approaches **sequentially** (to stay within Groq rate
limits) and questions **sequentially within each approach** (one API call
per question plus an optional LLM-judge call).

Resumability pattern
────────────────────
Groq's free tier imposes daily request and token limits.  If the runner
crashes at question 67, it must restart at question 68 — not question 1.

After **every single question** the runner writes a checkpoint JSON:

    results/<approach_name>_run<N>.json

The checkpoint contains all :class:`QuestionResult` objects collected so
far plus a ``status`` field (``"in_progress"`` or ``"complete"``).  When
the runner starts, it loads any existing checkpoint and skips questions
whose IDs already appear in the results list.  This makes the runner
fully idempotent: you can ``Ctrl-C`` and re-run at any time.

Rate-limit strategy
───────────────────
* A configurable ``delay_seconds`` pause between questions (defaults to
  ``settings.GROQ_CALL_DELAY_SECONDS``).
* Exponential backoff on 429 errors is handled inside the approach's
  ``run()`` method and the judge's ``compute_hallucination_score()``.
* The runner itself never retries — it trusts the lower layers to handle
  transient errors and records whatever they return.
"""

import glob
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from approaches.base import ApproachResult, BaseApproach
from config import settings
from data.eval_set.schemas import EvalQuestion, EvalSet
from evaluation.llm_judge import judge_single
from evaluation.metrics import compute_factual_accuracy
from evaluation.statistics import (
    compute_bootstrap_ci,
    compute_latency_percentiles,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_USD_TO_INR = 83.5


# ── result models ────────────────────────────────────────────────────────


class QuestionResult(BaseModel):
    """Scored result for a single question answered by a single approach."""

    question_id: str
    approach_name: str
    run_number: int
    question: str
    expected_answer: str
    predicted_answer: str
    category: str
    evaluation_type: str
    factual_accuracy: float
    hallucination_score: float
    schema_valid: bool
    correct: bool
    latency_ms: float
    tokens_total: int
    cost_usd: float
    retrieved_chunks: list[dict]
    judge_verdict: str
    timestamp: str
    error: str | None = None


class RunResults(BaseModel):
    """Aggregate results for one approach across one benchmark run."""

    approach_name: str
    run_number: int
    started_at: str
    completed_at: str | None = None
    total_questions: int
    completed_questions: int
    correct_count: int
    accuracy_pct: float
    hallucination_rate_pct: float
    avg_latency_ms: float
    total_cost_usd: float
    cost_per_correct_usd: float
    question_results: list[QuestionResult]
    status: str  # "in_progress" | "complete" | "failed"


# ── benchmark runner ─────────────────────────────────────────────────────


class BenchmarkRunner:
    """Orchestrates benchmark execution across approaches and runs.

    Args:
        approaches: The approach instances to benchmark.
        n_runs: Number of independent runs per approach.
        delay_seconds: Pause between questions (seconds).
        run_judge: Whether to call the LLM hallucination judge.
        results_dir: Directory to write checkpoint / result JSONs.
    """

    def __init__(
        self,
        approaches: list[BaseApproach],
        n_runs: int = 3,
        delay_seconds: float | None = None,
        run_judge: bool = True,
        results_dir: Path | None = None,
    ) -> None:
        self.approaches = approaches
        self.n_runs = n_runs
        self.delay_seconds = (
            delay_seconds
            if delay_seconds is not None
            else settings.GROQ_CALL_DELAY_SECONDS
        )
        self.run_judge = run_judge
        self.results_dir = results_dir or settings.RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.eval_set: EvalSet = EvalSet.from_file(settings.EVAL_SET_PATH)
        self._log = logger

    # ── public API ───────────────────────────────────────────────────

    def run_all(self) -> dict[str, list[RunResults]]:
        """Execute all approaches for all runs, returning results per approach."""
        all_results: dict[str, list[RunResults]] = {}

        for approach in self.approaches:
            name = approach.approach_name
            run_list: list[RunResults] = []

            for run_number in range(1, self.n_runs + 1):
                checkpoint = self._load_checkpoint(name, run_number)

                if checkpoint is not None and checkpoint.status == "complete":
                    run_list.append(checkpoint)
                    continue

                result = self._run_single(approach, run_number, checkpoint)
                run_list.append(result)

            all_results[name] = run_list

        return all_results

    # ── core execution ───────────────────────────────────────────────

    def _run_single(
        self,
        approach: BaseApproach,
        run_number: int,
        checkpoint: RunResults | None,
    ) -> RunResults:
        """Run one approach against all remaining questions for one run."""
        name = approach.approach_name
        questions = self.eval_set.questions

        # Resume from checkpoint if available
        if checkpoint is not None and checkpoint.status == "in_progress":
            completed_ids = {
                r.question_id for r in checkpoint.question_results
            }
            questions_remaining = [
                q for q in questions if q.id not in completed_ids
            ]
            results_so_far = list(checkpoint.question_results)
        else:
            questions_remaining = list(questions)
            results_so_far = []

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        run_result = RunResults(
            approach_name=name,
            run_number=run_number,
            started_at=checkpoint.started_at if checkpoint else now_iso,
            total_questions=len(questions),
            completed_questions=len(results_so_far),
            correct_count=0,
            accuracy_pct=0.0,
            hallucination_rate_pct=0.0,
            avg_latency_ms=0.0,
            total_cost_usd=0.0,
            cost_per_correct_usd=0.0,
            question_results=results_so_far,
            status="in_progress",
        )

        total_remaining = len(questions_remaining)
        self._log.info(
            "Starting %s run %d — %d questions remaining",
            name,
            run_number,
            total_remaining,
        )

        for i, question in enumerate(questions_remaining, start=1):
            self._log.info(
                "Run %d | %s | %d/%d | %s",
                run_number,
                name,
                i,
                total_remaining,
                question.id,
            )

            qr = self._process_question(
                approach=approach,
                question=question,
                run_number=run_number,
            )
            results_so_far.append(qr)

            # Update and checkpoint
            run_result.question_results = results_so_far
            run_result.completed_questions = len(results_so_far)
            self._save_checkpoint(run_result)

            # Delay between questions (skip after the last one)
            if i < total_remaining:
                time.sleep(self.delay_seconds)

        # ── mark complete and compute final metrics ──────────────────
        run_result.status = "complete"
        run_result.completed_at = datetime.now(tz=timezone.utc).isoformat()
        self._compute_run_metrics(run_result)
        self._save_checkpoint(run_result)

        self._log.info(
            "Completed %s run %d — accuracy %.1f%% | hallucination %.1f%% "
            "| avg latency %.0fms | cost $%.4f",
            name,
            run_number,
            run_result.accuracy_pct,
            run_result.hallucination_rate_pct,
            run_result.avg_latency_ms,
            run_result.total_cost_usd,
        )

        return run_result

    def _process_question(
        self,
        approach: BaseApproach,
        question: EvalQuestion,
        run_number: int,
    ) -> QuestionResult:
        """Run a single question through the approach and score it."""
        # Get the answer
        approach_result: ApproachResult = approach.run(question.question)

        # Score factual accuracy
        accuracy_result = compute_factual_accuracy(
            predicted=approach_result.answer,
            expected=question.expected_answer,
            evaluation_type=question.evaluation_type,
            question_category=question.category,
        )

        # LLM judge for hallucination
        if self.run_judge:
            self._log.debug("Calling judge for %s", question.id)
            judge = judge_single(
                question=question.question,
                predicted_answer=approach_result.answer,
                source_text=question.source.original_text,
                question_id=question.id,
            )
        else:
            judge = {
                "verdict": "SKIPPED",
                "hallucination_score": 0.5,
                "passed": True,
            }

        # Determine correctness
        is_classification = question.evaluation_type == "classification"
        correct = (
            (accuracy_result.score == 1.0)
            if is_classification
            else (accuracy_result.score >= 0.75)
        )

        # Schema validity (true if not schema_validation type, or if score > 0)
        schema_valid = (
            accuracy_result.score > 0.0
            if question.evaluation_type == "schema_validation"
            else True
        )

        # Retrieved chunks from RAG metadata
        retrieved_chunks: list[dict] = approach_result.metadata.get(
            "retrieved_chunks", []
        )

        return QuestionResult(
            question_id=question.id,
            approach_name=approach.approach_name,
            run_number=run_number,
            question=question.question,
            expected_answer=question.expected_answer,
            predicted_answer=approach_result.answer,
            category=question.category,
            evaluation_type=question.evaluation_type,
            factual_accuracy=accuracy_result.score,
            hallucination_score=judge.get("hallucination_score", 0.5),
            schema_valid=schema_valid,
            correct=correct,
            latency_ms=approach_result.latency_ms,
            tokens_total=approach_result.tokens_total,
            cost_usd=approach_result.cost_usd,
            retrieved_chunks=retrieved_chunks,
            judge_verdict=judge.get("verdict", "SKIPPED"),
            timestamp=approach_result.timestamp,
            error=approach_result.error,
        )

    # ── checkpoint persistence ───────────────────────────────────────

    def _save_checkpoint(self, run_result: RunResults) -> None:
        """Write run_result to a JSON checkpoint file.

        Never raises — a failed save must not crash the benchmark.
        """
        path = (
            self.results_dir
            / f"{run_result.approach_name}_run{run_result.run_number}.json"
        )
        try:
            data = run_result.model_dump()
            path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            self._log.debug("Checkpoint saved: %s", path)
        except Exception as exc:
            self._log.error("Failed to save checkpoint %s: %s", path, exc)

    def _load_checkpoint(
        self,
        approach_name: str,
        run_number: int,
    ) -> RunResults | None:
        """Load a checkpoint file if it exists."""
        path = self.results_dir / f"{approach_name}_run{run_number}.json"

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result = RunResults.model_validate(data)
        except Exception as exc:
            self._log.warning(
                "Corrupt checkpoint %s, starting fresh: %s", path, exc
            )
            return None

        if result.status == "complete":
            self._log.info(
                "Skipping %s run %d — already complete",
                approach_name,
                run_number,
            )
        else:
            self._log.info(
                "Resuming %s run %d from question %d/%d",
                approach_name,
                run_number,
                result.completed_questions,
                result.total_questions,
            )

        return result

    # ── metric computation ───────────────────────────────────────────

    @staticmethod
    def _compute_run_metrics(run_result: RunResults) -> None:
        """Populate aggregate metric fields on a completed RunResults."""
        qrs = run_result.question_results
        if not qrs:
            return

        correct = sum(1 for r in qrs if r.correct)
        total = len(qrs)
        latencies = [r.latency_ms for r in qrs]
        costs = [r.cost_usd for r in qrs]

        judged = [r for r in qrs if r.judge_verdict != "SKIPPED"]
        hallucinated = sum(
            1
            for r in judged
            if r.judge_verdict in ("MINOR_HALLUCINATION", "MAJOR_HALLUCINATION")
        )

        run_result.correct_count = correct
        run_result.accuracy_pct = round(correct / total * 100, 2)
        run_result.hallucination_rate_pct = (
            round(hallucinated / len(judged) * 100, 2) if judged else 0.0
        )
        run_result.avg_latency_ms = round(sum(latencies) / total, 1)
        run_result.total_cost_usd = round(sum(costs), 6)
        run_result.cost_per_correct_usd = (
            round(sum(costs) / correct, 6) if correct else 0.0
        )

    # ── summary generation ───────────────────────────────────────────

    def generate_summary(
        self,
        all_results: dict[str, list[RunResults]],
    ) -> dict:
        """Build and save the final benchmark summary across all approaches."""
        approaches_summary: dict[str, dict] = {}

        for name, run_list in all_results.items():
            completed_runs = [r for r in run_list if r.status == "complete"]
            if not completed_runs:
                continue

            accuracy_per_run = [r.accuracy_pct for r in completed_runs]
            overall_accuracy = (
                sum(accuracy_per_run) / len(accuracy_per_run)
                if accuracy_per_run
                else 0.0
            )

            # Bootstrap CI on per-run accuracies
            bootstrap = compute_bootstrap_ci(accuracy_per_run)

            # Latency percentiles across all questions × runs
            all_latencies = [
                qr.latency_ms
                for r in completed_runs
                for qr in r.question_results
            ]
            latency_stats = compute_latency_percentiles(all_latencies)

            # Hallucination rate across all questions × runs
            all_judged = [
                qr
                for r in completed_runs
                for qr in r.question_results
                if qr.judge_verdict != "SKIPPED"
            ]
            hallucinated = sum(
                1
                for qr in all_judged
                if qr.judge_verdict
                in ("MINOR_HALLUCINATION", "MAJOR_HALLUCINATION")
            )
            hallucination_rate = (
                round(hallucinated / len(all_judged) * 100, 2)
                if all_judged
                else 0.0
            )

            total_cost = sum(r.total_cost_usd for r in completed_runs)
            total_correct = sum(r.correct_count for r in completed_runs)
            cost_per_correct = (
                round(total_cost / total_correct, 6) if total_correct else 0.0
            )

            approaches_summary[name] = {
                "accuracy_per_run": [round(a, 2) for a in accuracy_per_run],
                "overall_accuracy_pct": round(overall_accuracy, 2),
                "bootstrap_ci": {
                    "ci_lower": bootstrap.get("ci_lower", 0.0),
                    "ci_upper": bootstrap.get("ci_upper", 0.0),
                    "margin_of_error": bootstrap.get("margin_of_error", 0.0),
                },
                "hallucination_rate_pct": hallucination_rate,
                "avg_latency_ms": round(latency_stats.get("mean_ms", 0.0), 1),
                "p95_latency_ms": round(
                    latency_stats.get("p95_ms", 0.0), 1
                ),
                "total_cost_usd": round(total_cost, 6),
                "cost_per_correct_usd": cost_per_correct,
                "cost_per_correct_inr": round(
                    cost_per_correct * _USD_TO_INR, 4
                ),
            }

        # Rank by overall accuracy descending
        ranking = sorted(
            approaches_summary.keys(),
            key=lambda n: approaches_summary[n]["overall_accuracy_pct"],
            reverse=True,
        )

        n_questions = self.eval_set.total_questions
        moe = (
            approaches_summary[ranking[0]]["bootstrap_ci"]["margin_of_error"]
            if ranking and approaches_summary[ranking[0]]["bootstrap_ci"]["margin_of_error"]
            else 0.0
        )

        summary = {
            "benchmark_version": "1.0.0",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "n_runs": self.n_runs,
            "total_questions": n_questions,
            "approaches": approaches_summary,
            "ranking": ranking,
            "best_approach": ranking[0] if ranking else "",
            "statistical_notes": (
                f"With {self.n_runs} runs of {n_questions} questions each, "
                f"confidence intervals have margin of error ≈ ±{moe:.1f}%. "
                f"Approaches with non-overlapping CIs can be considered "
                f"statistically different."
            ),
        }

        # Save
        path = self.results_dir / "benchmark_summary.json"
        try:
            path.write_text(
                json.dumps(summary, indent=2, default=str),
                encoding="utf-8",
            )
            self._log.info("Benchmark summary saved to %s", path)
        except Exception as exc:
            self._log.error("Failed to save summary: %s", exc)

        return summary

    # ── offline summary regeneration ─────────────────────────────────

    @staticmethod
    def regenerate_summary_from_existing_runs(
        results_dir: str | Path = "evaluation/results",
        min_questions: int = 10,
    ) -> dict:
        """Rebuild benchmark_summary.json from existing run files.

        Loads all ``*_run*.json`` files in *results_dir*, ignoring
        dry-run files (completed_questions <= *min_questions*), and
        writes a fresh summary.

        Callable from CLI::

            python3 -c "
            from evaluation.benchmark_runner import BenchmarkRunner
            BenchmarkRunner.regenerate_summary_from_existing_runs()
            "
        """
        results_dir = Path(results_dir)
        run_files = sorted(glob.glob(str(results_dir / "*_run*.json")))

        # Load valid (non-dry-run) results grouped by approach
        by_approach: dict[str, list[RunResults]] = {}
        for fpath in run_files:
            try:
                data = json.loads(Path(fpath).read_text(encoding="utf-8"))
                if data.get("status") != "complete":
                    continue
                if data.get("completed_questions", 0) <= min_questions:
                    continue
                rr = RunResults.model_validate(data)
                by_approach.setdefault(rr.approach_name, []).append(rr)
            except Exception as exc:
                logger.warning("Skipping %s: %s", fpath, exc)

        if not by_approach:
            logger.error("No valid run files found in %s", results_dir)
            return {}

        # Build summary (mirrors generate_summary logic)
        approaches_summary: dict[str, dict] = {}

        for name, run_list in by_approach.items():
            accuracy_per_run = [r.accuracy_pct for r in run_list]
            overall_accuracy = (
                sum(accuracy_per_run) / len(accuracy_per_run)
                if accuracy_per_run
                else 0.0
            )

            bootstrap = compute_bootstrap_ci(accuracy_per_run)

            all_latencies = [
                qr.latency_ms
                for r in run_list
                for qr in r.question_results
            ]
            latency_stats = compute_latency_percentiles(all_latencies)

            all_judged = [
                qr
                for r in run_list
                for qr in r.question_results
                if qr.judge_verdict != "SKIPPED"
            ]
            hallucinated = sum(
                1
                for qr in all_judged
                if qr.judge_verdict
                in ("MINOR_HALLUCINATION", "MAJOR_HALLUCINATION")
            )
            hallucination_rate = (
                round(hallucinated / len(all_judged) * 100, 2)
                if all_judged
                else 0.0
            )

            total_cost = sum(r.total_cost_usd for r in run_list)
            total_correct = sum(r.correct_count for r in run_list)
            cost_per_correct = (
                round(total_cost / total_correct, 6)
                if total_correct
                else 0.0
            )

            approaches_summary[name] = {
                "accuracy_per_run": [round(a, 2) for a in accuracy_per_run],
                "overall_accuracy_pct": round(overall_accuracy, 2),
                "bootstrap_ci": {
                    "ci_lower": bootstrap.get("ci_lower", 0.0),
                    "ci_upper": bootstrap.get("ci_upper", 0.0),
                    "margin_of_error": bootstrap.get("margin_of_error", 0.0),
                },
                "hallucination_rate_pct": hallucination_rate,
                "avg_latency_ms": round(
                    latency_stats.get("mean_ms", 0.0), 1
                ),
                "p95_latency_ms": round(
                    latency_stats.get("p95_ms", 0.0), 1
                ),
                "total_cost_usd": round(total_cost, 6),
                "cost_per_correct_usd": cost_per_correct,
                "cost_per_correct_inr": round(
                    cost_per_correct * _USD_TO_INR, 4
                ),
            }

        ranking = sorted(
            approaches_summary.keys(),
            key=lambda n: approaches_summary[n]["overall_accuracy_pct"],
            reverse=True,
        )

        total_questions = 100
        n_runs = max(len(v) for v in by_approach.values()) if by_approach else 1
        moe = (
            approaches_summary[ranking[0]]["bootstrap_ci"]["margin_of_error"]
            if ranking
            and approaches_summary[ranking[0]]["bootstrap_ci"]["margin_of_error"]
            else 0.0
        )

        summary = {
            "benchmark_version": "1.0.0",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "n_runs": n_runs,
            "total_questions": total_questions,
            "approaches": approaches_summary,
            "ranking": ranking,
            "best_approach": ranking[0] if ranking else "",
            "statistical_notes": (
                f"With {n_runs} run(s) of {total_questions} questions each, "
                f"confidence intervals have margin of error ≈ ±{moe:.1f}%. "
                f"Approaches with non-overlapping CIs can be considered "
                f"statistically different."
            ),
        }

        path = results_dir / "benchmark_summary.json"
        path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Regenerated benchmark summary → %s", path)

        # Print summary table
        print(f"\n{'='*60}")
        print("REGENERATED BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(
            f"{'Approach':<22} {'Accuracy':>10} {'Halluc.%':>10} "
            f"{'Avg Lat.':>10}"
        )
        print("-" * 60)
        for name in ranking:
            d = approaches_summary[name]
            print(
                f"{name:<22} {d['overall_accuracy_pct']:>9.1f}% "
                f"{d['hallucination_rate_pct']:>9.1f}% "
                f"{d['avg_latency_ms']:>8.0f}ms"
            )
        print(f"{'='*60}")
        print(f"Best: {ranking[0] if ranking else 'N/A'}")
        print()

        return summary
