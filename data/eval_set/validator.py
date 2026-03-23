"""Evaluation set validator for Financial Fine-Tuning Laboratory.

Runs ten integrity checks against data/eval_set/questions.json before
any benchmark run.  Every check is independent — a failure in one does
not skip subsequent checks.  Exit code is 0 on full pass, 1 on any
failure.

Usage::

    python3 -m data.eval_set.validator
"""

import json
import logging
import sys
from pathlib import Path

from config import settings
from data.eval_set.schemas import (
    EvalSet,
    QuestionCategory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# ── constants ─────────────────────────────────────────────────────────

_EXPECTED_TOTAL = 100
_EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    QuestionCategory.factual_extraction.value: 30,
    QuestionCategory.sentiment.value: 30,
    QuestionCategory.structured_output.value: 20,
    QuestionCategory.reasoning.value: 20,
}

_CATEGORY_ID_PREFIXES: dict[str, str] = {
    QuestionCategory.factual_extraction.value: "q_fe_",
    QuestionCategory.sentiment.value: "q_se_",
    QuestionCategory.structured_output.value: "q_so_",
    QuestionCategory.reasoning.value: "q_re_",
}

_VALID_EVAL_TYPES: dict[str, set[str]] = {
    QuestionCategory.factual_extraction.value: {"exact_match", "fuzzy_match"},
    QuestionCategory.sentiment.value: {"classification"},
    QuestionCategory.structured_output.value: {"schema_validation"},
    QuestionCategory.reasoning.value: {"reasoning_match"},
}

_VALID_SENTIMENTS = {"positive", "negative", "neutral"}


# ── result helpers ────────────────────────────────────────────────────

def _pass(check_num: int, msg: str) -> bool:
    logger.info("CHECK %d  PASS  %s", check_num, msg)
    return True


def _fail(check_num: int, msg: str) -> bool:
    logger.error("CHECK %d  FAIL  %s", check_num, msg)
    return False


# ── individual checks ─────────────────────────────────────────────────

def check_file_exists(path: Path) -> bool:
    if path.exists():
        return _pass(1, f"File exists: {path}")
    return _fail(1, f"File not found: {path}")


def check_valid_schema(path: Path) -> tuple[bool, EvalSet | None]:
    try:
        eval_set = EvalSet.from_file(path)
        return _pass(2, "JSON parses and validates against EvalSet schema"), eval_set
    except Exception as exc:
        _fail(2, f"Schema validation error: {exc}")
        return False, None


def check_question_count(eval_set: EvalSet) -> bool:
    n_list = len(eval_set.questions)
    n_total = eval_set.total_questions

    ok = True
    if n_total != _EXPECTED_TOTAL:
        _fail(3, f"total_questions={n_total}, expected {_EXPECTED_TOTAL}")
        ok = False
    if n_list != _EXPECTED_TOTAL:
        _fail(3, f"len(questions)={n_list}, expected {_EXPECTED_TOTAL}")
        ok = False
    if n_list != n_total:
        _fail(3, f"len(questions)={n_list} ≠ total_questions={n_total}")
        ok = False
    if ok:
        _pass(3, f"Question count correct: {n_total}")
    return ok


def check_category_distribution(eval_set: EvalSet) -> bool:
    actual: dict[str, int] = {}
    for q in eval_set.questions:
        actual[q.category.value] = actual.get(q.category.value, 0) + 1

    ok = True
    for cat, expected in _EXPECTED_CATEGORY_COUNTS.items():
        got = actual.get(cat, 0)
        if got != expected:
            _fail(4, f"{cat}: got {got}, expected {expected}")
            ok = False

    if ok:
        _pass(4, f"Category distribution correct: {actual}")
    return ok


def check_id_format(eval_set: EvalSet) -> bool:
    ok = True
    seen: set[str] = set()

    for q in eval_set.questions:
        prefix = _CATEGORY_ID_PREFIXES[q.category.value]
        if not q.id.startswith(prefix):
            _fail(5, f"ID '{q.id}' does not start with '{prefix}' "
                     f"(category={q.category.value})")
            ok = False
        if q.id in seen:
            _fail(5, f"Duplicate ID: '{q.id}'")
            ok = False
        seen.add(q.id)

    if ok:
        _pass(5, f"All {len(eval_set.questions)} IDs are unique and correctly prefixed")
    return ok


def check_no_empty_fields(eval_set: EvalSet) -> bool:
    ok = True
    for q in eval_set.questions:
        if not q.question.strip():
            _fail(6, f"{q.id}: question is empty")
            ok = False
        if not q.expected_answer.strip():
            _fail(6, f"{q.id}: expected_answer is empty")
            ok = False
        if not q.source.document_id.strip():
            _fail(6, f"{q.id}: source.document_id is empty")
            ok = False
        if not q.source.original_text.strip():
            _fail(6, f"{q.id}: source.original_text is empty")
            ok = False

    if ok:
        _pass(6, "No empty required fields found")
    return ok


def check_source_documents_exist(eval_set: EvalSet) -> bool:
    ok = True
    missing: list[str] = []
    verified = 0

    fpb_path = settings.PROCESSED_DIR / "financial_phrasebank.json"
    fiqa_path = settings.PROCESSED_DIR / "fiqa.json"

    for q in eval_set.questions:
        src = q.source
        if src.source_type == "edgar":
            raw_path = settings.RAW_DIR / f"{src.document_id}.txt"
            if raw_path.exists():
                verified += 1
            else:
                missing.append(str(raw_path))
                ok = False
        elif src.source_type == "financial_phrasebank":
            if fpb_path.exists():
                verified += 1
            else:
                if str(fpb_path) not in missing:
                    missing.append(str(fpb_path))
                ok = False
        elif src.source_type == "fiqa":
            if fiqa_path.exists():
                verified += 1
            else:
                if str(fiqa_path) not in missing:
                    missing.append(str(fiqa_path))
                ok = False

    if missing:
        for m in missing:
            _fail(7, f"Missing source file: {m}")
    else:
        _pass(7, f"All {verified} source documents exist on disk")
    return ok


def check_evaluation_type_consistency(eval_set: EvalSet) -> bool:
    ok = True
    for q in eval_set.questions:
        valid = _VALID_EVAL_TYPES[q.category.value]
        if q.evaluation_type not in valid:
            _fail(8, f"{q.id}: evaluation_type='{q.evaluation_type}' "
                     f"invalid for {q.category.value} (allowed: {valid})")
            ok = False

    if ok:
        _pass(8, "All evaluation_type values are consistent with their categories")
    return ok


def check_structured_output_json(eval_set: EvalSet) -> bool:
    ok = True
    for q in eval_set.questions:
        if q.category != QuestionCategory.structured_output:
            continue
        try:
            parsed = json.loads(q.expected_answer)
            if not isinstance(parsed, dict):
                raise ValueError("expected a JSON object, got " + type(parsed).__name__)
        except Exception as exc:
            _fail(9, f"{q.id}: expected_answer is not a valid JSON object: {exc}")
            ok = False

    if ok:
        _pass(9, "All structured_output expected_answers are valid JSON objects")
    return ok


def check_sentiment_answers(eval_set: EvalSet) -> bool:
    ok = True
    for q in eval_set.questions:
        if q.category != QuestionCategory.sentiment:
            continue
        if q.expected_answer not in _VALID_SENTIMENTS:
            _fail(10, f"{q.id}: expected_answer='{q.expected_answer}' "
                      f"must be one of {_VALID_SENTIMENTS}")
            ok = False

    if ok:
        _pass(10, f"All sentiment expected_answers are valid "
                  f"({_VALID_SENTIMENTS})")
    return ok


# ── orchestrator ──────────────────────────────────────────────────────

def run_validation() -> bool:
    """Run all ten checks.  Returns True if every check passes."""
    questions_path = settings.EVAL_SET_PATH
    results: list[bool] = []

    logger.info("=" * 60)
    logger.info("Validating eval set: %s", questions_path)
    logger.info("=" * 60)

    # Check 1 — file exists (gate: abort if missing)
    if not check_file_exists(questions_path):
        results.append(False)
        logger.error("Cannot continue — questions.json not found.")
        _print_summary(results, eval_set=None)
        return False

    results.append(True)

    # Check 2 — valid schema (gate: abort if unparseable)
    schema_ok, eval_set = check_valid_schema(questions_path)
    results.append(schema_ok)
    if not schema_ok or eval_set is None:
        logger.error("Cannot continue — schema validation failed.")
        _print_summary(results, eval_set=None)
        return False

    # Checks 3-10 — all independent
    results.append(check_question_count(eval_set))
    results.append(check_category_distribution(eval_set))
    results.append(check_id_format(eval_set))
    results.append(check_no_empty_fields(eval_set))
    results.append(check_source_documents_exist(eval_set))
    results.append(check_evaluation_type_consistency(eval_set))
    results.append(check_structured_output_json(eval_set))
    results.append(check_sentiment_answers(eval_set))

    _print_summary(results, eval_set)
    return all(results)


def _print_summary(results: list[bool], eval_set: EvalSet | None) -> None:
    passed = sum(results)
    total = len(results)

    logger.info("=" * 60)
    logger.info("Checks passed: %d/%d", passed, total)

    if eval_set is not None:
        logger.info("Questions validated: %d", len(eval_set.questions))
        edgar_docs = {
            q.source.document_id
            for q in eval_set.questions
            if q.source.source_type == "edgar"
        }
        logger.info("Source documents verified: %d", len(edgar_docs))

    if passed == total:
        logger.info("EVAL SET VALID ✓")
    else:
        logger.error("EVAL SET INVALID ✗  (%d check(s) failed)", total - passed)


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
