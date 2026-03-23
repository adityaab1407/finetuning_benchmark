"""Pydantic schemas for the Financial Fine-Tuning Laboratory evaluation set.

The eval set is the benchmark's single source of truth: 100 questions drawn
from real EDGAR filings and expert-labeled HuggingFace datasets.  Every
benchmark metric (accuracy, hallucination rate, cost per correct answer) is
computed against these question/answer pairs.

Four question categories with distinct evaluation logic:
  - factual_extraction  (30): exact numbers from SEC filings
  - sentiment           (30): positive / negative / neutral labels
  - structured_output   (20): multi-field JSON extraction
  - reasoning           (20): multi-step calculations and conclusions
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, field_validator


class QuestionCategory(str, Enum):
    """The four benchmark question types."""

    factual_extraction = "factual_extraction"
    sentiment = "sentiment"
    structured_output = "structured_output"
    reasoning = "reasoning"


class DifficultyLevel(str, Enum):
    """Subjective difficulty tier assigned per question."""

    easy = "easy"
    medium = "medium"
    hard = "hard"


class SourceReference(BaseModel):
    """Provenance record linking a question to its source document."""

    document_id: str
    """Filename stem of the source document, e.g. 'AAPL_10-Q_2025-08-01'."""

    source_type: str
    """One of 'edgar', 'financial_phrasebank', or 'fiqa'."""

    ticker: str | None = None
    """Company ticker symbol for EDGAR-sourced questions."""

    filing_date: str | None = None
    """Filing date string for EDGAR-sourced questions (e.g. '2025-08-01')."""

    section_hint: str | None = None
    """Rough location within the document, e.g. 'Revenue section, Q3 results'."""

    original_text: str
    """The verbatim source passage the ground-truth answer comes from.
    Capped at 1000 characters."""

    @field_validator("original_text")
    @classmethod
    def truncate_original_text(cls, v: str) -> str:
        return v[:1000]


class EvalQuestion(BaseModel):
    """A single benchmark question with its ground-truth answer and provenance."""

    id: str
    """Unique question ID.  Format: 'q_{prefix}_{number:03d}'
    Prefixes: fe=factual_extraction, se=sentiment, so=structured_output,
    re=reasoning.  Examples: 'q_fe_001', 'q_se_030', 'q_so_020'."""

    category: QuestionCategory

    difficulty: DifficultyLevel

    question: str
    """The question text presented verbatim to each approach."""

    expected_answer: str
    """Ground-truth answer string.
    - factual_extraction : exact value, e.g. '$94.9 billion'
    - sentiment          : 'positive', 'negative', or 'neutral'
    - structured_output  : JSON object string with exact field values
    - reasoning          : conclusion + supporting calculation summary
    """

    source: SourceReference

    requires_context: bool
    """True when the question cannot be answered without the source document
    (factual, structured, most reasoning).  False for self-contained
    questions (sentiment)."""

    evaluation_type: str
    """Scoring method for this question.
    - 'exact_match'        : factual numbers, precise string equality
    - 'fuzzy_match'        : factual written numbers, token overlap
    - 'classification'     : sentiment three-class label
    - 'schema_validation'  : structured_output JSON field checking
    - 'reasoning_match'    : reasoning conclusion + calculation match
    """

    created_at: str
    """ISO-8601 timestamp of when this question was added."""

    notes: str | None = None
    """Optional notes about the ground-truth value, edge cases, or
    ambiguities in the source text."""


class EvalSet(BaseModel):
    """The complete 100-question benchmark evaluation set."""

    version: str = "1.0.0"

    created_at: str

    total_questions: int

    questions: list[EvalQuestion]

    category_counts: dict[str, int]
    """Maps each category name to its question count."""

    source_documents: list[str]
    """Sorted list of all unique document_ids referenced by questions."""

    description: str

    # ── class methods ─────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: Path) -> "EvalSet":
        """Load and validate an EvalSet from a JSON file.

        Args:
            path: Path to the questions.json file.

        Returns:
            Validated EvalSet instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValidationError: If the JSON does not match the schema.
        """
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return cls.model_validate(data)

    def to_file(self, path: Path) -> None:
        """Serialise this EvalSet to a formatted JSON file.

        Args:
            path: Destination path.  Parent directories are created
                  automatically.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
