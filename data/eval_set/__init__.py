"""Evaluation set sub-package.

Contains the 100-question benchmark used to score all six LLM approaches.
Questions and sources are version-controlled JSON; this package exposes the
Pydantic schemas for loading, validating, and writing the eval set.
"""

from data.eval_set.schemas import (
    DifficultyLevel,
    EvalQuestion,
    EvalSet,
    QuestionCategory,
    SourceReference,
)

__all__ = [
    "DifficultyLevel",
    "EvalQuestion",
    "EvalSet",
    "QuestionCategory",
    "SourceReference",
]
