"""Synthetic data sub-package.

Manages teacher-model-generated training pairs for SFT and DPO.
Prompts are version-controlled; generated outputs live in
data/synthetic/generated/ (gitignored).
"""

from data.synthetic.formatter import DatasetFormatter
from data.synthetic.generator import SyntheticDataGenerator
from data.synthetic.quality_filter import QualityFilter

__all__ = [
    "DatasetFormatter",
    "QualityFilter",
    "SyntheticDataGenerator",
]
