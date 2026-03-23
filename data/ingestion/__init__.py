"""Data ingestion sub-package.

Handles fetching raw financial documents from SEC EDGAR and other
public sources for use in RAG and evaluation.
"""

from data.ingestion.edgar_client import EdgarClient
from data.ingestion.hf_loader import HuggingFaceLoader

__all__ = ["EdgarClient", "HuggingFaceLoader"]
