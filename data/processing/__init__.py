"""Data processing sub-package.

Handles cleaning, chunking, and embedding of raw financial documents
for storage in ChromaDB and use in RAG retrieval.
"""

from data.processing.chunker import TextChunker
from data.processing.cleaner import TextCleaner

__all__ = ["TextCleaner", "TextChunker"]
