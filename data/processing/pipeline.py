"""Data processing pipeline for Financial Fine-Tuning Laboratory.

Orchestrates the full flow: load HuggingFace datasets, clean and chunk
EDGAR filings, and persist everything to ``data/processed/`` as JSON
ready for RAG indexing and benchmark evaluation.

Usage::

    python3 -m data.processing.pipeline
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from data.ingestion.hf_loader import HuggingFaceLoader
from data.processing.chunker import TextChunker
from data.processing.cleaner import TextCleaner

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def _save_json(data: list | dict, path: Path) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Saved %s", path)
    except OSError:
        logger.exception("Failed to write %s", path)


def main() -> None:
    """Run the complete processing pipeline."""
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load HuggingFace datasets ─────────────────────────────
    logger.info("Step 1/5 — Loading HuggingFace datasets")
    hf = HuggingFaceLoader()
    hf_data = hf.load_all()

    fpb = hf_data.get("financial_phrasebank", [])
    fiqa = hf_data.get("fiqa", [])
    logger.info(
        "HuggingFace: %d FinancialPhraseBank sentences, %d FiQA pairs",
        len(fpb),
        len(fiqa),
    )

    # ── Step 2: Clean and chunk EDGAR files ───────────────────────────
    logger.info("Step 2/5 — Cleaning and chunking EDGAR filings")
    cleaner = TextCleaner()
    chunker = TextChunker()
    edgar_chunks = chunker.chunk_all_edgar(settings.RAW_DIR, cleaner)

    # ── Step 3: Save EDGAR chunks ─────────────────────────────────────
    logger.info("Step 3/5 — Saving EDGAR chunks")
    chunks_path = settings.PROCESSED_DIR / "edgar_chunks.json"
    _save_json(edgar_chunks, chunks_path)

    # ── Step 4: Save HF datasets to processed dir ────────────────────
    logger.info("Step 4/5 — Saving HuggingFace datasets to processed dir")

    fpb_path = settings.PROCESSED_DIR / "financial_phrasebank.json"
    _save_json(fpb, fpb_path)

    fiqa_path = settings.PROCESSED_DIR / "fiqa.json"
    _save_json(fiqa, fiqa_path)

    # ── Step 5: Build and save processing summary ─────────────────────
    logger.info("Step 5/5 — Building processing summary")

    # Compute per-ticker chunk counts
    ticker_counts: Counter[str] = Counter()
    chunks_per_file: Counter[str] = Counter()
    for chunk in edgar_chunks:
        ticker = chunk.get("metadata", {}).get("ticker", "unknown")
        ticker_counts[ticker] += 1
        chunks_per_file[chunk["doc_id"]] += 1

    file_chunk_counts = list(chunks_per_file.values()) if chunks_per_file else [0]
    n_files = len(chunks_per_file)

    summary = {
        "processed_at": datetime.now(tz=timezone.utc).isoformat(),
        "edgar_files_processed": n_files,
        "total_edgar_chunks": len(edgar_chunks),
        "avg_chunks_per_file": round(
            len(edgar_chunks) / n_files, 1
        ) if n_files else 0.0,
        "min_chunks_per_file": min(file_chunk_counts),
        "max_chunks_per_file": max(file_chunk_counts),
        "financial_phrasebank_count": len(fpb),
        "fiqa_count": len(fiqa),
        "chunk_size_words": settings.CHUNK_SIZE,
        "chunk_overlap_words": settings.CHUNK_OVERLAP,
        "chunks_per_ticker": dict(sorted(ticker_counts.items())),
    }

    summary_path = settings.PROCESSED_DIR / "processing_summary.json"
    _save_json(summary, summary_path)

    logger.info("Processing summary:")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
