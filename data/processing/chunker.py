"""Text chunker for financial documents.

Long SEC filings (a 10-Q can be 50 000+ words) must be split into
smaller overlapping pieces before they can be used for RAG retrieval
or fit within LLM context windows.

The chunker uses a sliding-window approach measured in **words** (not
characters).  Consecutive chunks share ``chunk_overlap`` words at
their boundary so that sentences straddling a split point are never
lost entirely.
"""

import logging
from pathlib import Path

from config import settings
from data.processing.cleaner import TextCleaner

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_MIN_CHUNK_WORDS = 50  # drop tail fragments shorter than this


class TextChunker:
    """Split cleaned text into overlapping word-level chunks."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._stride = chunk_size - chunk_overlap
        self._log = logger

    # ── public API ────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        source: str,
        metadata: dict | None = None,
    ) -> list[dict]:
        """Split *text* into overlapping word-level chunks.

        Args:
            text: Cleaned plain-text document.
            doc_id: Unique identifier for the source document.
            source: Human-readable source name (e.g. ``"AAPL_10-Q_2025-08-01"``).
            metadata: Optional extra metadata attached to every chunk.

        Returns:
            List of chunk dicts.  Each contains *chunk_id*, *doc_id*,
            *source*, *text*, *word_count*, *chunk_index*,
            *total_chunks*, and *metadata*.
        """
        words = text.split()
        if not words:
            return []

        raw_chunks: list[list[str]] = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            window = words[start:end]
            raw_chunks.append(window)
            start += self._stride

        # Filter short tail fragments
        raw_chunks = [c for c in raw_chunks if len(c) >= _MIN_CHUNK_WORDS]

        total = len(raw_chunks)
        extra = metadata or {}
        results: list[dict] = []
        for idx, chunk_words in enumerate(raw_chunks):
            results.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{idx:04d}",
                    "doc_id": doc_id,
                    "source": source,
                    "text": " ".join(chunk_words),
                    "word_count": len(chunk_words),
                    "chunk_index": idx,
                    "total_chunks": total,
                    "metadata": extra,
                }
            )

        return results

    def chunk_file(
        self,
        file_path: Path,
        cleaner: TextCleaner,
    ) -> list[dict]:
        """Read, clean, and chunk a single EDGAR filing.

        Args:
            file_path: Path to a raw ``.txt`` filing in ``data/raw/``.
            cleaner: :class:`TextCleaner` instance for preprocessing.

        Returns:
            List of chunk dicts produced from the file.
        """
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        cleaned = cleaner.clean_edgar(raw_text)

        stem = file_path.stem
        parts = stem.split("_")
        ticker = parts[0] if len(parts) > 0 else ""
        form_type = parts[1] if len(parts) > 1 else ""
        filing_date = parts[2] if len(parts) > 2 else ""

        meta = {
            "file_path": str(file_path),
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": filing_date,
        }

        chunks = self.chunk_text(
            text=cleaned, doc_id=stem, source=stem, metadata=meta
        )
        self._log.info(
            "Chunked %s → %d chunks (%d words each, %d overlap)",
            stem,
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks

    def chunk_all_edgar(
        self,
        raw_dir: Path,
        cleaner: TextCleaner,
    ) -> list[dict]:
        """Chunk every EDGAR ``.txt`` file in *raw_dir*.

        Skips non-filing files (``manifest.json``,
        ``financial_phrasebank.json``, ``fiqa.json``).

        Args:
            raw_dir: Directory containing raw EDGAR ``.txt`` files.
            cleaner: :class:`TextCleaner` instance.

        Returns:
            Flat list of all chunk dicts across all files.
        """
        skip_names = {"manifest.json", "financial_phrasebank.json", "fiqa.json"}
        txt_files = sorted(
            p for p in raw_dir.iterdir()
            if p.suffix == ".txt" and p.name not in skip_names
        )

        all_chunks: list[dict] = []
        for path in txt_files:
            chunks = self.chunk_file(path, cleaner)
            all_chunks.extend(chunks)

        self._log.info(
            "Chunked %d files → %d total chunks", len(txt_files), len(all_chunks)
        )
        return all_chunks
