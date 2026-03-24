"""Retrieval-Augmented Generation (RAG) approach for Financial Fine-Tuning Laboratory.

RAG is the only approach that reads actual documents at query time.
Zero-shot, few-shot, and chain-of-thought rely entirely on knowledge
the model absorbed during pre-training.  RAG retrieves the most
relevant chunks from 20 SEC 10-Q filings stored in ChromaDB, then
passes them as context to the LLM alongside the question.

**How it works:**

1. At ingestion time, each of the 1 182 EDGAR chunks is embedded by
   ChromaDB's built-in model (all-MiniLM-L6-v2) and stored in a
   persistent vector collection.
2. At query time the question is embedded with the same model.
   ChromaDB returns the top-k nearest chunks by cosine similarity.
3. The retrieved chunks are formatted into a prompt that instructs the
   LLM to answer *only* from the provided context.

**When RAG is expected to win:** factual extraction and structured
output questions about specific 10-Q filings — the retrieved context
contains the exact numbers.

**When RAG may lose:** sentiment classification (no document context
needed) and broad reasoning questions where the answer is not
localized in a single chunk.  A fine-tuned model that has
internalized financial patterns may outperform RAG on these.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from approaches.base import ApproachResult, BaseApproach
from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant with access to "
    "recent SEC 10-Q filings. Answer questions using ONLY "
    "the provided context from the filings. If the answer "
    "is not in the provided context, say 'Not found in "
    "provided context.' For factual questions, provide the "
    "specific figure from the context. For sentiment, "
    "respond with exactly one word: positive, negative, or "
    "neutral. For JSON extraction, return only valid JSON. "
    "For reasoning questions, show your calculation using "
    "the figures in the context."
)

_INGEST_BATCH_SIZE = 100


class RAGApproach(BaseApproach):
    """Retrieve relevant EDGAR chunks from ChromaDB, then generate an answer."""

    def __init__(self) -> None:
        super().__init__(
            approach_name="rag",
            model=settings.MODEL_RAG,
        )
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
        )
        self.collection: chromadb.Collection | None = None
        self.chunks_path: Path = settings.PROCESSED_DIR / "edgar_chunks.json"
        self._ingested: bool = False
        self._last_retrieved_chunks: list[dict] = []

    # ── setup & ingestion ─────────────────────────────────────────────

    def setup(self, force_reingest: bool = False) -> None:
        """Load or create the ChromaDB collection, ingesting chunks if needed.

        Idempotent: calling repeatedly with ``force_reingest=False``
        is a no-op once the collection is loaded.

        Args:
            force_reingest: If True, delete and rebuild the collection
                from scratch even if it already exists.
        """
        collection_name = settings.CHROMA_COLLECTION_NAME

        if force_reingest:
            try:
                self.chroma_client.delete_collection(name=collection_name)
                self._log.info("Deleted existing collection '%s' for reingest", collection_name)
            except Exception:
                pass  # collection did not exist yet — nothing to delete

        # Try loading an existing collection
        if not force_reingest:
            try:
                self.collection = self.chroma_client.get_collection(
                    name=collection_name,
                )
                count = self.collection.count()
                if count > 0:
                    self._log.info(
                        "Loaded existing ChromaDB collection: %d chunks",
                        count,
                    )
                    self._ingested = True
                    return
            except ValueError:
                pass  # collection does not exist yet

        # Create collection and ingest
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._ingest_chunks()

    def _ingest_chunks(self) -> None:
        """Load EDGAR chunks from JSON and add them to ChromaDB in batches."""
        raw = self.chunks_path.read_text(encoding="utf-8")
        chunks: list[dict] = json.loads(raw)
        total = len(chunks)
        self._log.info("Starting ingestion of %d chunks...", total)

        for start in range(0, total, _INGEST_BATCH_SIZE):
            batch = chunks[start : start + _INGEST_BATCH_SIZE]
            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "source": c["source"],
                        "ticker": c["metadata"]["ticker"],
                        "filing_date": c["metadata"]["filing_date"],
                        "chunk_index": c["chunk_index"],
                        "word_count": c["word_count"],
                    }
                    for c in batch
                ],
            )
            ingested = min(start + _INGEST_BATCH_SIZE, total)
            if ingested % 200 == 0 or ingested == total:
                self._log.info("Ingested %d/%d chunks...", ingested, total)

        self._log.info(
            "ChromaDB ingestion complete: %d chunks in collection '%s'",
            total,
            settings.CHROMA_COLLECTION_NAME,
        )
        self._ingested = True

    # ── retrieval ─────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
    ) -> list[dict]:
        """Query ChromaDB for the most relevant chunks.

        Args:
            question: The question to embed and search with.
            n_results: Number of chunks to return.  Defaults to
                ``settings.TOP_K_CHUNKS`` (3).

        Returns:
            List of chunk dicts sorted by relevance (descending),
            each containing *text*, *source*, *ticker*, *filing_date*,
            *distance*, and *relevance_score*.
        """
        if n_results is None:
            n_results = settings.TOP_K_CHUNKS

        results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        chunks: list[dict] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "ticker": meta.get("ticker", ""),
                    "filing_date": meta.get("filing_date", ""),
                    "distance": dist,
                    "relevance_score": 1.0 - dist,
                }
            )

        chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
        self._last_retrieved_chunks = chunks
        return chunks

    # ── prompt building ───────────────────────────────────────────────

    def _build_prompt(self, question: str) -> list[dict]:
        """Build the RAG prompt with retrieved context.

        Calls :meth:`retrieve` and stores results in
        ``_last_retrieved_chunks`` so that :meth:`run` can attach
        retrieval metadata without a second query.

        Args:
            question: The evaluation question text.

        Returns:
            Two-element message list for the Groq chat API.
        """
        chunks = self.retrieve(question)

        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {i}: {chunk['ticker']} {chunk['filing_date']}]\n"
                f"{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        user_content = (
            f"Context from SEC 10-Q Filings:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the context above:"
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ── run override ──────────────────────────────────────────────────

    def run(self, question: str) -> ApproachResult:
        """Execute RAG: retrieve chunks then generate an answer.

        Enriches the standard :class:`ApproachResult` metadata with
        information about the retrieved chunks.

        Args:
            question: The evaluation question text.

        Returns:
            An :class:`ApproachResult` with retrieval metadata.
        """
        if not self._ingested:
            self.setup()

        # _build_prompt (called inside super().run) will populate
        # self._last_retrieved_chunks via retrieve(), so we don't
        # need a separate retrieval call.
        result = super().run(question)

        chunks = self._last_retrieved_chunks
        result.metadata["retrieved_chunks"] = [
            {
                "source": c["source"],
                "ticker": c["ticker"],
                "relevance_score": round(c["relevance_score"], 4),
            }
            for c in chunks
        ]

        if chunks:
            top = chunks[0]
            self._log.debug(
                "Retrieved %d chunks for query, top source: %s (%.3f)",
                len(chunks),
                top["ticker"],
                top["relevance_score"],
            )

        return result
