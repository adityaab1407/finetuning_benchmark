"""ChromaDB ingestion script for the RAG approach.

Forces a fresh ingestion of all EDGAR chunks into ChromaDB, verifies
the collection count, and runs a test retrieval query to confirm
semantic search is working.

Usage::

    python3 -m approaches.ingest
"""

import logging
import sys

from approaches.rag import RAGApproach

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_TEST_QUERY = "What was Apple's revenue in the most recent quarter?"


def main() -> None:
    """Ingest chunks, verify, and run a test retrieval."""
    logger.info("Starting fresh ChromaDB ingestion...")

    rag = RAGApproach()
    rag.setup(force_reingest=True)

    # Verify collection count
    count = rag.collection.count()
    logger.info("Collection contains %d chunks", count)

    if count == 0:
        logger.error("Ingestion produced an empty collection — aborting")
        sys.exit(1)

    # Run test retrieval
    logger.info("Test query: '%s'", _TEST_QUERY)
    chunks = rag.retrieve(_TEST_QUERY, n_results=3)

    logger.info("")
    logger.info("Top %d retrieved chunks:", len(chunks))
    logger.info("-" * 80)
    for i, chunk in enumerate(chunks, start=1):
        logger.info(
            "  [%d] source=%-30s  ticker=%-6s  date=%-12s  relevance=%.4f",
            i,
            chunk["source"],
            chunk["ticker"],
            chunk["filing_date"],
            chunk["relevance_score"],
        )
        # Show first 150 chars of text
        preview = chunk["text"][:150].replace("\n", " ")
        logger.info("      text: %s...", preview)
    logger.info("-" * 80)

    logger.info("")
    logger.info("INGESTION COMPLETE — %d chunks indexed", count)


if __name__ == "__main__":
    main()
