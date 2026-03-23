"""Download SEC EDGAR filings for benchmark target companies.

Standalone script that fetches 10-Q filings for 10 major public
companies and saves them to data/raw/. Produces a manifest.json
summarizing what was downloaded.

Usage::

    python3 -m data.ingestion.download_transcripts
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from data.ingestion.edgar_client import EdgarClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "NVDA", "TSLA", "JPM", "V", "NFLX",
]
FORM_TYPE = "10-Q"
FILINGS_PER_COMPANY = 2


def build_manifest(
    results: dict[str, list[Path]],
    tickers: list[str],
    form_type: str,
) -> dict:
    """Build the manifest dict summarizing all downloads.

    Args:
        results: Mapping of ticker → list of downloaded file Paths.
        tickers: Original list of requested tickers.
        form_type: SEC form type that was fetched.

    Returns:
        Manifest dict ready for JSON serialization.
    """
    total = sum(len(paths) for paths in results.values())
    failed = [t for t in tickers if not results.get(t)]

    return {
        "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
        "tickers": tickers,
        "form_type": form_type,
        "files": {
            ticker: [str(p) for p in paths]
            for ticker, paths in results.items()
        },
        "total_files": total,
        "failed_tickers": failed,
    }


def print_summary(results: dict[str, list[Path]]) -> None:
    """Log a summary table of download results.

    Args:
        results: Mapping of ticker → list of downloaded file Paths.
    """
    header = f"{'Ticker':<8} {'Files':<7} {'Paths':<60} {'Status'}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for ticker, paths in results.items():
        count = len(paths)
        status = "OK" if count > 0 else "FAILED"
        path_str = ", ".join(str(p) for p in paths) if paths else "—"
        # Truncate long path strings for readability
        if len(path_str) > 58:
            path_str = path_str[:55] + "..."
        logger.info("%-8s %-7d %-60s %s", ticker, count, path_str, status)

    logger.info("=" * len(header))


def main() -> None:
    """Run the full download pipeline for all target tickers."""
    logger.info(
        "Starting EDGAR download: %d tickers, %s, %d filings each",
        len(TICKERS),
        FORM_TYPE,
        FILINGS_PER_COMPANY,
    )

    client = EdgarClient()
    results = client.download_all(
        tickers=TICKERS,
        form_type=FORM_TYPE,
        count=FILINGS_PER_COMPANY,
    )

    # Print summary table
    print_summary(results)

    # Save manifest
    manifest = build_manifest(results, TICKERS, FORM_TYPE)
    manifest_path = settings.RAW_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()
