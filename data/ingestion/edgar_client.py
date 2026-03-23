"""SEC EDGAR data ingestion client.

SEC EDGAR (Electronic Data Gathering, Analysis, and Retrieval) is the
US Securities and Exchange Commission's public filing system. Every
public company is required to file financial reports (10-K annual,
10-Q quarterly) and other disclosures here. The data is free, requires
no API key — only a descriptive User-Agent header identifying the
application making requests.

This module provides EdgarClient, which downloads 10-K and 10-Q
filings for given company tickers and saves them as plain text to
the project's data/raw/ directory for downstream processing, chunking,
and use in RAG retrieval.
"""

import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_USER_AGENT = "FinancialFineTuningLab/1.0 (research-project; contact@example.com)"
_REQUEST_DELAY = 0.15  # SEC allows max 10 req/s; 0.15s keeps us safe
_COMPANY_DELAY = 1.0   # extra courtesy gap between different companies
_RETRY_DELAY = 2.0     # wait before a single retry on connection error


class EdgarClient:
    """Downloads SEC EDGAR filings for given company tickers.

    Usage::

        client = EdgarClient()
        paths = client.download_company("AAPL", form_type="10-Q", count=2)
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._cik_cache: dict[str, str] = {}
        self._log = logger

        settings.RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────

    def get_cik(self, ticker: str) -> str:
        """Resolve a stock ticker to a zero-padded 10-digit CIK.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            CIK as a zero-padded 10-digit string.

        Raises:
            ValueError: If the ticker cannot be found on EDGAR.
        """
        ticker = ticker.upper()
        if ticker in self._cik_cache:
            self._log.debug("CIK cache hit for %s", ticker)
            return self._cik_cache[ticker]

        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={ticker}&type=10-K"
            "&dateb=&owner=include&count=1&search_text=&output=atom"
        )
        response = self._get(url)
        if response is None:
            raise ValueError(f"Failed to fetch CIK for ticker {ticker}")

        soup = BeautifulSoup(response.text, "lxml")
        cik_tag = soup.find("cik")
        if cik_tag is None:
            raise ValueError(
                f"Ticker '{ticker}' not found on SEC EDGAR"
            )

        cik = cik_tag.get_text().strip().zfill(10)
        self._cik_cache[ticker] = cik
        self._log.info("Resolved %s → CIK %s", ticker, cik)
        return cik

    def get_filings(
        self,
        cik: str,
        form_type: str = "10-K",
        count: int = 5,
    ) -> list[dict]:
        """Fetch recent filing metadata for a given CIK.

        Args:
            cik: Zero-padded 10-digit CIK string.
            form_type: SEC form type to filter (e.g. "10-K", "10-Q").
            count: Maximum number of filings to return.

        Returns:
            List of dicts with keys: accession_number, filing_date,
            form_type, primary_document — sorted by date descending.
        """
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = self._get(url)
        if response is None:
            self._log.warning("Could not fetch submissions for CIK %s", cik)
            return []

        data = response.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        documents = recent.get("primaryDocument", [])

        filings: list[dict] = []
        for i, form in enumerate(forms):
            if form == form_type:
                filings.append(
                    {
                        "accession_number": accessions[i],
                        "filing_date": dates[i],
                        "form_type": form,
                        "primary_document": documents[i],
                    }
                )
            if len(filings) >= count:
                break

        self._log.info(
            "Found %d %s filings for CIK %s", len(filings), form_type, cik
        )
        return filings

    def download_filing(
        self,
        cik: str,
        accession_number: str,
        ticker: str,
        filing_date: str,
        form_type: str,
    ) -> Path:
        """Download a single filing document and save it locally.

        Args:
            cik: Zero-padded 10-digit CIK.
            accession_number: EDGAR accession number (e.g. "0000320193-23-000077").
            ticker: Stock ticker for the output filename.
            filing_date: Filing date string for the output filename.
            form_type: Form type string for the output filename.

        Returns:
            Path to the saved file.
        """
        safe_date = filing_date.replace("/", "-")
        out_path = settings.RAW_DIR / f"{ticker}_{form_type}_{safe_date}.txt"

        if out_path.exists():
            self._log.info("Already exists: %s", out_path)
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Accession number without dashes for URL path construction
        acc_no_dashes = accession_number.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}"
            f"/{acc_no_dashes}/{accession_number}-index.htm"
        )

        # Fetch the filing index page to locate the primary document
        cik_int = str(int(cik))
        index_response = self._get(index_url)
        primary_doc_url: str | None = None

        if index_response is not None:
            primary_doc_url = self._select_document_from_index(
                index_response.text, cik_int, acc_no_dashes, form_type, ticker
            )

        # Guaranteed fallback: complete submission text file
        if primary_doc_url is None:
            primary_doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
                f"/{acc_no_dashes}/{acc_no_dashes}.txt"
            )
            self._log.info(
                "Using complete submission fallback for %s", ticker
            )

        doc_response = self._get(primary_doc_url)
        if doc_response is None:
            self._log.warning(
                "Failed to download document for %s %s %s",
                ticker,
                form_type,
                filing_date,
            )
            return out_path

        # Strip HTML tags to get plain text, removing iXBRL metadata first
        doc_soup = BeautifulSoup(doc_response.text, "lxml")
        text = self._extract_clean_text(doc_soup)

        out_path.write_text(text, encoding="utf-8")
        self._log.info(
            "Downloaded %s %s %s → %s", ticker, form_type, filing_date, out_path
        )
        return out_path

    def download_company(
        self,
        ticker: str,
        form_type: str = "10-K",
        count: int = 3,
    ) -> list[Path]:
        """Download multiple filings for a single company.

        Args:
            ticker: Stock ticker symbol.
            form_type: SEC form type to fetch.
            count: Number of most-recent filings to download.

        Returns:
            List of Paths to downloaded files.
        """
        ticker = ticker.upper()
        self._log.info("Starting download for %s (%s × %d)", ticker, form_type, count)

        try:
            cik = self.get_cik(ticker)
        except ValueError as exc:
            self._log.warning("Skipping %s: %s", ticker, exc)
            return []

        filings = self.get_filings(cik, form_type=form_type, count=count)
        if not filings:
            self._log.warning("No %s filings found for %s", form_type, ticker)
            return []

        paths: list[Path] = []
        for filing in filings:
            try:
                path = self.download_filing(
                    cik=cik,
                    accession_number=filing["accession_number"],
                    ticker=ticker,
                    filing_date=filing["filing_date"],
                    form_type=filing["form_type"],
                )
                paths.append(path)
            except Exception:
                self._log.exception(
                    "Failed to download %s filing dated %s",
                    ticker,
                    filing["filing_date"],
                )

        self._log.info(
            "Completed %s: %d/%d filings downloaded",
            ticker,
            len(paths),
            len(filings),
        )
        return paths

    def download_all(
        self,
        tickers: list[str],
        form_type: str = "10-K",
        count: int = 2,
    ) -> dict[str, list[Path]]:
        """Download filings for multiple companies.

        Args:
            tickers: List of stock ticker symbols.
            form_type: SEC form type to fetch.
            count: Number of filings per company.

        Returns:
            Dict mapping each ticker to its list of downloaded file Paths.
        """
        results: dict[str, list[Path]] = {}
        total_files = 0

        for i, ticker in enumerate(tickers):
            if i > 0:
                time.sleep(_COMPANY_DELAY)

            paths = self.download_company(
                ticker, form_type=form_type, count=count
            )
            results[ticker.upper()] = paths
            total_files += len(paths)

        self._log.info(
            "Downloaded %d files for %d companies", total_files, len(tickers)
        )
        return results

    # ── internal helpers ──────────────────────────────────────────────

    def _get(self, url: str) -> requests.Response | None:
        """Issue a GET request with rate-limiting and retry logic.

        Returns None on unrecoverable failure instead of raising.
        """
        time.sleep(_REQUEST_DELAY)

        for attempt in range(2):  # original + 1 retry
            try:
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                return resp

            except requests.exceptions.ConnectionError:
                if attempt == 0:
                    self._log.warning(
                        "Connection error for %s — retrying in %.1fs",
                        url,
                        _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                else:
                    self._log.error("Connection failed after retry: %s", url)
                    return None

            except requests.exceptions.Timeout:
                self._log.warning("Request timed out: %s", url)
                return None

            except requests.exceptions.HTTPError as exc:
                self._log.warning("HTTP %s for %s", exc.response.status_code, url)
                return None

        return None

    # Tag names whose entire subtree should be removed before text
    # extraction.  These carry iXBRL metadata (hidden context elements,
    # taxonomy references) that produce garbage when get_text() is called.
    _IXBRL_STRIP_TAGS = re.compile(
        r"^(ix:header|ix:hidden|ix:resources|ix:references"
        r"|xbrli:context|xbrli:unit|link:schemaref)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_clean_text(soup: BeautifulSoup) -> str:
        """Extract human-readable text from a filing HTML document.

        Modern EDGAR filings are Inline XBRL (iXBRL): the .htm file
        contains both readable HTML *and* embedded ``<ix:header>`` /
        ``<ix:hidden>`` sections with XBRL context metadata (entity
        info, period dates, unit definitions, ``xbrli:shares`` values,
        etc.).  A naive ``get_text()`` merges that metadata into the
        output, producing lines like
        ``"aapl-20250628 / false / 2025 / Q3 / xbrli:shares ..."``.

        This method strips all iXBRL/XBRL metadata elements *before*
        extracting text, so only human-readable prose, tables, and
        financial figures remain.

        Args:
            soup: Parsed BeautifulSoup tree of the filing document.

        Returns:
            Cleaned plain-text string with readable financial content.
        """
        # 1. Remove iXBRL metadata subtrees entirely.  These include
        #    <ix:header> (contains <ix:hidden>, <ix:resources>,
        #    <ix:references>), and any top-level XBRL context/unit
        #    definitions.  Tag names use namespace prefixes so we
        #    match with a regex.
        for tag in soup.find_all(EdgarClient._IXBRL_STRIP_TAGS):
            tag.decompose()

        # 2. Remove <script> and <style> blocks — no useful text.
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        # 3. Inline ix: value tags (<ix:nonfraction>, <ix:nonnumeric>)
        #    wrap actual financial figures/text in the body.  Unwrap
        #    them so their text content is preserved but the tags
        #    themselves don't appear in output.
        for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
            tag.unwrap()

        # 4. Extract text and collapse excessive blank lines.
        raw = soup.get_text(separator="\n", strip=True)
        lines = raw.splitlines()
        cleaned: list[str] = []
        blank_run = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_run += 1
                if blank_run <= 2:
                    cleaned.append("")
            else:
                blank_run = 0
                cleaned.append(stripped)

        return "\n".join(cleaned).strip()

    # Filenames to reject: XBRL viewers, R-numbered taxonomy files
    _SKIP_FILENAME = re.compile(
        r"(xbrl|viewer|R\d)", re.IGNORECASE
    )

    def _select_document_from_index(
        self,
        index_html: str,
        cik_int: str,
        acc_no_dashes: str,
        form_type: str,
        ticker: str,
    ) -> str | None:
        """Parse the filing index table and select the best document URL.

        The EDGAR index page contains a ``<table class="tableFile">``
        with columns: Seq, Description, Document, Type, Size.  The
        primary human-readable filing is the Seq-1 row whose Type
        matches the form (e.g. "10-Q").  Its ``<a>`` href goes through
        the ``/ix?doc=`` iXBRL viewer — we must extract the *real*
        document path from that href rather than following the viewer
        link directly.

        Strategy:
          1. Find the row where Type == form_type, filename ends in
             .htm/.html, and filename does not match XBRL/viewer
             patterns.  Extract the direct Archives URL from the
             ``/ix?doc=`` href (or from the filename if the href
             does not use the viewer prefix).
          2. If no suitable .htm row is found, return None so the
             caller falls back to the complete-submission .txt file.

        Args:
            index_html: Raw HTML of the filing index page.
            cik_int: CIK with leading zeros stripped (for URL building).
            acc_no_dashes: Accession number with dashes removed.
            form_type: The form type to match (e.g. "10-Q", "10-K").
            ticker: Ticker symbol (for log messages only).

        Returns:
            Absolute URL to the document, or None when no suitable
            document is found in the index table.
        """
        soup = BeautifulSoup(index_html, "lxml")

        # The page has two tables with class="tableFile":
        #   1. "Document Format Files" — contains the actual filings
        #   2. "Data Files" — contains XBRL taxonomy/schema files
        # We only want the first table.
        table = soup.find("table", class_="tableFile",
                          summary="Document Format Files")
        if table is None:
            # Fallback: grab the first tableFile if summary attr missing
            table = soup.find("table", class_="tableFile")
        if table is None:
            self._log.debug("No tableFile found in index for %s", ticker)
            return None

        base_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}"
        )

        for row in table.find_all("tr")[1:]:  # skip header row
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            doc_type = cells[3].get_text(strip=True)
            if doc_type != form_type:
                continue

            link = cells[2].find("a", href=True)
            if link is None:
                continue

            filename = link.get_text(strip=True)

            if not filename.lower().endswith((".htm", ".html")):
                continue

            if self._SKIP_FILENAME.search(filename):
                continue

            # The href may be a direct Archives path or go through the
            # iXBRL viewer ("/ix?doc=/Archives/...").  Extract the real
            # document path so we fetch raw HTML, not the JS viewer.
            href: str = link["href"]
            if "/ix?doc=" in href:
                # Extract the real path after /ix?doc=
                real_path = href.split("/ix?doc=", 1)[1]
                doc_url = f"https://www.sec.gov{real_path}"
            elif href.startswith("/Archives/"):
                doc_url = f"https://www.sec.gov{href}"
            elif href.startswith("http"):
                doc_url = href
            else:
                doc_url = f"{base_url}/{filename}"

            self._log.info(
                "Using primary %s document: %s", form_type, filename
            )
            return doc_url

        self._log.debug(
            "No clean %s .htm document found in index for %s",
            form_type,
            ticker,
        )
        return None
