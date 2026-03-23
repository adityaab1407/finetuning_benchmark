"""Text cleaner for raw financial documents.

SEC EDGAR filings arrive with HTML/XML artifacts, XBRL inline tags,
boilerplate legal disclaimers, repeated headers/footers, and irregular
whitespace.  HuggingFace sentences are mostly clean but may contain
HTML entities and extra whitespace.

This module provides :class:`TextCleaner` which normalises both kinds
of input into plain, human-readable text ready for chunking and
embedding.
"""

import html
import logging
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress the XMLParsedAsHTMLWarning globally — EDGAR files often
# contain mixed HTML/XML and BeautifulSoup's warning is just noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# ── Boilerplate phrases (case-insensitive match) ──────────────────────
_BOILERPLATE_PHRASES: list[str] = [
    "table of contents",
    "see accompanying notes",
    "see notes to condensed consolidated",
    "signatures",
    "pursuant to the requirements of the securities exchange act",
]


class TextCleaner:
    """Clean raw financial text for downstream NLP consumption."""

    def __init__(self) -> None:
        self._log = logger

        # Pre-compile all regex patterns once
        self._re_xbrl_tags = re.compile(
            r"<[^>]*(ix:|xbrl|contextref=)[^>]*>", re.IGNORECASE
        )
        self._re_tag_fragments = re.compile(r"<[^>]+>")
        self._re_multi_newlines = re.compile(r"\n{3,}")
        self._re_multi_spaces = re.compile(r" {3,}")
        self._re_tab_cr = re.compile(r"[\t\r]")
        self._re_punct_only = re.compile(r"^[\d\s\W]+$")

        self._boilerplate_lower = [p.lower() for p in _BOILERPLATE_PHRASES]

    # ── public API ────────────────────────────────────────────────────

    def clean_edgar(self, raw_text: str) -> str:
        """Clean a raw SEC EDGAR filing into readable plain text.

        Applies, in order: HTML/XML stripping, XBRL artifact removal,
        boilerplate deletion, whitespace normalisation, and short-line
        removal.

        Args:
            raw_text: The raw file contents of an EDGAR filing.

        Returns:
            Cleaned plain-text string.
        """
        # Step 1 — Strip HTML/XML tags via BeautifulSoup
        soup = BeautifulSoup(raw_text, "lxml")
        text = soup.get_text(separator="\n")

        # Step 2 — Remove leftover XBRL artifacts and tag fragments
        text = self._re_xbrl_tags.sub("", text)
        text = self._re_tag_fragments.sub("", text)

        # Step 3 — Remove SEC boilerplate lines
        lines = text.splitlines()
        lines = [
            line
            for line in lines
            if not self._is_boilerplate(line)
        ]
        text = "\n".join(lines)

        # Step 4 — Normalize whitespace
        text = self._re_tab_cr.sub(" ", text)
        text = self._re_multi_newlines.sub("\n\n", text)
        text = self._re_multi_spaces.sub(" ", text)

        cleaned_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            # Remove lines that are only punctuation/numbers (table separators)
            if stripped and self._re_punct_only.match(stripped):
                continue
            cleaned_lines.append(stripped)
        text = "\n".join(cleaned_lines)

        # Step 5 — Drop very short lines (stray headers, page numbers)
        lines = text.splitlines()
        lines = [line for line in lines if len(line) >= 20 or line == ""]
        text = "\n".join(lines)

        # Collapse runs of blank lines produced by removals
        text = self._re_multi_newlines.sub("\n\n", text)

        # Step 6 — Final strip
        return text.strip()

    def clean_sentence(self, text: str) -> str:
        """Clean a single sentence from FinancialPhraseBank / HF data.

        Lighter cleaning: whitespace normalisation and HTML entity
        decoding only.

        Args:
            text: A single sentence string.

        Returns:
            Cleaned sentence.
        """
        text = html.unescape(text)
        text = " ".join(text.split())
        return text.strip()

    def clean_batch(self, texts: list[str], mode: str = "edgar") -> list[str]:
        """Clean a list of texts using the specified mode.

        Args:
            texts: Raw text strings to clean.
            mode: ``"edgar"`` for SEC filings, ``"sentence"`` for
                  short HuggingFace sentences.

        Returns:
            List of cleaned strings in the same order.
        """
        fn = self.clean_edgar if mode == "edgar" else self.clean_sentence
        results = [fn(t) for t in texts]
        self._log.info("Cleaned %d texts in %s mode", len(results), mode)
        return results

    # ── internal helpers ──────────────────────────────────────────────

    def _is_boilerplate(self, line: str) -> bool:
        """Return True if *line* matches a known boilerplate phrase."""
        lower = line.strip().lower()
        return any(phrase in lower for phrase in self._boilerplate_lower)
