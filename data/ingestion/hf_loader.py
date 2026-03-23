"""HuggingFace financial dataset loader.

Downloads and caches two financial NLP datasets from HuggingFace Hub:

* **FinancialPhraseBank** — ~5 000 sentences labeled positive / neutral /
  negative by finance domain experts.  Used to build evaluation questions
  that test sentiment understanding.
* **FiQA** — financial question-answer pairs from analyst reports and
  earnings calls.  Used for Q&A evaluation and few-shot exemplars.

Both datasets are saved as local JSON on first download so subsequent
runs never hit the network.
"""

import json
import logging
import os
from pathlib import Path

from huggingface_hub import login

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


class HuggingFaceLoader:
    """Load and cache financial datasets from HuggingFace Hub."""

    def __init__(self) -> None:
        self._log = logger
        settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
        token = settings.HF_TOKEN
        if token:
            login(token=token, add_to_git_credential=False)
            self._log.info("Authenticated with HuggingFace Hub")
        else:
            self._log.warning("HF_TOKEN not set — using unauthenticated requests")

    # ── public API ────────────────────────────────────────────────────

    def load_financial_phrasebank(self) -> list[dict]:
        """Load FinancialPhraseBank via the Parquet mirror dataset.

        Uses ``warwickai/financial_phrasebank_mirror`` — a Parquet
        rehost of the original takala/financial_phrasebank corpus that
        works with all modern versions of the ``datasets`` library
        (no custom script required).  Contains 4 846 financial sentences
        labeled by domain experts.

        Returns:
            List of dicts with keys: id, sentence, label, source.
        """
        out_path = settings.RAW_DIR / "financial_phrasebank.json"

        cached = self._load_cached(out_path)
        if cached is not None:
            return cached

        try:
            from datasets import load_dataset

            ds = load_dataset(
                "warwickai/financial_phrasebank_mirror",
                split="train",
            )
        except Exception:
            self._log.exception("Failed to download FinancialPhraseBank")
            return []

        _label_map: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

        records: list[dict] = []
        for idx, item in enumerate(ds):
            label_str = _label_map.get(item["label"], "neutral")
            records.append(
                {
                    "id": f"fpb_{idx}",
                    "sentence": item["sentence"],
                    "label": label_str,
                    "source": "financial_phrasebank",
                }
            )

        self._save_json(records, out_path)
        self._log.info(
            "Loaded %d sentences from FinancialPhraseBank", len(records)
        )
        return records

    def load_fiqa(self) -> list[dict]:
        """Load FiQA question-answer pairs (train split, first 500).

        Returns:
            List of dicts with keys: id, question, answer, source.
        """
        out_path = settings.RAW_DIR / "fiqa.json"

        cached = self._load_cached(out_path)
        if cached is not None:
            return cached

        try:
            from datasets import load_dataset

            ds = load_dataset("LLukas22/fiqa", split="train")
        except Exception:
            self._log.exception("Failed to download FiQA")
            return []

        records: list[dict] = []
        limit = min(500, len(ds))
        for idx in range(limit):
            item = ds[idx]
            records.append(
                {
                    "id": f"fiqa_{idx}",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "source": "fiqa",
                }
            )

        self._save_json(records, out_path)
        self._log.info("Loaded %d Q&A pairs from FiQA", len(records))
        return records

    def load_all(self) -> dict[str, list[dict]]:
        """Load all HuggingFace datasets (cached when possible).

        Returns:
            Dict mapping dataset name to its list of records.
        """
        result = {
            "financial_phrasebank": self.load_financial_phrasebank(),
            "fiqa": self.load_fiqa(),
        }
        self._log.info("All HuggingFace datasets loaded")
        return result

    # ── internal helpers ──────────────────────────────────────────────

    def _load_cached(self, path: Path) -> list[dict] | None:
        """Return cached records from *path*, or None if unavailable."""
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                self._log.info("Loaded cached dataset from %s", path)
                return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._log.warning("Corrupted cache at %s — will re-download", path)
            path.unlink(missing_ok=True)

        return None

    def _save_json(self, data: list[dict], path: Path) -> None:
        """Persist *data* as JSON to *path*."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._log.info("Saved %d records to %s", len(data), path)
        except OSError:
            self._log.exception("Failed to save JSON to %s", path)
