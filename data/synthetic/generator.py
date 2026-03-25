"""Synthetic data generation for Financial Fine-Tuning Laboratory.

This module implements knowledge distillation: a large, capable teacher
model (Llama-3.3-70B via Groq) reads real financial documents and
generates instruction-response pairs that a smaller student model
(Mistral-7B) will learn from during fine-tuning.

Three prompt versions represent the iterative prompt-engineering process:

* **v1** — naive prompt, intentionally simple.  Produces generic,
  sometimes off-topic output.  This is the "before" in the Training
  Journey story.
* **v2** — structured prompt with explicit rules and question types.
  Meaningfully better quality and format compliance.
* **v3** — production prompt with few-shot examples, all four question
  categories, strict anti-hallucination rules, and length constraints.
  This version generates the actual training data.

The quality gap across versions is the point — the Streamlit dashboard
plots quality scores by version to visualize prompt-engineering impact.

Generation is fully resumable: raw outputs are checkpointed to disk
every 10 pairs so that Groq rate-limit interruptions lose at most
10 calls of work.
"""

import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from groq import Groq

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_DOLLAR_RE = re.compile(r"\$[\d,]+|\d+,\d{3}")


class SyntheticDataGenerator:
    """Generates financial instruction-response pairs using a teacher LLM.

    Args:
        model: Groq model ID for the teacher.  Defaults to
            ``settings.MODEL_TEACHER``.
    """

    def __init__(self, model: str | None = None) -> None:
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = model or settings.MODEL_TEACHER
        self.generated_dir = settings.SYNTHETIC_DIR / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir = settings.SYNTHETIC_DIR / "prompts"
        self._log = logger

    # ── source material ──────────────────────────────────────────────

    def load_source_material(self) -> dict:
        """Load and sample diverse source chunks and sentences.

        Returns:
            Dict with ``edgar_chunks`` (up to 150) and ``phrasebank``
            (90 balanced sentences).
        """
        # Load EDGAR chunks
        chunks_path = settings.PROCESSED_DIR / "edgar_chunks.json"
        all_chunks: list[dict] = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )

        # Group by ticker
        by_ticker: dict[str, list[dict]] = {}
        for c in all_chunks:
            ticker = c.get("metadata", {}).get("ticker", "UNKNOWN")
            by_ticker.setdefault(ticker, []).append(c)

        selected_chunks: list[dict] = []

        for ticker, ticker_chunks in by_ticker.items():
            # Prefer chunks with dollar figures and reasonable length
            scored = []
            for c in ticker_chunks:
                text = c.get("text", "")
                wc = c.get("word_count", len(text.split()))
                has_dollars = bool(_DOLLAR_RE.search(text))
                length_ok = 200 <= wc <= 500
                score = int(has_dollars) * 2 + int(length_ok)
                scored.append((score, c))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Take at least 2 per ticker, up to proportional share
            per_ticker = max(2, 150 // max(len(by_ticker), 1))
            for _, c in scored[:per_ticker]:
                selected_chunks.append(c)

        # Trim to 150 total
        random.seed(42)
        if len(selected_chunks) > 150:
            selected_chunks = random.sample(selected_chunks, 150)

        self._log.info(
            "Loaded %d EDGAR chunks (from %d tickers)",
            len(selected_chunks),
            len(by_ticker),
        )

        # Load phrasebank — balanced sample
        pb_path = settings.PROCESSED_DIR / "financial_phrasebank.json"
        all_sentences: list[dict] = json.loads(
            pb_path.read_text(encoding="utf-8")
        )

        by_label: dict[str, list[dict]] = {}
        for s in all_sentences:
            by_label.setdefault(s["label"], []).append(s)

        phrasebank: list[dict] = []
        for label in ("positive", "negative", "neutral"):
            pool = by_label.get(label, [])
            random.seed(42)
            sample = random.sample(pool, min(30, len(pool)))
            phrasebank.extend(sample)

        self._log.info(
            "Loaded %d phrasebank sentences (balanced)",
            len(phrasebank),
        )

        return {
            "edgar_chunks": selected_chunks,
            "phrasebank": phrasebank,
        }

    # ── prompt loading ───────────────────────────────────────────────

    def _load_prompt(self, prompt_version: str) -> dict:
        """Load a YAML prompt template by version name."""
        path = self.prompts_dir / f"{prompt_version}.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    # ── generation ───────────────────────────────────────────────────

    def generate_with_prompt(
        self,
        prompt_version: str,
        source_chunks: list[dict],
        source_sentences: list[dict],
        target_count: int = 100,
    ) -> list[dict]:
        """Generate instruction pairs using a specific prompt version.

        Resumable: loads existing raw file and generates only the
        remaining pairs.

        Args:
            prompt_version: One of ``"v1"``, ``"v2"``, ``"v3"``.
            source_chunks: EDGAR chunks to sample from.
            source_sentences: Phrasebank sentences to sample from.
            target_count: Total pairs to generate.

        Returns:
            List of raw pair dicts.
        """
        raw_path = self.generated_dir / f"{prompt_version}_raw.json"

        # Resume from existing file
        existing: list[dict] = []
        if raw_path.exists():
            existing = json.loads(raw_path.read_text(encoding="utf-8"))
            self._log.info(
                "Resuming %s: %d/%d pairs already generated",
                prompt_version,
                len(existing),
                target_count,
            )

        if len(existing) >= target_count:
            self._log.info(
                "%s already has %d pairs — skipping",
                prompt_version,
                len(existing),
            )
            return existing

        prompt_template = self._load_prompt(prompt_version)
        system_msg = prompt_template["system"]
        content_template = prompt_template["content"]

        # Combine sources for random sampling
        all_sources = []
        for c in source_chunks:
            all_sources.append(
                {
                    "text": c["text"],
                    "chunk_id": c["chunk_id"],
                    "ticker": c.get("metadata", {}).get("ticker", "UNKNOWN"),
                }
            )
        for s in source_sentences:
            all_sources.append(
                {
                    "text": s["sentence"],
                    "chunk_id": s["id"],
                    "ticker": "phrasebank",
                }
            )

        remaining = target_count - len(existing)
        start_idx = len(existing)

        self._log.info(
            "Generating %d pairs for %s (teacher: %s)",
            remaining,
            prompt_version,
            self.model,
        )

        for i in range(remaining):
            pair_num = start_idx + i + 1
            pair_id = f"{prompt_version}_{pair_num:03d}"

            # Pick a random source
            source = random.choice(all_sources)
            user_content = content_template.format(
                source_text=source["text"][:1500]
            )

            raw_response = self._call_teacher(system_msg, user_content)

            pair = {
                "pair_id": pair_id,
                "prompt_version": prompt_version,
                "source_chunk_id": source["chunk_id"],
                "source_ticker": source["ticker"],
                "raw_response": raw_response,
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            existing.append(pair)

            # Checkpoint every 10 pairs
            if pair_num % 10 == 0 or i == remaining - 1:
                raw_path.write_text(
                    json.dumps(existing, indent=2, default=str),
                    encoding="utf-8",
                )
                self._log.info(
                    "Generated %d/%d pairs for %s",
                    pair_num,
                    target_count,
                    prompt_version,
                )

            # Rate-limit delay (skip after last)
            if i < remaining - 1:
                time.sleep(settings.GROQ_CALL_DELAY_SECONDS)

        return existing

    def _call_teacher(self, system_msg: str, user_content: str) -> str:
        """Call the teacher model with exponential backoff on 429."""
        delays = [2, 4, 8]
        last_error = ""

        for attempt in range(len(delays) + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.7,
                    max_tokens=512,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = str(exc)
                is_rate_limit = (
                    "429" in last_error or "rate" in last_error.lower()
                )
                if is_rate_limit and attempt < len(delays):
                    wait = delays[attempt]
                    self._log.warning(
                        "Teacher rate limited (attempt %d), waiting %ds...",
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    self._log.error("Teacher call failed: %s", last_error)
                    return f"ERROR: {last_error}"

        return f"ERROR: {last_error}"

    # ── parsing ──────────────────────────────────────────────────────

    @staticmethod
    def parse_pair(raw_response: str) -> dict | None:
        """Parse a teacher response into instruction and response fields.

        Tries three strategies in order:

        1. **instruction_response** — ``INSTRUCTION: ... RESPONSE: ...``
           (v2/v3 structured format)
        2. **question_answer** — ``Question: ... Answer: ...`` or
           ``General question: ... Brief answer: ...`` (v1 variants)
        3. **freeform** — split on double newline; first paragraph is
           instruction, remainder is response (completely unstructured)

        Returns:
            Dict with ``instruction``, ``response``, and ``format_type``,
            or ``None`` if the text is under 20 characters total.
        """
        if not raw_response or raw_response.startswith("ERROR:"):
            return None

        if len(raw_response.strip()) < 20:
            return None

        # ── Strategy 1: INSTRUCTION: / RESPONSE: markers ────────────
        inst_match = re.search(
            r"INSTRUCTION:\s*(.+?)(?=\nRESPONSE:|\Z)",
            raw_response,
            re.DOTALL,
        )
        resp_match = re.search(
            r"RESPONSE:\s*(.+)",
            raw_response,
            re.DOTALL,
        )

        if inst_match and resp_match:
            instruction = inst_match.group(1).strip()
            response = resp_match.group(1).strip()
            if instruction and response:
                return {
                    "instruction": instruction,
                    "response": response,
                    "format_type": "instruction_response",
                }

        # ── Strategy 2: Question:/Answer: variants ──────────────────
        qa_match = re.search(
            r"(?:General\s+)?[Qq]uestion:\s*(.+?)(?=\n\s*(?:Brief\s+)?[Aa]nswer:|\Z)",
            raw_response,
            re.DOTALL,
        )
        ans_match = re.search(
            r"(?:Brief\s+)?[Aa]nswer:\s*(.+)",
            raw_response,
            re.DOTALL,
        )

        if qa_match and ans_match:
            instruction = qa_match.group(1).strip()
            response = ans_match.group(1).strip()
            if instruction and response:
                return {
                    "instruction": instruction,
                    "response": response,
                    "format_type": "question_answer",
                }

        # ── Strategy 3: freeform split on double newline ────────────
        paragraphs = re.split(r"\n\s*\n", raw_response.strip(), maxsplit=1)
        if len(paragraphs) >= 2:
            instruction = paragraphs[0].strip()
            response = paragraphs[1].strip()
            if instruction and response:
                return {
                    "instruction": instruction,
                    "response": response,
                    "format_type": "freeform",
                }

        return None

    # ── batch generation ─────────────────────────────────────────────

    def generate_all_versions(
        self,
        target_per_version: int = 100,
    ) -> dict[str, list[dict]]:
        """Generate pairs for all three prompt versions.

        Args:
            target_per_version: Pairs to generate per version.

        Returns:
            Dict mapping version name to list of raw pair dicts.
        """
        source = self.load_source_material()
        results: dict[str, list[dict]] = {}

        for version in ("v1", "v2", "v3"):
            self._log.info("Starting generation for %s", version)
            pairs = self.generate_with_prompt(
                prompt_version=version,
                source_chunks=source["edgar_chunks"],
                source_sentences=source["phrasebank"],
                target_count=target_per_version,
            )
            results[version] = pairs
            self._log.info(
                "Completed %s: %d pairs generated",
                version,
                len(pairs),
            )

        return results
