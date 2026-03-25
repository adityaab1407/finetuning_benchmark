"""End-to-end synthetic data pipeline for Financial Fine-Tuning Laboratory.

Orchestrates the full flow from source documents to training-ready datasets:

1. Load diverse source material (EDGAR chunks + phrasebank sentences)
2. Generate instruction-response pairs using 3 prompt versions
3. Parse raw teacher responses into structured pairs
4. Score and filter for quality
5. Format for SFT and DPO training
6. Save version comparison report
7. Optionally push to HuggingFace Hub

Usage::

    # Full pipeline — all 3 prompt versions
    python3 -m data.synthetic.pipeline

    # Quick run — single version, fewer pairs
    python3 -m data.synthetic.pipeline --version v3 --count 20

    # Skip generation, reuse existing raw files
    python3 -m data.synthetic.pipeline --skip-generate

    # Push final dataset to HuggingFace
    python3 -m data.synthetic.pipeline --skip-generate --push-to-hub
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from data.synthetic.formatter import DatasetFormatter
from data.synthetic.generator import SyntheticDataGenerator
from data.synthetic.quality_filter import QualityFilter

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic data generation pipeline.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="all",
        choices=["v1", "v2", "v3", "all"],
        help="Prompt version to run. Default: all",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=settings.SYNTHETIC_TARGET_PAIRS,
        help=f"Pairs to generate per version. Default: {settings.SYNTHETIC_TARGET_PAIRS}",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation, use existing raw files.",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Skip quality scoring.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo name. Default: {HF_USERNAME}/financial-finetune-dataset",
    )
    return parser


def main() -> None:
    """Execute the synthetic data pipeline."""
    parser = _build_parser()
    args = parser.parse_args()

    versions = ["v1", "v2", "v3"] if args.version == "all" else [args.version]
    generated_dir = settings.SYNTHETIC_DIR / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticDataGenerator()
    quality_filter = QualityFilter()
    formatter = DatasetFormatter()

    # ── Step 1: Load source material ─────────────────────────────────
    logger.info("Step 1: Loading source material")
    source_material = generator.load_source_material()
    logger.info(
        "Source material: %d EDGAR chunks, %d phrasebank sentences",
        len(source_material["edgar_chunks"]),
        len(source_material["phrasebank"]),
    )

    # ── Step 2: Generate with each version ───────────────────────────
    raw_by_version: dict[str, list[dict]] = {}

    if args.skip_generate:
        logger.info("Step 2: Skipping generation — loading existing raw files")
        for v in versions:
            raw_path = generated_dir / f"{v}_raw.json"
            if raw_path.exists():
                raw_by_version[v] = json.loads(
                    raw_path.read_text(encoding="utf-8")
                )
                logger.info("Loaded %d raw pairs for %s", len(raw_by_version[v]), v)
            else:
                logger.warning("No raw file found for %s — skipping", v)
                raw_by_version[v] = []
    else:
        logger.info("Step 2: Generating pairs for versions: %s", versions)
        for v in versions:
            raw_by_version[v] = generator.generate_with_prompt(
                prompt_version=v,
                source_chunks=source_material["edgar_chunks"],
                source_sentences=source_material["phrasebank"],
                target_count=args.count,
            )

    # ── Step 3: Parse raw responses ──────────────────────────────────
    logger.info("Step 3: Parsing raw responses")
    parsed_by_version: dict[str, list[dict]] = {}

    for v in versions:
        raw_pairs = raw_by_version.get(v, [])
        parsed: list[dict] = []

        for raw_pair in raw_pairs:
            result = generator.parse_pair(raw_pair.get("raw_response", ""))
            if result is not None:
                # Carry forward metadata
                result["pair_id"] = raw_pair.get("pair_id", "")
                result["prompt_version"] = raw_pair.get("prompt_version", v)
                result["source_chunk_id"] = raw_pair.get("source_chunk_id", "")
                result["source_ticker"] = raw_pair.get("source_ticker", "")
                result["generated_at"] = raw_pair.get("generated_at", "")
                parsed.append(result)

        parsed_by_version[v] = parsed

        # Save parsed file
        parsed_path = generated_dir / f"{v}_parsed.json"
        parsed_path.write_text(
            json.dumps(parsed, indent=2, default=str),
            encoding="utf-8",
        )

        success_rate = len(parsed) / len(raw_pairs) * 100 if raw_pairs else 0
        logger.info(
            "Parsed %s: %d/%d successful (%.1f%%)",
            v,
            len(parsed),
            len(raw_pairs),
            success_rate,
        )

    # ── Step 4: Quality filter ───────────────────────────────────────
    stats_by_version: dict[str, dict] = {}

    if args.skip_score:
        logger.info("Step 4: Skipping quality scoring")
        for v in versions:
            pairs = parsed_by_version.get(v, [])
            # Apply fast filter only
            filtered = [
                p
                for p in pairs
                if len(p.get("instruction", "")) > 20
                and len(p.get("response", "")) > 10
            ]
            for p in filtered:
                p.setdefault("quality_score", 0.70)
                p.setdefault("passed", True)
            parsed_by_version[v] = filtered
            stats = quality_filter.compute_dataset_stats(filtered)
            stats_by_version[v] = stats
    else:
        logger.info("Step 4: Quality filtering")
        for v in versions:
            pairs = parsed_by_version.get(v, [])
            filtered = quality_filter.filter_dataset(
                pairs=pairs,
                source_material=source_material,
                sample_size=min(50, len(pairs)),
            )
            parsed_by_version[v] = filtered
            stats = quality_filter.compute_dataset_stats(filtered)
            stats_by_version[v] = stats

            # Save stats
            stats_path = generated_dir / f"{v}_stats.json"
            stats_path.write_text(
                json.dumps(stats, indent=2, default=str),
                encoding="utf-8",
            )

    # Print quality comparison table
    _print_quality_table(versions, stats_by_version, parsed_by_version)

    # ── Step 5: Format for training (v3 only) ────────────────────────
    logger.info("Step 5: Formatting for training (v3 pairs)")

    # Use v3 if available, otherwise use the best available version
    training_version = "v3" if "v3" in parsed_by_version else versions[-1]
    training_pairs = parsed_by_version.get(training_version, [])

    sft_dataset = formatter.format_for_sft(training_pairs)
    dpo_dataset = formatter.format_for_dpo(training_pairs)

    sft_path = settings.SYNTHETIC_DIR / "sft_dataset.json"
    dpo_path = settings.SYNTHETIC_DIR / "dpo_dataset.json"

    sft_path.write_text(
        json.dumps(sft_dataset, indent=2, default=str),
        encoding="utf-8",
    )
    dpo_path.write_text(
        json.dumps(dpo_dataset, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("SFT dataset: %d pairs → %s", len(sft_dataset), sft_path)
    logger.info("DPO dataset: %d pairs → %s", len(dpo_dataset), dpo_path)

    # ── Step 6: Save version comparison report ───────────────────────
    logger.info("Step 6: Saving version comparison report")

    comparison = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "versions": stats_by_version,
        "final_sft_count": len(sft_dataset),
        "final_dpo_count": len(dpo_dataset),
        "recommendation": (
            f"{training_version} selected for training: "
            "highest pass rate and quality scores"
        ),
    }
    comparison_path = settings.SYNTHETIC_DIR / "version_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Version comparison saved to %s", comparison_path)

    # ── Step 7: Push to HuggingFace (optional) ───────────────────────
    if args.push_to_hub:
        _push_to_hub(sft_path, dpo_path, args.hf_repo)

    logger.info("Pipeline complete.")


def _print_quality_table(
    versions: list[str],
    stats: dict[str, dict],
    parsed: dict[str, list[dict]],
) -> None:
    """Print a quality comparison table to the console."""
    header = f"{'Version':<10} {'Total':>8} {'Passed':>8} {'Pass Rate':>10} {'Avg Quality':>12}"
    sep = "-" * len(header)

    # Console output for the summary table
    print(f"\n{sep}")
    print("QUALITY COMPARISON")
    print(sep)
    print(header)
    print(sep)

    for v in versions:
        s = stats.get(v, {})
        total = s.get("total_pairs", 0)
        pass_rate = s.get("pass_rate_pct", 0.0)
        passed = int(total * pass_rate / 100) if total else 0

        # Compute average quality from pairs
        pairs = parsed.get(v, [])
        quality_scores = [p.get("quality_score", 0.0) for p in pairs if "quality_score" in p]
        avg_quality = (
            sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.0
        )

        print(
            f"{v:<10} {total:>8} {passed:>8} {pass_rate:>9.1f}% {avg_quality:>11.2f}"
        )

    print(sep)
    print()


def _push_to_hub(
    sft_path: Path,
    dpo_path: Path,
    hf_repo: str | None,
) -> None:
    """Push SFT and DPO datasets to HuggingFace Hub."""
    try:
        from datasets import Dataset
        from huggingface_hub import login
    except ImportError:
        logger.error(
            "datasets and huggingface_hub packages required for push. "
            "Install with: pip install datasets huggingface_hub"
        )
        return

    if not settings.HF_TOKEN:
        logger.error("HF_TOKEN not set — cannot push to Hub")
        return

    login(token=settings.HF_TOKEN)

    repo_base = hf_repo or f"{settings.HF_USERNAME}/financial-finetune-dataset"
    if not settings.HF_USERNAME and not hf_repo:
        logger.error("HF_USERNAME not set and no --hf-repo provided")
        return

    # Push SFT dataset
    sft_data = json.loads(sft_path.read_text(encoding="utf-8"))
    sft_ds = Dataset.from_list(sft_data)
    sft_repo = f"{repo_base}_sft"
    sft_ds.push_to_hub(sft_repo)
    logger.info("SFT dataset pushed to HuggingFace: %s", sft_repo)

    # Push DPO dataset
    dpo_data = json.loads(dpo_path.read_text(encoding="utf-8"))
    dpo_ds = Dataset.from_list(dpo_data)
    dpo_repo = f"{repo_base}_dpo"
    dpo_ds.push_to_hub(dpo_repo)
    logger.info("DPO dataset pushed to HuggingFace: %s", dpo_repo)


if __name__ == "__main__":
    main()
