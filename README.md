# Financial Fine-Tuning Laboratory

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Platform Linux](https://img.shields.io/badge/platform-Linux-orange)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)

A head-to-head benchmark of **6 LLM approaches** on **100 curated financial questions**: zero-shot, few-shot, chain-of-thought, RAG, SFT fine-tuned, and DPO-aligned. The project answers *when* fine-tuning a small open-weight model (Mistral-7B with QLoRA) beats prompt engineering and retrieval — and when it doesn't.

---

## Benchmark Results

| Rank | Approach | Accuracy | Hallucination | Avg Latency |
|------|----------|----------|---------------|-------------|
| 1 | Chain-of-Thought | **37%** | 0% | 961 ms |
| 2 | Few-Shot | 35% | 0% | 417 ms |
| 3 | RAG | 8% | 0% | 1,409 ms |
| — | SFT Fine-Tuned | training... | — | — |
| — | DPO Aligned | training... | — | — |

*Single run over 100 questions; ±10% margin of error. Zero-shot not yet in summary.*

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/<your-username>/financial-finetuning-benchmark.git
cd financial-finetuning-benchmark
cp .env.example .env
# Add your GROQ_API_KEY to .env
make docker          # builds and starts backend (8083) + frontend (8503)
```

Open **http://localhost:8503** for the Streamlit dashboard.

### Local

```bash
python3 -m venv .finetune_env
source .finetune_env/bin/activate
make setup           # installs dependencies, creates .env
# Add your GROQ_API_KEY to .env
make app             # FastAPI on :8083 + Streamlit on :8503
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Streamlit Dashboard (port 8503)                    │
│  Live Demo · Benchmark · Training Lab · GPU Stats  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────┐
│  FastAPI Backend (port 8083)                        │
│  POST /api/run/all  ·  POST /api/run/{approach}     │
│  GET  /health                                       │
└──────┬──────┬──────┬──────────────────┬─────────────┘
       │      │      │                  │
  Zero  Few   CoT   RAG ──► ChromaDB   Groq API
  Shot  Shot        (1,182 chunks)   (Llama 4 Scout)
```

---

## Project Structure

```
├── approaches/         Six benchmark approach implementations
│   ├── base.py             Abstract base class (Groq client, retry, cost)
│   ├── zero_shot.py        No examples, direct question
│   ├── few_shot.py         4 worked financial examples in-context
│   ├── cot.py              3-step reasoning scaffold (UNDERSTAND→ANALYZE→ANSWER)
│   ├── rag.py              Top-3 chunk retrieval from ChromaDB
│   └── ingest.py           SEC filing ingestion into ChromaDB
│
├── backend/            FastAPI application
├── frontend/           Streamlit dashboard
├── config/             Pydantic settings (loaded from .env)
│
├── data/
│   ├── raw/                20 SEC 10-Q filings (10 tickers × 2 quarters)
│   ├── processed/          Chunked EDGAR + benchmark datasets
│   ├── eval_set/           100 expert-labeled questions + schemas
│   └── synthetic/          Generated SFT and DPO training datasets
│
├── evaluation/         Benchmark runner, scoring, LLM judge, results
├── training/           QLoRA SFT + DPO configs (trains on Kaggle GPU)
├── serving/            Ollama local model serving utilities
├── tests/              Pytest test suite
└── notebooks/          Kaggle training notebooks
```

---

## Evaluation Dataset

100 questions across 4 categories, sourced from real financial data:

| Category | Count | Evaluation |
|----------|-------|------------|
| Factual Extraction | 30 | fuzzy / exact match |
| Sentiment Classification | 30 | classification accuracy |
| Structured Output (JSON) | 20 | schema validation |
| Reasoning & Calculation | 20 | token-set match |

**Data sources:**
- 20 SEC 10-Q filings — AAPL, AMZN, GOOGL, JPM, META, MSFT, NFLX, NVDA, TSLA, V (Q3/Q4 FY2025–FY2026)
- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (sentiment labels)
- [FiQA](https://huggingface.co/datasets/BeIR/fiqa) (financial QA pairs)

---

## Approaches

| Approach | Model | Strategy |
|----------|-------|----------|
| Zero-Shot | Llama 4 Scout 17B | System prompt only — pure parametric knowledge |
| Few-Shot | Llama 4 Scout 17B | 4 worked financial examples in-context |
| Chain-of-Thought | Llama 4 Scout 17B | Structured reasoning; answers extracted via `FINAL ANSWER:` marker |
| RAG | Llama 4 Scout 17B | 1,182-chunk ChromaDB index of SEC filings; top-3 retrieval |
| SFT Fine-Tuned | Mistral-7B-Instruct-v0.3 | QLoRA fine-tuned on synthetic SFT dataset |
| DPO Aligned | Mistral-7B-Instruct-v0.3 | Preference-aligned on chosen/rejected pairs |

All Groq-based approaches use exponential-backoff retry on rate-limit errors (429). Parallel requests are staggered to stay within free-tier RPM limits.

---

## Training Pipeline

Fine-tuning runs on **Kaggle free GPU** (T4 / P100) via notebooks in `notebooks/`.

```
SEC filings + phrasebank + FiQA
        │
        ▼
Synthetic data generator (Llama 3.3-70B teacher via Groq)
        │
        ├── sft_dataset.json   → QLoRA SFT on Mistral-7B
        └── dpo_dataset.json   → DPO alignment (chosen/rejected pairs)
                │
                ▼
        Trained adapter weights
                │
                ▼
        Ollama local serving (port 11434)
```

**QLoRA config:** LoRA rank 16, alpha 32, lr 2e-4, 3 epochs, max_seq 2048, batch 1 + 8 gradient accumulation steps.

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Create venv, install deps, generate .env |
| `make data` | Download and chunk SEC filings |
| `make benchmark` | Run all approaches on 100 questions |
| `make app` | Start FastAPI + Streamlit locally |
| `make docker` | Build and start Docker stack |
| `make docker-down` | Stop Docker stack |
| `make train` | Training instructions for Kaggle |
| `make serve` | Start Ollama local model server |
| `make test` | Run pytest suite |
| `make lint` | ruff + black checks |
| `make format` | Auto-format with black + ruff |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
GROQ_API_KEY=          # Required — inference for all 4 prompt approaches
HF_TOKEN=              # Required — model downloads and dataset uploads
WANDB_API_KEY=         # Optional — training run tracking
HF_USERNAME=           # Optional — push trained adapter to Hub
```

---

## Key Findings

- **CoT edges out few-shot** (37% vs 35%) on this dataset. The structured three-step scaffold helps most on reasoning and calculation questions but adds overhead on simple factual lookups.
- **RAG underperforms** (8%) despite retrieving relevant chunks — the bottleneck is the 500-char chunk size losing surrounding context for many numerical facts.
- **Zero-shot is a surprisingly capable baseline** — the Llama 4 Scout model has strong financial knowledge from pretraining.
- **Fine-tuned models (SFT/DPO) are the hypothesis** — the benchmark is designed to test whether targeted fine-tuning on domain-specific examples can beat all prompt engineering approaches.

---

## License

MIT

