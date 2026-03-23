# Financial Fine-Tuning Laboratory

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Platform WSL/Linux](https://img.shields.io/badge/platform-WSL%20%2F%20Linux-orange)

A head-to-head benchmark of **6 LLM approaches** on **50 curated financial questions**: zero-shot, few-shot, chain-of-thought, RAG, SFT fine-tuned, and DPO-aligned. The project proves *when* fine-tuning a small open-weight model (Mistral-7B with QLoRA) beats prompt engineering and retrieval — and when it doesn't. Built for ML engineers and hiring managers who want to see fine-tuning done rigorously, with real evaluation, real data, and real trade-off analysis.

## Quick Start

```bash
git clone https://github.com/<your-username>/financial-finetuning-benchmark.git
cd financial-finetuning-benchmark
python3 -m venv .finetune_env
source .finetune_env/bin/activate
make setup          # install deps, create .env
# Edit .env with your API keys
make app            # launch FastAPI + Streamlit
```

## Project Structure

```
├── data/           Data ingestion, processing, eval set, synthetic data
├── approaches/     Six benchmark approach implementations
├── evaluation/     Scoring, judging, benchmark runner
├── training/       QLoRA SFT + DPO configs and scripts
├── serving/        Ollama model serving utilities
├── backend/        FastAPI application
├── frontend/       Streamlit dashboard
├── config/         Pydantic settings (loaded from .env)
├── tests/          Pytest test suite
└── notebooks/      Kaggle training notebooks
```

## Benchmark Results

> TODO — will be populated after Day 5 benchmark run.

## Model Training Details

> TODO — will be populated after Day 7 SFT training.

## Architecture Diagram

> TODO — will be added with the final write-up.

## Dataset

> TODO — will document the 50-question eval set and data sources.

## Learnings

> TODO — will capture key insights and trade-off analysis.

## License

MIT
