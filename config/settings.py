"""Centralized configuration for Financial Fine-Tuning Laboratory.

All settings are loaded from environment variables (via .env file) using
pydantic-settings. Every configurable value in the project flows through
this module — no hardcoded strings elsewhere.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from .env and environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── GROQ API ──────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""

    # ── MODEL ASSIGNMENTS ─────────────────────────────────────────────
    MODEL_ZERO_SHOT: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    MODEL_FEW_SHOT: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    MODEL_COT: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    MODEL_RAG: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    MODEL_JUDGE: str = "llama-3.1-8b-instant"
    MODEL_TEACHER: str = "llama-3.3-70b-versatile"
    MODEL_SCORER: str = "llama-3.1-8b-instant"
    MODEL_FALLBACK: str = "moonshotai/kimi-k2-instruct"

    # ── RATE LIMITING ─────────────────────────────────────────────────
    GROQ_CALL_DELAY_SECONDS: float = 2.5
    GROQ_MAX_RETRIES: int = 3
    GROQ_RPD_SOFT_LIMIT: int = 800
    BENCHMARK_THREADS: int = 4

    # ── DATA PATHS ────────────────────────────────────────────────────
    DATA_DIR: Path = Path("data")
    RAW_DIR: Path = Path("data/raw")
    PROCESSED_DIR: Path = Path("data/processed")
    EVAL_SET_PATH: Path = Path("data/eval_set/questions.json")
    SOURCES_PATH: Path = Path("data/eval_set/sources.json")
    SYNTHETIC_DIR: Path = Path("data/synthetic")
    RESULTS_DIR: Path = Path("evaluation/results")

    # ── CHUNKING ──────────────────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_CHUNKS: int = 3

    # ── CHROMADB ──────────────────────────────────────────────────────
    CHROMA_COLLECTION_NAME: str = "financial_docs"
    CHROMA_PERSIST_DIR: str = "data/chroma_db"

    # ── TRAINING ──────────────────────────────────────────────────────
    HF_TOKEN: str = ""
    WANDB_API_KEY: str = ""
    WANDB_PROJECT: str = "financial-finetuning"
    BASE_MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    MAX_SEQ_LENGTH: int = 2048
    TRAINING_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-4
    PER_DEVICE_BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 8

    # ── SERVING ───────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "mistral-financial"
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000

    # ── OBSERVABILITY ─────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"


settings = Settings()


def get_settings() -> Settings:
    """Return the module-level Settings singleton."""
    return settings
