"""Backend package for Financial Fine-Tuning Laboratory.

FastAPI application serving the benchmark API, evaluation endpoints,
and model inference proxy.
"""

from backend.main import app

__all__ = ["app"]
