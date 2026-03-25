"""Standalone runner for the Financial Fine-Tuning Laboratory API.

Usage::

    python3 -m backend.run
"""

import uvicorn

from config import settings


def main() -> None:
    """Start the FastAPI server with hot reload."""
    uvicorn.run(
        "backend.main:app",
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
