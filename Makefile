# ──────────────────────────────────────────────────────────────────────
# Financial Fine-Tuning Laboratory — Makefile (WSL / Linux)
# ──────────────────────────────────────────────────────────────────────

VENV       := .finetune_env
ACTIVATE   := source $(VENV)/bin/activate &&
PYTHON     := $(ACTIVATE) python3

.PHONY: setup data benchmark train serve app docker docker-down test clean lint format help

## setup        : Install dependencies and prepare .env
setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	else \
		echo "Virtual environment $(VENV) already exists."; \
	fi
	$(ACTIVATE) pip install --upgrade pip && pip install -r requirements.txt
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example"; \
	else \
		echo ".env already exists — skipping copy."; \
	fi
	@echo ""
	@echo "========================================"
	@echo "  Setup complete."
	@echo "  Open .env and add your API keys."
	@echo "  Then run: source $(VENV)/bin/activate"
	@echo "========================================"

## data         : Run data ingestion pipeline
data:
	$(PYTHON) -m data.ingestion.edgar_client

## benchmark    : Run the 6-approach benchmark on 50 questions
benchmark:
	$(PYTHON) -m evaluation.benchmark_runner

## train        : Instructions for Kaggle-based training
train:
	@echo "Training runs on Kaggle free GPU — not locally."
	@echo "Open notebooks/sft_training.ipynb in Kaggle."
	@echo "See README.md for step-by-step Kaggle instructions."

## serve        : Start Ollama local inference server
serve:
	ollama serve &
	@echo "Ollama started. Load adapter with: ollama run mistral-financial"

## app          : Start FastAPI backend + Streamlit frontend
app:
	@echo "Starting Financial Fine-Tuning Laboratory..."
	@trap 'kill 0' EXIT; \
	$(ACTIVATE) uvicorn backend.main:app \
		--host 0.0.0.0 --port 8000 \
		--log-level warning & \
	sleep 2 && \
	$(ACTIVATE) streamlit run frontend/app.py \
		--server.port 8501; \
	wait

## docker       : Build and start all services via Docker Compose (v2)
docker:
	docker compose up --build

## docker-down  : Stop and remove Docker Compose services
docker-down:
	docker compose down

## test         : Run pytest test suite
test:
	$(ACTIVATE) pytest tests/ -v

## clean        : Remove Python caches and temp files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache

## lint         : Check code style with ruff and black
lint:
	$(ACTIVATE) ruff check . && black --check .

## format       : Auto-format code with black and ruff
format:
	$(ACTIVATE) black . && ruff check . --fix

## help         : Show this help message
help:
	@echo ""
	@echo "Financial Fine-Tuning Laboratory"
	@echo "================================"
	@echo ""
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //' | column -t -s ':'
	@echo ""
