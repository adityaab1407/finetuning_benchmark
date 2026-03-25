#!/bin/bash
# ── Backend entrypoint ───────────────────────────────────────────────────
# Checks whether the ChromaDB collection is already populated.
# If empty (first run or after volume wipe), ingests all EDGAR chunks.
# Then hands off to uvicorn.
set -e

echo "[entrypoint] Checking ChromaDB collection..."

python3 - <<'PYEOF'
import sys
import chromadb
from config import settings

client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
try:
    col = client.get_collection(settings.CHROMA_COLLECTION_NAME)
    count = col.count()
    if count > 0:
        print(f"[entrypoint] Collection already has {count} chunks — skipping ingestion")
        sys.exit(0)
except Exception:
    pass

print("[entrypoint] Collection is empty — running ingestion")
sys.exit(1)
PYEOF

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "[entrypoint] Running: python3 -m approaches.ingest"
    python3 -m approaches.ingest
fi

echo "[entrypoint] Starting API server..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8083
