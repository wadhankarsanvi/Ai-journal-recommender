#!/usr/bin/env python3
"""
AI-Based Academic Journal Recommendation Assistant
Entrypoint for launching Gradio Web UI.
"""

import logging
import os
import sys
import threading

# Keep ONNX/BLAS from oversubscribing threads on small Render instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ChromaDB telemetry makes outbound HTTP calls we don't need.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.product.NoopTelemetryClient")

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("journal_ai.app")

from journal_ai.ui.gradio_app import create_gradio_app
from journal_ai.config.settings import settings
from journal_ai.rag.pipeline import warmup_rag

# Create the Gradio interface
demo = create_gradio_app()

# Keep the queue bounded so slow external APIs cannot exhaust the instance.
demo.queue(default_concurrency_limit=2, max_size=24)


def _warmup() -> None:
    """
    Download + load the embedding model in the background.

    This MUST NOT run inline at import: the first embedding call pulls an ~80 MB
    ONNX model, and blocking here would delay the port bind past Render's health
    check window and make the deploy look hung.
    """
    if settings.enable_rag:
        log.info("Starting RAG warmup in background...")
        warmup_rag()
    else:
        log.info("RAG warmup skipped because ENABLE_RAG is disabled.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 70)
    print("🎓 AI-Based Academic Journal Recommendation Assistant")
    print(f"🚀 Server listening on: http://{host}:{port}")
    print("=" * 70)

    threading.Thread(target=_warmup, name="rag-warmup", daemon=True).start()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=None,
        show_error=True,
    )