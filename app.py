#!/usr/bin/env python3
"""
AI-Based Academic Journal Recommendation Assistant
Entrypoint for launching Gradio Web UI.
"""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from journal_ai.ui.gradio_app import create_gradio_app
from journal_ai.config.settings import settings

# Create the Gradio interface
demo = create_gradio_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 70)
    print("🎓 AI-Based Academic Journal Recommendation Assistant")
    print(f"🚀 Server listening on: http://{host}:{port}")
    print("=" * 70)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=None,
        show_error=True,
    )

