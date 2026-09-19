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
    host = "127.0.0.1" if settings.host in ["0.0.0.0", "127.0.0.1", "localhost"] else settings.host
    port = settings.port

    print("=" * 70)
    print("🎓 AI-Based Academic Journal Recommendation Assistant")
    print(f"🚀 Open in your browser: http://127.0.0.1:{port}")
    print("=" * 70)

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        theme=None,
        show_error=True,
    )
