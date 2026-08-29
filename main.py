"""Cloud Run entrypoint.

Wraps the ADK agent in a FastAPI app using ADK's built-in helper, which
gives you the chat API plus ADK's dev UI for free. Cloud Run sets $PORT
automatically; we read it here so the container works both locally and
in production.
"""

import os

import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

# Directory containing the agent package (ai_barista/).
AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))

app = get_fast_api_app(
    agents_dir=AGENTS_DIR,
    web=True,  # serves the ADK dev UI at "/" -- handy for demos
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
