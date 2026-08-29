"""AI Barista agent for the coffee shop app.

Built with the Agent Development Kit (ADK). The agent retrieves menu
information through a Retrieval-Augmented Generation (RAG) pattern:
instead of embedding the entire menu inside the system prompt, the
`get_menu` tool embeds the user's query and the menu items, then
returns only the semantically closest matches to Gemini.
"""

import json
import os
from pathlib import Path

import numpy as np
from google.adk.agents import Agent

MENU_PATH = Path(__file__).parent / "menu.json"
TOP_K = 4  # how many menu items to return per query

# ---------------------------------------------------------------------------
# Embeddings backend
# ---------------------------------------------------------------------------
# Uses Vertex AI's text-embedding model when credentials/project are
# available (production path). Falls back to a lightweight local scorer so
# the agent still runs for local testing without a GCP project configured.
_embedding_model = None


def _get_embedding_model():
    """Lazily initializes the Vertex AI embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from vertexai.language_models import TextEmbeddingModel
        import vertexai

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertexai.init(project=project, location=location)
        _embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return _embedding_model


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embeds a list of strings, returning an (n, dim) numpy array."""
    model = _get_embedding_model()
    embeddings = model.get_embeddings(texts)
    return np.array([e.values for e in embeddings])


def _keyword_fallback_scores(query: str, items: list[dict]) -> list[float]:
    """A dependency-free fallback scorer used when Vertex AI embeddings
    aren't reachable (e.g. no ADC configured yet). Not semantic search --
    just enough to keep local development unblocked."""
    query_terms = set(query.lower().split())
    scores = []
    for item in items:
        haystack = " ".join(
            [item["name"], item["description"], item["category"], *item.get("tags", [])]
        ).lower()
        overlap = sum(1 for term in query_terms if term in haystack)
        scores.append(float(overlap))
    return scores


# ---------------------------------------------------------------------------
# Menu index (loaded once per process)
# ---------------------------------------------------------------------------
with open(MENU_PATH, "r", encoding="utf-8") as f:
    _MENU_ITEMS = json.load(f)

_menu_embeddings = None  # computed lazily, cached for the process lifetime


def _get_menu_embeddings():
    global _menu_embeddings
    if _menu_embeddings is None:
        corpus = [
            f"{item['name']}. {item['description']} Category: {item['category']}. "
            f"Tags: {', '.join(item.get('tags', []))}"
            for item in _MENU_ITEMS
        ]
        _menu_embeddings = _embed_texts(corpus)
    return _menu_embeddings


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


# ---------------------------------------------------------------------------
# Tool: get_menu
# ---------------------------------------------------------------------------
def get_menu(query: str) -> dict:
    """Retrieves the menu items most relevant to a user's request.

    This performs semantic (RAG-style) retrieval: the query and each menu
    item are embedded into vectors, and only the top-matching items are
    returned. This keeps the prompt small and lets the menu grow or change
    without ever needing to edit the agent's instructions.

    Args:
        query: The customer's request in natural language, e.g.
            "something sweet and cold" or "a strong coffee with no milk".

    Returns:
        A dict with a "results" key containing the top matching menu items
        (name, category, description, price, tags). Embedding vectors are
        deliberately excluded -- they're only useful for retrieval, not for
        the language model's response.
    """
    try:
        query_vec = _embed_texts([query])[0]
        menu_vecs = _get_menu_embeddings()
        scores = _cosine_similarity(query_vec, menu_vecs)
    except Exception:
        # No Vertex AI credentials/network available -- fall back so the
        # agent is still usable for local development.
        scores = np.array(_keyword_fallback_scores(query, _MENU_ITEMS))

    ranked_indices = np.argsort(-scores)[:TOP_K]
    results = []
    for idx in ranked_indices:
        item = _MENU_ITEMS[idx]
        results.append(
            {
                "name": item["name"],
                "category": item["category"],
                "description": item["description"],
                "price": item["price"],
                "tags": item.get("tags", []),
            }
        )

    return {"results": results}


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
root_agent = Agent(
    name="ai_barista",
    model="gemini-flash-latest",
    description="A friendly AI barista that recommends drinks from the coffee shop's live menu.",
    instruction="""
You are the AI Barista for a coffee shop. Be warm, concise, and helpful.

Always call the `get_menu` tool to find out what's actually available before
recommending or confirming any drink -- never invent menu items or prices
from general knowledge. Ground every recommendation in the tool's results.

If the customer asks for something that isn't in the retrieved results,
say so plainly, and optionally suggest the closest genuinely retrieved
alternatives instead of guessing.

Keep responses short and conversational, like a real barista taking an
order. When you recommend a drink, mention its name and price.
""",
    tools=[get_menu],
)
