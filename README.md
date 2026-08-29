# AI Barista — ADK + RAG on Cloud Run

A personalized AI assistant for a coffee shop app. Built with the Google
**Agent Development Kit (ADK)**, using a **RAG** pattern to recommend drinks
from a live menu, and deployed on **Cloud Run**.

## How it's grounded

Instead of pasting the whole menu into the system prompt, the agent calls a
`get_menu(query)` tool (see `ai_barista/agent.py`). That tool:

1. Embeds the customer's request and every menu item using Vertex AI's
   `text-embedding-004` model.
2. Ranks menu items by cosine similarity to the request.
3. Returns only the top matches (name, description, price, tags) —
   never the embedding vectors themselves, since those are only useful for
   retrieval, not for Gemini's response.

This keeps the prompt small, lets the menu change anytime without
redeploying, and keeps the model's answers grounded in what's *actually*
on the menu instead of its own pretrained knowledge.

## Project layout

```
ai-barista/
├── ai_barista/
│   ├── __init__.py
│   ├── agent.py       # root_agent + get_menu RAG tool
│   └── menu.json       # sample menu (swap for Firestore in production)
├── main.py              # Cloud Run entrypoint (FastAPI + ADK dev UI)
├── streamlit_app.py      # optional chat UI, uses st.session_state for history
├── requirements.txt
├── Dockerfile
└── deploy.sh             # creates a least-privilege SA + deploys to Cloud Run
```

## 1. Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI=True
```

No GCP project yet? The `get_menu` tool falls back to a simple keyword
matcher automatically if Vertex AI embeddings aren't reachable, so you can
still poke at the agent locally.

## 2. Run locally

```bash
adk web ai_barista
```

This opens ADK's built-in dev UI so you can chat with the agent and inspect
tool calls before deploying anything.

Or run the Cloud Run entrypoint directly:

```bash
python main.py
# then, in another terminal:
streamlit run streamlit_app.py
```

## 3. Deploy to Cloud Run

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-central1"
chmod +x deploy.sh
./deploy.sh
```

This script:
- Enables the Cloud Run, Vertex AI, Artifact Registry, and Cloud Build APIs.
- Creates a **dedicated service account** with only the `roles/aiplatform.user`
  role (principle of least privilege — no broad project-level access).
- Deploys the container from source and prints the live service URL.

Point `AGENT_URL` in `streamlit_app.py` at that URL to chat with the
deployed version.

## 4. Push to GitHub

For the APAC Skills submission, commit this whole folder to a public GitHub
repo and include the deployed Cloud Run URL in your README — that repo link
plus the live URL is typically what gets submitted (GitHub, not Kaggle,
since this lab's deliverable is a running service, not a notebook).

```bash
git init
git add .
git commit -m "AI Barista: ADK agent with RAG on Cloud Run"
git branch -M main
git remote add origin https://github.com/<you>/ai-barista.git
git push -u origin main
```

## Scaling beyond a static menu.json

For production (thousands of menu items), swap `menu.json` for **Firestore
with Vector Search**: store each item's embedding as a Firestore vector
field, and replace the in-memory cosine-similarity loop in `get_menu` with
a Firestore `find_nearest()` query. The agent's interface (`get_menu(query) ->
dict`) doesn't need to change — only the retrieval backend does.
