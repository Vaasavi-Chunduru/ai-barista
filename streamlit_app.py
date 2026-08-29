"""Optional chat UI for the AI Barista.

Run locally with:
    streamlit run streamlit_app.py

Point AGENT_URL at your deployed Cloud Run service, or run main.py locally
on port 8080 and leave the default.
"""

import os

import requests
import streamlit as st

AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8080")
APP_NAME = "ai_barista"
USER_ID = "streamlit-user"

st.set_page_config(page_title="AI Barista", page_icon="☕")
st.title("☕ AI Barista")

# st.session_state is what keeps the conversation alive across Streamlit's
# reruns within a browser session -- without it, history would reset on
# every message.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None


def ensure_session():
    if st.session_state.session_id is None:
        resp = requests.post(
            f"{AGENT_URL}/apps/{APP_NAME}/users/{USER_ID}/sessions",
            json={},
            timeout=30,
        )
        resp.raise_for_status()
        st.session_state.session_id = resp.json()["id"]


def send_message(text: str) -> str:
    ensure_session()
    resp = requests.post(
        f"{AGENT_URL}/run",
        json={
            "app_name": APP_NAME,
            "user_id": USER_ID,
            "session_id": st.session_state.session_id,
            "new_message": {"role": "user", "parts": [{"text": text}]},
        },
        timeout=60,
    )
    resp.raise_for_status()
    events = resp.json()
    reply_parts = []
    for event in events:
        for part in event.get("content", {}).get("parts", []):
            if "text" in part:
                reply_parts.append(part["text"])
    return "\n".join(reply_parts) if reply_parts else "(no response)"


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What can I get started for you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Brewing a response..."):
            reply = send_message(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
