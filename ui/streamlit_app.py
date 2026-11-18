"""Streamlit client for the PeopleOps API.

Demo only: login is out of scope, so the employee ID is typed in. The API stays the single
authority on identity, role and permissions; this UI never decides anything.
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_URL_DEFAULT = os.getenv("API_URL", "http://localhost:8000")

ROUTE_LABELS = {
    "rag": "Answered from HR documents",
    "tool": "Answered from leave data",
    "hybrid": "Documents and leave data",
    "clarify": "More information needed",
    "no_answer": "Not covered by the documents",
    "refuse": "Can't help with that",
}
STATUS_LABELS = {"success": "Done", "error": "Rejected", "pending_confirmation": "Waiting for your confirmation"}
HTTP_ERRORS = {
    400: "The message was empty or invalid.",
    401: "That employee ID is not recognised.",
    403: "This conversation belongs to another employee. Start a new conversation.",
    503: "The service is still starting up. Try again shortly.",
}
SAMPLES = [
    "How many sick days do I get?",
    "What is my annual leave balance?",
    "Which public holidays do I have this year?",
    "Apply annual leave next week",
]

st.set_page_config(page_title="Veltrix PeopleOps", page_icon="💬", layout="centered")


def new_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.queued = None


def api_health(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def send(url: str, employee_id: str, text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text})
    try:
        r = requests.post(
            f"{url}/chat",
            json={"session_id": st.session_state.session_id, "message": text},
            headers={"X-Employee-Id": employee_id},
            timeout=35,
        )
    except requests.RequestException:
        st.session_state.messages.append({"role": "error", "content": "Can't reach the API. Check the API URL."})
        return
    if r.status_code == 200:
        data = r.json()
        st.session_state.messages.append({"role": "assistant", "content": data["answer"], "data": data})
    else:
        st.session_state.messages.append(
            {"role": "error", "content": HTTP_ERRORS.get(r.status_code, f"Unexpected error ({r.status_code}).")}
        )


def render_details(data: dict) -> None:
    st.caption(ROUTE_LABELS.get(data["route"], data["route"]))
    if data["citations"]:
        with st.expander(f"Sources ({len(data['citations'])})"):
            for c in data["citations"]:
                st.markdown(f"**{c['doc_id']}**, {c['locator']}")
    if data["tool_calls"]:
        with st.expander(f"Actions ({len(data['tool_calls'])})"):
            for t in data["tool_calls"]:
                st.markdown(f"`{t['name']}`: {STATUS_LABELS.get(t['status'], t['status'])}")
                st.json(t["args"], expanded=False)
    st.caption(f"Request {data['request_id']}")


# ---- sidebar ----
if "session_id" not in st.session_state:
    new_conversation()

with st.sidebar:
    st.subheader("Connection")
    api_url = st.text_input("API URL", API_URL_DEFAULT).rstrip("/")
    employee_id = st.text_input("Employee ID", placeholder="E1001").strip()
    st.write("API status:", "ready" if api_health(api_url) else "unavailable")
    if st.button("New conversation", use_container_width=True):
        new_conversation()
        st.rerun()
    st.subheader("Try asking")
    for s in SAMPLES:
        if st.button(s, use_container_width=True, disabled=not employee_id):
            st.session_state.queued = s

# A session belongs to the employee who started it, so switching employee starts fresh.
if st.session_state.get("owner") != employee_id:
    new_conversation()
    st.session_state.owner = employee_id

st.title("Veltrix PeopleOps")
if not employee_id:
    st.info("Enter your employee ID in the sidebar to start.")

prompt = st.chat_input("Ask about HR policies, holidays or your leave", disabled=not employee_id)
text = st.session_state.pop("queued", None) or prompt
if text and employee_id:
    send(api_url, employee_id, text)

for m in st.session_state.messages:
    if m["role"] == "error":
        st.error(m["content"])
    else:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant":
                render_details(m["data"])

# Writes need an explicit click; the API enforces confirmation regardless of this UI.
last = st.session_state.messages[-1] if st.session_state.messages else None
if last and last["role"] == "assistant" and last["data"]["pending_confirmation"]:
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Confirm", type="primary"):
        st.session_state.queued = "yes, confirm"
        st.rerun()
    if c2.button("Cancel"):
        st.session_state.queued = "no, cancel that"
        st.rerun()
