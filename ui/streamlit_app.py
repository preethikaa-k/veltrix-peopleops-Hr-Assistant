"""Streamlit client for the Veltrix PeopleOps Assistant.

Enterprise AI HR Assistant UI with live status, sample profiles, animated loaders,
and confirmation state management.
"""
from __future__ import annotations

import os
import uuid
import requests
import streamlit as st

API_URL_DEFAULT = os.getenv("API_URL", "http://localhost:8000")

ROUTE_LABELS = {
    "rag": "📄 Policy Document",
    "tool": "🛠️ Leave Database",
    "hybrid": "📄 Policy & Leave Data",
    "clarify": "❓ Clarification Needed",
    "no_answer": "⚠️ Not Covered in Policies",
    "refuse": "🚫 Restricted / Out of Scope",
}

STATUS_LABELS = {
    "success": "✅ Completed",
    "error": "❌ Rejected",
    "pending_confirmation": "⏳ Waiting for your confirmation",
}

HTTP_ERRORS = {
    400: "The request was empty or invalid.",
    401: "That employee ID is not recognised in the HR database.",
    403: "This session belongs to another employee. Please start a new conversation.",
    503: "The assistant service is still starting up / ingesting policies. Please wait a moment.",
}

PRESET_PROFILES = {
    "E2004 - Sneha Kulkarni (Employee | India)": "E2004",
    "E3004 - Emily Carter (Employee | United States)": "E3004",
    "E4004 - Lukas Weber (Employee | Germany)": "E4004",
    "E2002 - Divya Menon (Manager | India)": "E2002",
    "E1001 - Meera Krishnan (HR Staff | India)": "E1001",
    "Custom Employee ID...": "",
}

SAMPLES = [
    "How many sick days do I get per year?",
    "What is my annual leave balance?",
    "Which public holidays do I have this year?",
    "Apply annual leave from 2026-07-06 to 2026-07-10",
    "What is the bereavement leave policy?",
]

# Set clean page configuration
st.set_page_config(
    page_title="Veltrix PeopleOps Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom modern CSS
st.markdown(
    """
    <style>
    /* Clean container styling */
    .stChatInputContainer {
        border-radius: 12px;
    }
    .user-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #e3f2fd;
        color: #0d47a1;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .status-badge-ready {
        display: inline-block;
        padding: 3px 10px;
        background-color: #e8f5e9;
        color: #1b5e20;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-badge-offline {
        display: inline-block;
        padding: 3px 10px;
        background-color: #ffebee;
        color: #b71c1c;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .route-chip {
        font-size: 0.75rem;
        color: #666;
        background: #f0f2f6;
        padding: 2px 8px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 4px;
    }
    .confirm-card {
        background-color: #fff8e1;
        border-left: 4px solid #ffb300;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.queued = None


def api_health(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def call_chat_api(url: str, employee_id: str, text: str) -> dict | None:
    try:
        r = requests.post(
            f"{url}/chat",
            json={"session_id": st.session_state.session_id, "message": text},
            headers={"X-Employee-Id": employee_id},
            timeout=35,
        )
        if r.status_code == 200:
            return r.json()
        error_msg = HTTP_ERRORS.get(r.status_code, f"API Error ({r.status_code}): {r.text}")
        st.session_state.messages.append({"role": "error", "content": error_msg})
        return None
    except requests.RequestException:
        st.session_state.messages.append(
            {"role": "error", "content": f"Unable to connect to API at {url}. Please verify the backend is running."}
        )
        return None


# ---- Sidebar ----
if "session_id" not in st.session_state:
    new_conversation()

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API status indicator
    is_ready = api_health(API_URL_DEFAULT)
    if is_ready:
        st.markdown('<span class="status-badge-ready">🟢 Backend Connected & Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-offline">🔴 Backend Offline</span>', unsafe_allow_html=True)

    api_url = st.text_input("API URL", API_URL_DEFAULT).rstrip("/")

    st.divider()
    st.subheader("👤 Requester Identity")
    
    # Profile selector helper
    selected_preset = st.selectbox(
        "Choose Employee Profile:",
        list(PRESET_PROFILES.keys()),
        index=0,
    )
    
    preset_id = PRESET_PROFILES[selected_preset]
    if preset_id:
        employee_id = preset_id
        st.caption(f"Active ID: `{employee_id}`")
    else:
        employee_id = st.text_input("Enter Custom Employee ID", placeholder="e.g. E2004").strip()

    if st.button("🔄 New Conversation", use_container_width=True):
        new_conversation()
        st.rerun()

    st.divider()
    st.subheader("💡 Quick Prompts")
    for sample in SAMPLES:
        if st.button(sample, use_container_width=True, disabled=not employee_id or not is_ready):
            st.session_state.queued = sample

# Switch conversation if employee changes
if st.session_state.get("owner") != employee_id:
    new_conversation()
    st.session_state.owner = employee_id

# ---- Main Chat View ----
st.title("🤖 Veltrix PeopleOps Assistant")
st.caption("AI-powered HR Assistant for Policy Guidance & Leave Management")

if not employee_id:
    st.warning("👈 Please select or enter an Employee ID in the sidebar to begin.")
elif not is_ready:
    st.error(f"Cannot reach backend API at `{api_url}`. Make sure Uvicorn is running: `uvicorn app.main:app --port 8000`")

# Render existing chat history
for m in st.session_state.messages:
    if m["role"] == "error":
        st.error(m["content"])
    else:
        avatar = "👤" if m["role"] == "user" else "🤖"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant" and "data" in m:
                data = m["data"]
                route_desc = ROUTE_LABELS.get(data.get("route", ""), data.get("route", ""))
                st.markdown(f'<span class="route-chip">{route_desc}</span>', unsafe_allow_html=True)

# Handle 2-Step Confirmation Buttons if waiting
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if (
    last_msg
    and last_msg.get("role") == "assistant"
    and last_msg.get("data", {}).get("pending_confirmation")
):
    with st.container():
        st.markdown(
            '<div class="confirm-card"><b>⚠️ Confirmation Needed:</b> Would you like to proceed with this leave action?</div>',
            unsafe_allow_html=True,
        )
        col1, col2, _ = st.columns([1, 1, 2])
        if col1.button("✅ Yes, Confirm", type="primary", use_container_width=True):
            st.session_state.queued = "yes, confirm"
            st.rerun()
        if col2.button("❌ Cancel", use_container_width=True):
            st.session_state.queued = "no, cancel that"
            st.rerun()

# User Input Box
prompt = st.chat_input("Ask a question about HR policies or manage your leave...", disabled=not employee_id or not is_ready)
user_query = st.session_state.pop("queued", None) or prompt

if user_query and employee_id:
    # 1. Immediately record and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # 2. Display live loader during retrieval & LLM generation
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Retrieving policy knowledge and checking records..."):
            response_data = call_chat_api(api_url, employee_id, user_query)

        if response_data:
            answer_text = response_data.get("answer", "")
            st.markdown(answer_text)
            
            # Display clean route category tag (No sources expander)
            route_desc = ROUTE_LABELS.get(response_data.get("route", ""), response_data.get("route", ""))
            st.markdown(f'<span class="route-chip">{route_desc}</span>', unsafe_allow_html=True)

            # Store in session state for re-rendering
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "data": response_data,
            })
            st.rerun()
