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
    .stChatInputContainer {
        border-radius: 12px;
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

api_url = API_URL_DEFAULT

with st.sidebar:
    st.subheader("👤 Employee Profile")
    
    # Profile selector helper
    selected_preset = st.selectbox(
        "Select Active Employee:",
        list(PRESET_PROFILES.keys()),
        index=0,
    )
    
    preset_id = PRESET_PROFILES[selected_preset]
    if preset_id:
        employee_id = preset_id
        st.caption(f"Logged in as: `{employee_id}`")
    else:
        employee_id = st.text_input("Enter Custom Employee ID", placeholder="e.g. E2004").strip()

    if st.button("🔄 New Conversation", use_container_width=True):
        new_conversation()
        st.rerun()

    st.divider()
    st.subheader("💡 Quick Prompts")
    for sample in SAMPLES:
        if st.button(sample, use_container_width=True, disabled=not employee_id):
            st.session_state.queued = sample

    st.divider()
    with st.expander("📊 Live Observability & Metrics", expanded=False):
        try:
            m_res = requests.get(f"{api_url}/metrics", timeout=2)
            if m_res.status_code == 200:
                m_data = m_res.json()
                st.metric("Total API Requests", m_data.get("total_requests", 0))
                lat = m_data.get("latency_ms", {})
                c1, c2 = st.columns(2)
                c1.metric("Avg Latency", f"{lat.get('avg', 0):.0f}ms")
                c2.metric("P95 Latency", f"{lat.get('p95', 0):.0f}ms")
                st.markdown(f"**🛡️ Guardrail Blocks:** `{m_data.get('routes', {}).get('refuse', 0)}`")
                st.markdown(f"**⚡ Vector DB:** `Qdrant (Metadata Pre-Filter)`")
                st.caption(f"Prometheus endpoint: `{api_url}/metrics?format=prometheus`")
            else:
                st.caption("Metrics endpoint unreachable.")
        except Exception:
            st.caption("Observability offline.")

# Switch conversation if employee changes
if st.session_state.get("owner") != employee_id:
    new_conversation()
    st.session_state.owner = employee_id

# ---- Main Chat View ----
st.title("🤖 Veltrix PeopleOps Assistant")
st.caption("AI-powered HR Assistant for Policy Guidance & Leave Management")

if not employee_id:
    st.warning("👈 Please select or enter an Employee ID in the sidebar to begin.")

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
prompt = st.chat_input("Ask a question about HR policies or manage your leave...", disabled=not employee_id)
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
