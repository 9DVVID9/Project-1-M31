"""
Conversational Traffic Assistant — Phase 2 Chatbot UI
Arago Street, Barcelona

Run with:
    streamlit run app.py

Requires GEMINI_API_KEY environment variable.
"""

import os
import streamlit as st

from src.connector import ModelConnector
from src.llm_enhancer import GeminiEnhancer

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Arago Traffic Assistant",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached model loading (runs once, shared across all sessions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading traffic models...")
def load_pipeline() -> tuple[ModelConnector, GeminiEnhancer]:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    mc  = ModelConnector()
    llm = GeminiEnhancer(api_key=api_key)
    return mc, llm

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚦 About")
    st.write(
        "Real-time traffic forecasts for **Arago Street, Barcelona** "
        "powered by a Random Forest model (~75% accuracy) and Gemini AI."
    )
    st.divider()

    st.subheader("Try asking")
    suggestions = [
        "Is it busy on Monday at 8am?",
        "How is traffic right now?",
        "Best time to drive on Friday?",
        "Hay trafico el sabado por la tarde?",
        "When should I avoid Arago Street?",
    ]
    for s in suggestions:
        st.caption(f"• {s}")

    st.divider()
    st.caption("Group 3 · UPC · Phase 2")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚦 Arago Street Traffic Assistant")
st.caption("Barcelona, Spain · Ask in English or Spanish")

# ---------------------------------------------------------------------------
# Session state — chat history
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------

_LEVEL_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            level = meta.get("traffic_level", "")
            icon  = _LEVEL_ICON.get(level, "")
            temp  = meta.get("weather", {}).get("temperature", "?")
            st.caption(
                f"{icon} **{level}** traffic · "
                f"{meta.get('day_name', '')} {meta.get('hour', 0):02d}:00 · "
                f"{temp}°C"
            )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about traffic on Arago Street..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Checking traffic conditions..."):
            try:
                mc, llm = load_pipeline()
                result  = llm.respond(prompt, mc)

                st.write(result["text"])

                level = result.get("traffic_level", "")
                icon  = _LEVEL_ICON.get(level, "")
                temp  = result.get("weather", {}).get("temperature", "?")
                st.caption(
                    f"{icon} **{level}** traffic · "
                    f"{result.get('day_name', '')} {result.get('hour', 0):02d}:00 · "
                    f"{temp}°C"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["text"],
                    "meta": result,
                })

            except ValueError as e:
                # Missing API key
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong. Please try again. ({e})")
