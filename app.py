"""
app.py — Forced full-width layout with explicit visible sidebar structure.
"""

import streamlit as st
from llm import ask_llm_stream

# Page configuration
st.set_page_config(
    page_title="SparkAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New chat"
if "recent_chats" not in st.session_state:
    st.session_state.recent_chats = []

# Clean CSS to ensure sidebar and header are never hidden
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="collapsedControl"] {
        display: block !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Sidebar Layout (Explicitly defined)
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✨ SparkAI")
    st.markdown("---")
    
    if st.button("New chat", key="btn_new", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat = "New chat"
        st.rerun()

    st.markdown("### Recent Chats")
    for chat in st.session_state.recent_chats:
        if st.button(chat, key=f"chat_{chat}", use_container_width=True):
            st.session_state.current_chat = chat
            st.rerun()

    st.markdown("---")
    st.caption("Sidebar controls active")

# -------------------------------------------------------------------------
# Main Chat Area
# -------------------------------------------------------------------------
st.markdown(f"### {st.session_state.current_chat}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask SparkAI"):
    if st.session_state.current_chat == "New chat" and prompt not in st.session_state.recent_chats:
        chat_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        st.session_state.current_chat = chat_title
        st.session_state.recent_chats.insert(0, chat_title)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        for chunk in ask_llm_stream(context="", question=prompt, history=st.session_state.messages[:-1]):
            full_response += chunk
            response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
