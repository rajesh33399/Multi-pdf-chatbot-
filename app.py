"""
app.py — SparkChat frontend featuring explicit sidebar controls and recovery layout.
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

# Custom CSS to guarantee layout rendering
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #1f1f1f;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
        padding-top: 10px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Sidebar Layout
# -------------------------------------------------------------------------
with st.sidebar:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("### ✨")
    with col_title:
        st.markdown("### SparkAI")
    
    st.markdown("---")
    
    if st.button("New chat", key="btn_new", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat = "New chat"
        st.rerun()

    if st.button("Search chats", key="btn_search", use_container_width=True):
        st.toast("Search feature coming soon!")

    if st.button("Library", key="btn_library", use_container_width=True):
        st.toast("Library view selected")
        
    st.markdown("### Notebooks")
    if st.button("➕ New notebook", use_container_width=True):
        st.toast("Notebook created!")
        
    st.markdown("### Recent")
    for chat in st.session_state.recent_chats:
        if st.button(chat, key=f"chat_{chat}", use_container_width=True):
            st.session_state.current_chat = chat
            st.rerun()

    st.markdown("---")

# -------------------------------------------------------------------------
# Main Chat Area
# -------------------------------------------------------------------------
# Top header layout to ensure navigation control is visible even if sidebar is closed
col_head1, col_head2 = st.columns([6, 1])
with col_head1:
    st.markdown(f"### {st.session_state.current_chat}")
with col_head2:
    if st.button("📂 Toggle Sidebar", use_container_width=True):
        st.toast("Use the top-left arrow in your browser toolbar to open/close the sidebar if needed.")

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
