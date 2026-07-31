"""
app.py — SparkChat frontend featuring the exact Google Gemini style sidebar with SVG icons and SparkAI branding.
"""

import streamlit as st
from llm import ask_llm_stream

# Page configuration
st.set_page_config(
    page_title="SparkAI",
    page_icon="✨",
    layout="wide"
)

# Initialize session state for sidebar toggle and chat history
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New chat"
if "recent_chats" not in st.session_state:
    st.session_state.recent_chats = []

# Custom CSS for styling, layout, and custom sidebar navigation items
st.markdown("""
    <style>
    /* Main background and font styling */
    .stApp {
        background-color: #ffffff;
        color: #1f1f1f;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
        padding-top: 10px;
    }
    
    /* Hide default streamlit header elements for clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom chat input styling at the bottom */
    .stChatInputContainer {
        border-radius: 24px;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Sidebar Layout (Gemini Style with SparkAI Branding & Exact SVG Icons)
# -------------------------------------------------------------------------
with st.sidebar:
    # Header: Spark Symbol and SparkAI Name
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("### ✨")
    with col_title:
        st.markdown("### SparkAI")
    
    st.markdown("---")
    
    # Navigation items
    if st.button("New chat", key="btn_new", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat = "New chat"
        st.rerun()

    if st.button("Search chats", key="btn_search", use_container_width=True):
        st.toast("Search feature coming soon!")

    if st.button("Images", key="btn_images", use_container_width=True):
        st.toast("Images view selected")

    if st.button("Videos", key="btn_videos", use_container_width=True):
        st.toast("Videos view selected")

    if st.button("Library", key="btn_library", use_container_width=True):
        st.toast("Library view selected")
        
    st.markdown("### Notebooks")
    if st.button("➕ New notebook", use_container_width=True):
        st.toast("Notebook created!")
        
    st.markdown("### Recent")
    
    # Dynamic recent chats list based on user activity
    for chat in st.session_state.recent_chats:
        col_item, col_menu = st.columns([5, 1])
        with col_item:
            if st.button(chat, key=f"chat_{chat}", use_container_width=True):
                st.session_state.current_chat = chat
                st.session_state.messages = [{"role": "assistant", "content": f"Loaded history for: {chat}"}]
                st.rerun()
        with col_menu:
            if st.button("⋮", key=f"menu_{chat}"):
                st.toast(f"Options for: {chat}")

    st.markdown("---")

# -------------------------------------------------------------------------
# Main Chat Area
# -------------------------------------------------------------------------
st.markdown(f"### {st.session_state.current_chat}")

# Render message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at the bottom
if prompt := st.chat_input("Ask SparkAI"):
    # If starting a new conversation from a prompt, add it to recent chats dynamically
    if st.session_state.current_chat == "New chat" and prompt not in st.session_state.recent_chats:
        chat_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        st.session_state.current_chat = chat_title
        st.session_state.recent_chats.insert(0, chat_title)

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response using streaming backend
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        for chunk in ask_llm_stream(context="", question=prompt, history=st.session_state.messages[:-1]):
            full_response += chunk
            response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
    
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
