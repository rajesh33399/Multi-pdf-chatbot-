"""
persistence.py — per-user chat history storage for SparkAI, backed by
Supabase (hosted Postgres).

Table expected in your Supabase project (run once in the SQL editor):

    create table if not exists user_chats (
        user_email text primary key,
        chats_json jsonb not null,
        updated_at timestamptz default now()
    );

Auth model: this app talks to Supabase from the Streamlit server only
(never from the browser), using the service_role key, which bypasses
Row Level Security entirely. That's appropriate here because access
control is already enforced upstream by Google login (st.user.email) —
Streamlit's server code is the only thing that ever calls this module.
Never ship the service_role key to any client-side/browser code.

Binary fields (generated images/videos, under message["data"]) are
base64-encoded before being written into the jsonb column, since JSON
has no native binary type, and decoded back to bytes on load.

Caveat: base64-encoded media inflates payload size by ~33% and Supabase
has per-request/row size limits on the free tier. Fine for a handful of
generated images per chat; if usage grows, move media to Supabase
Storage (a bucket) and store just the object path/URL in chats_json
instead of the raw bytes — save_user_chats/load_user_chats below are the
only place that would need to change.
"""
import base64

import streamlit as st
from supabase import create_client, Client

TABLE_NAME = "user_chats"


@st.cache_resource
def _get_client() -> Client:
    url = st.secrets["supabase"]["url"]
    service_key = st.secrets["supabase"]["service_key"]
    return create_client(url, service_key)


def _encode_bytes_fields(chats: dict) -> dict:
    """Return a deep copy of `chats` with any bytes/bytearray message
    fields (message["data"] for image/video messages) base64-encoded, so
    the whole structure is JSON-serializable for the jsonb column."""
    encoded = {}
    for chat_id, chat in chats.items():
        chat_copy = dict(chat)
        new_messages = []
        for msg in chat.get("messages", []):
            msg_copy = dict(msg)
            data = msg_copy.get("data")
            if isinstance(data, (bytes, bytearray)):
                msg_copy["data"] = base64.b64encode(bytes(data)).decode("ascii")
                msg_copy["_data_is_b64"] = True
            new_messages.append(msg_copy)
        chat_copy["messages"] = new_messages
        encoded[chat_id] = chat_copy
    return encoded


def _decode_bytes_fields(chats: dict) -> dict:
    """Inverse of _encode_bytes_fields — turns base64 strings back into
    real bytes for st.image / st.video to consume."""
    for chat in chats.values():
        for msg in chat.get("messages", []):
            if msg.pop("_data_is_b64", False):
                msg["data"] = base64.b64decode(msg["data"])
    return chats


def save_user_chats(user_email: str, chats: dict) -> None:
    """Upsert this user's entire chats dict. Safe to call every rerun."""
    if not user_email:
        return
    payload = _encode_bytes_fields(chats)
    client = _get_client()
    try:
        client.table(TABLE_NAME).upsert(
            {"user_email": user_email, "chats_json": payload}
        ).execute()
    except Exception as e:
        # Don't crash the whole app UI over a failed save — surface it
        # quietly so the user's current session keeps working even if
        # this particular write didn't land.
        st.toast(f"⚠️ Couldn't save chat history: {e}")


def load_user_chats(user_email: str) -> dict:
    """Load this user's chats dict, or {} if they have none saved yet
    (first-ever login) or the fetch fails."""
    if not user_email:
        return {}
    client = _get_client()
    try:
        res = (
            client.table(TABLE_NAME)
            .select("chats_json")
            .eq("user_email", user_email)
            .limit(1)
            .execute()
        )
    except Exception as e:
        st.toast(f"⚠️ Couldn't load chat history: {e}")
        return {}
    if not res.data:
        return {}
    return _decode_bytes_fields(res.data[0]["chats_json"])


def delete_user_chats(user_email: str) -> None:
    """Optional helper — wipe a user's saved history entirely."""
    if not user_email:
        return
    client = _get_client()
    client.table(TABLE_NAME).delete().eq("user_email", user_email).execute()
