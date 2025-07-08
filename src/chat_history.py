import streamlit as st
import uuid
from langchain_redis.chat_message_history import RedisChatMessageHistory
from src.config import get_redis_url

def get_session_id():
    """
    Ensures a unique session ID exists for the user's browser tab.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def get_message_history():
    """
    Returns a RedisChatMessageHistory object for the current session,
    which syncs chat messages with the Redis database.
    """
    session_id = get_session_id()
    redis_url = get_redis_url()
    
    # --- THIS IS THE FIX ---
    # Pass redis_url as the second positional argument, not a keyword argument.
    return RedisChatMessageHistory(session_id, redis_url)