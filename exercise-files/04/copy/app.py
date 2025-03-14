import streamlit as st
import openai
import os
from dotenv import load_dotenv
from main import query

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
MODEL_ENGINE = "gpt-3.5-turbo"

st.title("🤖 Chatbot App")
chat_placeholder = st.empty()


def init_chat_history():
    """Initialize chat history with a system message."""
    pass


def start_chat():
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    pass


if __name__ == "__main__":
    init_chat_history()
    start_chat()
