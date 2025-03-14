import streamlit as st
import openai
import os
from dotenv import load_dotenv
from query import query

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = API_KEY
MODEL_ENGINE = "gpt-3.5-turbo"

st.title("🤖 Q&A App")
chat_placeholder = st.empty()

# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

def init_chat_history():
    """Initialize chat history with a system message."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        # st.session_state["messages"].append(
        #     {"role": "system", "content": "You are a helpful assistant. Ask me anything!"}
        # )
        # print(st.session_state["messages"] )


def start_chat():
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for message in st.session_state.messages:
            # if message["role"] == "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # accept user input
    if prompt := st.chat_input("What is up?"):
        # add the user message to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # display the user mesage in
        with st.chat_message("user"):
            st.markdown(prompt)

            # print(st.session_state.messages)
            # print(content := st.session_state.messages[-1]["content"])

        # generate the response
        response = query(prompt) 
        
        # message_placeholder.markdown(response
        with st.chat_message("assistant"):
            st.markdown(response["answer"])


        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        # print(st.session_state.messages)    
            

if __name__ == "__main__":
    init_chat_history()
    start_chat()