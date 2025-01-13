import streamlit as st
import random
from streamlit_chat import message
from gptragbackend import get_response

@st.cache_data
def cached_get_response(user_input):
    response = get_response(user_input)
    return response
  
st.set_page_config(page_title="💬Chatbot")
st.header("Retrieval-Augmented Generation Chat Bot")

# Sidebar for chatbot information
with st.sidebar:
    st.title('Chat Bot')
    st.image('https://www.informatik.studium.fau.de/files/2023/01/EUMasterHPC-scaled.jpg')

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I am your chatbot, ask me your queries. I am here to assist you."}
    ]
# Set up the chat engine using the get_response function
    st.session_state.chat_engine = get_response

# Capture user input from the chat input widget    
if prompt := st.chat_input("Enter your Prompt here"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display all messages in the chat session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Check if the last message was from the user        
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Incorporating the conversation memory snippet
            user_input = st.session_state.messages[-1]["content"]
            response = cached_get_response(prompt)
            #response = get_response(user_input)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            