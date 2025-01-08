import streamlit as st
import random
from streamlit_chat import message
import os
from langchain_openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain.hub import pull
from agentsrag import get_response



@st.cache_data
def cached_get_response(user_input):
    response = get_response({"input": user_input})
    return response.get("output")

st.header("(Retrieval-Augmented Generation)AGENTS Chat Bot")

with st.sidebar:
    st.title('Chat Bot')
    st.image('https://www.informatik.studium.fau.de/files/2023/01/EUMasterHPC-scaled.jpg')
    
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"": "assistant", "content": "Hi, I am your chatbot. Ask me your questions."}
    ]

prompt = st.chat_input("Enter your Prompt here")

if prompt:
    st.session_state.messages.append({"": "user", "content": prompt})

for msg in st.session_state.messages:
    with st.chat_message(msg[""]):
        st.write(msg["content"])

if st.session_state.messages[-1][""] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_input = st.session_state.messages[-1]["content"]
            response = cached_get_response(user_input)
            st.write(response)
            st.session_state.messages.append({"": "assistant", "content": response})

