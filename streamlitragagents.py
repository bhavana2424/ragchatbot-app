import streamlit as st
from streamlit_chat import message
from agentsrag import AzureChatOpenAI
from langchain.agents import AgentExecutor
from langchain_community.vectorstores import Chroma

# Set up a simple retrieval-augmented generation chat bot with Streamlit

# Caching the AzureChatOpenAI responses for efficient processing
@st.cache_data
def cached_get_response(user_input):
    return chat_agent.invoke({"input": user_input})

# Set up page config
st.set_page_config(page_title="💬 RAG Chatbot")

# Header
st.header("Retrieval-Augmented Generation Chat Bot")

# Sidebar with some images related to EUMaster4HPC
with st.sidebar:
    st.title('Chat Bot')
    st.image('https://eurocc.fccn.pt/wp-content/uploads/2024/01/EUMaster4HPC.jpg')
    st.image('https://eurohpc-ju.europa.eu/sites/default/files/2022-03/EUMaster4HPC_logo.jpg')

# Initialize session state to hold messages and the chat agent
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I am your chatbot, ask me your queries. I am here to assist you."}
    ]

    # Assuming you've already created an agent using langchain's AgentExecutor
    from langchain import OpenAI, hub
    from langchain.tools.retriever import create_retriever_tool
    
    # Setup the agents, retrievers, and embeddings as you did in the previous code
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import WebBaseLoader

    # Load and process document data (same as before)
    loader = WebBaseLoader(["https://eumaster4hpc.uni.lu/", "https://www.kth.se/en/studies/master/computer-science/msc-computer-science-1.419974"])
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = Chroma.from_documents(chunked_docs, embed, persist_directory="C:/hpc_db")
    vectorstore.persist()

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Setting up the Azure Chat model
    gpt = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo-1106",
        openai_api_key="7c3f9550b69c419aa0f1830e338ff562",
        openai_api_type="azure",
        openai_api_version="2023-12-01-preview",
        azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/",
        verbose=True,
        temperature=0.1,
    )
    
    # Create the agent executor
    from langchain.agents import create_openai_tools_agent
    agent = create_openai_tools_agent(gpt, tools=[retriever], prompt=None)
    chat_agent = AgentExecutor(agent=agent, tools=[retriever], verbose=False)
    
# Chat interface
if prompt := st.chat_input("Enter your Prompt here"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# If last message is from user, generate a response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_input = st.session_state.messages[-1]["content"]
            response = cached_get_response(user_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            