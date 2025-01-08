import os
import bs4
from langchain_openai import OpenAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain.hub import pull
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.embeddings import HuggingFaceEmbeddings


#EUMASTER4HPC DATA
"""
url1 = "https://eumaster4hpc.uni.lu/application/"
response = requests.get(url1)
soup = BeautifulSoup(response.text, 'html.parser')

links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']
    # Convert relative URLs to absolute URLs
    full_link = urljoin(url1, link)
    links.append(full_link)

valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]

print("Valid Extracted Links:", valid_links)

loader = AsyncHtmlLoader([url1] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()

html2text = Html2TextTransformer()
eumaster4hpc_data = html2text.transform_documents(docs)


#FIB DATA



url2 = "https://www.fib.upc.edu/en/studies/masters/eumaster4hpc/"
response = requests.get(url2)
soup = BeautifulSoup(response.text, 'html.parser')


links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']
   
    full_link = urljoin(url2, link)
    links.append(full_link)


valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]

print("Valid Extracted Links:", valid_links)

loader = AsyncHtmlLoader([url2] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()


html2text = Html2TextTransformer()
fib_data = html2text.transform_documents(docs)


#USI DATA




url3 = "https://www.usi.ch/en/education/master/computational-science/structure-and-contents/high-performance-computing"
response = requests.get(url3)
soup = BeautifulSoup(response.text, 'html.parser')


links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']
   
    full_link = urljoin(url3, link)
    links.append(full_link)


valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]

print("Valid Extracted Links:", valid_links)


loader = AsyncHtmlLoader([url3] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()

html2text = Html2TextTransformer()
usi_data = html2text.transform_documents(docs)


#MILANO DATA



url4 = "https://masterhpc.polimi.it/"
response = requests.get(url4)
soup = BeautifulSoup(response.text, 'html.parser')


links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']
   
    full_link = urljoin(url4, link)
    links.append(full_link)


valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]


print("Valid Extracted Links:", valid_links)


loader = AsyncHtmlLoader([url4] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()

html2text = Html2TextTransformer()
milano_data = html2text.transform_documents(docs)


#FAU DATA



url5 = "https://www.eumaster4hpc.tf.fau.eu/"
response = requests.get(url5)
soup = BeautifulSoup(response.text, 'html.parser')


links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']

    full_link = urljoin(url5, link)
    links.append(full_link)


valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]


print("Valid Extracted Links:", valid_links)


loader = AsyncHtmlLoader([url5] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()


html2text = Html2TextTransformer()
fau_data = html2text.transform_documents(docs)


#ERUOHPC DATA




url6 = "https://eurohpc-ju.europa.eu/about/discover-eurohpc-ju_en"
response = requests.get(url6)
soup = BeautifulSoup(response.text, 'html.parser')


links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']
  
    full_link = urljoin(url6, link)
    links.append(full_link)


valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]


print("Valid Extracted Links:", valid_links)

loader = AsyncHtmlLoader([url6] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()


html2text = Html2TextTransformer()
eurohpc_data = html2text.transform_documents(docs)

#KTH DATA




url7 = "https://www.kth.se/en/studies/master/computer-science/msc-computer-science-1.419974"
response = requests.get(url7)
soup = BeautifulSoup(response.text, 'html.parser')


links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']
 
    full_link = urljoin(url7, link)
    links.append(full_link)


valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]


print("Valid Extracted Links:", valid_links)


loader = AsyncHtmlLoader([url7] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()


html2text = Html2TextTransformer()
kth_data = html2text.transform_documents(docs)


# LUXEMBOURG DATA

url8 = "https://www.uni.lu/en/"
response = requests.get(url8)
soup = BeautifulSoup(response.text, 'html.parser')


links = []
for a_tag in soup.find_all('a', href=True):
    link = a_tag['href']
   
    full_link = urljoin(url8, link)
    links.append(full_link)

valid_links = [link for link in links if urlparse(link).scheme in ["http", "https"]]

print("Valid Extracted Links:", valid_links)


loader = AsyncHtmlLoader([url8] + valid_links)  # Include the main URL and the extracted links
docs = loader.load()


html2text = Html2TextTransformer()
luxembourg_data = html2text.transform_documents(docs)



#splitting data 
eumaster4hpc_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


fib_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


usi_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


milano_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


fau_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


eurohpc_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


kth_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

luxembourg_data_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


chuncked_eumaster4hpc_data = eumaster4hpc_data_text_splitter.split_documents(eumaster4hpc_data)
chuncked_fib_data = fib_data_text_splitter.split_documents(fib_data)
chuncked_usi_data = usi_data_text_splitter.split_documents(usi_data)
chuncked_milano_data = milano_data_text_splitter.split_documents(milano_data)
chuncked_fau_data= fau_data_text_splitter.split_documents(fau_data)
chuncked_eurohpc_data = eurohpc_data_text_splitter.split_documents(eurohpc_data)
chuncked_kth_data= kth_data_text_splitter.split_documents(kth_data)
chuncked_luxembourg_data= luxembourg_data_text_splitter.split_documents(luxembourg_data)
"""

"""
modelPath = "intfloat/multilingual-e5-large"
embed = HuggingFaceEmbeddings(
    model_name=modelPath,  # Provide the pre-trained model's path 
)


eumaster4hpc_data_vectorstore = Chroma.from_documents(chuncked_eumaster4hpc_data, embed)
eumaster4hpc_data_vectorstore.persist()


fib_data_vectorstore = Chroma.from_documents(chuncked_fib_data, embed)
fib_data_vectorstore.persist()


usi_data_vectorstore = Chroma.from_documents(chuncked_usi_data, embed)
usi_data_vectorstore.persist()


milano_data_vectorstore = Chroma.from_documents(chuncked_milano_data, embed)
milano_data_vectorstore.persist()


fau_data_vectorstore = Chroma.from_documents(chuncked_fau_data, embed)
fau_data_vectorstore.persist()


eurohpc_data_vectorstore = Chroma.from_documents(chuncked_eurohpc_data, embed)
eurohpc_data_vectorstore.persist()


kth_data_vectorstore = Chroma.from_documents(chuncked_kth_data, embed)
kth_data_vectorstore.persist()

luxembourg_data_vectorstore = Chroma.from_documents(chuncked_luxembourg_data, embed)
luxembourg_data_vectorstore.persist()

"""

modelPath = "intfloat/multilingual-e5-large"
embed = HuggingFaceEmbeddings(
    model_name=modelPath,  # Provide the pre-trained model's path 
)

eumaster4hpc_data_vectorstore2 = Chroma(persist_directory="C:/euhpc_db", embedding_function=embed)
fib_data_vectorstore2 = Chroma(persist_directory="C:/fibdata_db", embedding_function=embed)
usi_data_vectorstore2 = Chroma(persist_directory="C:/usi_db", embedding_function=embed)
milano_data_vectorstore2 = Chroma(persist_directory="C:/milano_db", embedding_function=embed)
fau_data_vectorstore2 = Chroma(persist_directory="C:/fau_db", embedding_function=embed)
eurohpc_data_vectorstore2 = Chroma(persist_directory="C:/eurohpc_db", embedding_function=embed)
kth_data_vectorstore2 = Chroma(persist_directory="C:/kth_db", embedding_function=embed)
luxembourg_data_vectorstore2 = Chroma(persist_directory="C:/luxembourg_db", embedding_function=embed)




retriever1 = eumaster4hpc_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever2 = fib_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever3 = usi_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever4 = milano_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever5 = fau_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever6 = eurohpc_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever7 = kth_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever8 = luxembourg_data_vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k":5})


from langchain.tools.retriever import create_retriever_tool


eumaster4hpc_data_retriever_tool=create_retriever_tool(retriever1,"EuMaster4HPC_search",
                      "Search for information about EUMaster4HPC. For any questions about EUMaster4HPC, you must use this tool!")


fib_data_retriever_tool=create_retriever_tool(retriever2,"fib_search",
                      "Search for information about FIB University. For any questions about FIB University, you must use this tool!")


usi_data_retriever_tool=create_retriever_tool(retriever3,"usi_search",
                      "Search for information about usi University. For any questions about usi University, you must use this tool!")


milano_data_retriever_tool=create_retriever_tool(retriever4,"Milano_search",
                      "Search for information about Milano Univversity. For any questions about Milano University, you must use this tool!")


fau_data_retriever_tool=create_retriever_tool(retriever5,"fau_search",
                      "Search for information about FAU University. For any questions about FAU University, you must use this tool!")


eurohpc_data_retriever_tool=create_retriever_tool(retriever6,"Eurohpc_search",
                      "Search for information about Eurohpc. For any questions about Eurohpc, you must use this tool!")


kth_data_retriever_tool=create_retriever_tool(retriever7,"kth_search",
                      "Search for information about kth University. For any questions about Kth University, you must use this tool!")


luxembourg_data_retriever_tool=create_retriever_tool(retriever8,"luxembourg_search",
                      "Search for information about Luxembourg University. For any questions about Luxembourg University, you must use this tool!")


tools=[eumaster4hpc_data_retriever_tool, fib_data_retriever_tool, usi_data_retriever_tool, milano_data_retriever_tool, fau_data_retriever_tool,eurohpc_data_retriever_tool,kth_data_retriever_tool,luxembourg_data_retriever_tool]


tools


# Replace 'your_api_key_here' with your actual API key
prompt = pull("hwchase17/openai-functions-agent", api_key="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")
prompt.messages

gpt = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo-1106",
    openai_api_key="7c3f9550b69c419aa0f1830e338ff562",
    openai_api_type="azure",
    openai_api_version="2023-12-01-preview",
    azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/",
    verbose=True,
    temperature=0.1,
)

agent = create_openai_tools_agent(gpt, tools, prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=False)


def get_response(user_input):
    response = agent_executor.invoke({"input": user_input})
    return response

