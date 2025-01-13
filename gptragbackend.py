"""
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# List of initial URLs to scrape
urls = [
    "https://eumaster4hpc.uni.lu/application/",
    "https://www.fib.upc.edu/en/studies/masters/eumaster4hpc/",
    "https://www.usi.ch/en/education/master/computational-science/structure-and-contents/high-performance-computing",
    "https://masterhpc.polimi.it/",
    "https://www.eumaster4hpc.tf.fau.eu/",
    "https://eurohpc-ju.europa.eu/about/discover-eurohpc-ju_en",
    "https://www.kth.se/en/studies/master/computer-science/msc-computer-science-1.419974",
    "https://www.uni.lu/en/"
]

# Directory to save Markdown files
os.makedirs("scraped_data", exist_ok=True)

def extract_all_links(soup, base_url):
# Extracts and returns all absolute hyperlinks from the given BeautifulSoup object.
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = urljoin(base_url, a_tag["href"])
        if href.startswith("http"):  # Ensure the link is a valid HTTP URL
            links.append(href)
    return links

def scrape_and_collect(url, depth, max_depth, collected_data):
    if depth > max_depth or url in visited_urls:
        return

    try:
        # Mark the URL as visited
        visited_urls.add(url)
        print(f"Scraping: {url} (Depth: {depth})")

        # Fetch content using requests
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract page title and content
        title = soup.title.string if soup.title else "Untitled"
        main_content = soup.get_text()

        # Extract all hyperlinks
        links = extract_all_links(soup, url)

        # Append the content for this page to the collected data
        collected_data.append(f"### {title}\n\n**URL:** {url}\n\n{main_content}\n\n---\n\n")

        # Recursively scrape links
        for link in links:
            scrape_and_collect(link, depth + 1, max_depth, collected_data)

    except Exception as e:
        print(f"Failed to scrape {url}: {str(e)}")

# Set of visited URLs to avoid duplicating work
visited_urls = set()

# Set the maximum depth for recursive scraping
max_depth = 4

# Start the scraping process
for url in urls:
    # Reset the visited URLs for each initial URL
    visited_urls.clear()
    collected_data = []

    # Scrape and collect data
    scrape_and_collect(url, 0, max_depth, collected_data)

    # Save all collected data to a single .md file for this URL
    filename = os.path.join("scraped_data", url.replace("https://", "").replace("http://", "").replace("/", "_") + ".md")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# {url}\n\n" + "\n".join(collected_data))

print("All data has been scraped and saved successfully.")
"""

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from langchain_community.document_loaders import WebBaseLoader
import json
import os
import bs4
from langchain_openai import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableMap, RunnablePassthrough

gpt = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo-1106",
    openai_api_key="7c3f9550b69c419aa0f1830e338ff562",
    openai_api_type="azure",
    openai_api_version="2023-12-01-preview",
    azure_endpoint="https://chatbotopenaikeyswe.openai.azure.com/",
    verbose=True,
    temperature=0.0,
)

#loader = DirectoryLoader('/Users/bhavanas/Desktop/bhavsrag/scraped_data', glob="**/*.md")
#docs = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(
    #chunk_size=1000,
    #chunk_overlap=200,
    #length_function=len,
    #is_separator_regex=False,
#)
#texts = text_splitter.split_documents(docs)
#vectorstore = Chroma.from_documents(texts, embed, persist_directory="C:/ragevctorstore_db")
#vectorstore.persist()
#from langchain.vectorstores import Chroma
#from langchain.schema import Document  # Import the Document class

# Function to process data in smaller batches
#def process_in_batches(texts, embed, batch_size, persist_directory):
    # Initialize the Chroma vector store
    #vectorstore = Chroma(embedding_function=embed, persist_directory=persist_directory)
    
    # Extract the page_content from each Document object
    #texts = [doc.page_content for doc in texts if isinstance(doc, Document)]
    
    # Process documents in batches
    #for i in range(0, len(texts), batch_size):
        #batch_texts = texts[i:i + batch_size]
        
        # Ensure batch_texts contains only strings and filter out empty strings
        #batch_texts = [text for text in batch_texts if text and text.strip()]
        
        # Check if there are any texts left to process
        #if not batch_texts:
            #continue
        
        # Create Document objects
       #batch_docs = [Document(page_content=text) for text in batch_texts]
        
        # Debugging: Print the number of documents being added
        #print(f"Adding {len(batch_docs)} documents to the vector store.")
        
        # Add documents to the vector store
        #vectorstore.add_documents(batch_docs)
        #print("Documents added successfully.")
    
    # Persist the vector store
    #vectorstore.persist()
    #return vectorstore

# Define the batch size
#batch_size = 5461

# Call the function to process documents in batches
#vectorstore = process_in_batches(texts, embed, batch_size, persist_directory="C:/ragevctorstore_db")

# Verify the number of documents
#print(f"Number of documents in the vector store: {len(vectorstore)}")


# Initialize embeddings
modelPath = "intfloat/multilingual-e5-large"
embed = HuggingFaceEmbeddings(
model_name=modelPath,  # Provide the pre-trained model's path
)


template = """
You are an assistant for question-answering tasks named EU Portal Chatbot. 
For general questions, greetings (e.g., "Hi," "Hello"), or small talk, engage politely and warmly, 
using your knowledge to make the conversation pleasant and informative. 
For task-specific questions, rely solely on the provided pieces of retrieved context to answer and do not use outside information unless the query involves general knowledge or small-talk scenarios. 
If the question is out of scope or the context is insufficient, use your general knowledge to provide a helpful response if possible. 
Otherwise, respond with: "I'm sorry, but I don't have enough information to answer that question."

Be concise, approachable, and limit your responses to five sentences while using technical jargon only 
when necessary and ensuring it is understandable.

Question: {query}

Context:
{context}

Answer:
    
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(pages):
    return "\n\n".join(doc.page_content for doc in pages)

from langchain_community.vectorstores import Chroma

# Initialize each vector store and create a retriever for each
vectorstore1 = Chroma(persist_directory="C:/euhpc_db", embedding_function=embed)
retriever1 = vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore2 = Chroma(persist_directory="C:/eurohpc_db", embedding_function=embed)
retriever2 = vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore3 = Chroma(persist_directory="C:/fau_db", embedding_function=embed)
retriever3 = vectorstore3.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore4 = Chroma(persist_directory="C:/fib_db", embedding_function=embed)
retriever4 = vectorstore4.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore5 = Chroma(persist_directory="C:/fibdata_db", embedding_function=embed)
retriever5 = vectorstore5.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore6 = Chroma(persist_directory="C:/italinia_db", embedding_function=embed)
retriever6 = vectorstore6.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore7 = Chroma(persist_directory="C:/kth_db", embedding_function=embed)
retriever7 = vectorstore7.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore8 = Chroma(persist_directory="C:/luxembourg_db", embedding_function=embed)
retriever8 = vectorstore8.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore9 = Chroma(persist_directory="C:/milano_db", embedding_function=embed)
retriever9 = vectorstore9.as_retriever(search_type="similarity", search_kwargs={"k": 5})

vectorstore10 = Chroma(persist_directory="C:/usi_db", embedding_function=embed)
retriever10 = vectorstore10.as_retriever(search_type="similarity", search_kwargs={"k": 5})

"""
# Custom function to combine results from multiple retrievers
def combined_retriever(query):
    # Collect results from each retriever
    results = []
    for retriever in [retriever1, retriever2, retriever3, retriever4, retriever5, retriever6, retriever7, retriever8, retriever9, retriever10]:
        results.extend(retriever.get_relevant_documents(query))
    
    # Optionally, sort or filter the results based on relevance
    # Here, we simply return the combined list of results
    return results
"""

def combined_retriever(query):
    results = []
    for retriever in [retriever1, retriever2, retriever3, retriever4, retriever5,
                      retriever6, retriever7, retriever8, retriever9, retriever10]:
        results.extend(retriever.invoke(query))
    return results


def format_docs(pages):
    return "\n\n".join(doc.page_content for doc in pages)

from langchain_core.runnables import RunnableMap, RunnablePassthrough

rag_chain = RunnableMap(
    {
        "context": lambda inputs: format_docs(combined_retriever(inputs["query"])),
        "query": lambda inputs: inputs["query"],
    }
) | prompt | gpt | StrOutputParser()

"""
rag_chain = RunnableMap(
    {
        "context": lambda query: format_docs(combined_retriever(query)),
        "query": RunnablePassthrough(),
    }
) |prompt | gpt | StrOutputParser()
"""

def get_response(user_input):
    response = rag_chain.invoke({"query": user_input})
    return response



