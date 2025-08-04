from typing import TypedDict, List
import bs4
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
load_dotenv()

from langchain import hub
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load and chunk contents of the blog
def load_and_chunk_blog(url="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = init_chat_model(model="gpt-3.5-turbo", temperature=0, model_provider="openai")

# Load and index chunks
all_splits = load_and_chunk_blog()
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")
    

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

if __name__ == "__main__": 
    response = graph.invoke({"question": "What is Task Decomposition?"})
    print(response["answer"]) 
    