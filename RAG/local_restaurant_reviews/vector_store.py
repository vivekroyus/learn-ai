# from langchain_core.tools import retriever  # This was causing the conflict
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from ollama import embeddings
import pandas as pd

df = pd.read_csv("RAG/local_restaurant_reviews/synthetic_reviews.csv")
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")
db_location =  "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name = "restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content = row["title"] + " " + row["text_of_review"],
            metadata = {"rating":row["rating"], "date":row["date_of_review"]}
        )
        ids.append(str(i))
        documents.append(document)

    vector_store.add_documents(documents=documents, ids = ids)

retriever = vector_store.as_retriever(
        search_kwargs = {"k": 5}
)

# Export the retriever for use in other modules
__all__ = ['retriever']