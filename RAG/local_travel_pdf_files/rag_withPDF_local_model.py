from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store_PDF import retriever

model = OllamaLLM(model="llama3.2")
template = """
You are a helpful travel assistant that can answer questions about travel documents, reservations, and travel information.

Here are some relevant travel documents: {documents}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n ----------------------------------------")
    question = input("Type in your travel question (q to quit): ")
    print("\n\n ----------------------------------------")
    
    if question == 'q':
        break
    
    documents = retriever.invoke(question)
    result = chain.invoke({"documents": documents, "question": question})
    print(result)