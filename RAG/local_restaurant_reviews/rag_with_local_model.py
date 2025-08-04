
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retriever

model = OllamaLLM(model="llama3.2")
template = """
You are a helpful assistant that can answer questions about a restaurant.

Here are some relavant reviews : {reviews}

Here is the question to answer : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model 

while True:
    print("\n \n -----------------------------")
    question = input("What is your question (q to quit)" )
    print("\n \n -----------------------------")

    if question == 'q':
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, 
                           "question": question})
    print(result)