import openai
import os
from dotenv import load_dotenv  

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def google_search(query):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=f"Search Google for: {query}",
        max_tokens=100
    )
    return response.choices[0].text