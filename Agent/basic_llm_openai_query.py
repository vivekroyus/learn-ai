from openai import OpenAI
import os
from dotenv import load_dotenv  

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
åç
def basic_llm_query(query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Search Google for: {query}"}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Test the function
if __name__ == "__main__":
    result = basic_llm_query("Python programming")
    print(result)