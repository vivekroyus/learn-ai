from tabnanny import verbose
from openai import OpenAI
import os
from dotenv import load_dotenv  
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def google_search_agent(prompt):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent.invoke(prompt)


if __name__ == "__main__":
    result = google_search_agent("Who is the chief minister of Bihar?")
    print(result)