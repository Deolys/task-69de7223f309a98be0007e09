import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.utilities import SerpAPIWrapper

load_dotenv()

# Simple tool to write a file
class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a virtual file. Provide filename and content."

    def _run(self, filename: str, content: str) -> str:
        Path(filename).write_text(content)
        return f"File {filename} written."

# Tool for web search using SerpAPI (requires SERPAPI_API_KEY env var)
search = SerpAPIWrapper()

llm = OpenAI(temperature=0, model_name="gpt-4o-mini")

tools = [WriteFileTool(), search]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

if __name__ == "__main__":
    # Example query: find recent news about LangChain and write to file
    prompt = (
        "Search the web for the latest developments in LangChain. Summarize the findings and write a markdown file named 'langchain_update.md' with the summary."
    )
    agent.run(prompt)
