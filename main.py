import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.utilities import SerpAPIWrapper

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

class FileCreator(BaseTool):
    name = "FileCreator"
    description = "Creates a virtual file with given content. Returns the file path."

    def _run(self, file_name: str, content: str) -> str:
        # Write to local filesystem (virtual for this exercise)
        Path(file_name).write_text(content)
        return f"File {file_name} created."

# Initialize tools
search = SerpAPIWrapper()
file_creator = FileCreator()

tools = [search, file_creator]

# LLM and agent setup
llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Example query to demonstrate functionality
query = "deep agents from scratch langchain"
response = agent.run(query)
print("Agent response:", response)

# Create a file with the agent's response
file_name = "agent_output.txt"
file_creator._run(file_name, response)
print(f"Output written to {file_name}")
