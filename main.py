import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import Dict, Any

# Load environment variables (API keys)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# Simple virtual file system as a dict
virtual_fs: Dict[str, str] = {}

# Tool to write a virtual file
class WriteVirtualFileTool:
    name = "write_virtual_file"
    description = "Write content to a virtual file. The file will be saved locally at the end of execution."

    def __call__(self, file_name: str, content: str) -> str:
        virtual_fs[file_name] = content
        return f"Virtual file '{file_name}' written." 

# Tool to search the web (DuckDuckGo)
search_tool = DuckDuckGoSearchRun(name="duckduckgo_search", description="Search the web for information.")

# Define agent tools list
tools = [WriteVirtualFileTool(), search_tool]

# Prompt template for the deep agent
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can search the internet and write virtual files. Use the provided tools.")
])

# Create OpenAI function calling agent
agent = create_openai_functions_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    tools=[t for t in tools],
    prompt=prompt_template,
)

# Wrap into executor
executor = AgentExecutor(agent=agent, tools=[t for t in tools], verbose=True)

# Main workflow: ask user a question, agent searches and writes files
if __name__ == "__main__":
    # Example task: gather info about deep agents and save to file
    user_query = (
        "Write a brief summary of the DeepAgents from Scratch project and store it in a file named 'deepagents_summary.txt'."
    )
    print("Running agent...")
    result = executor.invoke({"input": user_query})
    print("Agent finished.")

    # Persist virtual files to disk
    output_dir = Path("output_files")
    output_dir.mkdir(exist_ok=True)
    for fname, content in virtual_fs.items():
        file_path = output_dir / fname
        file_path.write_text(content)
        print(f"Saved {file_path}")

    # Optionally, return the result of the last tool call
    print("Result:", result.get("output", "No output returned."))
