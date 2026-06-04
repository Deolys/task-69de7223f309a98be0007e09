import os
from typing import Dict, Any
from langchain.llms import OpenAI
from langchain.tools import BaseTool
import requests

# Simple web search tool using DuckDuckGo instant answer API
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for a query and return a short summary. Use only when you need up-to-date information."

    def _run(self, query: str) -> str:
        try:
            url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return data.get("AbstractText", "No summary available.")
        except Exception as e:
            return f"Error during web search: {e}"

# Virtual file system to store generated files in memory
class VirtualFileSystem:
    def __init__(self):
        self.files: Dict[str, str] = {}

    def write(self, path: str, content: str) -> None:
        self.files[path] = content

    def list_files(self) -> Dict[str, str]:
        return self.files

# DeepAgent that plans tasks using LLM and executes tools
class DeepAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self.tools = [WebSearchTool()]
        self.vfs = VirtualFileSystem()

    def run(self, goal: str) -> None:
        # Simple iterative loop: ask LLM for next step until finished
        steps = 0
        max_steps = 5
        context = ""
        while steps < max_steps:
            prompt = f"Goal: {goal}\nContext: {context}\nWhat is the next action? Provide JSON with keys 'action' (one of 'search', 'write_file', 'finish') and relevant parameters."
            response = self.llm(prompt)
            try:
                import json
                data = json.loads(response.strip())
            except Exception as e:
                print(f"Failed to parse LLM output: {e}\nResponse was:\n{response}")
                break
            action = data.get("action")
            if action == "search":
                query = data.get("query", "")
                result = self.tools[0]._run(query)
                context += f"\nSearch({query}) -> {result}"
            elif action == "write_file":
                path = data.get("path")
                content = data.get("content", "")
                if path:
                    self.vfs.write(path, content)
                    context += f"\nWrote file {path}"
            elif action == "finish":
                print("Agent finished. Virtual files created:\n")
                for p, c in self.vfs.list_files().items():
                    print(f"--- {p} ---\n{c}\n")
                break
            else:
                print(f"Unknown action: {action}")
                break
            steps += 1

if __name__ == "__main__":
    # Ensure OpenAI key is set in environment
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
    llm = OpenAI(temperature=0)
    agent = DeepAgent(llm)
    goal_text = "Create a README and example script that demonstrates searching the web for Python libraries."
    agent.run(goal_text)
