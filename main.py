import os
import json
import requests
from bs4 import BeautifulSoup
from deep_agents_from_scratch.agent import Agent
from openai import OpenAI

def search_perplexity(query: str) -> str:
    """Use Perplexity API to get a concise answer for the query."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY environment variable is required")
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai/v1")
    response = client.chat.completions.create(
        model="llama-3.1-sonar-large",
        messages=[{"role": "user", "content": query}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

class SimpleDeepAgent(Agent):
    def __init__(self, name: str = "PerplexitySearchAgent"):
        super().__init__(name=name)
        self.add_tool("search", search_perplexity)

    def run(self, query: str) -> dict:
        # Perform search and create virtual file content
        answer = self.run_tool("search", query=query)
        virtual_file = {
            "path": "output.txt",
            "content": f"Query: {query}\n\nAnswer:\n{answer}",
        }
        return virtual_file

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Agent that searches the web and creates a file")
    parser.add_argument("query", type=str, help="Search query for the agent")
    args = parser.parse_args()

    agent = SimpleDeepAgent()
    result = agent.run(args.query)
    # Write virtual file to disk (simulating upload)
    os.makedirs(os.path.dirname(result["path"]), exist_ok=True)
    with open(result["path"], "w", encoding="utf-8") as f:
        f.write(result["content"])
    print(f"Virtual file created at {result['path']}")
