import sys
from pathlib import Path
from deep_agents_from_scratch import DeepAgent, PerplexityTool

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <search query>")
        return
    query = " ".join(sys.argv[1:])
    agent = DeepAgent(tools=[PerplexityTool()], verbose=True)
    result = agent.run(query)
    out_path = Path("output.txt")
    out_path.write_text(result, encoding="utf-8")
    print(f"Result written to {out_path.resolve()}")

if __name__ == "__main__":
    main()
