#!/usr/bin/env python3
"""
Simple LangChain agent that generates a summary and writes it to a file.
"""
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def main():
    # Initialize the LLM (requires OPENAI_API_KEY env variable)
    llm = OpenAI(temperature=0.2, model="gpt-4o-mini")
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a concise summary about {topic}.",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    topic = "Python programming language"
    summary = chain.run(topic)

    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

if __name__ == "__main__":
    main()
