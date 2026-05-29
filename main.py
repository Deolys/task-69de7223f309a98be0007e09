import os
import json
import requests
from bs4 import BeautifulSoup

# Simple deep agent that searches the web and creates virtual files

def search_web(query: str, num_results: int = 3) -> list[dict]:
    """Return a list of search result dicts with title and snippet."""
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    resp = requests.get(url, params=params)
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.select("a.result__a")[:num_results]:
        title = a.get_text()
        link = a['href']
        snippet_tag = a.find_next_sibling("div", class_="result__snippet")
        snippet = snippet_tag.get_text() if snippet_tag else ""
        results.append({"title": title, "link": link, "snippet": snippet})
    return results


def fetch_page(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(['script', 'style']):
        script.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text


def create_virtual_file(name: str, content: str) -> None:
    # In a real scenario we might store this in memory or a database.
    # Here we simply write to disk for demonstration.
    with open(name, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    query = input("Enter search query: ")
    results = search_web(query)
    print(f"Found {len(results)} results.")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']}\n   {res['link']}")

    # Fetch first result and create a virtual file
    if results:
        url = results[0]["link"]
        html = fetch_page(url)
        text = extract_text(html)
        filename = "virtual_page.txt"
        create_virtual_file(filename, text)
        print(f"Virtual file '{filename}' created with extracted content.")

    # Example of creating multiple virtual files from queries
    topics = ["deep learning", "python programming", "data science"]
    for topic in topics:
        page_text = extract_text(fetch_page(f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}") or "")
        if page_text:
            create_virtual_file(f"{topic.replace(' ', '_')}.txt", page_text)
            print(f"Created file for topic: {topic}")

if __name__ == "__main__":
    main()
