"""Test script for FireCrawlWebReader with the updated API."""

import os
from llama_index.readers.web.firecrawl_web.base import FireCrawlWebReader

def test_firecrawl_reader():
    # Get API key from environment variable
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("Please set FIRECRAWL_API_KEY environment variable")

    # Initialize the reader with different modes
    print("\nTesting scrape mode...")
    scrape_reader = FireCrawlWebReader(
        api_key=api_key,
        mode="scrape",
        params={
            "timeout": 40000
        }
    )
    scrape_docs = scrape_reader.load_data(url="https://www.paulgraham.com/worked.html?v=123")
    print(f"Scrape mode documents: {len(scrape_docs)}")
    print(f"First document metadata: {scrape_docs[0].metadata}")

    print("\nTesting crawl mode...")
    crawl_reader = FireCrawlWebReader(
        api_key=api_key,
        mode="crawl",
        params={
            "delay": 2,
            "limit": 5
        }
    )
    crawl_docs = crawl_reader.load_data(url="https://mairistumpf.com")
    print(f"Crawl mode documents: {len(crawl_docs)}")
    print(f"First document metadata: {crawl_docs[0].metadata}")

    print("\nTesting search mode...")
    search_reader = FireCrawlWebReader(
        api_key=api_key,
        mode="search",
        params={
            "limit": 3
        }
    )
    search_docs = search_reader.load_data(query="Who is the president of the United States?")
    print(f"Search mode documents: {len(search_docs)}")
    print(f"First document metadata: {search_docs[0].metadata}")

    print("\nTesting extract mode...")
    extract_reader = FireCrawlWebReader(
        api_key=api_key,
        mode="extract",
        params={
            "prompt": "Extract the main topics and key points from this essay",
        },
    )
    extract_docs = extract_reader.load_data(urls=["https://www.paulgraham.com/worked.html"])
    print(f"Extract mode documents: {len(extract_docs)}")
    print(f"First document metadata: {extract_docs[0].metadata}")

if __name__ == "__main__":
    test_firecrawl_reader() 