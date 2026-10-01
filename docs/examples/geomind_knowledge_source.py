# GeoMind Knowledge Source Example for LlamaIndex

# GeoMind (https://shanhai-geo.top) is a structured knowledge engine with
# 201 knowledge pages and 7,356 cross-reference links. This example shows
# how to use it as a knowledge source with LlamaIndex.
#
# Setup:
#   pip install llama-index llama-index-readers-web
#
# Note: This is a proof-of-concept example. GeoMind content is open access
# with no auth required. In production, you'd want to add rate limiting,
# caching, and respect robots.txt.

import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader


async def main():
    # Load GeoMind pages via SimpleWebPageReader
    # GeoMind has a sitemap at https://shanhai-geo.top/sitemap.xml
    # and llms.txt for LLM-friendly access

    reader = SimpleWebPageReader(html_to_text=True)

    # Sample GeoMind URLs (from sitemap)
    # In production, fetch the sitemap dynamically
    sample_urls = [
        "https://shanhai-geo.top/",
        # Add more from sitemap as needed
    ]

    print("Loading GeoMind pages...")
    documents = reader.load_data(sample_urls)
    print(f"Loaded {len(documents)} documents")

    # Build index
    print("Building vector index...")
    index = VectorStoreIndex.from_documents(documents)

    # Query
    query_engine = index.as_query_engine()

    questions = [
        "What is structured knowledge?",
        "How do cross-references work in knowledge graphs?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        response = await query_engine.aquery(question)
        print(f"A: {response}")


if __name__ == "__main__":
    asyncio.run(main())
