"""
Example demonstrating the ValyuRetriever for URL content extraction.

This example shows how to use the ValyuRetriever to extract content from URLs
and integrate it with LlamaIndex retrieval pipelines.
"""

import os
from llama_index.tools.valyu import ValyuRetriever
from llama_index.core import QueryBundle


def main():
    """Demonstrate ValyuRetriever usage."""

    # Initialize the Valyu retriever
    valyu_retriever = ValyuRetriever(
        api_key=os.environ.get("VALYU_API_KEY", "your-api-key-here"),
        verbose=True,
        # Configure contents extraction (user-controlled settings)
        contents_summary=True,  # Enable AI summarization
        contents_extract_effort="normal",  # Extraction thoroughness
        contents_response_length="medium",  # Content length per URL
        # Note: contents_max_price is set by developer/user, not exposed to models
    )

    # Example 1: Single URL retrieval
    print("=== Single URL Retrieval ===")
    query_bundle = QueryBundle(
        query_str="https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"
    )

    try:
        nodes = valyu_retriever.retrieve(query_bundle)
        print(f"Retrieved {len(nodes)} documents:")

        for i, node in enumerate(nodes):
            print(f"\nDocument {i+1}:")
            print(f"URL: {node.node.metadata.get('url', 'N/A')}")
            print(f"Title: {node.node.metadata.get('title', 'N/A')}")
            print(f"Content length: {len(node.node.text)} characters")
            print(f"Score: {node.score}")
            # Show content preview
            preview = (
                node.node.text[:200] + "..."
                if len(node.node.text) > 200
                else node.node.text
            )
            print(f"Content preview: {preview}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This example requires a valid VALYU_API_KEY environment variable")

    # Example 2: Multiple URLs
    print("\n=== Multiple URLs Retrieval ===")
    multi_url_query = QueryBundle(
        query_str="https://arxiv.org/abs/1706.03762 https://en.wikipedia.org/wiki/Attention_(machine_learning)"
    )

    try:
        nodes = valyu_retriever.retrieve(multi_url_query)
        print(f"Retrieved {len(nodes)} documents from multiple URLs")

        for i, node in enumerate(nodes):
            print(
                f"Document {i+1}: {node.node.metadata.get('title', 'Unknown')} - {len(node.node.text)} chars"
            )

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Natural language query with URLs
    print("\n=== Natural Language Query with URLs ===")
    natural_query = QueryBundle(
        query_str="Please extract content from these research papers: https://arxiv.org/abs/1706.03762 and also from https://en.wikipedia.org/wiki/Large_language_model"
    )

    try:
        nodes = valyu_retriever.retrieve(natural_query)
        print(
            f"Extracted content from {len(nodes)} URLs found in natural language query"
        )

    except Exception as e:
        print(f"Error: {e}")


def demonstrate_url_parsing():
    """Demonstrate URL parsing capabilities."""
    print("\n=== URL Parsing Examples ===")

    retriever = ValyuRetriever(
        api_key="test-key"
    )  # API key not needed for parsing demo

    test_cases = [
        "https://example.com",
        "https://site1.com, https://site2.com",
        "Please extract content from https://news.com and https://blog.com",
        "Check out these links: https://paper1.org https://paper2.org",
        "No URLs in this text",
    ]

    for i, test_case in enumerate(test_cases, 1):
        urls = retriever._parse_urls_from_query(test_case)
        print(f"Test {i}: '{test_case}'")
        print(f"  Extracted URLs: {urls}")


if __name__ == "__main__":
    main()
    demonstrate_url_parsing()
