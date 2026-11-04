"""Example usage of the Serpex tool with LlamaIndex."""

import os

from llama_index.tools.serpex import SerpexToolSpec


def basic_search_example() -> None:
    """Basic search: title, URL and snippet per result."""
    print("=" * 60)
    print("Basic Search Example")
    print("=" * 60)

    tool = SerpexToolSpec()
    results = tool.search("latest developments in artificial intelligence", num_results=5)

    for doc in results:
        print(doc.text)
    print()


def operators_example() -> None:
    """Search operators go inside the query: site:, -site:, after:, before:."""
    print("=" * 60)
    print("Search Operators Example")
    print("=" * 60)

    tool = SerpexToolSpec()
    query = "llamaindex release notes site:github.com after:2026-09-01"
    results = tool.search(query, num_results=5)

    for doc in results:
        print(doc.metadata["title"], doc.metadata["url"])
    print()


def page_content_example() -> None:
    """Also fetch the top pages as markdown (3 credits for 5 pages, 6 for 10)."""
    print("=" * 60)
    print("Page Content Example")
    print("=" * 60)

    tool = SerpexToolSpec(include_content=True, content_results=5)
    results = tool.search("what is retrieval augmented generation", num_results=5)

    for doc in results:
        if "content_error" in doc.metadata:
            print(f"{doc.metadata['url']}: no content ({doc.metadata['content_error']})")
        else:
            print(f"{doc.metadata['url']}: {len(doc.text)} characters")
    print()


def agent_example() -> None:
    """Example with a LlamaIndex agent."""
    print("=" * 60)
    print("Agent Example")
    print("=" * 60)

    try:
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI
    except ImportError:
        print("OpenAI dependencies not installed. Install with:")
        print("pip install llama-index-llms-openai")
        return

    import asyncio

    agent = FunctionAgent(
        tools=SerpexToolSpec().to_tool_list(),
        llm=OpenAI(model="gpt-4.1"),
    )
    response = asyncio.run(
        agent.run("What are the latest features announced for LlamaIndex? Search the web.")
    )
    print(response)
    print()


def main() -> None:
    """Run all examples."""
    if not os.environ.get("SERPEX_API_KEY"):
        print("Please set SERPEX_API_KEY environment variable")
        print("Get your API key at: https://serpex.dev/dashboard")
        return

    basic_search_example()
    operators_example()
    page_content_example()
    agent_example()


if __name__ == "__main__":
    main()
