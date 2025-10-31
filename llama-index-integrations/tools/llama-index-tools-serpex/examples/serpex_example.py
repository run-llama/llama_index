"""Example usage of SERPEX tool with LlamaIndex."""

import os

from llama_index.tools.serpex import SerpexToolSpec

# Set your API key (or use environment variable SERPEX_API_KEY)
os.environ["SERPEX_API_KEY"] = "your_api_key_here"


def basic_search_example():
    """Basic search example."""
    print("=" * 60)
    print("Basic Search Example")
    print("=" * 60)

    # Initialize tool
    tool = SerpexToolSpec()

    # Perform search
    results = tool.search("latest developments in artificial intelligence", num_results=5)

    for doc in results:
        print(doc.text)
    print()


def location_search_example():
    """Location-based search example."""
    print("=" * 60)
    print("Location-Based Search Example")
    print("=" * 60)

    # Initialize tool
    tool = SerpexToolSpec()

    # Search with location
    results = tool.search_with_location(
        query="best Italian restaurants", location="San Francisco, CA", num_results=5
    )

    for doc in results:
        print(doc.text)
    print()


def international_search_example():
    """International search with country and language."""
    print("=" * 60)
    print("International Search Example")
    print("=" * 60)

    # Initialize tool
    tool = SerpexToolSpec()

    # Search with country and language
    results = tool.search(
        query="noticias de tecnología",
        num_results=5,
        gl="es",  # Spain
        hl="es",  # Spanish
    )

    for doc in results:
        print(doc.text)
    print()


def agent_example():
    """Example with LlamaIndex agent."""
    print("=" * 60)
    print("Agent Example")
    print("=" * 60)

    try:
        from llama_index.agent.openai import OpenAIAgent
        from llama_index.llms.openai import OpenAI

        # Initialize SERPEX tool
        serpex_tool = SerpexToolSpec()

        # Create agent
        llm = OpenAI(model="gpt-4")
        agent = OpenAIAgent.from_tools(serpex_tool.to_tool_list(), llm=llm, verbose=True)

        # Ask question that requires web search
        response = agent.chat(
            "What are the latest features announced for LlamaIndex? Search the web for recent news."
        )
        print(response)

    except ImportError:
        print("OpenAI dependencies not installed. Install with:")
        print("pip install llama-index-agent-openai llama-index-llms-openai")
    print()


def main():
    """Run all examples."""
    # Make sure API key is set
    if not os.environ.get("SERPEX_API_KEY"):
        print("Please set SERPEX_API_KEY environment variable")
        print("Get your API key at: https://serpex.dev/dashboard")
        return

    # Run examples
    basic_search_example()
    location_search_example()
    international_search_example()
    agent_example()


if __name__ == "__main__":
    main()
