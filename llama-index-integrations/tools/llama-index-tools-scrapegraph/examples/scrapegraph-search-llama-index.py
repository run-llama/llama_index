"""
Example demonstrating ScrapeGraph Search integration with LlamaIndex.

This example shows how to use the ScrapegraphToolSpec for search functionality
both standalone and integrated with LlamaIndex agents.
"""

import os
from llama_index.tools.scrapegraph import ScrapegraphToolSpec


def standalone_search_examples():
    """Demonstrate standalone search functionality."""
    scrapegraph_tool = ScrapegraphToolSpec()

    print("🔍 ScrapeGraph Search Examples")
    print("=" * 32)

    # Example 1: Basic search
    print("\n1. Basic AI search query:")
    try:
        response = scrapegraph_tool.scrapegraph_search(
            query="What are the latest developments in artificial intelligence?"
        )

        if "failed" not in str(response).lower():
            print("✅ Search successful:")
            print(f"Response: {response[:500]}...")  # Show first 500 chars
        else:
            print(f"❌ Error: {response}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 2: Specific technical search
    print("\n2. Technical search with max results:")
    try:
        response = scrapegraph_tool.scrapegraph_search(
            query="Python 3.12 new features type hints",
            max_results=5
        )

        if "failed" not in str(response).lower():
            print("✅ Technical search successful:")
            print(f"Response: {response[:500]}...")
        else:
            print(f"❌ Error: {response}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 3: Market research search
    print("\n3. Market research search:")
    try:
        response = scrapegraph_tool.scrapegraph_search(
            query="ScrapeGraph AI pricing and competitors"
        )

        if "failed" not in str(response).lower():
            print("✅ Market research search successful:")
            print(f"Response: {response[:500]}...")
        else:
            print(f"❌ Error: {response}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")


def agent_integration_example():
    """Demonstrate integration with LlamaIndex agents."""
    try:
        from llama_index.core.agent import ReActAgent
        from llama_index.core.llms import OpenAI

        print("\n🤖 Agent Integration Example")
        print("=" * 30)

        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Skipping agent example: OPENAI_API_KEY not set")
            return

        # Initialize the tool spec
        tool_spec = ScrapegraphToolSpec()

        # Create a list of tools from the tool spec
        tools = tool_spec.to_tool_list()

        # Initialize the LLM
        llm = OpenAI(temperature=0)

        # Create an agent with the tools
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

        # Example agent query
        query = "Find information about web scraping best practices and tools"
        print(f"\nAgent Query: {query}")

        response = agent.chat(
            f"Use the scrapegraph search tool to find information about: {query}"
        )

        print("✅ Agent search completed:")
        print(f"Response: {response}")

    except ImportError as e:
        print(f"⚠️  Agent integration requires additional dependencies: {e}")
    except Exception as e:
        print(f"❌ Agent integration error: {str(e)}")


def main():
    """Run all search examples."""
    # Run standalone examples
    standalone_search_examples()

    # Run agent integration if available
    agent_integration_example()

    print("\n📚 Tips:")
    print("• Set your SGAI_API_KEY environment variable")
    print("• Use specific queries for better search results")
    print("• Combine with LlamaIndex agents for advanced workflows")
    print("• Use max_results parameter to limit response size")


if __name__ == "__main__":
    main()
