"""
Example demonstrating ScrapeGraph Markdownify integration with LlamaIndex.

This example shows how to use the ScrapegraphToolSpec to convert web pages
to clean, structured markdown format.
"""

from llama_index.tools.scrapegraph import ScrapegraphToolSpec


def main():
    """Demonstrate Markdownify functionality for converting web content to markdown."""
    # Initialize the tool spec (will use SGAI_API_KEY from environment)
    scrapegraph_tool = ScrapegraphToolSpec()

    print("📄 ScrapeGraph Markdownify Example")
    print("=" * 37)

    # Example 1: Convert main page to markdown
    print("\n1. Converting website to markdown:")
    try:
        response = scrapegraph_tool.scrapegraph_markdownify(
            url="https://scrapegraphai.com/",
        )

        if "failed" not in str(response).lower():
            print("✅ Successfully converted to markdown:")
            # Show first 500 characters as preview
            preview = response[:500] if len(response) > 500 else response
            print(f"Preview:\n{preview}")
            if len(response) > 500:
                print(f"\n... (truncated, total length: {len(response)} characters)")
        else:
            print(f"❌ Error: {response}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 2: Convert blog page to markdown
    print("\n2. Converting a documentation page:")
    try:
        response = scrapegraph_tool.scrapegraph_markdownify(
            url="https://docs.scrapegraphai.com/",
        )

        if "failed" not in str(response).lower():
            print("✅ Successfully converted documentation to markdown:")
            preview = response[:300] if len(response) > 300 else response
            print(f"Preview:\n{preview}")
            if len(response) > 300:
                print(f"\n... (truncated, total length: {len(response)} characters)")
        else:
            print(f"❌ Error: {response}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 3: Convert news article
    print("\n3. Converting a sample news article:")
    try:
        response = scrapegraph_tool.scrapegraph_markdownify(
            url="https://example.com/",
        )

        if "failed" not in str(response).lower():
            print("✅ Successfully converted article to markdown:")
            preview = response[:400] if len(response) > 400 else response
            print(f"Preview:\n{preview}")
            if len(response) > 400:
                print(f"\n... (truncated, total length: {len(response)} characters)")
        else:
            print(f"❌ Error: {response}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    print("\n📚 Tips:")
    print("• Set your SGAI_API_KEY environment variable")
    print("• Markdownify preserves structure and formatting")
    print("• Great for content analysis and documentation")
    print("• Works well with static and dynamic content")


if __name__ == "__main__":
    main()
