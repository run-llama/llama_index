"""
Comprehensive example showcasing all ScrapeGraph tool functionalities with LlamaIndex.

This example demonstrates all available methods in the ScrapegraphToolSpec:
- SmartScraper for intelligent data extraction
- Markdownify for content conversion
- Search for web search functionality
- Basic Scrape for HTML extraction
- Agentic Scraper for complex navigation
"""

from typing import List

from pydantic import BaseModel, Field
from llama_index.tools.scrapegraph import ScrapegraphToolSpec


class NewsArticle(BaseModel):
    """Schema for news article information."""

    title: str = Field(description="Article title")
    author: str = Field(description="Article author", default="N/A")
    date: str = Field(description="Publication date", default="N/A")
    summary: str = Field(description="Article summary", default="N/A")


def demonstrate_all_tools():
    """Demonstrate all ScrapeGraph tool functionalities."""
    # Initialize the tool spec (will use SGAI_API_KEY from environment)
    scrapegraph_tool = ScrapegraphToolSpec()

    print("🚀 Complete ScrapeGraph Tools Demonstration")
    print("=" * 47)

    # 1. SmartScraper Example
    print("\n🤖 1. SmartScraper - AI-Powered Data Extraction")
    print("-" * 50)
    try:
        response = scrapegraph_tool.scrapegraph_smartscraper(
            prompt="Extract the main headline, key points, and any important information from this page",
            url="https://example.com/",
        )

        if "error" not in response:
            print("✅ SmartScraper extraction successful:")
            print(f"Result: {str(response)[:300]}...")
        else:
            print(f"❌ SmartScraper error: {response['error']}")
    except Exception as e:
        print(f"❌ SmartScraper exception: {str(e)}")

    # 2. Markdownify Example
    print("\n📄 2. Markdownify - Content to Markdown Conversion")
    print("-" * 54)
    try:
        response = scrapegraph_tool.scrapegraph_markdownify(
            url="https://example.com/",
        )

        if "failed" not in str(response).lower():
            print("✅ Markdownify conversion successful:")
            print(f"Markdown preview: {response[:200]}...")
            print(f"Total length: {len(response)} characters")
        else:
            print(f"❌ Markdownify error: {response}")
    except Exception as e:
        print(f"❌ Markdownify exception: {str(e)}")

    # 3. Search Example
    print("\n🔍 3. Search - Web Search Functionality")
    print("-" * 39)
    try:
        response = scrapegraph_tool.scrapegraph_search(
            query="ScrapeGraph AI web scraping tools",
            max_results=3
        )

        if "failed" not in str(response).lower():
            print("✅ Search successful:")
            print(f"Search results: {str(response)[:300]}...")
        else:
            print(f"❌ Search error: {response}")
    except Exception as e:
        print(f"❌ Search exception: {str(e)}")

    # 4. Basic Scrape Example
    print("\n🌐 4. Basic Scrape - HTML Content Extraction")
    print("-" * 46)
    try:
        response = scrapegraph_tool.scrapegraph_scrape(
            url="https://httpbin.org/html",
            render_heavy_js=False,
            headers={"User-Agent": "ScrapeGraph-Demo/1.0"}
        )

        if "error" not in response:
            html_content = response.get("html", "")
            print("✅ Basic scrape successful:")
            print(f"HTML length: {len(html_content):,} characters")
            print(f"Request ID: {response.get('request_id', 'N/A')}")

            # Extract title if present
            if "<title>" in html_content:
                title_start = html_content.find("<title>") + 7
                title_end = html_content.find("</title>", title_start)
                if title_end != -1:
                    title = html_content[title_start:title_end]
                    print(f"Page title: {title}")
        else:
            print(f"❌ Basic scrape error: {response['error']}")
    except Exception as e:
        print(f"❌ Basic scrape exception: {str(e)}")

    # 5. Agentic Scraper Example
    print("\n🤖 5. Agentic Scraper - Intelligent Navigation")
    print("-" * 47)
    try:
        response = scrapegraph_tool.scrapegraph_agentic_scraper(
            prompt="Navigate through this website and find any contact information, company details, or important announcements. Look in multiple sections if needed.",
            url="https://example.com/",
        )

        if "error" not in response:
            print("✅ Agentic scraper successful:")
            if isinstance(response, dict):
                for key, value in response.items():
                    print(f"  {key}: {str(value)[:100]}...")
            else:
                print(f"Navigation result: {str(response)[:300]}...")
        else:
            print(f"❌ Agentic scraper error: {response['error']}")
    except Exception as e:
        print(f"❌ Agentic scraper exception: {str(e)}")

    # 6. Integration with LlamaIndex Agent Example
    print("\n🔗 6. LlamaIndex Agent Integration")
    print("-" * 35)
    try:
        # Create tools list
        tools = scrapegraph_tool.to_tool_list()
        print(f"✅ Created {len(tools)} tools for LlamaIndex integration:")
        for tool in tools:
            print(f"  • {tool.metadata.name}: {tool.metadata.description[:60]}...")

        print("\n💡 These tools can be used with LlamaIndex agents:")
        print("   from llama_index.core.agent import ReActAgent")
        print("   agent = ReActAgent.from_tools(tools, llm=your_llm)")

    except Exception as e:
        print(f"❌ Integration setup error: {str(e)}")

    # Performance and Usage Summary
    print("\n📊 Tool Comparison Summary")
    print("-" * 28)
    print("SmartScraper:    🎯 Best for structured data extraction with AI")
    print("Markdownify:     📄 Best for content analysis and documentation")
    print("Search:          🔍 Best for finding information across the web")
    print("Basic Scrape:    ⚡ Fastest for simple HTML content extraction")
    print("Agentic Scraper: 🧠 Most powerful for complex navigation tasks")

    print("\n🎯 Use Case Recommendations:")
    print("• Data Mining: SmartScraper + Agentic Scraper")
    print("• Content Analysis: Markdownify + SmartScraper")
    print("• Research: Search + SmartScraper")
    print("• Monitoring: Basic Scrape (fastest)")
    print("• Complex Sites: Agentic Scraper")

    print("\n📚 Next Steps:")
    print("• Set SGAI_API_KEY environment variable")
    print("• Choose the right tool for your use case")
    print("• Combine tools for comprehensive workflows")
    print("• Integrate with LlamaIndex agents for advanced automation")


def main():
    """Run the complete demonstration."""
    demonstrate_all_tools()


if __name__ == "__main__":
    main()
