"""
Example demonstrating ScrapeGraph basic scraping integration with LlamaIndex.

This example shows how to use the ScrapegraphToolSpec for basic HTML scraping
with various options like JavaScript rendering and custom headers.
"""

from llama_index.tools.scrapegraph import ScrapegraphToolSpec


def main():
    """Demonstrate basic scraping functionality with various options."""
    # Initialize the tool spec (will use SGAI_API_KEY from environment)
    scrapegraph_tool = ScrapegraphToolSpec()

    print("🌐 ScrapeGraph Basic Scraping Examples")
    print("=" * 42)

    # Example 1: Basic HTML scraping
    print("\n1. Basic HTML scraping:")
    try:
        response = scrapegraph_tool.scrapegraph_scrape(
            url="https://example.com/",
        )

        if "error" not in response:
            html_content = response.get("html", "")
            print(f"✅ Successfully scraped {len(html_content):,} characters of HTML")
            print(f"Request ID: {response.get('request_id', 'N/A')}")

            # Show a preview of the HTML
            if html_content:
                preview = html_content[:200].replace('\n', ' ').strip()
                print(f"HTML Preview: {preview}...")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 2: Scraping with JavaScript rendering
    print("\n2. Scraping with JavaScript rendering enabled:")
    try:
        response = scrapegraph_tool.scrapegraph_scrape(
            url="https://httpbin.org/html",
            render_heavy_js=True
        )

        if "error" not in response:
            html_content = response.get("html", "")
            print(f"✅ Successfully scraped with JS rendering: {len(html_content):,} characters")
            print(f"Request ID: {response.get('request_id', 'N/A')}")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 3: Scraping with custom headers
    print("\n3. Scraping with custom headers:")
    try:
        custom_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive"
        }

        response = scrapegraph_tool.scrapegraph_scrape(
            url="https://httpbin.org/headers",
            headers=custom_headers
        )

        if "error" not in response:
            html_content = response.get("html", "")
            print(f"✅ Successfully scraped with custom headers: {len(html_content):,} characters")
            print(f"Request ID: {response.get('request_id', 'N/A')}")

            # Check if our headers were included (httpbin.org/headers shows sent headers)
            if "Mozilla/5.0" in html_content:
                print("✅ Custom User-Agent header was successfully used")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 4: Complex scraping with multiple options
    print("\n4. Complex scraping with multiple options:")
    try:
        response = scrapegraph_tool.scrapegraph_scrape(
            url="https://scrapegraphai.com/",
            render_heavy_js=False,
            headers={
                "User-Agent": "ScrapeGraph-LlamaIndex-Bot/1.0",
                "Accept": "text/html,application/xhtml+xml"
            }
        )

        if "error" not in response:
            html_content = response.get("html", "")
            print(f"✅ Successfully performed complex scraping: {len(html_content):,} characters")
            print(f"Request ID: {response.get('request_id', 'N/A')}")

            # Extract some basic info from the HTML
            if "<title>" in html_content:
                title_start = html_content.find("<title>") + 7
                title_end = html_content.find("</title>", title_start)
                if title_end != -1:
                    title = html_content[title_start:title_end]
                    print(f"Page Title: {title}")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    print("\n📚 Tips:")
    print("• Set your SGAI_API_KEY environment variable")
    print("• Use render_heavy_js=True for dynamic content that requires JavaScript")
    print("• Custom headers help avoid blocking and improve compatibility")
    print("• Check response metadata like request_id for tracking")
    print("• Basic scraping is faster than SmartScraper for simple HTML extraction")


if __name__ == "__main__":
    main()
