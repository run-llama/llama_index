"""
Example demonstrating ScrapeGraph Agentic Scraper integration with LlamaIndex.

This example shows how to use the ScrapegraphToolSpec for agentic scraping
that can navigate and interact with websites intelligently.
"""

from typing import List

from pydantic import BaseModel, Field
from llama_index.tools.scrapegraph import ScrapegraphToolSpec


class ProductInfo(BaseModel):
    """Schema for representing product information."""

    name: str = Field(description="Product name")
    price: str = Field(description="Product price", default="N/A")
    description: str = Field(description="Product description", default="N/A")
    features: List[str] = Field(description="List of key features", default_factory=list)


class ProductsListSchema(BaseModel):
    """Schema for representing multiple products."""

    products: List[ProductInfo] = Field(description="List of products found")


class ContactInfo(BaseModel):
    """Schema for contact information."""

    email: str = Field(description="Contact email", default="N/A")
    phone: str = Field(description="Contact phone", default="N/A")
    address: str = Field(description="Contact address", default="N/A")


def main():
    """Demonstrate agentic scraping functionality for complex navigation tasks."""
    # Initialize the tool spec (will use SGAI_API_KEY from environment)
    scrapegraph_tool = ScrapegraphToolSpec()

    print("🤖 ScrapeGraph Agentic Scraper Examples")
    print("=" * 43)

    # Example 1: Navigate and extract product information
    print("\n1. Extracting product information with navigation:")
    try:
        response = scrapegraph_tool.scrapegraph_agentic_scraper(
            prompt="Navigate to the products or services section and extract information about the main offerings, including names, features, and any pricing information available.",
            url="https://scrapegraphai.com/",
            schema=ProductsListSchema,
        )

        if "error" not in response:
            print("✅ Successfully extracted product data using agentic navigation:")
            if "products" in response:
                for product in response["products"]:
                    print(f"  • Product: {product.get('name', 'N/A')}")
                    print(f"    Price: {product.get('price', 'N/A')}")
                    print(f"    Description: {product.get('description', 'N/A')[:100]}...")
                    if product.get('features'):
                        print(f"    Features: {', '.join(product['features'][:3])}...")
                    print()
            else:
                print(f"Response: {response}")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 2: Navigate to find contact information
    print("\n2. Finding contact information through navigation:")
    try:
        response = scrapegraph_tool.scrapegraph_agentic_scraper(
            prompt="Navigate through the website to find contact information, including email addresses, phone numbers, and physical addresses. Look in contact pages, footer, about sections, etc.",
            url="https://scrapegraphai.com/",
            schema=ContactInfo,
        )

        if "error" not in response:
            print("✅ Successfully found contact information:")
            print(f"  Email: {response.get('email', 'Not found')}")
            print(f"  Phone: {response.get('phone', 'Not found')}")
            print(f"  Address: {response.get('address', 'Not found')}")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 3: Complex navigation for documentation
    print("\n3. Navigating documentation to find specific information:")
    try:
        response = scrapegraph_tool.scrapegraph_agentic_scraper(
            prompt="Navigate to the documentation or help section and find information about API usage, getting started guides, or tutorials. Extract the main steps and any code examples mentioned.",
            url="https://scrapegraphai.com/",
        )

        if "error" not in response:
            print("✅ Successfully navigated and extracted documentation:")
            if isinstance(response, dict):
                for key, value in response.items():
                    print(f"  {key}: {str(value)[:200]}...")
            else:
                print(f"Response: {str(response)[:500]}...")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 4: Multi-step navigation for comprehensive data
    print("\n4. Multi-step navigation for comprehensive site analysis:")
    try:
        response = scrapegraph_tool.scrapegraph_agentic_scraper(
            prompt="Perform a comprehensive analysis of this website by navigating through different sections. Extract: 1) Main value proposition, 2) Key features or services, 3) Pricing information if available, 4) Company background, 5) Contact or support options. Navigate to multiple pages as needed.",
            url="https://scrapegraphai.com/",
        )

        if "error" not in response:
            print("✅ Successfully completed comprehensive site analysis:")
            if isinstance(response, dict):
                for key, value in response.items():
                    print(f"  {key}: {str(value)[:150]}...")
            else:
                print(f"Analysis: {str(response)[:600]}...")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 5: E-commerce style navigation
    print("\n5. E-commerce style product discovery:")
    try:
        response = scrapegraph_tool.scrapegraph_agentic_scraper(
            prompt="Navigate this website as if it were an e-commerce site. Look for product catalogs, pricing pages, feature comparisons, or service offerings. If you find multiple items or services, list them with their characteristics.",
            url="https://example.com/",
        )

        if "error" not in response:
            print("✅ Successfully performed e-commerce style navigation:")
            print(f"Discovery results: {str(response)[:400]}...")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    print("\n📚 Tips:")
    print("• Set your SGAI_API_KEY environment variable")
    print("• Agentic scraper can navigate multiple pages and follow links")
    print("• Use detailed prompts to guide the navigation behavior")
    print("• Combine with schemas for structured data extraction")
    print("• Great for complex sites requiring multi-step interaction")
    print("• More powerful but slower than basic scraping methods")


if __name__ == "__main__":
    main()
