"""
Example demonstrating ScrapeGraph SmartScraper integration with LlamaIndex.

This example shows how to use the ScrapegraphToolSpec to extract structured data
from web pages using AI-powered scraping with defined schemas.
"""

import os
from typing import List

from pydantic import BaseModel, Field
from llama_index.tools.scrapegraph import ScrapegraphToolSpec


class FounderSchema(BaseModel):
    """Schema for representing a company founder."""

    name: str = Field(description="Name of the founder")
    role: str = Field(description="Role of the founder")
    social_media: str = Field(description="Social media URL of the founder", default="N/A")


class ListFoundersSchema(BaseModel):
    """Schema for representing a list of company founders."""

    founders: List[FounderSchema] = Field(description="List of founders")


def main():
    """Demonstrate SmartScraper functionality with structured data extraction."""
    # Initialize the tool spec (will use SGAI_API_KEY from environment)
    scrapegraph_tool = ScrapegraphToolSpec()

    print("🤖 ScrapeGraph SmartScraper Example")
    print("=" * 40)

    # Example 1: Extract founders with schema
    print("\n1. Extracting founders with structured schema:")
    try:
        response = scrapegraph_tool.scrapegraph_smartscraper(
            prompt="Extract information about the founders and their roles",
            url="https://scrapegraphai.com/",
            schema=ListFoundersSchema,
        )

        if "error" not in response:
            print("✅ Successfully extracted founder data:")
            if "founders" in response:
                for founder in response["founders"]:
                    print(f"  • {founder.get('name', 'N/A')} - {founder.get('role', 'N/A')}")
            else:
                print(f"Response: {response}")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 2: Extract general information without schema
    print("\n2. Extracting general product information:")
    try:
        response = scrapegraph_tool.scrapegraph_smartscraper(
            prompt="Extract the main product features and benefits described on this page",
            url="https://scrapegraphai.com/",
        )

        if "error" not in response:
            print("✅ Successfully extracted product information:")
            print(f"Response: {response}")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    # Example 3: Extract pricing information
    print("\n3. Extracting pricing information:")
    try:
        response = scrapegraph_tool.scrapegraph_smartscraper(
            prompt="Find and extract any pricing information, plans, or cost details",
            url="https://scrapegraphai.com/",
        )

        if "error" not in response:
            print("✅ Successfully extracted pricing information:")
            print(f"Response: {response}")
        else:
            print(f"❌ Error: {response['error']}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")

    print("\n📚 Tips:")
    print("• Set your SGAI_API_KEY environment variable")
    print("• Use Pydantic schemas for structured data extraction")
    print("• Be specific in your prompts for better results")
    print("• The tool handles errors gracefully with try-catch")


if __name__ == "__main__":
    main()
