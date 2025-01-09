"""scrapegraph tool specification module for web scraping operations."""

from typing import Dict, List, Optional

from pydantic import BaseModel
from scrapegraph_py import Client

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ScrapegraphToolSpec(BaseToolSpec):
    """scrapegraph tool specification for web scraping operations."""

    spec_functions = [
        "scrapegraph_smartscraper",
        "scrapegraph_markdownify",
        "scrapegraph_local_scrape",
    ]

    def scrapegraph_smartscraper(
        self,
        prompt: str,
        url: str,
        api_key: str,
        schema: Optional[List[BaseModel]] = None,
    ) -> List[Dict]:
        """Perform synchronous web scraping using scrapegraph.

        Args:
            prompt (str): User prompt describing the scraping task
            url (str): Target website URL to scrape
            api_key (str): scrapegraph API key
            schema (Optional[List[BaseModel]]): Pydantic models defining the output structure

        Returns:
            List[Dict]: Scraped data matching the provided schema
        """
        client = Client(api_key=api_key)

        # Basic usage
        return client.smartscraper(
            website_url=url, user_prompt=prompt, output_schema=schema
        )

    def scrapegraph_markdownify(self, url: str, api_key: str) -> str:
        """Convert webpage content to markdown format using scrapegraph.

        Args:
            url (str): Target website URL to convert
            api_key (str): scrapegraph API key

        Returns:
            str: Markdown representation of the webpage content
        """
        client = Client(api_key=api_key)

        return client.markdownify(website_url=url)

    def scrapegraph_local_scrape(self, text: str, api_key: str) -> str:
        """Extract structured data from raw text using scrapegraph.

        Args:
            text (str): Raw text to process and extract data from
            api_key (str): scrapegraph API key

        Returns:
            str: Structured data extracted from the input text
        """
        client = Client(api_key=api_key)

        return client.local_scrape(text=text)
