"""ScrapeGraph tool specification module for web scraping operations."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
from scrapegraph_py import Client

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ScrapegraphToolSpec(BaseToolSpec):
    """
    ScrapeGraph tool specification for web scraping operations.

    This tool provides access to ScrapeGraph AI's web scraping capabilities,
    including smart scraping, content conversion to markdown, search functionality,
    and basic HTML scraping with various options.
    """

    spec_functions = [
        "scrapegraph_smartscraper",
        "scrapegraph_markdownify",
        "scrapegraph_search",
        "scrapegraph_scrape",
        "scrapegraph_agentic_scraper",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the ScrapeGraph tool specification.

        Args:
            api_key (Optional[str]): ScrapeGraph API key. If not provided,
                                   will attempt to load from environment variable SGAI_API_KEY.

        """
        if api_key:
            self.client = Client(api_key=api_key)
        else:
            self.client = Client.from_env()

    def scrapegraph_smartscraper(
        self,
        prompt: str,
        url: str,
        schema: Optional[Union[List[BaseModel], Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[List[Dict], Dict]:
        """
        Perform intelligent web scraping using ScrapeGraph's SmartScraper.

        Args:
            prompt (str): User prompt describing what data to extract from the webpage
            url (str): Target website URL to scrape
            schema (Optional[Union[List[BaseModel], Dict]]): Pydantic models or dict defining output structure
            **kwargs: Additional parameters for the SmartScraper

        Returns:
            Union[List[Dict], Dict]: Scraped data matching the provided schema or prompt requirements

        """
        try:
            return self.client.smartscraper(
                website_url=url, user_prompt=prompt, output_schema=schema, **kwargs
            )
        except Exception as e:
            return {"error": f"SmartScraper failed: {e!s}"}

    def scrapegraph_markdownify(self, url: str, **kwargs) -> str:
        """
        Convert webpage content to markdown format using ScrapeGraph.

        Args:
            url (str): Target website URL to convert to markdown
            **kwargs: Additional parameters for the markdownify operation

        Returns:
            str: Markdown representation of the webpage content

        """
        try:
            return self.client.markdownify(website_url=url, **kwargs)
        except Exception as e:
            return f"Markdownify failed: {e!s}"

    def scrapegraph_search(
        self, query: str, max_results: Optional[int] = None, **kwargs
    ) -> str:
        """
        Perform a search query using ScrapeGraph's search functionality.

        Args:
            query (str): Search query to execute
            max_results (Optional[int]): Maximum number of search results to return
            **kwargs: Additional parameters for the search operation

        Returns:
            str: Search results from ScrapeGraph

        """
        try:
            search_params = {"query": query}
            if max_results:
                search_params["max_results"] = max_results
            search_params.update(kwargs)

            return self.client.search(**search_params)
        except Exception as e:
            return f"Search failed: {e!s}"

    def scrapegraph_scrape(
        self,
        url: str,
        render_heavy_js: bool = False,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform basic HTML scraping using ScrapeGraph's scrape functionality.

        Args:
            url (str): Target website URL to scrape
            render_heavy_js (bool): Whether to enable JavaScript rendering for dynamic content
            headers (Optional[Dict[str, str]]): Custom HTTP headers to include in the request
            **kwargs: Additional parameters for the scrape operation

        Returns:
            Dict[str, Any]: Dictionary containing scraped HTML content and metadata

        """
        try:
            scrape_params = {"website_url": url, "render_heavy_js": render_heavy_js}
            if headers:
                scrape_params["headers"] = headers
            scrape_params.update(kwargs)

            return self.client.scrape(**scrape_params)
        except Exception as e:
            return {"error": f"Scrape failed: {e!s}"}

    def scrapegraph_agentic_scraper(
        self,
        prompt: str,
        url: str,
        schema: Optional[Union[List[BaseModel], Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[List[Dict], Dict]:
        """
        Perform agentic web scraping that can navigate and interact with websites.

        Args:
            prompt (str): User prompt describing the scraping task and navigation requirements
            url (str): Target website URL to start scraping from
            schema (Optional[Union[List[BaseModel], Dict]]): Pydantic models or dict defining output structure
            **kwargs: Additional parameters for the agentic scraper

        Returns:
            Union[List[Dict], Dict]: Scraped data from the agentic navigation and extraction

        """
        try:
            return self.client.agentic_scraper(
                website_url=url, user_prompt=prompt, output_schema=schema, **kwargs
            )
        except Exception as e:
            return {"error": f"Agentic scraper failed: {e!s}"}
