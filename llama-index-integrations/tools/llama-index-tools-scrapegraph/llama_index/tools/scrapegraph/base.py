"""scrapegraph tool specification module for web scraping and feedback operations."""

import importlib.util
from typing import List, Dict

from pydantic import BaseModel
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from scrapegraph_py import SyncClient, AsyncClient


class ScrapegraphToolSpec(BaseToolSpec):
    """scrapegraph tool specification for web scraping and feedback operations."""

    spec_functions = [
        "scrapegraph_feedback",
        "scrapegraph_smartscraper",
        "scrapegraph_smartscraper_async",
        "scrapegraph_get_credits"
    ]

    def __init__(self) -> None:
        """Initialize scrapegraph tool spec and verify dependencies."""
        if not importlib.util.find_spec("scrapegraph-py"):
            raise ImportError(
                "scrapegraphToolSpec requires the scrapegraph-py package to be installed."
            )
        super().__init__()

    def scrapegraph_feedback(
        self,
        request_id: str,
        api_key: str,
        rating: int,
        feedback_text: str
    ) -> str:
        """Submit feedback for a previous scrapegraph request.

        Args:
            request_id (str): The ID of the request to provide feedback for
            api_key (str): scrapegraph API key
            rating (int): Numerical rating for the request
            feedback_text (str): Detailed feedback text

        Returns:
            str: Response from the feedback submission
        """
        sgai_client = SyncClient(api_key=api_key)
        feedback_response = sgai_client.submit_feedback(
            request_id=request_id,
            rating=rating,
            feedback_text=feedback_text,
        )
        sgai_client.close()
        return feedback_response

    def scrapegraph_smartscraper(
        self,
        prompt: str,
        url: str,
        api_key: str,
        schema: List[BaseModel]
    ) -> List[Dict]:
        """Perform synchronous web scraping using scrapegraph.

        Args:
            prompt (str): User prompt describing the scraping task
            url (str): Target website URL to scrape
            api_key (str): scrapegraph API key
            schema (List[BaseModel]): Pydantic models defining the output structure

        Returns:
            List[Dict]: Scraped data matching the provided schema
        """
        sgai_client = SyncClient(api_key=api_key)
        response = sgai_client.smartscraper(
            website_url=url,
            user_prompt=prompt,
            output_schema=schema,
        )
        sgai_client.close()
        return response

    async def scrapegraph_smartscraper_async(
        self,
        prompt: str,
        url: str,
        api_key: str,
        schema: List[BaseModel]
    ) -> Dict:
        """Perform asynchronous web scraping using scrapegraph.

        Args:
            prompt (str): User prompt describing the scraping task
            url (str): Target website URL to scrape
            api_key (str): scrapegraph API key
            schema (List[BaseModel]): Pydantic models defining the output structure

        Returns:
            Dict: Scraped data matching the provided schema
        """
        sgai_client = AsyncClient(api_key=api_key)
        try:
            response = await sgai_client.smartscraper(
                website_url=url,
                user_prompt=prompt,
                output_schema=schema,
            )
            return response
        finally:
            await sgai_client.close()

    def scrapegraph_get_credits(self, api_key: str) -> str:
        """Retrieve remaining API credits for the scrapegraph account.

        Args:
            api_key (str): scrapegraph API key

        Returns:
            str: Remaining credits information
        """
        sgai_client = SyncClient(api_key=api_key)
        credits = sgai_client.get_credits()
        sgai_client.close()
        return credits
