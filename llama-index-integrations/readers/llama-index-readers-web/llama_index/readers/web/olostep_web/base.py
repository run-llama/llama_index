"""Olostep Web Reader."""

import requests
from typing import List, Optional, Dict, Callable

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class OlostepWebReader(BasePydanticReader):
    """
    A web reader that uses Olostep API to scrape web pages.

    Args:
        api_key (str): The Olostep API key.
        mode (str): The mode to run the loader in. One of "scrape" or "search".
                    Default is "scrape".
        params (Optional[dict]): Additional parameters for the API call.

    """

    api_key: str
    mode: str
    params: Optional[dict]

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        mode: str = "scrape",
        params: Optional[dict] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(
            api_key=api_key,
            mode=mode,
            params=params or {},
        )

    @classmethod
    def class_name(cls) -> str:
        return "OlostepWebReader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> List[Document]:
        """
        Load data from the input URL or query.

        Args:
            url (Optional[str]): URL to scrape or for sitemap.
            query (Optional[str]): Query for search.
            params (Optional[dict]): Additional parameters for the API call.

        Returns:
            List[Document]: List of documents.

        """
        if self.mode == "scrape":
            if not url:
                raise ValueError("URL must be provided for scrape mode.")
            return self._scrape(url, params=params)
        elif self.mode == "search":
            if not query:
                raise ValueError("Query must be provided for search mode.")
            return self._search(query, params=params)
        else:
            raise ValueError("Invalid mode. Choose from 'scrape' or 'search'.")

    def _search(self, query: str, params: Optional[dict] = None) -> List[Document]:
        """
        Perform a search using Olostep's Google Search parser.

        Args:
            query (str): The search query.
            params (Optional[dict]): Additional parameters for the API call.

        Returns:
            List[Document]: A list containing a single document with the search results.

        """
        import json

        combined_params = {**(self.params or {}), **(params or {})}

        search_url = f"https://www.google.com/search?q={query}"

        api_url = "https://api.olostep.com/v1/scrapes"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "url_to_scrape": search_url,
            "formats": ["json"],
            "parser": {"id": "@olostep/google-search"},
            "wait_before_scraping": 0,
        }
        data.update(combined_params)

        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json().get("result", {})
        json_content = result.get("json_content")

        metadata = {
            "source": search_url,
            "query": query,
            "page_metadata": result.get("page_metadata", {}),
        }

        return [Document(text=json.dumps(json_content, indent=4), metadata=metadata)]

    def _scrape(self, url: str, params: Optional[dict] = None) -> List[Document]:
        """
        Scrape a single URL.

        Args:
            url (str): The URL to scrape.
            params (Optional[dict]): Additional parameters for the API call.

        Returns:
            List[Document]: A list containing a single document with the scraped content.

        """
        api_url = "https://api.olostep.com/v1/scrapes"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {"url_to_scrape": url, "formats": ["markdown"]}

        # Combine and add parameters
        combined_params = {**(self.params or {}), **(params or {})}
        data.update(combined_params)

        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json().get("result", {})

        import json

        content_parts = []
        requested_formats = data.get("formats", [])
        for format in requested_formats:
            content_key = f"{format}_content"
            if content_key in result:
                content_value = result[content_key]
                if format == "json" and isinstance(content_value, (dict, list)):
                    content_parts.append(json.dumps(content_value, indent=4))
                else:
                    content_parts.append(str(content_value))

        content = "\n\n".join(content_parts)

        metadata = {"source": url, "page_metadata": result.get("page_metadata", {})}

        return [Document(text=content, metadata=metadata)]
