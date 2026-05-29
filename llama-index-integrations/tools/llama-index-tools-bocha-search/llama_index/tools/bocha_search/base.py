"""Bocha Search tool spec."""

import json
import httpx
import requests
from typing import Dict, List, Optional
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

BOCHA_SEARCH_URL = "https://api.bocha.cn/v1/web-search"


class BochaSearchToolSpec(BaseToolSpec):
    """Bocha Search tool spec."""

    spec_functions = [
        "bocha_search",
        "abocha_search",
    ]

    def __init__(self, api_key: str) -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): The Bocha AI API key.

        """
        self.api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for the Bocha AI API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _parse_results(self, response_json: Dict) -> List[Document]:
        """Parse Bocha web search response into a list of Document objects."""
        return [Document(text=json.dumps(response_json, ensure_ascii=False))]

    def bocha_search(
        self,
        query: str,
        freshness: Optional[str] = "noLimit",
        summary: Optional[bool] = False,
        count: Optional[int] = 10,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> List[Document]:
        """
        Search the web using Bocha AI Web Search API and retrieve a list of documents.

        Args:
            query (str): The search query keywords.
            freshness (Optional[str]): Time range for search (noLimit, oneDay, oneWeek, oneMonth, oneYear, or YYYY-MM-DD..YYYY-MM-DD). Default is "noLimit".
            summary (Optional[bool]): Whether to include a text summary for each page. Default is False.
            count (Optional[int]): Number of search results to return (1-50). Default is 10.
            include (Optional[str]): Domain names to include, separated by '|' or ','.
            exclude (Optional[str]): Domain names to exclude, separated by '|' or ','.

        Returns:
            List[Document]: A list of Document objects containing search results.

        """
        payload = {
            "query": query,
            "freshness": freshness,
            "summary": summary,
            "count": count,
        }
        if include is not None:
            payload["include"] = include
        if exclude is not None:
            payload["exclude"] = exclude

        response = requests.post(
            BOCHA_SEARCH_URL,
            headers=self._get_headers(),
            json=payload,
        )
        response.raise_for_status()
        return self._parse_results(response.json())

    async def abocha_search(
        self,
        query: str,
        freshness: Optional[str] = "noLimit",
        summary: Optional[bool] = False,
        count: Optional[int] = 10,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> List[Document]:
        """
        Asynchronously search the web using Bocha AI Web Search API and retrieve a list of documents.

        Args:
            query (str): The search query keywords.
            freshness (Optional[str]): Time range for search (noLimit, oneDay, oneWeek, oneMonth, oneYear, or YYYY-MM-DD..YYYY-MM-DD). Default is "noLimit".
            summary (Optional[bool]): Whether to include a text summary for each page. Default is False.
            count (Optional[int]): Number of search results to return (1-50). Default is 10.
            include (Optional[str]): Domain names to include, separated by '|' or ','.
            exclude (Optional[str]): Domain names to exclude, separated by '|' or ','.

        Returns:
            List[Document]: A list of Document objects containing search results.

        """
        payload = {
            "query": query,
            "freshness": freshness,
            "summary": summary,
            "count": count,
        }
        if include is not None:
            payload["include"] = include
        if exclude is not None:
            payload["exclude"] = exclude

        async with httpx.AsyncClient() as client:
            response = await client.post(
                BOCHA_SEARCH_URL,
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            return self._parse_results(response.json())
