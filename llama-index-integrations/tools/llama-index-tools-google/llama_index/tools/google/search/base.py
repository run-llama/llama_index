"""Google Search tool spec."""

import json
import httpx
import urllib.parse
from typing import Dict, List, Optional, Union

from llama_index.core.tools.tool_spec.base import BaseToolSpec

QUERY_URL_TMPL = (
    "https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}"
)


class GoogleSearchToolSpec(BaseToolSpec):
    """Google Search tool spec."""

    spec_functions = [("google_search", "agoogle_search")]

    def __init__(self, key: str, engine: str, num: Optional[int] = None) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num

    def _get_url(self, query: str) -> str:
        url = QUERY_URL_TMPL.format(
            key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query)
        )

        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"

        return url

    def _parse_results(self, results: List[Dict]) -> Union[List[Dict], str]:
        cleaned_results = []
        if len(results) == 0:
            return "No search results available"

        for result in results:
            if "snippet" in result:
                cleaned_results.append(
                    {
                        "title": result["title"],
                        "link": result["link"],
                        "snippet": result["snippet"],
                    }
                )

        return cleaned_results

    def google_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.

        """
        url = self._get_url(query)

        with httpx.Client() as client:
            response = client.get(url)

        results = json.loads(response.text).get("items", [])

        return self._parse_results(results)

    async def agoogle_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.

        """
        url = self._get_url(query)

        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        results = json.loads(response.text).get("items", [])

        return self._parse_results(results)
