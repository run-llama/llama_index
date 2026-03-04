"""Bing Search tool spec."""

from typing import List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

ENDPOINT_BASE_URL = "https://api.bing.microsoft.com/v7.0/"


class BingSearchToolSpec(BaseToolSpec):
    """Bing Search tool spec."""

    spec_functions = ["bing_news_search", "bing_image_search", "bing_video_search"]

    def __init__(
        self, api_key: str, lang: Optional[str] = "en-US", results: Optional[int] = 3
    ) -> None:
        """Initialize with parameters."""
        self.api_key = api_key
        self.lang = lang
        self.results = results

    def _bing_request(self, endpoint: str, query: str, keys: List[str]):
        response = requests.get(
            ENDPOINT_BASE_URL + endpoint,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            params={"q": query, "mkt": self.lang, "count": self.results},
        )
        response_json = response.json()
        return [[result[key] for key in keys] for result in response_json["value"]]

    def bing_news_search(self, query: str):
        """
        Make a query to bing news search. Useful for finding news on a query.

        Args:
            query (str): The query to be passed to bing.

        """
        return self._bing_request("news/search", query, ["name", "description", "url"])

    def bing_image_search(self, query: str):
        """
        Make a query to bing images search. Useful for finding an image of a query.

        Args:
            query (str): The query to be passed to bing.

        returns a url of the images found

        """
        return self._bing_request("images/search", query, ["name", "contentUrl"])

    def bing_video_search(self, query: str):
        """
        Make a query to bing video search. Useful for finding a video related to a query.

        Args:
            query (str): The query to be passed to bing.

        """
        return self._bing_request("videos/search", query, ["name", "contentUrl"])
