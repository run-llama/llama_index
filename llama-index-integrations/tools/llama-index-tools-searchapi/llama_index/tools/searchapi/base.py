"""SearchApi tool spec."""

from typing import List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

ENDPOINT_URL = "https://www.searchapi.io/api/v1/search"

# Different engines return their primary results under different keys.
# We try these in order, then fall back to the raw payload for anything else.
RESULT_KEYS = [
    "organic_results",
    "news_results",
    "scholar_results",
    "videos",
    "jobs",
]


class SearchApiToolSpec(BaseToolSpec):
    """
    SearchApi tool spec.

    Queries SearchApi.io, a real-time SERP API that wraps 100+ engines
    (Google, Google News, Google Scholar, YouTube, Google Jobs, and more).
    """

    spec_functions = ["search"]

    def __init__(self, api_key: str, engine: str = "google") -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): Your SearchApi.io API key.
            engine (str): Default engine to query, e.g. "google",
                "google_news", "google_scholar", "youtube_transcripts".
                Defaults to "google".

        """
        self.api_key = api_key
        self.engine = engine

    def _make_request(self, params: dict) -> dict:
        """
        Make a request to the SearchApi.io API.

        Args:
            params (dict): The query parameters to send to the API.

        Returns:
            dict: The parsed JSON response from the API.

        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(ENDPOINT_URL, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def search(
        self,
        query: str,
        engine: Optional[str] = None,
        num_results: int = 10,
    ) -> List[Document]:
        """
        Run a real-time web search through SearchApi.io and return the results.

        Use this for questions about current events, recent information, or
        anything that needs fresh, up-to-date data from the web.

        Args:
            query (str): The search query.
            engine (Optional[str]): Override the engine for this query (e.g.
                "google_news", "google_scholar"). Falls back to the engine
                configured on the tool.
            num_results (int): Maximum number of results to return. Defaults to 10.

        Returns:
            List[Document]: One document per search result.

        """
        params = {
            "engine": engine or self.engine,
            "q": query,
            "num": num_results,
        }
        data = self._make_request(params)

        for key in RESULT_KEYS:
            results = data.get(key)
            if results:
                documents = []
                for result in results:
                    text = result.get("snippet") or result.get("title") or ""
                    metadata = {
                        field: result[field]
                        for field in ("title", "link", "source", "position")
                        if result.get(field) is not None
                    }
                    documents.append(Document(text=text, metadata=metadata))
                return documents

        # Unknown engine shape: return the raw payload so nothing is lost.
        return [Document(text=str(data))]
