from typing import Dict, List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

SEARCH_URL = "https://www.searchapi.io/api/v1/search"


class SearchApiToolSpec(BaseToolSpec):
    """
    SearchApi tool spec.

    SearchApi (https://www.searchapi.io) is a real-time SERP API that returns
    structured results from 100+ engines, including Google, Google News,
    Google Scholar, Bing, Baidu, YouTube, and more.
    """

    spec_functions = ["search"]

    def __init__(self, api_key: str, engine: str = "google") -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): SearchApi API key, obtained from https://www.searchapi.io.
            engine (str): The default engine to query, e.g. "google", "google_news",
                "bing", "youtube". Defaults to "google".

        """
        self.api_key = api_key
        self.engine = engine

    def _make_request(self, params: Dict) -> requests.Response:
        """
        Make a request to the SearchApi API.

        Args:
            params (dict): The parameters to be passed to the API.

        Returns:
            requests.Response: The response from the API.

        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        return response

    def search(
        self,
        query: str,
        engine: Optional[str] = None,
        num_results: int = 5,
    ) -> List[Document]:
        """
        Make a query to SearchApi to receive a list of results.

        Args:
            query (str): The query to be passed to the search engine.
            engine (Optional[str]): The engine to use for this query, overriding the
                default set on the tool. Defaults to None.
            num_results (int): The number of search results to return. Defaults to 5.

        Returns:
            List[Document]: A list of documents containing search results.

        """
        params = {
            "engine": engine or self.engine,
            "q": query,
            "num": num_results,
        }
        response = self._make_request(params)
        data = response.json()

        results = data.get("organic_results", [])
        documents: List[Document] = []
        for result in results[:num_results]:
            text = "\n".join(
                str(result.get(field, ""))
                for field in ("title", "snippet", "link")
                if result.get(field)
            )
            documents.append(Document(text=text, metadata=result))

        # Fall back to the raw response if no organic results are present
        # (some engines return other result shapes, e.g. videos or knowledge graph).
        if not documents:
            return [Document(text=response.text)]

        return documents
