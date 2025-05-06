import requests
from yarl import URL
from typing import Dict, List
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

JINA_SEARCH_URL_ENDPOINT = "https://s.jina.ai/"


class JinaToolSpec(BaseToolSpec):
    """
    Jina tool spec.
    """

    spec_functions = ["jina_search"]

    def _make_request(self, params: Dict) -> requests.Response:
        """
        Make a request to the Jina Search API.

        Args:
            params (dict): The parameters to be passed to the API.

        Returns:
            requests.Response: The response from the API.

        """
        headers = {
            "Accept": "application/json",
        }
        url = str(URL(JINA_SEARCH_URL_ENDPOINT + params.get("query")))
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def jina_search(self, query: str) -> List[Document]:
        """
        Make a query to the Jina Search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Jina Search.

        Returns:
            [Document]: A list of documents containing search results.

        """
        search_params = {
            "query": query,
        }
        response = self._make_request(search_params)
        return [
            Document(
                text=result["content"],
                extra_info={
                    "url": result["url"],
                    "title": result["title"],
                    "description": result["description"],
                },
            )
            for result in response["data"]
        ]
