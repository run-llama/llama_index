import requests
import urllib.parse
from typing import Dict
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

SEARCH_URL_TMPL = "https://api.search.brave.com/res/v1/web/search?{params}"

DEFAULT_TIMEOUT = 10.0


class BraveSearchToolSpec(BaseToolSpec):
    """
    Brave Search tool spec.
    """

    spec_functions = ["brave_search"]

    def __init__(self, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): The Brave Search API subscription token.
            timeout (float): Timeout in seconds for the HTTP request, default is 10.0.

        """
        self.api_key = api_key
        self.timeout = timeout

    def _make_request(self, params: Dict) -> requests.Response:
        """
        Make a request to the Brave Search API.

        Args:
            params (dict): The parameters to be passed to the API.

        Returns:
            requests.Response: The response from the API.

        """
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        url = SEARCH_URL_TMPL.format(params=urllib.parse.urlencode(params))

        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response

    def brave_search(
        self, query: str, search_lang: str = "en", num_results: int = 5
    ) -> [Document]:
        """
        Make a query to the Brave Search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Brave Search.
            search_lang (str): The search language preference (ISO 639-1), default is "en".
            num_results (int): The number of search results returned in response, default is 5.

        Returns:
            [Document]: A list of documents containing search results.

        """
        search_params = {
            "q": query,
            "search_lang": search_lang,
            "count": num_results,
        }

        response = self._make_request(search_params)
        return [Document(text=response.text)]
