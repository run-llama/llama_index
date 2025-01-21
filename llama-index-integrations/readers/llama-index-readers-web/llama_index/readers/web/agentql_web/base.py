"""AgentQL Web Reader."""
import httpx
from typing import Optional, List

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

import logging

logging.getLogger("root").setLevel(logging.INFO)

QUERY_DATA_ENDPOINT = "https://api.agentql.com/v1/query-data"
API_TIMEOUT_SECONDS = 900


class AgentQLWebReader(BasePydanticReader):
    """
    Scrape a URL with or without a agentql query and returns document in json format.

    Args:
        api_key (str): The AgentQL API key, get one at https://dev.agentql.com
        params (dict): Additional parameters to pass to the AgentQL API. Visit https://docs.agentql.com/rest-api/api-reference for details.
    """

    api_key: str
    params: Optional[dict]

    def __init__(
        self,
        api_key: str,
        params: Optional[dict] = None,
    ) -> None:
        super().__init__(api_key=api_key, params=params)

    def load_data(
        self, url: str, query: Optional[str] = None, prompt: Optional[str] = None
    ) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            url (str): URL to scrape or crawl.
            query (Optional[str]): AgentQL query used to specify the scraped data.
            prompt (Optional[str]): Natural language description of the data you want to scrape.
            Either query or prompt must be provided.
            params (Optional[dict]): Additional parameters to pass to the AgentQL API. Visit https://docs.agentql.com/rest-api/api-reference for details.

        Returns:
            List[Document]: List of documents.
        """
        payload = {"url": url, "query": query, "prompt": prompt, "params": self.params}

        headers = {"X-API-Key": f"{self.api_key}", "Content-Type": "application/json"}

        try:
            response = httpx.post(
                QUERY_DATA_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            response = e.response
            if response.status_code in [401, 403]:
                raise ValueError(
                    "Please, provide a valid API Key. You can create one at https://dev.agentql.com."
                ) from e
            else:
                try:
                    error_json = response.json()
                    msg = (
                        error_json["error_info"]
                        if "error_info" in error_json
                        else error_json["detail"]
                    )
                except (ValueError, TypeError):
                    msg = f"HTTP {e}."
                raise ValueError(msg) from e
        else:
            json = response.json()

            return [Document(text=str(json["data"]), metadata=json["metadata"])]
