"""Outcrawl Web Reader."""

import os
from typing import Any, Dict, List, Optional

import requests

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

OUTCRAWL_API_URL = "https://api.outcrawl.ai/v1"


class OutcrawlWebReader(BasePydanticReader):
    """
    Web reader backed by the Outcrawl API (https://outcrawl.ai).

    Two modes:

    - ``scrape``: fetch one URL and return its main content as a single Markdown
      document.
    - ``search``: run a web search and return one document per hit. With
      ``scrape=True`` in ``params`` every hit is rendered and its Markdown becomes
      the document text; otherwise the text is the hit's snippet.

    Args:
        api_key (Optional[str]): Outcrawl API key. Defaults to the
            ``OUTCRAWL_API_KEY`` environment variable.
        mode (str): ``"scrape"`` or ``"search"``. Default is ``"scrape"``.
        params (Optional[dict]): Extra fields sent with every request, for example
            ``{"onlyMainContent": False}`` for scrape or ``{"limit": 5}`` for search.
        api_url (str): API base URL. Default is ``https://api.outcrawl.ai/v1``.
        timeout (int): Request timeout in seconds. Default is 60.

    """

    api_key: str
    mode: str
    params: Optional[dict]
    api_url: str
    timeout: int

    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: str = "scrape",
        params: Optional[dict] = None,
        api_url: str = OUTCRAWL_API_URL,
        timeout: int = 60,
    ) -> None:
        """Initialize with parameters."""
        api_key = api_key or os.getenv("OUTCRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "An Outcrawl API key is required: pass api_key or set OUTCRAWL_API_KEY."
            )
        if mode not in ("scrape", "search"):
            raise ValueError("Invalid mode. Choose from 'scrape' or 'search'.")
        super().__init__(
            api_key=api_key,
            mode=mode,
            params=params or {},
            api_url=api_url.rstrip("/"),
            timeout=timeout,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OutcrawlWebReader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> List[Document]:
        """
        Load data from a URL (scrape mode) or a query (search mode).

        Args:
            url (Optional[str]): URL to scrape.
            query (Optional[str]): Search query.
            params (Optional[dict]): Extra fields for this request; they override
                the reader's ``params``.

        Returns:
            List[Document]: The loaded documents.

        """
        if self.mode == "scrape":
            if not url:
                raise ValueError("URL must be provided for scrape mode.")
            return self._scrape(url, params=params)
        if not query:
            raise ValueError("Query must be provided for search mode.")
        return self._search(query, params=params)

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.api_url}/{path}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _scrape(self, url: str, params: Optional[dict] = None) -> List[Document]:
        body: Dict[str, Any] = {"url": url, "formats": ["markdown"]}
        body.update({**(self.params or {}), **(params or {})})
        result = self._post("scrape", body)

        page_metadata = result.get("metadata") or {}
        metadata = {
            "source": result.get("url") or url,
            "status_code": result.get("statusCode"),
            "title": page_metadata.get("title"),
            "description": page_metadata.get("description"),
            "language": page_metadata.get("language"),
        }
        return [
            Document(
                text=result.get("markdown") or "",
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
        ]

    def _search(self, query: str, params: Optional[dict] = None) -> List[Document]:
        body: Dict[str, Any] = {"query": query}
        body.update({**(self.params or {}), **(params or {})})
        if body.get("scrape") and "formats" not in body:
            body["formats"] = ["markdown"]
        result = self._post("search", body)

        documents = []
        for hit in result.get("hits") or []:
            metadata = {
                "source": hit.get("url"),
                "title": hit.get("title"),
                "snippet": hit.get("snippet"),
                "rank": hit.get("rank"),
                "query": query,
            }
            documents.append(
                Document(
                    text=hit.get("markdown") or hit.get("snippet") or "",
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
        return documents
