"""You Retriever."""

import os
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

_SEARCH_ENDPOINT = "https://ydc-index.io/v1/search"
_DEFAULT_TIMEOUT = 30


class YouRetriever(BaseRetriever):
    """
    Retriever for You.com's Search API (unified web and news search).

    [API reference](https://docs.you.com/api-reference/search/v1-search)

    Args:
        api_key: You.com API key, if `YDC_API_KEY` is not set in the environment
        callback_manager: Callback manager for instrumentation
        count: Maximum number of search results to return per section (web/news), range 1-100, defaults to 10
        safesearch: Safesearch settings, one of "off", "moderate", "strict"
        country: Country code (ISO 3166-2), ex: 'US' for United States
        language: Language of results in BCP 47 format, ex: 'en' for English, defaults to 'EN'
        freshness: Recency of results - "day", "week", "month", "year", or custom range "YYYY-MM-DDtoYYYY-MM-DD"
        offset: Offset for pagination (in multiples of count), range 0-9
        livecrawl: Which section(s) to live crawl - "web", "news", or "all"
        livecrawl_formats: Format of live-crawled content - "html" or "markdown"

    """

    _api_key: str
    count: Optional[int]
    safesearch: Optional[Literal["off", "moderate", "strict"]]
    country: Optional[str]
    language: Optional[str]
    freshness: Optional[str]
    offset: Optional[int]
    livecrawl: Optional[Literal["web", "news", "all"]]
    livecrawl_formats: Optional[Literal["html", "markdown"]]

    def __init__(
        self,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        count: Optional[int] = None,
        safesearch: Optional[Literal["off", "moderate", "strict"]] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        freshness: Optional[str] = None,
        offset: Optional[int] = None,
        livecrawl: Optional[Literal["web", "news", "all"]] = None,
        livecrawl_formats: Optional[Literal["html", "markdown"]] = None,
    ) -> None:
        """Initialize YouRetriever with API key and search parameters."""
        self._api_key = api_key or os.getenv("YDC_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "You.com API key is required. Please provide it as an argument "
                "or set the YDC_API_KEY environment variable."
            )
        super().__init__(callback_manager)

        self.count = count
        self.safesearch = safesearch
        self.country = country
        self.language = language
        self.freshness = freshness
        self.offset = offset
        self.livecrawl = livecrawl
        self.livecrawl_formats = livecrawl_formats

    def _generate_params(self, query: str) -> Dict[str, Union[str, int]]:
        """Generate query parameters for the API request."""
        params: Dict[str, Any] = {
            "query": query,
            "count": self.count,
            "safesearch": self.safesearch,
            "country": self.country,
            "language": self.language,
            "freshness": self.freshness,
            "offset": self.offset,
            "livecrawl": self.livecrawl,
            "livecrawl_formats": self.livecrawl_formats,
        }

        # Remove `None` values
        return {k: v for k, v in params.items() if v is not None}

    def _process_result(self, result: Dict[str, Any], source_type: str) -> TextNode:
        """Process a single search result into a TextNode."""
        # Use snippets if available, fall back to description
        snippets = result.get("snippets", [])
        text = "\n".join(snippets) if snippets else result.get("description", "")

        metadata: Dict[str, Any] = {
            "url": result.get("url"),
            "title": result.get("title"),
            "description": result.get("description"),
            "page_age": result.get("page_age"),
            "thumbnail_url": result.get("thumbnail_url"),
            "favicon_url": result.get("favicon_url"),
            "authors": result.get("authors"),
            "source_type": source_type,
        }

        # Livecrawl content is additional full-page content when requested
        contents = result.get("contents") or {}
        if contents.get("markdown"):
            metadata["content_markdown"] = contents["markdown"]
        if contents.get("html"):
            metadata["content_html"] = contents["html"]

        return TextNode(
            text=text,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def _process_response(self, data: Dict[str, Any]) -> List[NodeWithScore]:
        """Process API response data into NodeWithScore list."""
        results = data.get("results", {})
        nodes: List[TextNode] = []

        # Process web results if present
        for hit in results.get("web", []):
            nodes.append(self._process_result(hit, "web"))

        # Process news results if present
        for article in results.get("news", []):
            nodes.append(self._process_result(article, "news"))

        return [NodeWithScore(node=node, score=1.0) for node in nodes]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from You.com Search API."""
        headers = {"X-API-Key": self._api_key, "Accept": "application/json"}
        params = self._generate_params(query_bundle.query_str)

        try:
            with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
                response = client.get(
                    _SEARCH_ENDPOINT,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            raise ValueError(f"You.com API request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ValueError(f"You.com API request failed: {e}") from e
        except Exception as e:
            raise ValueError(f"You.com API request failed: {e}") from e

        return self._process_response(data)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from You.com Search API asynchronously."""
        headers = {"X-API-Key": self._api_key, "Accept": "application/json"}
        params = self._generate_params(query_bundle.query_str)

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                response = await client.get(
                    _SEARCH_ENDPOINT,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as e:
            raise ValueError(f"You.com API request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ValueError(f"You.com API request failed: {e}") from e
        except Exception as e:
            raise ValueError(f"You.com API request failed: {e}") from e

        return self._process_response(data)
