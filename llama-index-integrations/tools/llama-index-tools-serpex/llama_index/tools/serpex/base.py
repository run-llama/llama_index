"""Serpex search tool specification."""

import os
import re
import warnings
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

import requests

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

USER_AGENT = "llama-index-tools-serpex/0.3.1"
SEARCH_URL = "https://api.serpex.dev/api/search"
BILLING_URL = "https://app.serpex.dev/dashboard/billing"

# Client timeouts (seconds), above the server's own budget for each call
# (search 30 s upstream, 45 s with page content), so a search the server still
# finishes and bills is not abandoned client-side.
SEARCH_TIMEOUT = 60.0
SEARCH_CONTENT_TIMEOUT = 100.0

# The API returns at most 10 results per search.
MAX_RESULTS = 10

# Accepted for backward compatibility; ignored by the Serpex API and never sent.
_DEPRECATED_SEARCH_ARGS = ("engine", "time_range")


class SerpexSearchResult(TypedDict, total=False):
    """One result of ``GET /api/search``."""

    title: str
    url: str
    snippet: str
    position: int
    #: Deprecated: always "auto".
    engine: str
    img_src: str
    duration: str
    score: float
    #: Page markdown. Only with include_content, only on the first
    #: content_results results, never together with ``content_error``.
    content: str
    #: Why page content is missing (e.g. "robots_disallowed"). Never with ``content``.
    content_error: str


class SerpexSearchMetadata(TypedDict, total=False):
    """``metadata`` of ``GET /api/search``."""

    number_of_results: int
    response_time: int
    timestamp: str
    #: Credits charged: 0, 1, 3 or 6.
    credits_used: int
    from_cache: bool
    status: Literal["success", "no_results"]
    content_requested: int
    content_delivered: int
    no_results_verified: bool
    charged: bool
    message: str


class SerpexSearchResponse(TypedDict, total=False):
    """Body of a successful ``GET /api/search`` (HTTP 200)."""

    id: str
    query: str
    #: Deprecated: always ["auto"].
    engines: List[str]
    results: List[SerpexSearchResult]
    metadata: SerpexSearchMetadata
    #: Present only when no results were found.
    message: str


class SerpexErrorBody(TypedDict, total=False):
    """Error body. Fields other than ``error`` depend on the status code."""

    error: Union[str, Dict[str, Any]]
    message: str
    retryable: bool
    request_id: str
    remainingCredits: int
    limit: int
    retryAfterMs: int


class SerpexAPIError(requests.exceptions.HTTPError):
    """
    The Serpex API answered with an error status.

    Subclasses ``requests.exceptions.HTTPError``, so existing ``except`` clauses
    keep working. The message says what happened, whether it was charged and
    whether to retry.

    Attributes:
        status_code: HTTP status.
        details: the parsed error body.
        retryable: True for 429 and 5xx.

    """

    def __init__(
        self,
        message: str,
        status_code: int,
        details: SerpexErrorBody,
        response: Optional[requests.Response] = None,
    ) -> None:
        super().__init__(message, response=response)
        self.status_code = status_code
        self.details = details
        self.retryable = status_code == 429 or status_code >= 500


def _sentence(text: str) -> str:
    return text if re.search(r"[.!?]$", text) else f"{text}."


def _error_text(status: int, body: SerpexErrorBody) -> str:
    """One actionable sentence per HTTP status of the Serpex API."""
    error = body.get("error")
    if isinstance(error, str) and error:
        msg = error
    elif isinstance(error, dict) and isinstance(error.get("message"), str):
        msg = str(error["message"])
    elif isinstance(body.get("message"), str):
        msg = str(body["message"])
    else:
        msg = f"HTTP {status}"
    if status == 400:
        return f"Serpex rejected the request (HTTP 400): {_sentence(msg)}"
    if status == 401:
        return f"Serpex rejected the API key (HTTP 401): {_sentence(msg)} Check SERPEX_API_KEY."
    if status == 402:
        balance = body.get("remainingCredits")
        balance_text = f" Balance: {balance} credits." if isinstance(balance, int) else ""
        return f"Out of Serpex credits (HTTP 402).{balance_text} Top up at {BILLING_URL}."
    if status == 429:
        limit = body.get("limit")
        limit_text = f" (limit: {limit} requests per second)" if isinstance(limit, int) else ""
        wait = body.get("retryAfterMs")
        wait_ms = wait if isinstance(wait, int) else 1000
        return f"Serpex rate limit exceeded{limit_text}. Retry in {wait_ms} ms."
    if status >= 500:
        charged = "" if re.search("charged", msg, re.I) else " Not charged."
        retry = "" if re.search("retry", msg, re.I) else " Retry shortly."
        return (
            f"Serpex is temporarily unavailable (HTTP {status}): {_sentence(msg)}{charged}{retry}"
        )
    return f"Serpex request failed (HTTP {status}): {_sentence(msg)}"


def _warn_deprecated(names: List[str]) -> None:
    warnings.warn(
        f"SerpexToolSpec: {', '.join(names)} {'is' if len(names) == 1 else 'are'} deprecated "
        "and ignored by the Serpex API; the value is not sent. For a date filter put "
        "after:YYYY-MM-DD / before:YYYY-MM-DD in the query.",
        DeprecationWarning,
        stacklevel=3,
    )


class SerpexToolSpec(BaseToolSpec):
    """
    Serpex tool spec for web search.

    This tool searches the web with the Serpex API, a real-time web search
    API for AI agents. Each search returns ranked web results (title, URL,
    snippet) and, optionally, the top pages' content as markdown.

    Args:
        api_key (Optional[str]): Serpex API key. If not provided, will look
            for SERPEX_API_KEY environment variable.
        include_content (bool): Also fetch page content (markdown) for the top
            results. Costs 3 credits (5 pages) or 6 (10 pages) instead of 1.
        content_results (int): How many top results get page content: 5 or 10.
        timeout (Optional[float]): Request timeout in seconds. Default: 60, or
            100 with include_content.
        engine (Optional[str]): Deprecated and ignored; the Serpex API routes
            every search itself.

    Examples:
        >>> from llama_index.tools.serpex import SerpexToolSpec
        >>> tool = SerpexToolSpec(api_key="your_api_key")
        >>> results = tool.search("latest AI news")
        >>> for doc in results:
        ...     print(doc.text)

    """

    spec_functions = ["search"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        engine: Optional[str] = None,
        include_content: bool = False,
        content_results: int = 5,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize Serpex tool.

        Args:
            api_key: Serpex API key. If not provided, reads from
                SERPEX_API_KEY environment variable.
            engine: Deprecated and ignored by the API (not sent). Kept so
                existing code keeps working.
            include_content: Also fetch page content (markdown) for the top
                results.
            content_results: How many top results get page content: 5 or 10.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If API key is not provided and not found in environment,
                or content_results is not 5 or 10.

        """
        self.api_key = api_key or os.environ.get("SERPEX_API_KEY")

        if not self.api_key:
            raise ValueError(
                "SERPEX_API_KEY not found. Please set it as an environment "
                "variable or pass it as an argument. "
                "Get your API key at: https://serpex.dev/dashboard"
            )
        if content_results not in (5, 10):
            raise ValueError("content_results must be 5 or 10")
        if engine is not None:
            _warn_deprecated(["engine"])

        self.base_url = SEARCH_URL
        self.engine = engine
        self.include_content = include_content
        self.content_results = content_results
        self.timeout = timeout

    def search(self, query: str, num_results: int = 10, **kwargs: Any) -> List[Document]:
        """
        Search the web using the Serpex API.

        Returns ranked web results: title, URL and snippet, plus the page
        content when the tool was created with include_content=True.

        Args:
            query: Search query string (max 500 characters). Operators such as
                site:example.com, -site:example.com, after:YYYY-MM-DD and
                before:YYYY-MM-DD go inside the query.
            num_results: How many results to return, 1 to 10 (default: 10).

        Returns:
            List of Document objects, one per search result. Each document
            contains the title, URL and snippet (and page content, when
            requested) in its text, with metadata including search details.

        Examples:
            >>> tool = SerpexToolSpec(api_key="your_key")
            >>> results = tool.search("LlamaIndex tutorial", num_results=5)
            >>> for doc in results:
            ...     print(f"Title: {doc.metadata['title']}")
            ...     print(f"URL: {doc.metadata['url']}")
            ...     print(doc.text)

            >>> # Only results from the last month
            >>> results = tool.search("AI news after:2026-09-23")

        """
        passed = [name for name in _DEPRECATED_SEARCH_ARGS if kwargs.get(name) is not None]
        if passed:
            _warn_deprecated(passed)
        include_content = bool(kwargs.get("include_content", self.include_content))
        content_results = kwargs.get("content_results", self.content_results)
        if content_results not in (5, 10):
            raise ValueError("content_results must be 5 or 10")

        # Only q, include_content and content_results exist on the API.
        params: Dict[str, Any] = {"q": query}
        if include_content:
            params["include_content"] = "true"
            params["content_results"] = content_results

        if self.timeout is not None:
            timeout = self.timeout
        else:
            timeout = SEARCH_CONTENT_TIMEOUT if include_content else SEARCH_TIMEOUT

        response = requests.get(
            self.base_url,
            params=params,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
            },
            timeout=timeout,
        )
        if not response.ok:
            try:
                body = response.json()
            except ValueError:
                body = None
            error_body = cast(
                SerpexErrorBody,
                body if isinstance(body, dict) else {"error": response.reason or ""},
            )
            raise SerpexAPIError(
                _error_text(response.status_code, error_body),
                response.status_code,
                error_body,
                response=response,
            )

        data: SerpexSearchResponse = response.json()
        results_list = data.get("results") or []
        if not results_list:
            return []

        api_metadata = data.get("metadata") or {}
        limit = max(1, min(int(num_results or MAX_RESULTS), MAX_RESULTS))

        documents = []
        for result in results_list[:limit]:
            title = result.get("title", "")
            url = result.get("url", "")
            snippet = result.get("snippet", "")

            text = f"{title}\nURL: {url}\n{snippet}"
            content = result.get("content")
            if content is not None:
                text += f"\n\n{content}"

            metadata: Dict[str, Any] = {
                "title": title,
                "url": url,
                "snippet": snippet,
                "position": result.get("position"),
                "number_of_results": api_metadata.get("number_of_results", 0),
                "response_time": api_metadata.get("response_time", 0),
                "credits_used": api_metadata.get("credits_used", 0),
                "query": query,
                # Deprecated: always "auto".
                "engine": "auto",
            }
            if "content_error" in result:
                metadata["content_error"] = result["content_error"]

            documents.append(Document(text=text, metadata=metadata))

        return documents
