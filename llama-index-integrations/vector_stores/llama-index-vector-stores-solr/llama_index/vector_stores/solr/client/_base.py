"""Sync/async clients for interacting with Apache Solr."""

from typing import Any, Optional

from llama_index.vector_stores.solr.constants import SolrConstants


class _BaseSolrClient:
    """Base Solr client for shared functionality."""

    def __init__(
        self,
        base_url: str,
        request_timeout_sec: int = SolrConstants.DEFAULT_TIMEOUT_SEC,
        headers: Optional[dict[str, str]] = None,
        **client_kwargs: Any,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url:
                The base URL of the target Solr collection or core.
            request_timeout_sec: The timeout for requests to Solr.
            headers: Additional headers to include in all requests.
            **client_kwargs:
                Additional keyword arguments to pass to the internal client
                constructor.

        """
        if not base_url.strip():
            raise ValueError(
                f"Parameter 'base_url' cannot be empty, input='{base_url}'"
            )
        if request_timeout_sec < 0:
            raise ValueError(
                f"Parameter 'request_timeout_sec' cannot be negative, "
                f"input='{request_timeout_sec}'"
            )

        self._base_url = base_url.rstrip("/")
        self._request_timeout_sec = request_timeout_sec
        self._headers = headers or {}
        self._client_kwargs = client_kwargs

        # client will be created in implementations
        self._client: Any = None

    def __str__(self) -> str:
        """String representation of the client."""
        return f"{self.__class__.__name__}(base_url='{self.base_url}')"

    def __repr__(self) -> str:
        """String representation of the client."""
        return str(self)

    @property
    def base_url(self) -> str:
        """The base URL of the target Solr collection."""
        return self._base_url
