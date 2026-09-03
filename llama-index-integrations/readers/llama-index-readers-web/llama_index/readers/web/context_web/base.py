"""Context.dev web reader."""

import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import httpx
from llama_index.core.bridge.pydantic import PrivateAttr, SecretStr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


ContextMode = Literal["scrape", "crawl", "sitemap", "search", "extract"]
JsonObject = Dict[str, Any]
QueryParams = List[Tuple[str, str]]


class ContextAPIError(RuntimeError):
    """Error returned by the Context.dev API."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
    ) -> None:
        suffix = f" [{error_code}]" if error_code else ""
        super().__init__(
            f"Context.dev API request failed ({status_code}){suffix}: {message}"
        )
        self.status_code = status_code
        self.error_code = error_code


class ContextWebReader(BasePydanticReader):
    """
    Load web content through Context.dev as LlamaIndex documents.

    Args:
        api_key: Context.dev API key. Defaults to ``CONTEXT_DEV_API_KEY``.
        api_url: Context.dev API base URL.
        mode: One of ``scrape``, ``crawl``, ``sitemap``, ``search``, or
            ``extract``.
        params: Parameters forwarded to the selected Context.dev endpoint.
        timeout: HTTP timeout in seconds.
        metadata_fn: Optional function that returns additional metadata for a URL.

    Examples:
        Scrape a page into a document:

        .. code-block:: python

            from llama_index.readers.web import ContextWebReader

            reader = ContextWebReader(mode="scrape")
            documents = reader.load_data(url="https://www.context.dev")

        Search the web and include page markdown:

        .. code-block:: python

            reader = ContextWebReader(
                mode="search",
                params={"markdownOptions": {"enabled": True}},
            )
            documents = reader.load_data(query="latest AI infrastructure news")

    """

    is_remote: bool = True
    api_key: SecretStr
    api_url: str
    mode: ContextMode
    params: JsonObject
    timeout: float

    _client: Optional[httpx.Client] = PrivateAttr(default=None)
    _async_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)
    _metadata_fn: Optional[Callable[[str], Dict[str, Any]]] = PrivateAttr(default=None)

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: ContextMode = "scrape",
        params: Optional[JsonObject] = None,
        timeout: float = 180.0,
        metadata_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the reader."""
        resolved_api_key = (
            api_key or os.getenv("CONTEXT_DEV_API_KEY") or os.getenv("CONTEXT_API_KEY")
        )
        if not resolved_api_key or not resolved_api_key.strip():
            raise ValueError(
                "Context.dev API key missing. Pass api_key or set CONTEXT_DEV_API_KEY."
            )
        resolved_api_key = resolved_api_key.strip()

        resolved_api_url = (
            api_url
            or os.getenv("CONTEXT_DEV_API_BASE")
            or os.getenv("CONTEXT_API_BASE")
            or "https://api.context.dev/v1"
        ).strip()
        if not resolved_api_url.startswith(("http://", "https://")):
            raise ValueError("Context.dev API URL must use HTTP or HTTPS.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        super().__init__(
            api_key=SecretStr(resolved_api_key),
            api_url=resolved_api_url.rstrip("/"),
            mode=mode,
            params=dict(params or {}),
            timeout=timeout,
        )
        self._client = client
        self._async_client = async_client
        self._metadata_fn = metadata_fn

    @classmethod
    def class_name(cls) -> str:
        return "ContextWebReader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        params: Optional[JsonObject] = None,
    ) -> List[Document]:
        """Load documents from a URL, domain, or search query."""
        method, path, query_params, body = self._request_parts(
            url=url,
            query=query,
            params=params,
        )
        response = self._request(
            method=method,
            path=path,
            query_params=query_params,
            body=body,
        )
        return self._documents(response)

    async def aload_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        params: Optional[JsonObject] = None,
    ) -> List[Document]:
        """Asynchronously load documents from a URL, domain, or search query."""
        method, path, query_params, body = self._request_parts(
            url=url,
            query=query,
            params=params,
        )
        response = await self._arequest(
            method=method,
            path=path,
            query_params=query_params,
            body=body,
        )
        return self._documents(response)

    def _request_parts(
        self,
        url: Optional[str],
        query: Optional[str],
        params: Optional[JsonObject],
    ) -> Tuple[str, str, QueryParams, Optional[JsonObject]]:
        arguments = {**self.params, **(params or {})}

        if self.mode == "search":
            if not query or url is not None:
                raise ValueError("Search mode requires query and does not accept url.")
            return "POST", "/web/search", [], {**arguments, "query": query}

        if not url or query is not None:
            raise ValueError(
                f"{self.mode.capitalize()} mode requires url and does not accept query."
            )

        if self.mode == "scrape":
            return (
                "GET",
                "/web/scrape/markdown",
                self._query_params({**arguments, "url": url}),
                None,
            )
        if self.mode == "sitemap":
            return (
                "GET",
                "/web/scrape/sitemap",
                self._query_params({**arguments, "domain": url}),
                None,
            )
        if self.mode == "crawl":
            return "POST", "/web/crawl", [], {**arguments, "url": url}
        if self.mode == "extract":
            schema = arguments.get("schema")
            if not isinstance(schema, dict) or not schema:
                raise ValueError(
                    "Extract mode requires a non-empty JSON Schema in params['schema']."
                )
            return "POST", "/web/extract", [], {**arguments, "url": url}
        raise ValueError(f"Unsupported Context.dev reader mode: {self.mode}")

    def _request(
        self,
        method: str,
        path: str,
        query_params: QueryParams,
        body: Optional[JsonObject],
    ) -> JsonObject:
        request_kwargs = self._request_kwargs(method, path, query_params, body)
        try:
            if self._client is not None:
                response = self._client.request(**request_kwargs)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(**request_kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Unable to reach the Context.dev API: {exc}") from exc
        return self._response_json(response)

    async def _arequest(
        self,
        method: str,
        path: str,
        query_params: QueryParams,
        body: Optional[JsonObject],
    ) -> JsonObject:
        request_kwargs = self._request_kwargs(method, path, query_params, body)
        try:
            if self._async_client is not None:
                response = await self._async_client.request(**request_kwargs)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(**request_kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Unable to reach the Context.dev API: {exc}") from exc
        return self._response_json(response)

    def _request_kwargs(
        self,
        method: str,
        path: str,
        query_params: QueryParams,
        body: Optional[JsonObject],
    ) -> JsonObject:
        request: JsonObject = {
            "method": method,
            "url": f"{self.api_url}{path}",
            "params": query_params,
            "headers": {
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
                "User-Agent": "llama-index-readers-web/context",
            },
        }
        if body is not None:
            request["json"] = body
        return request

    def _response_json(self, response: httpx.Response) -> JsonObject:
        try:
            payload = response.json()
        except ValueError as exc:
            if response.is_error:
                raise ContextAPIError(
                    response.status_code,
                    response.text.strip() or "Request failed",
                ) from exc
            raise RuntimeError(
                "Context.dev API returned an invalid JSON response."
            ) from exc

        if response.is_error:
            record = payload if isinstance(payload, dict) else {}
            message = next(
                (
                    record[key]
                    for key in ("message", "error", "error_description")
                    if isinstance(record.get(key), str) and record[key]
                ),
                "Request failed",
            )
            error_code = (
                record.get("error_code")
                if isinstance(record.get("error_code"), str)
                else None
            )
            raise ContextAPIError(response.status_code, message, error_code)

        if not isinstance(payload, dict):
            raise RuntimeError("Context.dev API returned an unexpected response shape.")
        return {key: value for key, value in payload.items() if key != "key_metadata"}

    def _documents(self, response: JsonObject) -> List[Document]:
        if self.mode == "scrape":
            return self._scrape_documents(response)
        if self.mode == "crawl":
            return self._crawl_documents(response)
        if self.mode == "sitemap":
            return self._sitemap_documents(response)
        if self.mode == "search":
            return self._search_documents(response)
        if self.mode == "extract":
            return self._extract_documents(response)
        raise ValueError(f"Unsupported Context.dev reader mode: {self.mode}")

    def _scrape_documents(self, response: JsonObject) -> List[Document]:
        markdown = response.get("markdown")
        url = response.get("url")
        if not isinstance(markdown, str) or not isinstance(url, str):
            raise RuntimeError(
                "Context.dev scrape response is missing markdown or url."
            )
        metadata = self._metadata(
            url,
            response.get("metadata"),
            mode="scrape",
            content_length=response.get("contentLength"),
        )
        return [Document(text=markdown, metadata=metadata)]

    def _crawl_documents(self, response: JsonObject) -> List[Document]:
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Context.dev crawl response is missing results.")

        documents: List[Document] = []
        for result in results:
            if not isinstance(result, dict) or not isinstance(
                result.get("markdown"), str
            ):
                continue
            page_metadata = result.get("metadata")
            page_url_value = (
                page_metadata.get("url") if isinstance(page_metadata, dict) else None
            )
            page_url = page_url_value if isinstance(page_url_value, str) else ""
            metadata = self._metadata(page_url, page_metadata, mode="crawl")
            documents.append(Document(text=result["markdown"], metadata=metadata))
        return documents

    def _sitemap_documents(self, response: JsonObject) -> List[Document]:
        urls = response.get("urls")
        domain = response.get("domain")
        if not isinstance(urls, list) or not isinstance(domain, str):
            raise RuntimeError(
                "Context.dev sitemap response is missing urls or domain."
            )
        crawl_metadata = response.get("meta")
        return [
            Document(
                text=url,
                metadata=self._metadata(
                    url,
                    {
                        "domain": domain,
                        "sitemap": crawl_metadata,
                    },
                    mode="sitemap",
                ),
            )
            for url in urls
            if isinstance(url, str)
        ]

    def _search_documents(self, response: JsonObject) -> List[Document]:
        results = response.get("results")
        query = response.get("query")
        if not isinstance(results, list) or not isinstance(query, str):
            raise RuntimeError(
                "Context.dev search response is missing results or query."
            )

        documents: List[Document] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            result_url = result.get("url")
            if not isinstance(result_url, str):
                continue
            markdown_field = result.get("markdown")
            markdown = (
                markdown_field.get("markdown")
                if isinstance(markdown_field, dict)
                and isinstance(markdown_field.get("markdown"), str)
                else None
            )
            title = result.get("title") if isinstance(result.get("title"), str) else ""
            description = (
                result.get("description")
                if isinstance(result.get("description"), str)
                else ""
            )
            text = (
                markdown
                or "\n\n".join(value for value in (title, description) if value)
                or result_url
            )
            metadata = self._metadata(
                result_url,
                {
                    "title": title,
                    "description": description,
                    "query": query,
                    "relevance": result.get("relevance"),
                    "markdown_status": (
                        markdown_field.get("code")
                        if isinstance(markdown_field, dict)
                        else None
                    ),
                },
                mode="search",
            )
            documents.append(Document(text=text, metadata=metadata))
        return documents

    def _extract_documents(self, response: JsonObject) -> List[Document]:
        data = response.get("data")
        url = response.get("url")
        if not isinstance(data, dict) or not isinstance(url, str):
            raise RuntimeError("Context.dev extract response is missing data or url.")
        metadata = self._metadata(
            url,
            {
                "urls_analyzed": response.get("urls_analyzed"),
                "extract": response.get("metadata"),
            },
            mode="extract",
        )
        return [
            Document(
                text=json.dumps(data, indent=2, ensure_ascii=False),
                metadata=metadata,
            )
        ]

    def _metadata(
        self,
        url: str,
        api_metadata: Any,
        mode: ContextMode,
        **extra: Any,
    ) -> JsonObject:
        metadata = dict(api_metadata) if isinstance(api_metadata, dict) else {}
        custom_metadata = self._metadata_fn(url) if self._metadata_fn else {}
        return {
            **metadata,
            **custom_metadata,
            **extra,
            "url": url,
            "source": url,
            "mode": mode,
        }

    @classmethod
    def _query_params(cls, values: JsonObject) -> QueryParams:
        return [
            pair
            for name, value in values.items()
            for pair in cls._query_value(name, value)
        ]

    @classmethod
    def _query_value(cls, name: str, value: Any) -> QueryParams:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [pair for item in value for pair in cls._query_value(name, item)]
        if isinstance(value, dict):
            return [
                pair
                for key, item in value.items()
                for pair in cls._query_value(f"{name}[{key}]", item)
            ]
        if isinstance(value, bool):
            return [(name, "true" if value else "false")]
        return [(name, str(value))]
