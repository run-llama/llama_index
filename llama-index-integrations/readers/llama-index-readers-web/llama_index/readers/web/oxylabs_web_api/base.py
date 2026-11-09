"""Oxylabs Web API Reader."""

import asyncio
import os
import random
import time
import warnings
from importlib.metadata import PackageNotFoundError, version
from platform import architecture, python_version
from typing import Any, Dict, List, Literal, Optional

import httpx
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

# Transient statuses worth another attempt. A spent quota, a faulted request and
# a scrape timeout are verdicts, and resending them changes nothing.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
TIMEOUT_TITLE = "REQUEST_FAILED_SCRAPE_TIMEOUT"
QUOTA_TITLE = "QUOTA_EXCEEDED"


class OxylabsWebAPIError(Exception):
    """A Web API failure that affects every url, such as a rejected API key."""


class _UrlFailed(Exception):
    """The Web API could not return this url; the others may still load."""


class _ScrapeTimeout(Exception):
    """The synchronous scrape ran out of time; the request can be queued instead."""


def _sdk_type() -> str:
    try:
        package_version = version("llama-index-readers-web")
    except PackageNotFoundError:
        package_version = "0.0.0"
    bits, _ = architecture()
    return (
        f"oxylabs-llama-index-web-sdk-python/{package_version} "
        f"({python_version()}; {bits})"
    )


def _json_body(response: httpx.Response) -> Optional[Dict[str, Any]]:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _detail(body: Optional[Dict[str, Any]], response: httpx.Response) -> str:
    """The useful part of an error body: per-field errors first, then `detail`."""
    if body is None:
        return response.text[:300]
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(
            ": ".join(str(part) for part in (e.get("pointer"), e.get("detail")) if part)
            for e in errors
            if isinstance(e, dict)
        )
    return str(body.get("detail") or body.get("message") or body)[:300]


def _retryable(response: httpx.Response, body: Optional[Dict[str, Any]]) -> bool:
    if response.status_code not in RETRY_STATUS:
        return False
    if body is None:
        return True
    return body.get("title") not in (TIMEOUT_TITLE, QUOTA_TITLE) and (
        body.get("state") != "faulted"
    )


def _check(response: httpx.Response, path: str) -> Dict[str, Any]:
    """Return the response body, or raise what went wrong."""
    body = _json_body(response)
    title = (body or {}).get("title")
    status = response.status_code

    if status in (401, 403):
        raise OxylabsWebAPIError(
            f"Oxylabs Web API rejected the API key ({status}). Check"
            " OXYLABS_WEB_API_KEY: it must come from a Web API instance in the"
            " Oxylabs dashboard, and keys from other Oxylabs products don't work."
        )
    if title == QUOTA_TITLE:
        raise OxylabsWebAPIError(
            "Oxylabs Web API quota is spent. Check the plan in the Oxylabs dashboard."
        )
    if title == TIMEOUT_TITLE:
        raise _ScrapeTimeout
    if body is not None and body.get("state") == "faulted":
        request_id = (body.get("metadata") or {}).get("request_id")
        reason = ": ".join(str(p) for p in (title, body.get("detail")) if p)
        raise _UrlFailed(
            f"the request faulted ({reason or 'no reason given'}; request_id"
            f" {request_id or 'unknown'})"
        )
    if status >= 400:
        raise _UrlFailed(f"{path} returned {status}: {_detail(body, response)}")
    if body is None:
        raise _UrlFailed(f"{path} returned a non-JSON response")
    return body


def _request_id(queued: Dict[str, Any]) -> str:
    request_id = (queued.get("metadata") or {}).get("request_id")
    if not request_id:
        raise _UrlFailed("the scrape was queued but no request id came back")
    return str(request_id)


class OxylabsWebAPIReader(BasePydanticReader):
    """
    Load web pages with the Oxylabs Web API, as Markdown or HTML.

    The Web API opens JavaScript-heavy and bot-protected pages, and
    authenticates with a single API key. Get one by creating a Web API instance
    in the [Oxylabs dashboard](https://dashboard.oxylabs.io).

    A url the API could not return is skipped with a warning, so the other
    urls still load. A rejected API key or a spent quota raises
    `OxylabsWebAPIError`, since no url can succeed.

    Args:
        api_key: Oxylabs Web API key. Defaults to `OXYLABS_WEB_API_KEY`.
        output: `markdown` (default) or `html`.
        location: Two-letter country code to fetch the pages from, e.g. `DE`.
        device: `desktop` (the API's default) or `mobile`.
        run_js: Render the pages' JavaScript. Slower; use it for pages that
            render client-side.

    Example:
        .. code-block:: python

            from llama_index.readers.web import OxylabsWebAPIReader

            reader = OxylabsWebAPIReader(location="DE")

            docs = reader.load_data(
                [
                    "https://sandbox.oxylabs.io/products/1",
                    "https://sandbox.oxylabs.io/products/2",
                ]
            )

            print(docs[0].text)

    """

    is_remote: bool = True
    output: Literal["markdown", "html"] = "markdown"
    location: Optional[str] = Field(default=None, pattern="^[A-Za-z]{2}$")
    device: Optional[Literal["desktop", "mobile"]] = None
    run_js: bool = False
    base_url: str = "https://webapi.oxylabs.io"
    request_timeout: float = Field(
        default=200, description="Seconds to wait for one HTTP response."
    )
    max_retries: int = Field(
        default=2, description="Extra attempts for 429, 5xx and connection errors."
    )
    max_wait: float = Field(
        default=600, description="Seconds to wait for a slow page the API queued."
    )
    poll_interval: float = Field(
        default=5, description="Seconds between checks on a queued page."
    )
    max_concurrency: int = Field(
        default=5, description="Pages `aload_data` fetches at the same time."
    )

    # Kept out of the model's fields so it is never serialized or printed.
    _api_key: str = PrivateAttr()

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # A key pasted or read from a file often carries a trailing newline.
        api_key = (api_key or os.environ.get("OXYLABS_WEB_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "Pass api_key or set the OXYLABS_WEB_API_KEY environment variable."
            )
        if any(char.isspace() for char in api_key):
            # httpx would refuse the header with an error that quotes the key.
            raise ValueError(
                "OXYLABS_WEB_API_KEY contains whitespace or a line break; pass the"
                " key on its own."
            )
        self._api_key = api_key

    @classmethod
    def class_name(cls) -> str:
        return "OxylabsWebAPIReader"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "x-oxylabs-sdk": _sdk_type(),
        }

    def _payload(self, url: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"url": url, "output": [self.output]}
        if self.location:
            payload["location"] = self.location.upper()
        if self.device:
            payload["device"] = self.device
        if self.run_js:
            payload["run_js"] = True
        return payload

    @staticmethod
    def _backoff(attempt: int) -> float:
        # Full jitter, so concurrent loads that hit the same 429 don't retry in step.
        return random.uniform(0, 2 ** (attempt - 1))

    def _request(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error = "no attempt was made"
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self._backoff(attempt))
            try:
                response = client.request(
                    method,
                    f"{self.base_url}{path}",
                    json=payload,
                    headers=self._headers(),
                )
            except httpx.LocalProtocolError:
                # Never echo this one: httpx quotes the rejected header, the key.
                raise OxylabsWebAPIError(
                    "The request headers were rejected: check that"
                    " OXYLABS_WEB_API_KEY holds only the key."
                ) from None
            except httpx.HTTPError as exc:
                error = f"could not reach the Oxylabs Web API: {exc}"
                continue
            if attempt < self.max_retries and _retryable(
                response, _json_body(response)
            ):
                error = f"{path} returned {response.status_code}"
                continue
            return _check(response, path)
        raise _UrlFailed(error)

    async def _arequest(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error = "no attempt was made"
        for attempt in range(self.max_retries + 1):
            if attempt:
                await asyncio.sleep(self._backoff(attempt))
            try:
                response = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    json=payload,
                    headers=self._headers(),
                )
            except httpx.LocalProtocolError:
                raise OxylabsWebAPIError(
                    "The request headers were rejected: check that"
                    " OXYLABS_WEB_API_KEY holds only the key."
                ) from None
            except httpx.HTTPError as exc:
                error = f"could not reach the Oxylabs Web API: {exc}"
                continue
            if attempt < self.max_retries and _retryable(
                response, _json_body(response)
            ):
                error = f"{path} returned {response.status_code}"
                continue
            return _check(response, path)
        raise _UrlFailed(error)

    def _scrape(self, client: httpx.Client, url: str) -> Dict[str, Any]:
        payload = self._payload(url)
        try:
            return self._request(client, "POST", "/v1/scrape", payload)
        except _ScrapeTimeout:
            # Too slow for the synchronous call: the same request goes to the
            # API's queue, and the result is collected there.
            queued = self._request(client, "POST", "/v1/async/scrape", payload)
            request_id = _request_id(queued)
            deadline = time.monotonic() + self.max_wait
            while True:
                body = self._request(client, "GET", f"/v1/async/scrape/{request_id}")
                if body.get("state") != "pending":
                    return body
                if time.monotonic() >= deadline:
                    raise _UrlFailed(
                        f"still loading after {self.max_wait:.0f}s (raise `max_wait`)"
                    ) from None
                time.sleep(self.poll_interval)

    async def _ascrape(self, client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
        payload = self._payload(url)
        try:
            return await self._arequest(client, "POST", "/v1/scrape", payload)
        except _ScrapeTimeout:
            queued = await self._arequest(client, "POST", "/v1/async/scrape", payload)
            request_id = _request_id(queued)
            deadline = time.monotonic() + self.max_wait
            while True:
                body = await self._arequest(
                    client, "GET", f"/v1/async/scrape/{request_id}"
                )
                if body.get("state") != "pending":
                    return body
                if time.monotonic() >= deadline:
                    raise _UrlFailed(
                        f"still loading after {self.max_wait:.0f}s (raise `max_wait`)"
                    ) from None
                await asyncio.sleep(self.poll_interval)

    def _document(self, url: str, response: Dict[str, Any]) -> Document:
        results = [r for r in response.get("results") or [] if isinstance(r, dict)]
        if not results:
            raise _UrlFailed("the API returned no results")
        text = "\n\n".join(str(r.get(self.output) or "") for r in results)

        metadata: Dict[str, Any] = {"source": url}
        result_meta = results[0].get("metadata")
        if isinstance(result_meta, dict):
            if result_meta.get("url"):
                metadata["final_url"] = result_meta["url"]
            http = (result_meta.get("statuses") or {}).get("http_request") or {}
            if http.get("code"):
                metadata["status_code"] = http["code"]
        request_id = (response.get("metadata") or {}).get("request_id")
        if request_id:
            metadata["request_id"] = request_id
        return Document(text=text, metadata=metadata)

    @staticmethod
    def _skip(url: str, reason: str) -> None:
        # Without the warning a failed url is indistinguishable from one that was
        # never requested, so a partially successful load looks complete.
        warnings.warn(
            f"Oxylabs Web API could not load {url!r}: {reason}. Skipping it.",
            stacklevel=3,
        )

    def load_data(self, urls: List[str]) -> List[Document]:
        """
        Load urls into Documents, one request at a time.

        Args:
            urls: Absolute http(s) urls to load.

        """
        documents = []
        with httpx.Client(timeout=self.request_timeout) as client:
            for url in urls:
                try:
                    documents.append(self._document(url, self._scrape(client, url)))
                except _UrlFailed as exc:
                    self._skip(url, str(exc))
        return documents

    async def aload_data(self, urls: List[str]) -> List[Document]:
        """
        Load urls into Documents, up to `max_concurrency` at a time.

        Args:
            urls: Absolute http(s) urls to load.

        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def load(client: httpx.AsyncClient, url: str) -> Optional[Document]:
            async with semaphore:
                try:
                    return self._document(url, await self._ascrape(client, url))
                except _UrlFailed as exc:
                    self._skip(url, str(exc))
                    return None

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            loaded = await asyncio.gather(*(load(client, url) for url in urls))
        return [document for document in loaded if document is not None]
