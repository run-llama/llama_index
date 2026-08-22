from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from pydantic import Field, PrivateAttr

from hydrafetch import AsyncHydrafetch, Hydrafetch

Mode = Literal["scrape", "crawl", "map"]

CONTENT_FORMATS = ("markdown", "html", "rawHtml", "summary")

_METADATA_KEYS = (
    "title",
    "description",
    "language",
    "siteName",
    "author",
    "publishedTime",
    "wordCount",
    "pageType",
    "image",
)


class HydrafetchReader(BasePydanticReader):
    """Load web pages into LlamaIndex documents via the Hydrafetch API.

    Three modes. `scrape` loads one page, `crawl` walks a site and loads every page it
    finds, and `map` returns one document per discovered URL without fetching the bodies.
    """

    is_remote: bool = True

    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    content_format: str = "markdown"
    raise_for_status: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    _client: Any = PrivateAttr(default=None)
    _aclient: Any = PrivateAttr(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "HydrafetchReader"

    def _sync(self) -> Hydrafetch:
        if self._client is None:
            self._client = Hydrafetch(self.api_key, **self._kwargs())
        return self._client

    def _async(self) -> AsyncHydrafetch:
        if self._aclient is None:
            self._aclient = AsyncHydrafetch(self.api_key, **self._kwargs())
        return self._aclient

    def _kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        return kwargs

    def load_data(
        self,
        urls: str | Iterable[str],
        *,
        mode: Mode = "scrape",
        **params: Any,
    ) -> list[Document]:
        return list(self.lazy_load_data(urls, mode=mode, **params))

    def lazy_load_data(
        self,
        urls: str | Iterable[str],
        *,
        mode: Mode = "scrape",
        **params: Any,
    ) -> Iterator[Document]:
        for url in _as_list(urls):
            yield from self._load_one(url, _check_mode(mode), params)

    async def aload_data(
        self,
        urls: str | Iterable[str],
        *,
        mode: Mode = "scrape",
        **params: Any,
    ) -> list[Document]:
        checked = _check_mode(mode)
        documents: list[Document] = []
        for url in _as_list(urls):
            documents.extend(await self._aload_one(url, checked, params))
        return documents

    def _load_one(self, url: str, mode: Mode, params: Mapping[str, Any]) -> Iterator[Document]:
        merged = {**self.params, **params}
        if mode == "map":
            yield from _map_documents(_unwrap(self._sync().map(url, **merged)))
        elif mode == "crawl":
            job = _unwrap(self._sync().crawl(url, **self._scrape_params(merged)))
            yield from self._crawl_documents(job)
        else:
            data = _unwrap(self._sync().scrape(url, **self._scrape_params(merged)))
            self._guard(url, data)
            yield self._document(data)

    async def _aload_one(self, url: str, mode: Mode, params: Mapping[str, Any]) -> list[Document]:
        merged = {**self.params, **params}
        if mode == "map":
            return list(_map_documents(_unwrap(await self._async().map(url, **merged))))
        if mode == "crawl":
            job = _unwrap(await self._async().crawl(url, **self._scrape_params(merged)))
            return list(self._crawl_documents(job))
        data = _unwrap(await self._async().scrape(url, **self._scrape_params(merged)))
        self._guard(url, data)
        return [self._document(data)]

    def _crawl_documents(self, job: Mapping[str, Any]) -> Iterator[Document]:
        for page in job.get("pages") or []:
            if not isinstance(page, dict):
                continue
            data = page.get("data")
            if not isinstance(data, dict):
                continue
            data.setdefault("url", page.get("url") or page.get("requestedUrl"))
            if self.raise_for_status and not _ok(data):
                continue
            document = self._document(data)
            if page.get("depth") is not None:
                document.metadata["depth"] = page["depth"]
            yield document

    def _guard(self, url: str, data: Mapping[str, Any]) -> None:
        if self.raise_for_status and not _ok(data):
            raise ValueError(
                f"{url} returned HTTP {data.get('status')}. The body of an error page is not "
                "the page you asked for; pass raise_for_status=False to load it anyway."
            )

    def _document(self, data: Mapping[str, Any]) -> Document:
        return _to_document(data, self.content_format)

    def _scrape_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(params)
        merged.setdefault("formats", [self.content_format])
        return merged


def _as_list(urls: str | Iterable[str]) -> list[str]:
    return [urls] if isinstance(urls, str) else list(urls)


def _check_mode(mode: str) -> Mode:
    if mode not in ("scrape", "crawl", "map"):
        raise ValueError(f"mode must be 'scrape', 'crawl' or 'map', got {mode!r}")
    return mode  # type: ignore[return-value]


def _unwrap(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload if isinstance(payload, dict) else {}


def _ok(data: Mapping[str, Any]) -> bool:
    status = data.get("status")
    if status is None:
        return True
    try:
        return 200 <= int(status) < 300
    except (TypeError, ValueError):
        return True


def _map_documents(data: Mapping[str, Any]) -> Iterator[Document]:
    for entry in data.get("links") or []:
        if not isinstance(entry, dict) or not entry.get("url"):
            continue
        metadata: dict[str, Any] = {"source": entry["url"]}
        if entry.get("lastmod"):
            metadata["lastmod"] = entry["lastmod"]
        yield Document(text=entry["url"], metadata=metadata)


def _to_document(data: Mapping[str, Any], content_format: str) -> Document:
    text = data.get(content_format)
    if text is None and content_format != "markdown":
        text = data.get("markdown")
    if text is None:
        structured = data.get("structured") or data.get("json")
        text = str(structured) if structured is not None else ""

    metadata: dict[str, Any] = {"source": data.get("url") or data.get("finalUrl") or ""}
    if data.get("finalUrl") and data.get("finalUrl") != metadata["source"]:
        metadata["final_url"] = data["finalUrl"]
    if data.get("status") is not None:
        metadata["status"] = data["status"]

    raw = data.get("metadata")
    if isinstance(raw, Mapping):
        for key in _METADATA_KEYS:
            value = raw.get(key)
            if value is not None:
                metadata[_snake(key)] = value

    return Document(text=str(text), metadata=metadata)


def _snake(key: str) -> str:
    out: list[str] = []
    for char in key:
        if char.isupper():
            out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out)
