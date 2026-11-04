"""AlterLab Web Reader."""

import os
import time
from typing import Any, Dict, List, Optional

import requests

from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

DEFAULT_BASE_URL = "https://api.alterlab.io"
_JOB_DONE = {"succeeded", "completed"}
_JOB_FAILED = {"failed", "rejected", "cancelled"}


def _content_text(page: Dict[str, Any]) -> str:
    """Return markdown, text or html from a scrape or crawl page result."""
    content = page.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("markdown", "text", "html"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


class AlterLabWebReader(BasePydanticReader):
    """
    Load web pages with the AlterLab API.

    AlterLab is a web scraping API with anti-bot handling and JavaScript
    rendering. It returns markdown and picks the lowest-cost scraping tier that
    can fetch each page.

    Args:
        api_key (Optional[str]): AlterLab API key. Defaults to the
            ``ALTERLAB_API_KEY`` environment variable.
        mode (str): One of ``"scrape"``, ``"crawl"`` or ``"search"``.
            Default is ``"scrape"``.
        params (Optional[dict]): Extra fields for the API request, for example
            ``{"mode": "js"}`` for scrape or ``{"max_pages": 20}`` for crawl.
            LLM-based extraction fields (``extraction_prompt``,
            ``extraction_schema``) need your own LLM provider key registered in
            AlterLab (bring your own key).
        base_url (str): API base URL. Default is ``https://api.alterlab.io``.
        poll_timeout (float): Maximum seconds to wait for a job. Default 300.

    Example:
        >>> reader = AlterLabWebReader(mode="scrape")
        >>> documents = reader.load_data(url="https://example.com")

    """

    api_key: Optional[str] = Field(default=None, exclude=True)
    mode: str = "scrape"
    params: Dict[str, Any] = Field(default_factory=dict)
    base_url: str = DEFAULT_BASE_URL
    poll_timeout: float = 300.0

    @classmethod
    def class_name(cls) -> str:
        return "AlterLabWebReader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> List[Document]:
        """
        Load documents from a URL or search query.

        Args:
            url (Optional[str]): URL to scrape, or start URL to crawl.
            query (Optional[str]): Search query (``search`` mode).
            params (Optional[dict]): Extra request fields, merged over the
                reader's ``params``.

        Returns:
            List[Document]: One document per page or search result.

        """
        merged = {**self.params, **(params or {})}
        if self.mode == "scrape":
            if not url:
                raise ValueError("url must be provided for scrape mode.")
            return self._scrape(url, merged)
        if self.mode == "crawl":
            if not url:
                raise ValueError("url must be provided for crawl mode.")
            return self._crawl(url, merged)
        if self.mode == "search":
            if not query:
                raise ValueError("query must be provided for search mode.")
            return self._search(query, merged)
        raise ValueError("Invalid mode. Choose from 'scrape', 'crawl' or 'search'.")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        key = self.api_key or os.environ.get("ALTERLAB_API_KEY")
        if not key:
            raise ValueError(
                "AlterLab API key is required. Pass api_key or set ALTERLAB_API_KEY."
            )
        return {"X-API-Key": key}

    def _request(self, method: str, path: str, **kwargs: Any) -> "requests.Response":
        response = requests.request(
            method,
            f"{self.base_url.rstrip('/')}{path}",
            headers=self._headers(),
            timeout=60,
            **kwargs,
        )
        if not response.ok:
            raise RuntimeError(
                f"AlterLab API error {response.status_code}: {response.text[:500]}"
            )
        return response

    def _wait(self, path: str, done: set, failed: set, **kwargs: Any) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        delay = 1.0
        while True:
            data = self._request("GET", path, **kwargs).json()
            status = data.get("status")
            if status in done:
                return data
            if status in failed:
                raise RuntimeError(
                    f"AlterLab job {status}: {data.get('error') or 'no detail'}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError("AlterLab job did not finish before poll_timeout")
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def _scrape(self, url: str, params: dict) -> List[Document]:
        payload = {
            "url": url,
            "mode": "auto",
            "sync": True,
            "formats": ["markdown"],
            **params,
        }
        response = self._request("POST", "/api/v1/scrape", json=payload)
        result = response.json()
        if response.status_code == 202 and result.get("job_id"):
            job = self._wait(f"/api/v1/jobs/{result['job_id']}", _JOB_DONE, _JOB_FAILED)
            result = job.get("result") or job
        metadata = {"source": url, **self._page_metadata(result)}
        return [Document(text=_content_text(result), metadata=metadata)]

    def _crawl(self, url: str, params: dict) -> List[Document]:
        payload = {"url": url, "formats": ["markdown"], **params}
        crawl_id = self._request("POST", "/api/v1/crawl", json=payload).json()[
            "crawl_id"
        ]
        result = self._wait(
            f"/api/v1/crawl/{crawl_id}",
            {"completed", "partial"},
            {"failed"},
            params={"include_results": "true"},
        )
        documents = []
        for page in result.get("pages") or []:
            text = _content_text(page)
            if text:
                metadata = {"source": page.get("url", url), **self._page_metadata(page)}
                documents.append(Document(text=text, metadata=metadata))
        return documents

    def _search(self, query: str, params: dict) -> List[Document]:
        payload = {"query": query, **params}
        results = (
            self._request("POST", "/api/v1/search", json=payload).json().get("results")
            or []
        )
        documents = []
        for item in results:
            text = item.get("snippet") or item.get("description") or ""
            title = item.get("title", "")
            metadata = {
                "source": item.get("url", ""),
                "title": title,
                "query": query,
                "position": item.get("position"),
            }
            documents.append(
                Document(
                    text=f"{title}\n{text}".strip(),
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
        return documents

    @staticmethod
    def _page_metadata(page: Dict[str, Any]) -> Dict[str, Any]:
        meta = page.get("metadata")
        if not isinstance(meta, dict):
            return {}
        return {k: meta[k] for k in ("title", "status_code") if meta.get(k) is not None}
