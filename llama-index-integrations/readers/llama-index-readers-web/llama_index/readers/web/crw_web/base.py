"""CRW Web Reader."""

import time
from typing import Any, Dict, List, Optional

import requests

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

VALID_MODES = {"scrape", "crawl", "map"}


class CrwWebReader(BasePydanticReader):
    """Load web pages as markdown Documents using a self-hosted CRW server (https://github.com/us/crw)."""

    is_remote: bool = True

    base_url: str = "http://localhost:3000"
    mode: str = "scrape"
    api_key: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    poll_interval: float = 2.0
    poll_timeout: float = 300.0

    _session: requests.Session = PrivateAttr()

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        mode: str = "scrape",
        api_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        poll_interval: float = 2.0,
        poll_timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {sorted(VALID_MODES)}."
            )

        super().__init__(
            base_url=base_url,
            mode=mode,
            api_key=api_key,
            params=params,
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
            **kwargs,
        )

        session = requests.Session()
        if api_key is not None:
            session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._session = session

    @classmethod
    def class_name(cls) -> str:
        return "CrwWeb_reader"

    def load_data(self, url: str, mode: Optional[str] = None) -> List[Document]:
        if not url:
            raise ValueError("url must not be empty.")

        effective_mode = self.mode if mode is None else mode
        if effective_mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{effective_mode}'. Must be one of "
                f"{sorted(VALID_MODES)}."
            )

        if effective_mode == "scrape":
            return self._scrape(url)
        if effective_mode == "crawl":
            return self._crawl(url)
        return self._map(url)

    def _base_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"formats": ["markdown"]}
        if self.params:
            body.update(self.params)
        return body

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        response = self._session.post(url, json=body, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        response = self._session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def _raise_if_error(self, data: Dict[str, Any], context: str) -> None:
        if not data.get("success", True):
            error = data.get("error", "unknown error")
            raise RuntimeError(f"CRW {context} failed: {error}")

    def _doc_from_page(self, page: Dict[str, Any]) -> Document:
        text = page.get("markdown") or page.get("content") or ""
        metadata = dict(page.get("metadata") or {})
        url_val = page.get("url") or metadata.get("sourceURL") or ""
        metadata.setdefault("source_url", url_val)
        metadata.setdefault("title", page.get("title", ""))
        metadata.setdefault("statusCode", page.get("statusCode"))
        return Document(text=text, metadata=metadata)

    def _scrape(self, url: str) -> List[Document]:
        body = self._base_body()
        body["url"] = url
        data = self._post("/v1/scrape", body)
        self._raise_if_error(data, "scrape")
        page = data.get("data", data)
        return [self._doc_from_page(page)]

    def _crawl(self, url: str) -> List[Document]:
        body = self._base_body()
        body["url"] = url
        submit = self._post("/v1/crawl", body)
        self._raise_if_error(submit, "crawl submit")

        job_id = submit.get("id")
        if not job_id:
            raise RuntimeError("CRW crawl response did not include a job id.")

        deadline = time.monotonic() + self.poll_timeout
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Crawl job '{job_id}' did not complete within "
                    f"{self.poll_timeout}s."
                )

            status_data = self._get(f"/v1/crawl/{job_id}")
            status = status_data.get("status", "")

            if status == "completed":
                break

            if status not in {"pending", "running", "scraping"}:
                raise RuntimeError(f"Crawl job ended with status '{status}'.")

            time.sleep(self.poll_interval)

        pages = status_data.get("data", [])
        return [self._doc_from_page(p) for p in pages]

    def _map(self, url: str) -> List[Document]:
        body = self._base_body()
        body["url"] = url
        data = self._post("/v1/map", body)
        self._raise_if_error(data, "map")

        documents: List[Document] = []
        for link in data.get("links", []):
            if isinstance(link, str):
                link_url, title = link, ""
            else:
                link_url = link.get("url", "")
                title = link.get("title", "")

            documents.append(
                Document(
                    text=title or link_url,
                    metadata={
                        "source_url": link_url,
                        "title": title,
                        "statusCode": link.get("statusCode")
                        if isinstance(link, dict)
                        else None,
                        "source": "map",
                    },
                )
            )
        return documents
