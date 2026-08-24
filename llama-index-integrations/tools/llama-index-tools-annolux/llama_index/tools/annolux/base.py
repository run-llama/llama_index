import os
from typing import Any, Dict, List, Optional
import requests

try:
    from llama_index.core.schema import Document
    from llama_index.core.tools.tool_spec.base import BaseToolSpec
except ImportError:
    try:
        from llama_index.schema import Document  # type: ignore
        from llama_index.tools.tool_spec.base import BaseToolSpec  # type: ignore
    except ImportError:
        # Minimal fallback class when llama-index is not installed locally
        class BaseToolSpec:  # type: ignore
            pass

        class Document:  # type: ignore
            def __init__(self, text: str, extra_info: Optional[Dict[str, Any]] = None):
                self.text = text
                self.extra_info = extra_info or {}


class AnnoluxToolSpec(BaseToolSpec):
    """Annolux Curated Search Tool Spec for LlamaIndex."""

    spec_functions = ["annolux_search"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.annolux.com",
    ) -> None:
        """Initialize with Annolux API key."""
        self.api_key = api_key or os.environ.get("ANNOLUX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Annolux API key must be provided via `api_key` argument "
                "or `ANNOLUX_API_KEY` environment variable."
            )
        self.base_url = base_url.rstrip("/")

    def annolux_search(
        self,
        query: str,
        limit: int = 5,
        domains: Optional[List[str]] = None,
        deduplicate: bool = True,
        ranking: str = "default",
    ) -> List[Document]:
        """Search curated English and Chinese technical web corpus with explicit fetched_at dates.

        Args:
            query (str): The search query to retrieve information for.
            limit (int, optional): Max results to return (1-10). Defaults to 5.
            domains (List[str], optional): Domain whitelist filters (e.g. ['docs.rs']).
            deduplicate (bool, optional): Deduplicate results. Defaults to True.
            ranking (str, optional): Ranking strategy ('default' or 'provider'). Defaults to 'default'.

        Returns:
            List[Document]: List of LlamaIndex Document objects with metadata.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "llama-index-tools-annolux/0.1.0",
        }
        payload: Dict[str, Any] = {
            "query": query,
            "limit": min(max(1, limit), 10),
            "deduplicate": deduplicate,
            "ranking": ranking,
        }
        if domains:
            payload["domains"] = domains

        response = requests.post(
            f"{self.base_url}/v1/search",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        documents = []
        for item in data.get("results", []):
            text = f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nSnapshot Date: {item.get('fetched_at', '')}\n\n{item.get('snippet', '')}"
            doc = Document(
                text=text,
                extra_info={
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "fetched_at": item.get("fetched_at", ""),
                    "domain": item.get("domain", ""),
                    "score": item.get("score", 0.0),
                },
            )
            documents.append(doc)

        return documents
