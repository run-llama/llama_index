"""ClearFetch reader for LlamaIndex."""
from typing import List, Optional
import os
import requests

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ClearFetchReader(BaseReader):
    """
    ClearFetch Reader.

    Extracts clean, token-optimized Markdown content from web URLs
    with automated anti-bot and Cloudflare Turnstile bypass.

    Args:
        api_key (Optional[str]): ClearFetch API key from https://clearfetch.dev/dashboard.
            Defaults to CLEARFETCH_API_KEY environment variable.
        api_base (str): Base URL of ClearFetch API. Defaults to https://api.clearfetch.dev.
        stealth (bool): Enable stealth residential proxy routing. Defaults to True.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.clearfetch.dev",
        stealth: bool = True,
    ):
        """Initialize ClearFetchReader."""
        self.api_key = api_key or os.getenv("CLEARFETCH_API_KEY", "")
        self.api_base = api_base.rstrip("/")
        self.stealth = stealth

    def load_data(
        self,
        urls: List[str],
        stealth: Optional[bool] = None,
    ) -> List[Document]:
        """
        Load web pages as LlamaIndex Document objects.

        Args:
            urls (List[str]): List of webpage URLs to read.
            stealth (Optional[bool]): Override default stealth proxy behavior.

        Returns:
            List[Document]: Extracted LlamaIndex Document objects with token-optimized Markdown.
        """
        documents = []
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LlamaIndex-ClearFetchReader/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        use_stealth = self.stealth if stealth is None else stealth

        for url in urls:
            try:
                response = requests.post(
                    f"{self.api_base}/v1/read",
                    json={"url": url, "stealth": use_stealth},
                    headers=headers,
                    timeout=45.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content") or data.get("markdown") or ""
                    title = data.get("title", "")
                    tokens_saved = data.get("tokens_saved", 0)

                    documents.append(
                        Document(
                            text=content,
                            extra_info={
                                "url": url,
                                "title": title,
                                "source": "clearfetch",
                                "tokens_saved": tokens_saved,
                            },
                        )
                    )
            except Exception as e:
                continue

        return documents
