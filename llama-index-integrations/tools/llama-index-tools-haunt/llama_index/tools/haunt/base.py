"""Haunt tool spec."""

import json
from typing import List, Optional, Union

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

_READER_PROMPT = "Return the readable page content."


class HauntToolSpec(BaseToolSpec):
    """Haunt web extraction tool spec.

    Haunt reads public web pages and returns structured JSON or clean
    markdown. When a page cannot be read, it returns an honest error code
    (access_denied, login_required, not_found) instead of invented content,
    so an agent can branch on the failure. Failed reads are not charged.
    """

    spec_functions = [
        "extract",
        "load",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize with parameters."""
        from hauntapi import Haunt

        self.client = Haunt(api_key=api_key)

    def extract(self, url: str, prompt: str) -> str:
        """
        Extract structured data from a public web page.

        Args:
            url: The public web page URL to read.
            prompt: Plain-language description of the data to return,
                for example "the product name, price and stock status".

        Returns:
            A JSON string with the extracted data, or a JSON string with an
            honest error_code and message when the page cannot be read.

        """
        result = self.client.extract(url, prompt)
        if not result.success:
            return json.dumps(
                {
                    "error_code": result.error_code or "extraction_failed",
                    "message": result.message or result.error or "extraction failed",
                }
            )
        data = result.data
        return data if isinstance(data, str) else json.dumps(data)

    def load(self, urls: Union[str, List[str]]) -> List[Document]:
        """
        Load one or more public web pages as clean markdown Documents.

        Args:
            urls: One URL or a list of URLs.

        Returns:
            A list of Documents with the page content as markdown. An
            unreadable page raises ValueError with the honest reason,
            never invented content.

        """
        if isinstance(urls, str):
            urls = [urls]
        documents = []
        for url in urls:
            result = self.client.extract(
                url, _READER_PROMPT, response_format="markdown"
            )
            if not result.success:
                raise ValueError(
                    f"Haunt could not read {url}: "
                    f"{result.error_code or 'extraction_failed'}: "
                    f"{result.message or result.error or 'no detail'}"
                )
            data = result.data
            if isinstance(data, dict) and isinstance(data.get("markdown"), str):
                text = data["markdown"]
            elif isinstance(data, str):
                text = data
            else:
                text = json.dumps(data)
            documents.append(Document(text=text, extra_info={"url": url}))
        return documents
