"""Plasmate reader for LlamaIndex - fetch web pages as semantic content."""

import json
import subprocess
from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PlasmateWebReader(BaseReader):
    """Read web pages using Plasmate's Semantic Object Model (SOM).

    Plasmate compiles HTML into structured semantic content, using 10-16x
    fewer tokens than raw HTML. Ideal for RAG pipelines that need web content.

    Requires the Plasmate binary: pip install plasmate

    Args:
        timeout: Maximum seconds to wait for each page fetch. Defaults to 30.
        javascript: Whether to enable JavaScript rendering. Defaults to True.
    """

    def __init__(self, timeout: int = 30, javascript: bool = True) -> None:
        self.timeout = timeout
        self.javascript = javascript

    def load_data(self, urls: List[str], **kwargs) -> List[Document]:
        """Load documents from web URLs using Plasmate.

        Args:
            urls: List of URLs to fetch and convert to documents.

        Returns:
            List of LlamaIndex Document objects with extracted content.
        """
        documents = []
        for url in urls:
            try:
                cmd = ["plasmate", "fetch", url]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout
                )
                if result.returncode != 0:
                    continue

                som = json.loads(result.stdout)

                # Extract text content from SOM regions
                text_parts = []
                for region in som.get("regions", []):
                    for element in region.get("elements", []):
                        text = element.get("text", "")
                        if text:
                            role = element.get("role", "")
                            if role == "heading":
                                level = element.get("attrs", {}).get("level", 2)
                                text_parts.append(f"{'#' * level} {text}")
                            else:
                                text_parts.append(text)

                content = "\n\n".join(text_parts)

                metadata = {
                    "url": url,
                    "title": som.get("title", ""),
                    "lang": som.get("lang", ""),
                    "som_version": som.get("som_version", ""),
                    "source": "plasmate",
                }

                documents.append(Document(text=content, metadata=metadata))
            except Exception:
                continue

        return documents
