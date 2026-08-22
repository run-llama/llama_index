from typing import List, Optional
import urllib.request
import json

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class BareWireWebReader(BaseReader):
    """BareWire web reader.

    Uses BareWire's agentic proxy to strip bloat, bypass anti-bot measures,
    and return semantic Markdown.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize with BareWire API key."""
        self.api_key = api_key
        self.api_endpoint = "https://api.barewire.ai/v1/connect"

    def load_data(
        self, urls: List[str], custom_hostname: Optional[str] = None
    ) -> List[Document]:
        """Load data from the URLs."""
        documents = []
        for url in urls:
            req = urllib.request.Request(
                self.api_endpoint,
                data=json.dumps({"url": url}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            try:
                with urllib.request.urlopen(req) as response:
                    content = response.read().decode("utf-8")
                    documents.append(
                        Document(text=content, metadata={"source": url})
                    )
            except Exception as e:
                raise ValueError(f"Error fetching {url} via BareWire: {e}")
        return documents
