"""Seltz tool spec."""

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class SeltzToolSpec(BaseToolSpec):
    """
    Seltz web knowledge tool spec.

    Seltz provides fast, up-to-date web data with context-engineered
    web content and sources for real-time AI reasoning.
    """

    spec_functions = ["search"]

    def __init__(self, api_key: str) -> None:
        """
        Initialize with parameters.

        Args:
            api_key: Seltz API key. Obtain one at https://www.seltz.ai/

        """
        from seltz import Seltz

        self.client = Seltz(api_key=api_key)

    def search(self, query: str, max_documents: Optional[int] = 10) -> List[Document]:
        """
        Search the web using Seltz and return relevant documents with sources.

        Args:
            query: The search query text.
            max_documents: Maximum number of documents to return (default: 10).

        Returns:
            A list of Document objects containing web content and source URLs.

        """
        from seltz import Includes

        includes = Includes(max_documents=max_documents) if max_documents else None
        response = self.client.search(query, includes=includes)
        return [
            Document(text=doc.content or "", metadata={"url": doc.url or ""})
            for doc in response.documents
        ]
