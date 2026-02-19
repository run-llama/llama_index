"""iGPT Email Intelligence reader."""

import json
from typing import List, Optional

from igptai import IGPT

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class IGPTEmailReader(BaseReader):
    """
    iGPT Email Intelligence Reader.

    Loads structured, reasoning-ready email context from the iGPT API as
    LlamaIndex Documents for indexing and retrieval.

    Args:
        api_key (str): iGPT API key. See https://docs.igpt.ai for details.
        user (str): User identifier for the connected mailbox.

    Example:
        .. code-block:: python

            from llama_index.readers.igpt_email import IGPTEmailReader
            from llama_index.core import VectorStoreIndex

            reader = IGPTEmailReader(api_key="your-key", user="user-id")
            documents = reader.load_data(
                query="project Alpha", date_from="2025-01-01"
            )
            index = VectorStoreIndex.from_documents(documents)

    """

    def __init__(self, api_key: str, user: str) -> None:
        """Initialize with parameters."""
        super().__init__()
        self.client = IGPT(api_key=api_key, user=user)

    def load_data(
        self,
        query: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Document]:
        """
        Load email context as Documents from iGPT recall.search().

        Each result from the iGPT API is returned as a separate Document.
        Thread metadata (subject, participants, date, thread ID) is preserved
        in metadata for filtering and attribution during retrieval.

        Args:
            query (str): Search query to run against connected email data.
            date_from (str, optional): Filter results from this date (YYYY-MM-DD).
            date_to (str, optional): Filter results up to this date (YYYY-MM-DD).
            max_results (int): Maximum number of results to return. Default is 50.

        Returns:
            List[Document]: One Document per email result, ready for indexing.
                Thread metadata is stored in metadata (subject, from, to,
                date, thread_id, id).

        """
        response = self.client.recall.search(
            query=query,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
        )

        if isinstance(response, dict) and "error" in response:
            raise ValueError(f"iGPT API error: {response['error']}")

        if not response:
            return []

        results = (
            response if isinstance(response, list) else response.get("results", [])
        )

        documents = []
        for item in results:
            if isinstance(item, dict):
                text = item.get("content", item.get("body", json.dumps(item)))
                metadata = {
                    "source": "igpt_email",
                    "subject": item.get("subject"),
                    "from": item.get("from"),
                    "to": item.get("to"),
                    "date": item.get("date"),
                    "thread_id": item.get("thread_id"),
                    "id": item.get("id"),
                }
            else:
                text = str(item)
                metadata = {"source": "igpt_email"}

            documents.append(Document(text=text, metadata=metadata))

        return documents
