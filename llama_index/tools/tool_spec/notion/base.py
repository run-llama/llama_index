"""Notion tool spec."""

from llama_index.tools.tool_spec.base import BaseToolSpec
from llama_index.readers.notion import NotionPageReader
from typing import Optional, List


class NotionToolSpec(BaseToolSpec):
    """Notion tool spec.

    Currently a simple wrapper around the data loader.
    TODO: add more methods to the Notion spec.

    """

    spec_functions = ["load_data"]

    def __init__(self, integration_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        self.reader = NotionPageReader(integration_token=integration_token)

    def load_data(
        self, page_ids: Optional[List[str]] = None, database_id: Optional[str] = None
    ) -> str:
        """load data from the input directory.

        Args:
            page_ids (List[str]): List of page ids to load.
            database_id (str): Database id to load.

        Returns:
            str: Loaded data.

        """
        page_ids = page_ids or []
        docs = self.reader.load_data(page_ids=page_ids, database_id=database_id)
        return "\n".join([doc.get_text() for doc in docs])
