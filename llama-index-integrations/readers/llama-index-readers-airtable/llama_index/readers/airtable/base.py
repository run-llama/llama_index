"""Airtable reader."""

from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pyairtable import Table


class AirtableReader(BaseReader):
    """
    Airtable reader. Reads data from a table in a base.

    Args:
        api_key (str): Airtable API key.

    """

    def __init__(self, api_key: str) -> None:
        """Initialize Airtable reader."""
        self.api_key = api_key

    def load_data(self, base_id: str, table_id: str) -> List[Document]:
        """
        Load data from a table in a base.

        Args:
            table_id (str): Table ID.
            base_id (str): Base ID.


        Returns:
            List[Document]: List of documents.

        """
        table = Table(self.api_key, base_id, table_id)
        all_records = table.all()
        return [Document(text=f"{all_records}", extra_info={})]
