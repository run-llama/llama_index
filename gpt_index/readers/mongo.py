"""Mongo client."""

from typing import List, Dict, Optional

from pymongo import MongoClient

from gpt_index.readers.base import BaseReader
from gpt_index.schema import Document


class SimpleMongoReader(BaseReader):
    """Simple mongo reader.

    Concatenates all files into one document text.

    """

    def __init__(self, host: str, port: int, max_docs: int = 1000) -> None:
        """Initialize with parameters."""
        self.client = MongoClient(host, port)
        self.max_docs = max_docs

    def load_data(
        self, db_name: str, collection_name: str, query_dict: Optional[Dict] = None
    ) -> List[Document]:
        """Load data from the input directory."""

        documents = []
        db = self.client[db_name]
        if query_dict is None:
            cursor = db[collection_name].find()
        else:
            cursor = db[collection_name].find(query_dict)

        for item in cursor:
            if "text" not in item:
                raise ValueError("`text` field not found in Mongo document.")
            documents.append(Document(item["text"]))
        return documents
