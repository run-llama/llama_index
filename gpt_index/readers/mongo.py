"""Mongo client."""

from typing import Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class SimpleMongoReader(BaseReader):
    """Simple mongo reader.

    Concatenates each Mongo doc into Document used by LlamaIndex.

    Args:
        host (str): Mongo host.
        port (int): Mongo port.
        max_docs (int): Maximum number of documents to load.

    """

    def __init__(self, host: str, port: int, max_docs: int = 1000) -> None:
        """Initialize with parameters."""
        try:
            import pymongo  # noqa: F401
            from pymongo import MongoClient  # noqa: F401
        except ImportError:
            raise ValueError(
                "`pymongo` package not found, please run `pip install pymongo`"
            )
        self.client: MongoClient = MongoClient(host, port)
        self.max_docs = max_docs

    def load_data(
        self, db_name: str, collection_name: str, query_dict: Optional[Dict] = None
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            db_name (str): name of the database.
            collection_name (str): name of the collection.
            query_dict (Optional[Dict]): query to filter documents.
                Defaults to None

        Returns:
            List[Document]: A list of documents.

        """
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
