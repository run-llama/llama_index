"""Mongo client."""

from typing import Any, Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class SimpleMongoReader(BaseReader):
    """Simple mongo reader.

    Concatenates each Mongo doc into Document used by GPT Index.

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

    def _load_data(
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

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            db_name (str): name of the database.
            collection_name (str): name of the collection.

        Returns:
            List[Document]: A list of documents.

        """
        if "db_name" not in load_kwargs:
            raise ValueError("`db_name` not found in load_kwargs.")
        else:
            db_name = load_kwargs["db_name"]
        if "collection_name" not in load_kwargs:
            raise ValueError("`collection_name` not found in load_kwargs.")
        else:
            collection_name = load_kwargs["collection_name"]
        query_dict = load_kwargs.get("query_dict", None)
        return self._load_data(db_name, collection_name, query_dict=query_dict)
