"""Mongo client."""

from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class SimpleMongoReader(BaseReader):
    """Simple mongo reader.

    Concatenates each Mongo doc into Document used by LlamaIndex.

    Args:
        host (str): Mongo host.
        port (int): Mongo port.
        max_docs (int): Maximum number of documents to load.

    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        uri: Optional[str] = None,
        max_docs: int = 1000,
    ) -> None:
        """Initialize with parameters."""
        try:
            import pymongo  # noqa: F401
            from pymongo import MongoClient  # noqa: F401
        except ImportError:
            raise ImportError(
                "`pymongo` package not found, please run `pip install pymongo`"
            )
        if uri:
            if uri is None:
                raise ValueError("Either `host` and `port` or `uri` must be provided.")
            self.client: MongoClient = MongoClient(uri)
        else:
            if host is None or port is None:
                raise ValueError("Either `host` and `port` or `uri` must be provided.")
            self.client = MongoClient(host, port)
        self.max_docs = max_docs

    def load_data(
        self,
        db_name: str,
        collection_name: str,
        field_names: List[str] = ["text"],
        field_names_metadata: Optional[List[str]] = None,
        query_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            db_name (str): name of the database.
            collection_name (str): name of the collection.
            field_names(List[str]): names of the fields to be concatenated.
                Defaults to ["text"]
            field_names_metadata (Optional[List[str]]): field names for 
                metadata to be included in Document. e.g. "_id".
                Defaults to None
            query_dict (Optional[Dict]): query to filter documents.
                Defaults to None

        Returns:
            List[Document]: A list of documents.

        """
        documents = []
        db = self.client[db_name]
        field_names_metadata = field_names_metadata or []
        if query_dict is None:
            cursor = db[collection_name].find()
        else:
            cursor = db[collection_name].find(query_dict)

        for item in cursor:
            text = ""
            for field_name in field_names:
                if field_name not in item:
                    raise ValueError(
                        f"`{field_name}` field not found in Mongo document."
                    )
                text += item[field_name]

            # Metadata
            extra_info = {
                field_name: item[field_name] for field_name in field_names_metadata
            }
            # Convert mongodb id (ObjectId) to string representation
            if "_id" in extra_info:
                extra_info["_id"] = str(extra_info["_id"])

            documents.append(Document(text, extra_info=extra_info))

        return documents
