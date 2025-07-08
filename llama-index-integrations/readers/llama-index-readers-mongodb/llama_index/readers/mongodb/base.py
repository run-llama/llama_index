"""Mongo client."""

from collections.abc import Callable
from typing import Dict, Iterable, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class SimpleMongoReader(BaseReader):
    """
    Simple mongo reader.

    Concatenates each Mongo doc into Document used by LlamaIndex.

    Args:
        host (str): Mongo host.
        port (int): Mongo port.

    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        uri: Optional[str] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from pymongo import MongoClient
        except ImportError as err:
            raise ImportError(
                "`pymongo / motor` package not found, please run `pip install pymongo motor`"
            ) from err

        if uri:
            client_args = (uri,)
        elif host and port:
            client_args = (host, port)
        else:
            raise ValueError("Either `host` and `port` or `uri` must be provided.")

        self.client = MongoClient(*client_args)
        self.async_client = AsyncIOMotorClient(*client_args)

    def lazy_load_data(
        self,
        db_name: str,
        collection_name: str,
        field_names: List[str] = ["text"],
        separator: str = "",
        query_dict: Optional[Dict] = None,
        max_docs: int = 0,
        metadata_names: Optional[List[str]] = None,
        field_extractors: Optional[Dict[str, Callable[..., str]]] = None,
    ) -> Iterable[Document]:
        """
        Lazy load data from MongoDB.

        Args:
            db_name (str): name of the database.
            collection_name (str): name of the collection.
            field_names(List[str]): names of the fields to be concatenated.
                Defaults to ["text"]
            separator (str): separator to be used between fields.
                Defaults to ""
            query_dict (Optional[Dict]): query to filter documents. Read more
                at [docs](https://docs.mongodb.com/manual/reference/method/db.collection.find/)
                Defaults to empty dict
            max_docs (int): maximum number of documents to load.
                Defaults to 0 (no limit)
            metadata_names (Optional[List[str]]): names of the fields to be added
                to the metadata attribute of the Document. Defaults to None
            field_extractors (Optional[Dict[str, Callable[..., str]]]): a dictionary
                of functions to use when extracting a field from a document.
                Defaults to None

        Yields:
            Document: a document object with the concatenated text and metadata.

        Raises:
            ValueError: if a field is not found in a document.

        """
        db = self.client[db_name]
        cursor = db[collection_name].find(
            filter=query_dict or {},
            limit=max_docs,
            projection=dict.fromkeys(field_names + (metadata_names or []), 1),
        )

        field_extractors = field_extractors or {}

        for item in cursor:
            try:
                texts = [
                    field_extractors.get(name, str)(item[name]) for name in field_names
                ]
            except KeyError as err:
                raise ValueError(
                    f"{err.args[0]} field not found in Mongo document."
                ) from err

            text = separator.join(texts)

            if metadata_names is None:
                yield Document(text=text, id_=str(item["_id"]))
            else:
                try:
                    metadata = {name: item.get(name) for name in metadata_names}
                    metadata["collection"] = collection_name
                except KeyError as err:
                    raise ValueError(
                        f"{err.args[0]} field not found in Mongo document."
                    ) from err
                yield Document(text=text, id_=str(item["_id"]), metadata=metadata)

    async def alazy_load_data(
        self,
        db_name: str,
        collection_name: str,
        field_names: List[str] = ["text"],
        separator: str = "",
        query_dict: Optional[Dict] = None,
        max_docs: int = 0,
        metadata_names: Optional[List[str]] = None,
        field_extractors: Optional[Dict[str, Callable[..., str]]] = None,
    ):
        """
        Asynchronously lazy load data from a MongoDB collection.

        Args:
            db_name (str): The name of the database to connect to.
            collection_name (str): The name of the collection to query.
            field_names (List[str]): The fields to concatenate into the document's text. Defaults to ["text"].
            separator (str): The separator to use between concatenated fields. Defaults to "".
            query_dict (Optional[Dict]): A dictionary to filter documents. Defaults to None.
            max_docs (int): The maximum number of documents to load. Defaults to 0 (no limit).
            metadata_names (Optional[List[str]]): The fields to include in the document's metadata. Defaults to None.
            field_extractors (Optional[Dict[str, Callable[..., str]]]): A dictionary of field-specific extractor functions. Defaults to None.

        Yields:
            Document: An asynchronous generator of Document objects with concatenated text and optional metadata.

        Raises:
            ValueError: If the async_client is not initialized or if a specified field is not found in a document.

        """
        db = self.async_client[db_name]
        cursor = db[collection_name].find(
            filter=query_dict or {},
            limit=max_docs,
            projection=dict.fromkeys(field_names + (metadata_names or []), 1),
        )

        field_extractors = field_extractors or {}

        async for item in cursor:
            try:
                texts = [
                    field_extractors.get(name, str)(item[name]) for name in field_names
                ]
            except KeyError as err:
                raise ValueError(
                    f"{err.args[0]} field not found in Mongo document."
                ) from err

            text = separator.join(texts)

            if metadata_names is None:
                yield Document(text=text, id_=str(item["_id"]))
            else:
                try:
                    metadata = {name: item.get(name) for name in metadata_names}
                    metadata["collection"] = collection_name
                except KeyError as err:
                    raise ValueError(
                        f"{err.args[0]} field not found in Mongo document."
                    ) from err
                yield Document(text=text, id_=str(item["_id"]), metadata=metadata)
