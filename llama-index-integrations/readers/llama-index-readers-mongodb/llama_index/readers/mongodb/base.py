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
            from pymongo import MongoClient
        except ImportError as err:
            raise ImportError(
                "`pymongo` package not found, please run `pip install pymongo`"
            ) from err

        client: MongoClient
        if uri:
            client = MongoClient(uri)
        elif host and port:
            client = MongoClient(host, port)
        else:
            raise ValueError("Either `host` and `port` or `uri` must be provided.")

        self.client = client

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
        Load data from the input directory.

        Args:
            db_name (str): name of the database.
            collection_name (str): name of the collection.
            field_names(List[str]): names of the fields to be concatenated.
                Defaults to ["text"]
            separator (str): separator to be used between fields.
                Defaults to ""
            query_dict (Optional[Dict]): query to filter documents. Read more
            at [official docs](https://www.mongodb.com/docs/manual/reference/method/db.collection.find/#std-label-method-find-query)
                Defaults to None
            max_docs (int): maximum number of documents to load.
                Defaults to 0 (no limit)
            metadata_names (Optional[List[str]]): names of the fields to be added
                to the metadata attribute of the Document. Defaults to None
            field_extractors (Optional[Dict[str, Callable[..., str]]]): dictionary
                containing field name and a function to extract text from the field.
                The default extractor function is `str`. Defaults to None.

        Returns:
            List[Document]: A list of documents.

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
