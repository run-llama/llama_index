"""ArangoDB client."""

from typing import Any, Dict, Iterator, List, Optional, Union, cast

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class SimpleArangoDBReader(BaseReader):
    """
    Simple arangodb reader.
    Concatenates each ArangoDB doc into Document used by LlamaIndex.

    Args:
        host: (Union[str, List[str]]) list of urls or url for connecting to the db
        client: (Any) ArangoDB client

    """

    def __init__(
        self, host: Optional[Union[str, List[str]]] = None, client: Optional[Any] = None
    ) -> None:
        """Initialize with parameters."""
        try:
            from arango import ArangoClient
        except ImportError as err:
            raise ImportError(
                "`arango` package not found, please run `pip install python-arango`"
            ) from err

        host = host or "http://127.0.0.1:8529"
        self.client = client or ArangoClient(hosts=host)
        self.client = cast(ArangoClient, self.client)

    def _flatten(self, texts: List[Union[str, List[str]]]) -> List[str]:
        result = []
        for text in texts:
            result += text if isinstance(text, list) else [text]
        return result

    def lazy_load(
        self,
        username: str,
        password: str,
        db_name: str,
        collection_name: str,
        field_names: List[str] = ["text"],
        separator: str = " ",
        query_dict: Optional[Dict] = {},
        max_docs: int = None,
        metadata_names: Optional[List[str]] = None,
    ) -> Iterator[Document]:
        """
        Lazy load data from ArangoDB.

        Args:
            username (str): for credentials.
            password (str): for credentials.
            db_name (str): name of the database.
            collection_name (str): name of the collection.
            field_names(List[str]): names of the fields to be concatenated.
                Defaults to ["text"]
            separator (str): separator to be used between fields.
                Defaults to " "
            query_dict (Optional[Dict]): query to filter documents. Read more
            at [docs](https://docs.python-arango.com/en/main/specs.html#arango.collection.StandardCollection.find)
                Defaults to empty dict
            max_docs (int): maximum number of documents to load.
                Defaults to None (no limit)
            metadata_names (Optional[List[str]]): names of the fields to be added
                to the metadata attribute of the Document. Defaults to None
        Returns:
            List[Document]: A list of documents.

        """
        db = self.client.db(name=db_name, username=username, password=password)
        collection = db.collection(collection_name)
        cursor = collection.find(filters=query_dict, limit=max_docs)
        for item in cursor:
            try:
                texts = [str(item[name]) for name in field_names]
            except KeyError as err:
                raise ValueError(
                    f"{err.args[0]} field not found in arangodb document."
                ) from err
            texts = self._flatten(texts)
            text = separator.join(texts)

            if metadata_names is None:
                yield Document(text=text)
            else:
                try:
                    metadata = {name: item[name] for name in metadata_names}
                except KeyError as err:
                    raise ValueError(
                        f"{err.args[0]} field not found in arangodb document."
                    ) from err
                yield Document(text=text, metadata=metadata)

    def load_data(
        self,
        username: str,
        password: str,
        db_name: str,
        collection_name: str,
        field_names: List[str] = ["text"],
        separator: str = " ",
        query_dict: Optional[Dict] = {},
        max_docs: int = None,
        metadata_names: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load data from the ArangoDB.

        Args:
            username (str): for credentials.
            password (str): for credentials.
            db_name (str): name of the database.
            collection_name (str): name of the collection.
            field_names(List[str]): names of the fields to be concatenated.
                Defaults to ["text"]
            separator (str): separator to be used between fields.
                Defaults to ""
            query_dict (Optional[Dict]): query to filter documents. Read more
            at [docs](https://docs.python-arango.com/en/main/specs.html#arango.collection.StandardCollection.find)
                Defaults to empty dict
            max_docs (int): maximum number of documents to load.
                Defaults to 0 (no limit)
            metadata_names (Optional[List[str]]): names of the fields to be added
                to the metadata attribute of the Document. Defaults to None
        Returns:
            List[Document]: A list of documents.

        """
        return list(
            self.lazy_load(
                username,
                password,
                db_name,
                collection_name,
                field_names,
                separator,
                query_dict,
                max_docs,
                metadata_names,
            )
        )
