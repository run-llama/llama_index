"""Couchbase document loader."""

from typing import Any, Iterable, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class CouchbaseReader(BaseReader):
    """
    Couchbase document loader.

    Loads data from a Couchbase cluster into Document used by LlamaIndex.

    Args:
        client(Optional[Any]): A Couchbase client to use.
            If not provided, the client will be created based on the connection_string
            and database credentials.
        connection_string (Optional[str]): The connection string to the Couchbase cluster.
        db_username (Optional[str]): The username to connect to the Couchbase cluster.
        db_password (Optional[str]): The password to connect to the Couchbase cluster.

    """

    def __init__(
        self,
        client: Optional[Any] = None,
        connection_string: Optional[str] = None,
        db_username: Optional[str] = None,
        db_password: Optional[str] = None,
    ) -> None:
        """Initialize Couchbase document loader."""
        import_err_msg = "`couchbase` package not found, please run `pip install --upgrade couchbase`"
        try:
            from couchbase.auth import PasswordAuthenticator
            from couchbase.cluster import Cluster
            from couchbase.options import ClusterOptions
        except ImportError:
            raise ImportError(import_err_msg)

        if not client:
            if not connection_string or not db_username or not db_password:
                raise ValueError(
                    "You need to pass either a couchbase client or connection_string and credentials must be provided."
                )
            else:
                auth = PasswordAuthenticator(
                    db_username,
                    db_password,
                )

                self._client: Cluster = Cluster(connection_string, ClusterOptions(auth))
        else:
            self._client = client

    def lazy_load_data(
        self,
        query: str,
        text_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = [],
    ) -> Iterable[Document]:
        """
        Load data from the Couchbase cluster lazily.

        Args:
            query (str): The SQL++ query to execute.
            text_fields (Optional[List[str]]): The columns to write into the
                `text` field of the document. By default, all columns are
                written.
            metadata_fields (Optional[List[str]]): The columns to write into the
                `metadata` field of the document. By default, no columns are written.

        """
        from datetime import timedelta

        if not query:
            raise ValueError("Query must be provided.")

        # Ensure connection to Couchbase cluster
        self._client.wait_until_ready(timedelta(seconds=5))

        # Run SQL++ Query
        result = self._client.query(query)
        for row in result:
            if not text_fields:
                text_fields = list(row.keys())

            metadata = {field: row[field] for field in metadata_fields}

            document = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in text_fields
            )

            yield (Document(text=document, metadata=metadata))

    def load_data(
        self,
        query: str,
        text_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load data from the Couchbase cluster.

        Args:
            query (str): The SQL++ query to execute.
            text_fields (Optional[List[str]]): The columns to write into the
                `text` field of the document. By default, all columns are
                written.
            metadata_fields (Optional[List[str]]): The columns to write into the
                `metadata` field of the document. By default, no columns are written.

        """
        return list(self.lazy_load_data(query, text_fields, metadata_fields))
