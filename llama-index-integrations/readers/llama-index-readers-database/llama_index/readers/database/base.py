"""Database Reader."""

from typing import Any, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import text
from sqlalchemy.engine import Engine


class DatabaseReader(BaseReader):
    """Simple Database reader.

    Concatenates each row into Document used by LlamaIndex.

    Args:
        sql_database (Optional[SQLDatabase]): SQL database to use,
            including table names to specify.
            See :ref:`Ref-Struct-Store` for more details.

        OR

        engine (Optional[Engine]): SQLAlchemy Engine object of the database connection.

        OR

        uri (Optional[str]): uri of the database connection.

        OR

        scheme (Optional[str]): scheme of the database connection.
        host (Optional[str]): host of the database connection.
        port (Optional[int]): port of the database connection.
        user (Optional[str]): user of the database connection.
        password (Optional[str]): password of the database connection.
        dbname (Optional[str]): dbname of the database connection.

    Returns:
        DatabaseReader: A DatabaseReader object.
    """

    def __init__(
        self,
        sql_database: Optional[SQLDatabase] = None,
        engine: Optional[Engine] = None,
        uri: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        if sql_database:
            self.sql_database = sql_database
        elif engine:
            self.sql_database = SQLDatabase(engine, *args, **kwargs)
        elif uri:
            self.uri = uri
            self.sql_database = SQLDatabase.from_uri(uri, *args, **kwargs)
        elif scheme and host and port and user and password and dbname:
            uri = f"{scheme}://{user}:{password}@{host}:{port}/{dbname}"
            self.uri = uri
            self.sql_database = SQLDatabase.from_uri(uri, *args, **kwargs)
        else:
            raise ValueError(
                "You must provide either a SQLDatabase, "
                "a SQL Alchemy Engine, a valid connection URI, or a valid "
                "set of credentials."
            )

    def load_data(self, query: str, metadata: dict=None, doc_cols: list=None, metadata_cols: list=None) -> List[Document]:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): Query parameter to filter tables and rows.
            metadata (dict): Metadata to be added to the documents. Default is None, which means no metadata.
            doc_cols (list): Columns to be used as documents. Default is None, which means all columns.
            metadata_cols (list): Columns to be used as metadata. Default is None, which means no columns as metadata.

        Returns:
            List[Document]: A list of Document objects.
        """
        documents = []
        if not metadata:
            metadata = {}
        with self.sql_database.engine.connect() as connection:
            if query is None:
                raise ValueError("A query parameter is necessary to filter the data")
            else:
                result = connection.execute(text(query))

            for item in result.fetchall():
                # fetch each item
                doc_str = ""
                for col, entry in zip(result.keys(), item):
                    if doc_cols is None or col in doc_cols:
                        doc_str += ", ".join([f"{col}: {entry}"])
                    if metadata_cols and col in metadata_cols:
                        metadata[col] = entry

                documents.append(Document(text=doc_str, metadata=metadata))
        return documents
