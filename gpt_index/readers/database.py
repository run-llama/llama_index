"""Database Reader."""

from typing import List, Optional
from sqlalchemy import create_engine, text

from gpt_index.readers.schema.base import Document
from gpt_index.langchain_helpers.sql_wrapper import SQLDatabase

class DatabaseReader(SQLDatabase):
    """Simple Database reader.

    Concatenates each row into Document used by GPT Index.

    Args:
        uri (str): uri of the database connection.

        OR

        scheme (str): scheme of the database connection.
        host (str): host of the database connection.
        port (int): port of the database connection.
        user (str): user of the database connection.
        password (str): password of the database connection.
        dbname (str): dbname of the database connection.
        
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
    ) -> None:
        """Initialize with parameters."""

        if uri:
            self._engine = create_engine(uri)
        else:
            if scheme and host and port and user and password and dbname:
                self._engine = create_engine(f"{scheme}://{user}:{password}@{host}:{port}/{dbname}")
            else:
                raise ValueError("You must provide a valid connection URI or a valid set of credentials")

    def load_data(self, query: str) -> List[Document]:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): Query parameter to filter rows.

        Returns:
            List[Document]: A list of documents.

        """
        documents = []
        with self.engine.connect() as connection:
            if query is None:
                raise ValueError("A query parameter is necessary to filter the data")
            else:
                result = connection.execute(text(query))

            for item in result.fetchall():
                documents.append(Document(item[0]))
        return documents