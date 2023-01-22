"""PostgreSQL client."""

from typing import Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document

class PostgreSQLReader(BaseReader):
    """Simple PostgreSQL reader.

    Concatenates each PostgreSQL row into Document used by GPT Index.

    Args:
        host (str): PostgreSQL host.
        port (int): PostgreSQL port.
        user (str): PostgreSQL user.
        password (str): PostgreSQL password.
        dbname (str): PostgreSQL dbname.
        max_docs (int): Maximum number of documents to load.

    """

    def __init__(
        self, host: str, port: int, user: str, password: str, dbname: str, max_docs: int = 1000
    ) -> None:
        """Initialize with parameters."""
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ValueError(
                "`psycopg2` package not found, please run `pip install psycopg2`"
            )
        self.connection = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=dbname
        )
        self.max_docs = max_docs

    def load_data(self, query: str) -> List[Document]:
        """Load data from the input table.

        Args:
            query (str): Query parameter (PostgreSQL Query) to filter rows.

        Returns:
            List[Document]: A list of documents.

        """
        documents = []
        cursor = self.connection.cursor()
        if query is None:
            raise ValueError("A query parameter (PostgreSQL Query) is necessary to filter the data")
        else:
            cursor.execute(query)

        for item in cursor.fetchall():
            documents.append(Document(item[0]))
        return documents