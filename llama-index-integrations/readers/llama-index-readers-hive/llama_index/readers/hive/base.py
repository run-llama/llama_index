"""Hive data reader."""

try:
    from pyhive import hive
except ImportError:
    raise ImportError("`hive` package not found, please run `pip install pyhive`")
try:
    import sqlglot
except ImportError:
    raise ImportError("`sqlglot` package not found, please run `pip install sqlglot`")

from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class InvalidSqlError(Exception):
    """Raise when invalid SQL is passed."""


def _validate_sql_query(statements: List[str]):
    if len(statements) > 1:
        raise InvalidSqlError("You cannot pass multiple statements into the query")
    if "or" in statements[0].lower():
        raise InvalidSqlError("The use of OR is not allowed to prevent SQL Injections")
    return statements[0]


class HiveReader(BaseReader):
    """
    Read documents from a Hive.

    These documents can then be used in a downstream Llama Index data structure.

    Args:
        host : What host HiveServer2 runs on
        port : The port Hive Server runs on. Defaults to 10000.
        auth : The value of hive.server2.authentication used by HiveServer2.
               Defaults to ``NONE``
        database: the database name
        password: Use with auth='LDAP' or auth='CUSTOM' only

    """

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth: Optional[str] = None,
    ):
        """Initialize with parameters."""
        self.con = hive.Connection(
            host=host,
            port=port,
            username=username,
            database=database,
            auth=auth,
            password=password,
        )

    def load_data(self, table: str, conditions: Optional[str] = None) -> List[Document]:
        """
        Read data from the Hive.

        Args:
            table (str): table from where to perform the selection
            conditions (Optional[str]): conditions for the selection
        Returns:
            List[Document]: A list of documents.

        """
        try:
            if not conditions:
                query = f"SELECT * FROM {table}"
            else:
                query = f"SELECT * FROM {table} {conditions}"
            parsed_query = sqlglot.parse(query)
            statements = [statement.sql() for statement in parsed_query]
            query_to_exec = _validate_sql_query(statements)
            cursor = self.con.cursor()
            cursor.execute(query_to_exec)
            rows = cursor.fetchall()
        except Exception:
            raise Exception(
                "Throws Exception in execution, please check your connection params and query."
            )

        documents: List[Document] = []
        for row in rows:
            documents.append(Document(text=str(row)))
        return documents
