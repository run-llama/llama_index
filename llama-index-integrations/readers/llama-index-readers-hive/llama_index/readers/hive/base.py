"""Hive data reader."""

try:
    from pyhive import hive
except ImportError:
    raise ImportError("`hive` package not found, please run `pip install pyhive`")
try:
    import sqlglot
except ImportError:
    raise ImportError("`sqlglot` package not found, please run `pip install sqlglot`")

from typing import List, Optional, Tuple
from deprecated import deprecated
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class InvalidSqlError(Exception):
    """Raise when invalid SQL is passed."""


def _validate_sql_query(statements: List[str]) -> None:
    if len(statements) > 1:
        raise InvalidSqlError("You cannot pass multiple statements into the query")
    if not statements[0].lower().startswith("select"):
        raise InvalidSqlError("You must provide a SELECT query")
    if "or" in statements[0].lower():
        raise InvalidSqlError("The use of OR is not allowed to prevent SQL Injections")


@deprecated(
    reason="llama-index-readers-hive has been deprecated since v0.3.1 on the grounds of security concerns for SQL query handling, and will thus no longer be maintained. Use this package with caution.",
    version="0.3.1",
)
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

    def load_data(
        self, query: str, params: Optional[Tuple[str, ...]] = None
    ) -> List[Document]:
        """
        Read data from the Hive.

        Args:
            query (str): Query with which to extract data from Hive. Parametrized values must be represented as '%s'.
            params (Optional[Tuple[str, ...]): Parametrized values.

        Returns:
            List[Document]: A list of documents.

        """
        try:
            if params:
                filled_query = query % tuple(repr(p) for p in params)
            else:
                filled_query = query
            parsed_query = sqlglot.parse(filled_query)
            statements = [statement.sql() for statement in parsed_query]
            _validate_sql_query(statements=statements)
            cursor = self.con.cursor()
            cursor.execute(operation=query, parameters=params)
            rows = cursor.fetchall()
        except Exception:
            raise Exception(
                "Throws Exception in execution, please check your connection params and query."
            )

        documents: List[Document] = []
        for row in rows:
            documents.append(Document(text=str(row)))
        return documents
