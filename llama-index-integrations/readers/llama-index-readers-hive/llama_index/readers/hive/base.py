"""Hive data reader."""

from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


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
        try:
            from pyhive import hive
        except ImportError:
            raise ImportError(
                "`hive` package not found, please run `pip install pyhive`"
            )

        self.con = hive.Connection(
            host=host,
            port=port,
            username=username,
            database=database,
            auth=auth,
            password=password,
        )

    def load_data(self, query: str) -> List[Document]:
        """Read data from the Hive.

        Args:
            query (str): The query used to query data from Hive
        Returns:
            List[Document]: A list of documents.

        """
        try:
            cursor = self.con.cursor().execute(query)
            cursor.execute(query)
            rows = cursor.fetchall()
        except Exception:
            raise Exception(
                "Throws Exception in execution, please check your connection params and query "
            )

        documents = []
        for row in rows:
            documents = Document(text=row)
        return documents
