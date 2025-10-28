from typing import Dict, List, Any, Tuple
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import trino
import logging

logger = logging.getLogger(__name__)


class TrinoReader(BaseReader):
    """

    Trino database reader.

    Loads data from a Trino cluster into Document used by LlamaIndex.

    Args:
        host (str): server that's running Trino
        schema (str): reside within a catalog and serve as a wa to organize tables and other database objects
        port (int): network port number used for communication with a Trino cluster
        catalog (str): A catalog in trino specifies a connector



    """

    def __init__(
        self,
        user: str,
        schema: str,
        host: str,
        port: int = 8080,
        catalog: str = "hive",
        **kwargs: Any,
    ) -> None:
        """Initialize with Trino connection parameters."""
        # Store connection parameters (self.host, self.port, etc.)
        # self.conn_params = {...}

        self.host = host
        self.port = port
        self.catalog = catalog
        self.user = user
        self.schema = schema
        self._conn = None
        self._cursor = None

        self._conn_paramse = {
            "host": host,
            "port": port,
            "catalog": catalog,
            "user": user,
            "schema": schema,
        }

    def configure_Connection(self) -> Tuple[trino.dbapi.Connection, trino.dbapi.Cursor]:
        """
        Configure Connection for Trino Datawarehouse
        """
        if self._conn is None or self._conn.closed:
            try:
                self._conn = trino.dbapi.connect(
                    host=self.host,
                    port=self.port,
                    catalog=self.catalog,
                    user=self.user,
                    schema=self.schema,
                )

                self._cursor = self._conn.cursor()
            except trino.dbapi.DatabaseError as e:
                print(f"Trino connection failed:{e}")
                raise
        return self._conn, self._cursor

    def execute_query(self, query: str):
        """
        Executes Query againg Trino instance

        Args:
            query (str): The SQL++ query to execute.
            conn (trino.dbapi.Connection) The trino connection used to build the cursor
            cursor (trino.dbapi.Cursor) an Object used to execute SQL queries against Trino

        """
        try:
            self._cursor.execute(query)

            rows = self._cursor.fetchall()
            return [rows, self._cursor.description]
        except trino.dbapi.DatabaseError as e:
            print(f"Trino connection failed: {e}")
            raise

    def load_data(self, query: str) -> List[Document]:
        """
        Loads data from Trino by executing a single SQL query.

        Args:
            query: The SQL query to execute against the Trino cluster.

        """
        all_documents = []

        conn = None
        cur = None
        try:
            conn, cur = self.configure_Connection()
            if not conn or not cur:
                logger.warning("Could not establish connection; returning empty list")
                return []
            # 1. Connect to Trino using self.conn_params

            results = self.execute_query(query, conn, cur)

            column_names = [desc[0] for desc in results[1]]

            if results[0] is None:
                return []

            # 3 Document Transformation
            for row_index, row in enumerate(results[0]):
                # Ensure all elements in row are cast to str before joining for content
                content = ", ".join(
                    f"{name}: {value}" for name, value in zip(column_names, row)
                )

                # Metadata must be a mapping (Dict[str, Any])
                metadata: Dict[str, Any] = dict(zip(column_names, row))
                metadata.update(
                    {
                        "source": "raw_data",
                        "catalog": self.catalog,
                        "schema": self.schema,
                        "row_id": row_index,
                    }
                )

                all_documents.append(Document(text=content, metadata=metadata))

            return all_documents

        except Exception as e:
            logger.error(f"FATAL ERROR during data loading: {e}")
            raise
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
