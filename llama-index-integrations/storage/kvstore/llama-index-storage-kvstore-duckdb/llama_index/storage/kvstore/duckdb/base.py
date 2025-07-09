import json
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.storage.kvstore.types import (
    DEFAULT_COLLECTION,
    BaseKVStore,
)
import logging
from pathlib import Path
from typing import Any, Optional

import duckdb
from duckdb import (
    ColumnExpression,
    ConstantExpression,
    Expression,
)
import pyarrow

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 128


class DuckDBTableIncorrectColumnsError(Exception):
    def __init__(
        self, table_name: str, expected_columns: list[str], actual_columns: list[str]
    ):
        self.table_name = table_name
        self.expected_columns = expected_columns
        self.actual_columns = actual_columns
        super().__init__(
            f"Table {table_name} has incorrect columns. Expected {expected_columns}, got {actual_columns}."
        )


class DuckDBKVStore(BaseKVStore):
    """
    DuckDB KV Store.

    Args:
        duckdb_uri (str): DuckDB URI
        duckdb_client (Any): DuckDB client
        async_duckdb_client (Any): Async DuckDB client

    Raises:
            ValueError: If duckdb-py is not installed

    Examples:
        >>> from llama_index.storage.kvstore.duckdb import DuckDBKVStore
        >>> # Create a DuckDBKVStore
        >>> duckdb_kv_store = DuckDBKVStore(
        >>>     duckdb_url="duckdb://127.0.0.1:6379")

    """

    database_name: str
    table_name: str
    persist_dir: str

    _conn: Optional[duckdb.DuckDBPyConnection] = None
    _table: Optional[duckdb.DuckDBPyRelation] = None

    def __init__(
        self,
        database_name: str = ":memory:",
        table_name: str = "keyvalue",
        # https://duckdb.org/docs/extensions/full_text_search
        persist_dir: str = "./storage",
        client: Optional[duckdb.DuckDBPyConnection] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Init params."""
        if client is not None:
            self._conn = client.cursor()

        self.database_name = database_name
        self.table_name = table_name
        self.persist_dir = persist_dir

    @classmethod
    def from_vector_store(
        cls, duckdb_vector_store, table_name: str = "keyvalue"
    ) -> "DuckDBKVStore":
        """
        Load a DuckDBKVStore from a DuckDB Client.

        Args:
            client (DuckDB): DuckDB client

        """
        from llama_index.vector_stores.duckdb.base import DuckDBVectorStore

        assert isinstance(duckdb_vector_store, DuckDBVectorStore)

        return cls(
            database_name=duckdb_vector_store.database_name,
            table_name=table_name,
            persist_dir=duckdb_vector_store.persist_dir,
            client=duckdb_vector_store.client,
        )

    @property
    def client(self) -> duckdb.DuckDBPyConnection:
        """Return client."""
        if self._conn is None:
            self._conn = self._connect(self.database_name, self.persist_dir)

        return self._conn

    @classmethod
    def _connect(
        cls, database_name: str, persist_dir: str
    ) -> duckdb.DuckDBPyConnection:
        """Connect to the DuckDB database -- create the data persistence directory if it doesn't exist."""
        database_connection = database_name

        if database_name != ":memory:":
            persist_path = Path(persist_dir)

            if not persist_path.exists():
                persist_path.mkdir(parents=True, exist_ok=True)

            database_connection = str(persist_path / database_name)

        return duckdb.connect(database_connection)

    @property
    def table(self) -> duckdb.DuckDBPyRelation:
        """Return the table."""
        if self._table is None:
            self._table = self._initialize_table(self.client, self.table_name)

        return self._table

    @classmethod
    def _initialize_table(
        cls, conn: duckdb.DuckDBPyConnection, table_name: str
    ) -> duckdb.DuckDBPyRelation:
        """Initialize the DuckDB Database, extensions, and documents table."""
        home_dir = Path.home()
        conn.execute(f"SET home_directory='{home_dir}';")
        conn.install_extension("json")
        conn.load_extension("json")

        _ = conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name}  (
                key VARCHAR,
                collection VARCHAR,
                value JSON,
                PRIMARY KEY (key, collection)
            );

            CREATE INDEX IF NOT EXISTS key_collection_idx ON {table_name} (key, collection);
        """)

        table = conn.table(table_name)

        required_columns = ["key", "value"]
        table_columns = table.describe().columns

        for column in required_columns:
            if column not in table_columns:
                raise DuckDBTableIncorrectColumnsError(
                    table_name, required_columns, table_columns
                )

        return table

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self.put_all([(key, val)], collection)

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self.put(key, val, collection)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self.put_all(kv_pairs, collection, batch_size)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Put a dictionary of key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): collection name

        """
        if len(kv_pairs) == 0:
            return

        rows = [
            {"key": key, "collection": collection, "value": json.dumps(value)}
            for key, value in kv_pairs
        ]
        arrow_table = pyarrow.Table.from_pylist(rows)

        _ = self.client.from_arrow(arrow_table).query(
            "put_all",
            sql_query=f"""
            INSERT OR REPLACE INTO {self.table.alias}
            SELECT * from put_all;
            """,
        )

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        expression: Expression = (
            ColumnExpression("collection")
            .__eq__(ConstantExpression(collection))
            .__and__(ColumnExpression("key").__eq__(ConstantExpression(key)))
        )
        row_result = self.table.filter(filter_expr=expression).fetchone()

        if row_result is None:
            return None

        return json.loads(row_result[2])

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        return self.get(key, collection)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        filter_expr: Expression = ColumnExpression("collection").__eq__(
            ConstantExpression(collection)
        )

        table: pyarrow.Table = self.table.filter(
            filter_expr=filter_expr
        ).fetch_arrow_table()

        as_list = table.to_pylist()

        return {row["key"]: json.loads(row["value"]) for row in as_list}

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        return self.get_all(collection)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        filter_expression = (
            ColumnExpression("collection")
            .__eq__(ConstantExpression(collection))
            .__and__(ColumnExpression("key").__eq__(ConstantExpression(key)))
        )

        command = f"DELETE FROM {self.table.alias} WHERE {filter_expression}"
        _ = self.client.execute(command)
        return True

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        return self.delete(key, collection)
