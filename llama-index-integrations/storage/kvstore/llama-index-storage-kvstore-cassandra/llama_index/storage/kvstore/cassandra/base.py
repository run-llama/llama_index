"""Cassandra Key-Value Store implementation for LlamaIndex."""

import json
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

IMPORT_ERROR_MSG = "`cassandra-driver` package not found, please run `pip install cassandra-driver`"

DEFAULT_KEYSPACE = "llama_index_kvstore"
DEFAULT_TABLE = "kv_store"


class CassandraKVStore(BaseKVStore):
    """
    Cassandra Key-Value store.

    Args:
        session (Any): Cassandra session
        keyspace (Optional[str]): Cassandra keyspace name
        table (Optional[str]): Table name for key-value storage
        contact_points (Optional[List[str]]): Cassandra contact points
        port (Optional[int]): Cassandra port
        username (Optional[str]): Username for authentication
        password (Optional[str]): Password for authentication

    """

    def __init__(
        self,
        session: Any = None,
        keyspace: Optional[str] = None,
        table: Optional[str] = None,
        contact_points: Optional[List[str]] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Init a CassandraKVStore."""
        # Check for availability
        try:
            import cassandra  # noqa: F401
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        self._keyspace = keyspace or DEFAULT_KEYSPACE
        self._table = table or DEFAULT_TABLE

        # Initialize session
        if session is not None:
            self._session = session
            self._owns_session = False
        else:
            self._session = self._create_session(
                contact_points or ["127.0.0.1"],
                port or 9042,
                username,
                password,
            )
            self._owns_session = True

        # Ensure keyspace and table exist
        self._ensure_keyspace_and_table()

    def _create_session(
        self,
        contact_points: List[str],
        port: int,
        username: Optional[str],
        password: Optional[str],
    ) -> Any:
        """Create a Cassandra session."""
        from cassandra.auth import PlainTextAuthProvider
        from cassandra.cluster import Cluster

        auth_provider = None
        if username and password:
            auth_provider = PlainTextAuthProvider(username=username, password=password)

        cluster = Cluster(contact_points, port=port, auth_provider=auth_provider)
        return cluster.connect()

    def _ensure_keyspace_and_table(self) -> None:
        """Create keyspace and table if they don't exist."""
        # Create keyspace
        self._session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {self._keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """
        )

        # Use keyspace
        self._session.set_keyspace(self._keyspace)

        # Create table
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                collection text,
                key text,
                value text,
                PRIMARY KEY (collection, key)
            )
            """
        )

        # Prepare statements for performance
        self._put_stmt = self._session.prepare(
            f"INSERT INTO {self._table} (collection, key, value) VALUES (?, ?, ?)"
        )
        self._get_stmt = self._session.prepare(
            f"SELECT value FROM {self._table} WHERE collection = ? AND key = ?"
        )
        self._get_all_stmt = self._session.prepare(
            f"SELECT key, value FROM {self._table} WHERE collection = ?"
        )
        self._delete_stmt = self._session.prepare(
            f"DELETE FROM {self._table} WHERE collection = ? AND key = ?"
        )
        self._delete_all_stmt = self._session.prepare(
            f"DELETE FROM {self._table} WHERE collection = ?"
        )

    @classmethod
    def from_params(
        cls,
        contact_points: Optional[List[str]] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keyspace: Optional[str] = None,
        table: Optional[str] = None,
    ) -> "CassandraKVStore":
        """
        Load a CassandraKVStore from connection parameters.

        Args:
            contact_points (Optional[List[str]]): Cassandra contact points
            port (Optional[int]): Cassandra port
            username (Optional[str]): Username for authentication
            password (Optional[str]): Password for authentication
            keyspace (Optional[str]): Cassandra keyspace name
            table (Optional[str]): Table name for key-value storage

        """
        return cls(
            contact_points=contact_points,
            port=port,
            username=username,
            password=password,
            keyspace=keyspace,
            table=table,
        )

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self.put_all([(key, val)], collection=collection)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Put multiple key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): list of key-value pairs
            collection (str): collection name
            batch_size (int): batch size for insertion

        """
        from cassandra.query import BatchStatement

        # Process in batches
        for i in range(0, len(kv_pairs), batch_size):
            batch = kv_pairs[i : i + batch_size]
            batch_stmt = BatchStatement()
            for key, val in batch:
                # Serialize value to JSON
                val_json = json.dumps(val)
                batch_stmt.add(self._put_stmt, (collection, key, val_json))
            self._session.execute(batch_stmt)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        Returns:
            Optional[dict]: value if found, None otherwise

        """
        result = self._session.execute(self._get_stmt, (collection, key))
        row = result.one()
        if row is not None:
            return json.loads(row.value)
        return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        Returns:
            Dict[str, dict]: all key-value pairs in the collection

        """
        result = self._session.execute(self._get_all_stmt, (collection,))
        output = {}
        for row in result:
            output[row.key] = json.loads(row.value)
        return output

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        Returns:
            bool: True if deleted, False if not found

        """
        result = self._session.execute(self._delete_stmt, (collection, key))
        # Cassandra doesn't return deleted count for DELETE, so we check before/after
        return True  # Best effort

    def close(self) -> None:
        """Close the Cassandra session and cluster."""
        if self._owns_session and self._session is not None:
            cluster = self._session.cluster
            self._session.shutdown()
            cluster.shutdown()
            self._session = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    # Async support (using the same sync implementation for now)
    # In the future, this could be enhanced with async cassandra driver
    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Async put - delegates to sync for now."""
        self.put(key, val, collection=collection)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Async put_all - delegates to sync for now."""
        self.put_all(kv_pairs, collection=collection, batch_size=batch_size)

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """Async get - delegates to sync for now."""
        return self.get(key, collection=collection)

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Async get_all - delegates to sync for now."""
        return self.get_all(collection=collection)

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Async delete - delegates to sync for now."""
        return self.delete(key, collection=collection)
