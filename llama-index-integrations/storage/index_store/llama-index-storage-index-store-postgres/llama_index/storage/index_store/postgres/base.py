from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.postgres import PostgresKVStore


class PostgresIndexStore(KVIndexStore):
    """
    Postgres Index store.

    Args:
        postgres_kvstore (PostgresKVStore): Postgres key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        postgres_kvstore: PostgresKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a PostgresIndexStore."""
        super().__init__(
            postgres_kvstore, namespace=namespace, collection_suffix=collection_suffix
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        namespace: Optional[str] = None,
        table_name: str = "indexstore",
        schema_name: str = "public",
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        collection_suffix: Optional[str] = None,
    ) -> "PostgresIndexStore":
        """Load a PostgresIndexStore from a PostgresURI."""
        postgres_kvstore = PostgresKVStore.from_uri(
            uri=uri,
            table_name=table_name,
            schema_name=schema_name,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
        )
        return cls(postgres_kvstore, namespace, collection_suffix)

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        namespace: Optional[str] = None,
        table_name: str = "indexstore",
        schema_name: str = "public",
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        collection_suffix: Optional[str] = None,
    ) -> "PostgresIndexStore":
        """Load a PostgresIndexStore from a Postgres host and port."""
        postgres_kvstore = PostgresKVStore.from_params(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
            schema_name=schema_name,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
        )
        return cls(postgres_kvstore, namespace, collection_suffix)
