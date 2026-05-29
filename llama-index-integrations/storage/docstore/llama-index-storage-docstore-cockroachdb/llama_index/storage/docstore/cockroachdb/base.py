"""CockroachDB Document (Node) store for LlamaIndex."""

from __future__ import annotations

from typing import Any

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE

from llama_index.storage.kvstore.cockroachdb import CockroachDBKVStore


class CockroachDBDocumentStore(KVDocumentStore):
    """CockroachDB-backed Document (Node) store.

    Thin wrapper over :class:`CockroachDBKVStore`.
    """

    def __init__(
        self,
        cockroachdb_kvstore: CockroachDBKVStore,
        namespace: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__(cockroachdb_kvstore, namespace=namespace, batch_size=batch_size)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        namespace: str | None = None,
        table_name: str = "docstore",
        schema_name: str = "public",
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = True,
        create_engine_kwargs: dict[str, Any] | None = None,
    ) -> CockroachDBDocumentStore:
        kvstore = CockroachDBKVStore.from_uri(
            uri=uri,
            table_name=table_name,
            schema_name=schema_name,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            create_engine_kwargs=create_engine_kwargs,
        )
        return cls(kvstore, namespace)

    @classmethod
    def from_params(
        cls,
        host: str | None = None,
        port: int | str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        sslmode: str = "verify-full",
        sslrootcert: str | None = None,
        namespace: str | None = None,
        table_name: str = "docstore",
        schema_name: str = "public",
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = True,
        create_engine_kwargs: dict[str, Any] | None = None,
    ) -> CockroachDBDocumentStore:
        kvstore = CockroachDBKVStore.from_params(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            sslmode=sslmode,
            sslrootcert=sslrootcert,
            table_name=table_name,
            schema_name=schema_name,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            create_engine_kwargs=create_engine_kwargs,
        )
        return cls(kvstore, namespace)
